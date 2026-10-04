import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from mlops_playground.features import make_daily_orders, make_daily_raw_features, make_features
from mlops_playground.pipeline import FeatureSelector

ROOT = Path(__file__).resolve().parent


def file_md5(path):
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(path):
    data = pd.read_parquet(path, dtype_backend="pyarrow", use_threads=False)
    data["created_at"] = pd.to_datetime(data["created_at"])
    unique_orders = (
        data[["increment_id", "created_at"]]
        .dropna()
        .drop_duplicates(subset="increment_id")
        .sort_values("created_at")
    )
    cumulative = unique_orders.groupby("created_at").size().sort_index().cumsum()
    if cumulative.empty:
        raise ValueError("No orders found in the dataset")
    train_end = cumulative[cumulative >= len(unique_orders) * 0.70].index[0]
    validation_end = cumulative[cumulative >= len(unique_orders) * 0.85].index[0]
    if train_end >= validation_end:
        raise ValueError("The dataset needs separate training and validation dates")
    available = data[data["created_at"] <= validation_end].copy()
    orders = make_daily_orders(available)
    features = make_features(orders, make_daily_raw_features(available)).drop(columns="orders")
    train_mask = features.index <= train_end
    names = features.loc[train_mask].dropna(axis=1, how="all").columns.tolist()
    features = features[names].astype("float64")
    if train_mask.sum() < 29 or (~train_mask).sum() < 2:
        raise ValueError("At least 29 training days and two validation days are required")
    return features.loc[train_mask], orders.loc[train_mask], features.loc[~train_mask], orders.loc[~train_mask]


def build_pipeline(features, max_iter):
    return Pipeline([
        ("preprocessing", FeatureSelector(list(features))),
        ("model", HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.03,
            max_iter=max_iter,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=5,
            early_stopping=False,
            random_state=42,
        )),
    ])


def get_champion(client, model_name):
    try:
        return client.get_model_version_by_alias(model_name, "champion")
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/cleaned_types.parquet")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.localhost"))
    parser.add_argument("--experiment-name", default="daily-orders-training")
    parser.add_argument("--model-name", default="daily-orders")
    parser.add_argument("--run-name")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--min-gain", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/block-b")
    args = parser.parse_args()
    if args.max_iter < 1:
        parser.error("--max-iter must be positive")
    if not math.isfinite(args.min_gain) or args.min_gain <= 0:
        parser.error("--min-gain must be finite and positive")
    return args


def main():
    args = parse_args()
    data_path = args.data.resolve()
    data_md5 = file_md5(data_path)
    X_train, y_train, X_val, y_val = prepare_data(data_path)
    if file_md5(data_path) != data_md5:
        raise ValueError("The dataset changed while being read")
    print(f"DATA: {data_path.name}, data_md5={data_md5}", flush=True)
    print(f"SPLIT: train_days={len(y_train)}, validation_days={len(y_val)}, features={len(X_train.columns)}", flush=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    client = MlflowClient()
    champion = get_champion(client, args.model_name)
    champion_mae = None
    if champion is not None:
        champion_pipeline = mlflow.sklearn.load_model(f"models:/{args.model_name}/{champion.version}")
        champion_predictions = champion_pipeline.predict(X_val)
        if not np.isfinite(champion_predictions).all():
            raise ValueError("Champion returned non-finite predictions")
        champion_mae = float(mean_absolute_error(y_val, champion_predictions))
    with mlflow.start_run(run_name=args.run_name or f"max-iter-{args.max_iter}") as run:
        output = args.output_dir / run.info.run_id
        output.mkdir(parents=True, exist_ok=True)
        pipeline = build_pipeline(X_train.columns, args.max_iter)
        mlflow.log_params({
            **pipeline.named_steps["model"].get_params(),
            "data_md5": data_md5,
            "data_file": data_path.relative_to(ROOT).as_posix() if data_path.is_relative_to(ROOT) else data_path.name,
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "train_end": str(y_train.index.max().date()),
            "validation_start": str(y_val.index.min().date()),
            "validation_end": str(y_val.index.max().date()),
            "train_days": len(y_train),
            "validation_days": len(y_val),
            "n_features": len(X_train.columns),
            "gate_metric": "validation_mae",
            "min_gain": args.min_gain,
            "evaluation": "one_step_daily_forecast",
        })
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)
        if not np.isfinite(predictions).all():
            raise ValueError("Candidate returned non-finite predictions")
        candidate_mae = float(mean_absolute_error(y_val, predictions))
        gain = None if champion_mae is None else champion_mae - candidate_mae
        passed = champion_mae is None or gain >= args.min_gain
        mlflow.log_metrics({
            "train_mae": float(mean_absolute_error(y_train, pipeline.predict(X_train))),
            "validation_mae": candidate_mae,
            "validation_rmse": float(math.sqrt(mean_squared_error(y_val, predictions))),
            "validation_r2": float(r2_score(y_val, predictions)),
            "gate_passed": int(passed),
        })
        if champion_mae is not None:
            mlflow.log_metrics({"champion_mae": champion_mae, "mae_gain": gain})
        metadata = {
            "model_version": run.info.run_id,
            "features": list(X_train.columns),
            "threshold": None,
            "task": "daily_orders_regression",
            "target": "orders",
            "data_md5": data_md5,
            "gate": {"metric": "validation_mae", "direction": "lower", "min_gain": args.min_gain},
            "evaluation": "one_step_daily_forecast",
        }
        metadata_path = output / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        figure = Figure(figsize=(12, 7))
        axes = figure.subplots(2, 1, sharex=True)
        axes[0].plot(y_val.index, y_val, label="Actual orders")
        axes[0].plot(y_val.index, predictions, label="Predicted orders")
        axes[0].set_ylabel("Orders per day")
        axes[0].set_title(f"Daily orders: validation MAE = {candidate_mae:.2f}, max_iter = {args.max_iter}")
        axes[0].legend()
        axes[1].plot(y_val.index, np.abs(y_val.to_numpy() - predictions), color="tab:red")
        axes[1].set_ylabel("Absolute error")
        axes[1].set_xlabel("Validation date")
        figure.tight_layout()
        figure.savefig(output / "daily_orders_forecast.png")
        mlflow.log_figure(figure, "daily_orders_forecast.png")
        comparison = pd.DataFrame({"actual": y_val, "prediction": predictions})
        comparison["absolute_error"] = (comparison["actual"] - comparison["prediction"]).abs()
        comparison.to_csv(output / "validation_predictions.csv", index_label="date")
        mlflow.log_artifact(str(output / "validation_predictions.csv"))
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=args.model_name,
            skops_trusted_types=[
                "mlops_playground.pipeline.FeatureSelector",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            ],
            signature=infer_signature(X_train, pipeline.predict(X_train)),
            input_example=X_val.head(3),
            code_paths=[str(ROOT / "src/mlops_playground")],
            metadata=metadata,
            extra_files=[str(metadata_path)],
        )
        version = str(model_info.registered_model_version)
        client.set_registered_model_alias(args.model_name, "challenger", version)
        if passed:
            client.set_registered_model_alias(args.model_name, "champion", version)
        decision = {
            "run_id": run.info.run_id,
            "model_name": args.model_name,
            "version": version,
            "max_iter": args.max_iter,
            "data_md5": data_md5,
            "candidate_mae": candidate_mae,
            "previous_champion_version": None if champion is None else champion.version,
            "champion_mae": champion_mae,
            "mae_gain": gain,
            "min_gain": args.min_gain,
            "gate_passed": passed,
            "reason": "first_model" if champion is None else "sufficient_improvement" if passed else "insufficient_improvement",
            "champion_version_after": version if passed else champion.version,
            "run_url": f"{args.tracking_uri.rstrip('/')}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}",
        }
        metadata["registry_name"] = args.model_name
        metadata["registry_version"] = version
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(metadata_path))
        decision_path = output / "gate_decision.json"
        decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(decision_path))
        for key in ("data_md5", "gate_passed", "reason"):
            client.set_model_version_tag(args.model_name, version, key, str(decision[key]))
        mlflow.set_tags({"gate_decision": "promoted" if passed else "rejected", "registry_version": version})
        lines = [
            f"DATA: {data_path.name}, data_md5={data_md5}",
            f"SPLIT: train_days={len(y_train)}, validation_days={len(y_val)}, features={len(X_train.columns)}",
            f"VERSION: {version}, max_iter={args.max_iter}, validation_mae={candidate_mae:.6f}",
        ]
        if champion_mae is None:
            lines.append("GATE: PASS (first model, no champion yet)")
        else:
            lines.append(f"GATE: {'PASS' if passed else 'REJECT'}; champion_mae={champion_mae:.6f}; gain={gain:.6f}; MIN_GAIN={args.min_gain}")
        lines.extend([
            f"ALIASES: challenger={version}, champion={decision['champion_version_after']}",
            f"RUN: {decision['run_url']}",
            f"EVIDENCE: {output}",
        ])
        console_output = "\n".join(lines) + "\n"
        (output / "train_output.txt").write_text(console_output, encoding="utf-8")
        mlflow.log_artifact(str(output / "train_output.txt"))
        print(console_output, flush=True)


if __name__ == "__main__":
    main()
