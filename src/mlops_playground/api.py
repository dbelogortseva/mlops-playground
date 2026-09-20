from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    estimator: object
    columns: tuple[str, ...]
    version: str


def load_model(path: Path) -> LoadedModel:
    bundle = joblib.load(path)
    metadata = bundle["metadata"]
    estimator = bundle["pipeline"].named_steps["model"]
    columns = tuple(metadata["feature_columns"])
    schema_columns = tuple(
        field.alias or name for name, field in PredictRequest.model_fields.items()
    )
    if columns != schema_columns or columns != tuple(estimator.feature_names_in_):
        raise ValueError("Model features do not match the API contract")
    if metadata["sklearn_version"] != sklearn.__version__:
        raise ValueError("Model and runtime scikit-learn versions differ")
    version = metadata["version"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Missing model version")
    history = metadata["history"]
    raw = metadata["raw_daily_features"]
    frame = history.rename("orders").to_frame().join(raw, how="left")
    features = bundle["pipeline"].named_steps["preprocessing"].transform(frame)
    prediction = float(estimator.predict(features.tail(1))[0])
    if not math.isfinite(prediction) or prediction < 0:
        raise ValueError("Model warm-up returned an invalid prediction")
    return LoadedModel(estimator, columns, version)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        path = Path(model_path or os.environ.get("MODEL_PATH", "artifact/model.joblib"))
        try:
            app.state.model = load_model(path)
            logger.info("Model %s loaded once from %s", app.state.model.version, path)
        except Exception:
            logger.exception("Model loading failed")
        try:
            yield
        finally:
            app.state.model = None

    app = FastAPI(
        title="Daily orders prediction API",
        version="1.0.0",
        description="Predict daily order count from one row of precomputed features.",
        lifespan=lifespan,
    )
    app.state.model = None

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        errors = [
            {key: error[key] for key in ("loc", "msg", "type")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    def require_model() -> LoadedModel:
        model = app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return model

    @app.get("/health", tags=["Status"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["Status"], responses={503: {"description": "Model unavailable"}})
    def ready():
        model = require_model()
        return {"status": "ready", "model_version": model.version}

    @app.post(
        "/v1/predict",
        response_model=PredictResponse,
        tags=["Prediction"],
        responses={503: {"description": "Model unavailable"}},
    )
    def predict(payload: PredictRequest):
        started = perf_counter()
        request_id = uuid4()
        model = require_model()
        values = payload.model_dump(by_alias=True)
        frame = pd.DataFrame(
            [[np.nan if values[name] is None else values[name] for name in model.columns]],
            columns=model.columns,
            dtype=float,
        )
        try:
            prediction = float(model.estimator.predict(frame)[0])
            if not math.isfinite(prediction) or prediction < 0:
                raise ValueError("Invalid model output")
        except Exception:
            logger.exception("Prediction failed: request_id=%s", request_id)
            raise HTTPException(status_code=500, detail="Prediction failed") from None
        return PredictResponse(
            prediction=prediction,
            model_version=model.version,
            request_id=request_id,
            latency_ms=(perf_counter() - started) * 1000,
        )

    return app


app = create_app()
