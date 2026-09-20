import numpy as np
import pandas as pd


def make_daily_orders(df):
    daily = (
        df.groupby(df["created_at"].dt.floor("D"))["increment_id"]
          .nunique()
          .sort_index()
    )

    full_index = pd.date_range(
        daily.index.min(),
        daily.index.max(),
        freq="D"
    )

    return daily.reindex(full_index, fill_value=0)


def make_daily_raw_features(df):
    data = df.copy()
    data["date"] = data["created_at"].dt.floor("D")

    result = pd.DataFrame(
        index=pd.date_range(
            data["date"].min(),
            data["date"].max(),
            freq="D"
        )
    )

    numeric_cols = (
        data
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    numeric_cols = [
        col for col in numeric_cols
        if col != "increment_id"
    ]

    for col in numeric_cols:
        grouped = data.groupby("date")[col]

        result[f"{col}_mean"] = grouped.mean()
        result[f"{col}_median"] = grouped.median()
        result[f"{col}_std"] = grouped.std()
        result[f"{col}_min"] = grouped.min()
        result[f"{col}_max"] = grouped.max()
        result[f"{col}_sum"] = grouped.sum()

    categorical_cols = (
        data
        .select_dtypes(include=["object", "category", "string"])
        .columns
        .tolist()
    )

    categorical_cols = [
        col for col in categorical_cols
        if col != "increment_id"
    ]

    for col in categorical_cols:
        result[f"{col}_nunique"] = (
            data.groupby("date")[col].nunique()
        )

    result["rows_count"] = (
        data.groupby("date").size()
    )

    return result


def make_features(series, raw_daily_features=None):
    features = pd.DataFrame({
        "orders": series
    })

    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month
    features["is_weekend"] = (
        features["day_of_week"] >= 5
    ).astype(int)

    features["dow_sin"] = np.sin(
        2 * np.pi * features["day_of_week"] / 7
    )

    features["dow_cos"] = np.cos(
        2 * np.pi * features["day_of_week"] / 7
    )

    day_of_year = features.index.dayofyear

    features["year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    features["year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    for lag in [1, 7, 14, 28]:
        features[f"lag_{lag}"] = (
            features["orders"].shift(lag)
        )

    shifted_orders = features["orders"].shift(1)

    for window in [7, 28]:
        features[f"rolling_mean_{window}"] = (
            shifted_orders.rolling(window).mean()
        )

        features[f"rolling_std_{window}"] = (
            shifted_orders.rolling(window).std()
        )

    if raw_daily_features is not None:
        raw_features = (
            raw_daily_features
            .reindex(features.index)
            .copy()
        )

        raw_features = raw_features.shift(1)

        raw_features = raw_features.add_prefix("raw_")

        features = features.join(
            raw_features,
            how="left"
        )

    return features


def recursive_forecast(
    model,
    history,
    forecast_dates,
    feature_columns,
    raw_daily_features
):
    history = history.copy()
    predictions = []

    for date in forecast_dates:
        temp = pd.concat([
            history,
            pd.Series(
                [np.nan],
                index=[date]
            )
        ])

        features = make_features(
            temp,
            raw_daily_features
        )

        X_date = features.loc[
            [date],
            feature_columns
        ]

        prediction = model.predict(X_date)[0]

        prediction = max(0, prediction)

        predictions.append(prediction)

        history.loc[date] = prediction

    return pd.Series(
        predictions,
        index=forecast_dates,
        name="prediction"
    )
