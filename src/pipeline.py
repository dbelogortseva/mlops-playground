import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .features import make_features


def daily_input(history, raw_daily_features):
    return history.rename("orders").to_frame().join(raw_daily_features, how="left")


class DailyFeatures(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        features = self._features(X).dropna(axis=1, how="all")
        self.feature_columns_ = list(features.columns)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_columns_")
        return self._features(X).loc[:, self.feature_columns_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_columns_")
        return np.asarray(self.feature_columns_, dtype=object)

    def _features(self, X):
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Daily input must have a DatetimeIndex.")
        return make_features(X["orders"], X.drop(columns="orders")).drop(columns="orders")


def make_bundle(model, history, raw_daily_features):
    preprocessor = DailyFeatures().fit(daily_input(history, raw_daily_features))
    if preprocessor.feature_columns_ != list(model.feature_names_in_):
        raise ValueError("Preprocessing feature order differs from the fitted model.")
    return {
        "pipeline": Pipeline([("preprocessing", preprocessor), ("model", model)]),
        "metadata": {
            "version": "1.0.0",
            "feature_columns": list(model.feature_names_in_),
            "threshold": None,
            "task": "regression",
            "threshold_reason": "Regression of daily order counts; classification is not applicable.",
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "training_end": str(history.index.max().date()),
            "history": history.copy(),
            "raw_daily_features": raw_daily_features.reindex(history.index).copy(),
        },
    }


def recursive_pipeline_forecast(pipeline, history, forecast_dates, raw_daily_features):
    history = history.copy()
    predictions = []
    for date in forecast_dates:
        temp = pd.concat([history, pd.Series([np.nan], index=[date])])
        prediction = max(0, pipeline.predict(daily_input(temp, raw_daily_features))[-1])
        predictions.append(prediction)
        history.loc[date] = prediction
    return pd.Series(predictions, index=forecast_dates, name="prediction")
