from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
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
from starlette.concurrency import run_in_threadpool

from .schemas import PredictRequest, PredictResponse
from .request_log import PostgresRequestLog, PredictionLog, request_features

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
        database_url = os.environ.get("DATABASE_URL", "").strip()
        app.state.request_log = PostgresRequestLog(database_url) if database_url else None
        if app.state.request_log is not None:
            try:
                await run_in_threadpool(app.state.request_log.initialize)
            except Exception:
                logger.error("Request log initialization failed; check PostgreSQL availability")
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
            app.state.request_log = None

    app = FastAPI(
        title="Daily orders prediction API",
        version="1.0.0",
        description="Predict daily order count from one row of precomputed features.",
        lifespan=lifespan,
    )
    app.state.model = None
    app.state.request_log = None

    @app.middleware("http")
    async def log_prediction(request: Request, call_next):
        if request.url.path != "/v1/predict":
            return await call_next(request)
        request.state.request_id = uuid4()
        request.state.started = perf_counter()
        requested_at = datetime.now(timezone.utc)
        model = app.state.model
        body = await request.body()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled prediction error: request_id=%s", request.state.request_id)
            response = JSONResponse(status_code=500, content={"detail": "Prediction failed"})
        response.headers["X-Request-ID"] = str(request.state.request_id)
        latency_ms = getattr(request.state, "latency_ms", (perf_counter() - request.state.started) * 1000)
        if app.state.request_log is not None:
            record = PredictionLog(
                request_id=request.state.request_id,
                requested_at=requested_at,
                model_version=model.version if model is not None else None,
                features=request_features(body),
                prediction=getattr(request.state, "prediction", None) if response.status_code == 200 else None,
                latency_ms=latency_ms,
                status_code=response.status_code,
            )
            try:
                await run_in_threadpool(app.state.request_log.write, record)
            except Exception:
                logger.error("Request log write failed: request_id=%s", request.state.request_id)
        return response

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
    def predict(payload: PredictRequest, request: Request):
        started = request.state.started
        request_id = request.state.request_id
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
        request.state.prediction = prediction
        request.state.latency_ms = (perf_counter() - started) * 1000
        return PredictResponse(
            prediction=prediction,
            model_version=model.version,
            request_id=request_id,
            latency_ms=request.state.latency_ms,
        )

    return app


app = create_app()
