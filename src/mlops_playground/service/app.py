import json
import math
import time
import uuid
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from mlops_playground import db
from mlops_playground.config import settings


class Features(BaseModel):
    model_config = {"extra": "forbid", "populate_by_name": True}

    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    is_weekend: int = Field(ge=0, le=1)
    dow_sin: float = Field(ge=-1, le=1)
    dow_cos: float = Field(ge=-1, le=1)
    year_sin: float = Field(ge=-1, le=1)
    year_cos: float = Field(ge=-1, le=1)
    lag_1: int | None = Field(default=None, ge=0)
    lag_7: int | None = Field(default=None, ge=0)
    lag_14: int | None = Field(default=None, ge=0)
    lag_28: int | None = Field(default=None, ge=0)
    rolling_mean_7: float | None = Field(default=None, ge=0)
    rolling_std_7: float | None = Field(default=None, ge=0)
    rolling_mean_28: float | None = Field(default=None, ge=0)
    rolling_std_28: float | None = Field(default=None, ge=0)
    raw_price_mean: float | None = Field(default=None, ge=0)
    raw_price_median: float | None = Field(default=None, ge=0)
    raw_price_std: float | None = Field(default=None, ge=0)
    raw_price_min: float | None = Field(default=None, ge=0)
    raw_price_max: float | None = Field(default=None, ge=0)
    raw_price_sum: float | None = Field(default=None, ge=0)
    raw_qty_ordered_mean: float | None = Field(default=None, ge=0)
    raw_qty_ordered_median: float | None = Field(default=None, ge=0)
    raw_qty_ordered_std: float | None = Field(default=None, ge=0)
    raw_qty_ordered_min: int | None = Field(default=None, ge=0)
    raw_qty_ordered_max: int | None = Field(default=None, ge=0)
    raw_qty_ordered_sum: int | None = Field(default=None, ge=0)
    raw_grand_total_mean: float | None = None
    raw_grand_total_median: float | None = None
    raw_grand_total_std: float | None = Field(default=None, ge=0)
    raw_grand_total_min: float | None = None
    raw_grand_total_max: float | None = None
    raw_grand_total_sum: float | None = None
    raw_discount_amount_mean: float | None = None
    raw_discount_amount_median: float | None = None
    raw_discount_amount_std: float | None = Field(default=None, ge=0)
    raw_discount_amount_min: float | None = None
    raw_discount_amount_max: float | None = None
    raw_discount_amount_sum: float | None = None
    raw_MV_mean: float | None = Field(default=None, ge=0)
    raw_MV_median: float | None = Field(default=None, ge=0)
    raw_MV_std: float | None = Field(default=None, ge=0)
    raw_MV_min: int | None = Field(default=None, ge=0)
    raw_MV_max: int | None = Field(default=None, ge=0)
    raw_MV_sum: int | None = Field(default=None, ge=0)
    raw_Year_mean: float | None = Field(default=None, ge=1, le=9999)
    raw_Year_median: float | None = Field(default=None, ge=1, le=9999)
    raw_Year_std: float | None = Field(default=None, ge=0)
    raw_Year_min: int | None = Field(default=None, ge=1, le=9999)
    raw_Year_max: int | None = Field(default=None, ge=1, le=9999)
    raw_Year_sum: int | None = Field(default=None, ge=0)
    raw_Month_mean: float | None = Field(default=None, ge=1, le=12)
    raw_Month_median: float | None = Field(default=None, ge=1, le=12)
    raw_Month_std: float | None = Field(default=None, ge=0)
    raw_Month_min: int | None = Field(default=None, ge=1, le=12)
    raw_Month_max: int | None = Field(default=None, ge=1, le=12)
    raw_Month_sum: int | None = Field(default=None, ge=0)
    raw_item_id_nunique: int | None = Field(default=None, ge=0)
    raw_status_nunique: int | None = Field(default=None, ge=0)
    raw_sku_nunique: int | None = Field(default=None, ge=0)
    raw_category_name_1_nunique: int | None = Field(default=None, ge=0)
    raw_sales_commission_code_nunique: int | None = Field(default=None, ge=0)
    raw_payment_method_nunique: int | None = Field(default=None, ge=0)
    raw_bi_status_nunique: int | None = Field(default=None, ge=0, alias="raw_BI Status_nunique")
    raw_my_nunique: int | None = Field(default=None, ge=0, alias="raw_M-Y_nunique")
    raw_FY_nunique: int | None = Field(default=None, ge=0)
    raw_customer_id_nunique: int | None = Field(default=None, ge=0, alias="raw_Customer ID_nunique")
    raw_rows_count: int | None = Field(default=None, ge=0)


class Prediction(BaseModel):
    prediction: float
    model_version: str
    request_id: str
    latency_ms: float


def body_as_json(body: bytes) -> dict:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"body": body.decode("utf-8", errors="replace")}
    return value if isinstance(value, dict) else {"body": value}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.model_name:
        from mlops_playground.service.model_loader import load_registered_model

        bundle = load_registered_model(settings.model_name, settings.mlflow_tracking_uri)
        app.state.model_source = "mlflow"
        app.state.model_uri = bundle["model_uri"]
    else:
        bundle = joblib.load(settings.model_path)
        app.state.model_source = "file"
        app.state.model_uri = None
    app.state.pipeline = bundle["pipeline"]
    app.state.meta = bundle["metadata"]
    app.state.version = bundle["metadata"]["model_version"]
    db.init()
    yield
    app.state.pipeline = None


app = FastAPI(title="mlops-playground", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def log_prediction(request: Request, call_next):
    if request.url.path != "/v1/predict":
        return await call_next(request)

    request.state.request_id = str(uuid.uuid4())
    request.state.started_at = time.perf_counter()
    features = body_as_json(await request.body())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    latency_ms = getattr(
        request.state,
        "latency_ms",
        round((time.perf_counter() - request.state.started_at) * 1000, 2),
    )
    response.background = BackgroundTask(
        db.save_prediction,
        request.state.request_id,
        features,
        getattr(request.state, "prediction", None),
        app.state.version,
        latency_ms,
        response.status_code,
    )
    return response


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": getattr(app.state, "version", "unknown"),
        "model_path": settings.model_path if getattr(app.state, "model_source", None) == "file" else None,
        "model_source": getattr(app.state, "model_source", "unknown"),
        "model_uri": getattr(app.state, "model_uri", None),
    }


@app.get("/ready")
def ready():
    if getattr(app.state, "pipeline", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/v1/predict")
def predict(x: Features, request: Request) -> Prediction:
    payload = x.model_dump(by_alias=True)
    frame = pd.DataFrame([payload]).reindex(columns=app.state.meta["features"])
    prediction = float(app.state.pipeline.predict(frame)[0])
    if not math.isfinite(prediction):
        raise HTTPException(status_code=500, detail="Invalid model output")
    prediction = max(0.0, prediction)
    latency_ms = round((time.perf_counter() - request.state.started_at) * 1000, 2)
    request.state.prediction = prediction
    request.state.latency_ms = latency_ms
    return Prediction(
        prediction=prediction,
        model_version=app.state.version,
        request_id=request.state.request_id,
        latency_ms=latency_ms,
    )
