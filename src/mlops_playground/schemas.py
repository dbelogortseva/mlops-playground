from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Nonnegative = Annotated[float, Field(ge=0)]
Count = Annotated[int, Field(ge=0)]
Cycle = Annotated[float, Field(ge=-1, le=1)]
Month = Annotated[float, Field(ge=1, le=12)]
Year = Annotated[float, Field(ge=1, le=9999)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        strict=True,
        json_schema_extra={"examples": [{
            "day_of_week": 4,
            "month": 7,
            "is_weekend": 0,
            "dow_sin": -0.433883739117558,
            "dow_cos": -0.9009688679024191,
            "year_sin": -0.00645086419810515,
            "year_cos": -0.9999791929590823,
        }]},
    )

    day_of_week: Annotated[int, Field(ge=0, le=6, description="Monday=0, Sunday=6")]
    month: Annotated[int, Field(ge=1, le=12)]
    is_weekend: Annotated[int, Field(ge=0, le=1)]
    dow_sin: Cycle
    dow_cos: Cycle
    year_sin: Cycle
    year_cos: Cycle
    lag_1: Count | None = None
    lag_7: Count | None = None
    lag_14: Count | None = None
    lag_28: Count | None = None
    rolling_mean_7: Nonnegative | None = None
    rolling_std_7: Nonnegative | None = None
    rolling_mean_28: Nonnegative | None = None
    rolling_std_28: Nonnegative | None = None
    raw_price_mean: Nonnegative | None = None
    raw_price_median: Nonnegative | None = None
    raw_price_std: Nonnegative | None = None
    raw_price_min: Nonnegative | None = None
    raw_price_max: Nonnegative | None = None
    raw_price_sum: Nonnegative | None = None
    raw_qty_ordered_mean: Nonnegative | None = None
    raw_qty_ordered_median: Nonnegative | None = None
    raw_qty_ordered_std: Nonnegative | None = None
    raw_qty_ordered_min: Count | None = None
    raw_qty_ordered_max: Count | None = None
    raw_qty_ordered_sum: Count | None = None
    raw_grand_total_mean: float | None = None
    raw_grand_total_median: float | None = None
    raw_grand_total_std: Nonnegative | None = None
    raw_grand_total_min: float | None = None
    raw_grand_total_max: float | None = None
    raw_grand_total_sum: float | None = None
    raw_discount_amount_mean: float | None = None
    raw_discount_amount_median: float | None = None
    raw_discount_amount_std: Nonnegative | None = None
    raw_discount_amount_min: float | None = None
    raw_discount_amount_max: float | None = None
    raw_discount_amount_sum: float | None = None
    raw_MV_mean: Nonnegative | None = None
    raw_MV_median: Nonnegative | None = None
    raw_MV_std: Nonnegative | None = None
    raw_MV_min: Count | None = None
    raw_MV_max: Count | None = None
    raw_MV_sum: Count | None = None
    raw_Year_mean: Year | None = None
    raw_Year_median: Year | None = None
    raw_Year_std: Nonnegative | None = None
    raw_Year_min: Annotated[int, Field(ge=1, le=9999)] | None = None
    raw_Year_max: Annotated[int, Field(ge=1, le=9999)] | None = None
    raw_Year_sum: Count | None = None
    raw_Month_mean: Month | None = None
    raw_Month_median: Month | None = None
    raw_Month_std: Nonnegative | None = None
    raw_Month_min: Annotated[int, Field(ge=1, le=12)] | None = None
    raw_Month_max: Annotated[int, Field(ge=1, le=12)] | None = None
    raw_Month_sum: Count | None = None
    raw_item_id_nunique: Count | None = None
    raw_status_nunique: Count | None = None
    raw_sku_nunique: Count | None = None
    raw_category_name_1_nunique: Count | None = None
    raw_sales_commission_code_nunique: Count | None = None
    raw_payment_method_nunique: Count | None = None
    raw_bi_status_nunique: Count | None = Field(None, alias="raw_BI Status_nunique")
    raw_my_nunique: Count | None = Field(None, alias="raw_M-Y_nunique")
    raw_FY_nunique: Count | None = None
    raw_customer_id_nunique: Count | None = Field(None, alias="raw_Customer ID_nunique")
    raw_rows_count: Count | None = None


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    prediction: Nonnegative
    model_version: str
    request_id: UUID
    latency_ms: Nonnegative
