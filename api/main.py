"""REST API for the Kaohsiung housing valuation model.

Run locally:
    uvicorn api.main:app --reload            # then open http://127.0.0.1:8000/docs

Endpoints
    GET  /health                    liveness check (used by Docker / Render)
    GET  /model/info                training period, features, test-set accuracy
    GET  /districts                 districts and building types the model knows
    GET  /districts/{name}/roads    roads with a road-level price encoding
    POST /predict                   value one home (+ SHAP explanation)
    POST /predict/batch             value up to 100 homes in one call
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # make `khp` importable without installing the package

from khp.predictor import HouseInput, HousePricePredictor  # noqa: E402

API_VERSION = "1.0.0"
MAX_BATCH = 100

predictor = HousePricePredictor(os.environ.get("MODEL_DIR", ROOT / "models"))

app = FastAPI(
    title="Kaohsiung Housing Price API",
    version=API_VERSION,
    description=(
        "高雄市住宅估價 API。模型為 XGBoost，以內政部實價登錄 "
        f"{predictor.meta['data_from'][:7]}–{predictor.meta['data_through'][:7]} 共 "
        f"{predictor.meta['n_training_rows']:,} 筆交易訓練；2026 年樣本外測試總價誤差中位數 "
        f"{predictor.meta['test_metrics']['MdAPE_%']}%。價格單位：萬元，面積單位：坪。"
        "僅供參考，不構成不動產估價報告。"
    ),
)


# --------------------------------------------------------------------- schemas
class HouseRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "district": "鳳山區", "btype": "大樓", "area_ping": 35, "age_years": 10,
        "total_floors": 15, "floor": 7, "rooms": 3, "halls": 2, "baths": 2,
        "road": "中山東路", "parking": "separate",
    }]})

    district: str = Field(description="行政區，例如 鳳山區；完整清單見 GET /districts")
    btype: str = Field(description="建物型態：大樓／華廈／公寓／透天厝")
    area_ping: float = Field(gt=0, ge=8, le=200, description="建坪（不含車位）")
    age_years: float = Field(ge=0, le=60, description="屋齡（年）")
    total_floors: int = Field(ge=1, le=50, description="總樓層；透天厝填樓層數")
    floor: int | None = Field(default=None, ge=1, le=50, description="所在樓層；透天厝不用填")
    rooms: int = Field(default=3, ge=0, le=8)
    halls: int = Field(default=2, ge=0, le=5)
    baths: int = Field(default=2, ge=0, le=6)
    road: str | None = Field(default=None, description="路名；不填或不在訓練資料中則用行政區平均行情")
    land_area_ping: float | None = Field(default=None, ge=0, le=200,
                                         description="土地持分（坪）；不填用同區同型態中位數")
    parking: Literal["none", "separate", "included"] = Field(
        default="none", description="none 無車位／separate 車位分開計價／included 車位含在總價內")
    management: bool | None = Field(default=None, description="有無管理組織；不填依型態預設")
    elevator: bool | None = Field(default=None, description="有無電梯；只有透天厝會用到")
    has_addition: bool = Field(default=False, description="有無增建／頂樓加蓋／未登記空間")

    @model_validator(mode="after")
    def _check(self) -> "HouseRequest":
        if self.district not in predictor.districts:
            raise ValueError(f"district must be one of {predictor.districts}")
        if self.btype not in predictor.building_types:
            raise ValueError(f"btype must be one of {predictor.building_types}")
        if self.btype != "透天厝":
            if self.floor is None:
                raise ValueError("floor is required unless btype is 透天厝")
            if self.floor > self.total_floors:
                raise ValueError("floor cannot be higher than total_floors")
        return self

    def to_input(self) -> HouseInput:
        return HouseInput(**self.model_dump())


class Factor(BaseModel):
    feature: str
    effect_pct: float = Field(description="這個條件讓單價比基準高（正）或低（負）多少 %")


class Interval(BaseModel):
    unit_low: float
    unit_high: float
    total_low: float
    total_high: float


class Prediction(BaseModel):
    unit_price_10k_per_ping: float = Field(description="預估單價（萬元／坪，不含車位）")
    total_price_10k: float = Field(description="預估總價（萬元，不含車位）")
    interval_80: Interval = Field(description="80% 預測區間")
    road_used: str | None = Field(description="實際用到的路段行情；null 表示用行政區平均")
    road_warning: str | None = None
    land_area_ping_default: float | None = Field(default=None, description="未提供土地持分時填入的預設值")
    baseline_unit_price: float | None = Field(default=None, description="SHAP 基準：模型對全部成交物件的平均估價")
    top_factors: list[Factor] | None = None


class BatchRequest(BaseModel):
    houses: list[HouseRequest] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


# ------------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": API_VERSION}


@app.get("/model/info")
def model_info() -> dict:
    m = predictor.meta
    return {
        "model": "XGBoost (log unit price)",
        "data_from": m["data_from"], "data_through": m["data_through"],
        "n_training_rows": m["n_training_rows"],
        "test_period_start": m["test_period_start"],
        "test_metrics": m["test_metrics"],
        "features": m["features"],
        "n_estimators": m["n_estimators"],
    }


@app.get("/districts")
def districts() -> dict:
    return {"districts": predictor.districts, "building_types": predictor.building_types}


@app.get("/districts/{name}/roads")
def roads(name: str) -> dict:
    if name not in predictor.districts:
        raise HTTPException(status_code=404, detail=f"unknown district: {name}")
    return {"district": name, "roads": predictor.roads_in(name)}


@app.post("/predict", response_model=Prediction)
def predict(house: HouseRequest, explain: bool = Query(True, description="是否回傳 SHAP 影響因素")) -> dict:
    return predictor.predict([house.to_input()], explain=explain)[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest, explain: bool = Query(False)) -> dict:
    return {"predictions": predictor.predict([h.to_input() for h in req.houses], explain=explain)}
