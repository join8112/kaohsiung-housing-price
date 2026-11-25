"""Load the trained XGBoost valuation model and score one home at a time.

This module is the serving layer used by the FastAPI service (``api/main.py``).
It only needs numpy, pandas and xgboost, so the Docker image stays small.

The feature row is built exactly the same way as in ``app/streamlit_app.py``
(and as in training, see ``khp.features``):

* ``road_te`` / ``road_n`` come from ``models/road_encoding.csv``; an unknown or
  missing road falls back to the district mean log price with ``road_n = 0``.
* ``t`` is fixed at the latest month in the training data, i.e. the estimate is
  "what would this home sell for at today's price level".
* The 80% interval uses the 10% / 90% quantiles of the test-period log residuals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# Building types that always have (大樓, 華廈) or never have (公寓) an elevator in
# the training data; only 透天厝 lets the user choose.
ELEVATOR_BY_TYPE = {"大樓": True, "華廈": True, "公寓": False}


@dataclass(frozen=True)
class HouseInput:
    district: str
    btype: str
    area_ping: float                 # building area excluding parking
    age_years: float
    total_floors: int
    floor: int | None = None         # None for 透天厝
    rooms: int = 3
    halls: int = 2
    baths: int = 2
    road: str | None = None
    land_area_ping: float | None = None   # None -> district x type median
    parking: str = "none"            # none | separate | included
    management: bool | None = None   # None -> default by building type
    elevator: bool | None = None     # only used for 透天厝
    has_addition: bool = False


class HousePricePredictor:
    """Wraps the booster plus the lookup tables saved next to it."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        model_dir = Path(model_dir)
        self.booster = xgb.Booster()
        self.booster.load_model(model_dir / "xgb_unit_price.json")
        self.meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.roads = pd.read_csv(model_dir / "road_encoding.csv", encoding="utf-8-sig")
        self.defaults = pd.read_csv(model_dir / "district_defaults.csv", encoding="utf-8-sig")
        self.features: list[str] = self.meta["features"]
        self.categories: dict[str, list[str]] = self.meta["categories"]
        self._road_lookup = {
            (d, r): (float(te), float(n))
            for d, r, te, n in self.roads[["district", "road", "te", "count"]].itertuples(index=False)
        }

    # ------------------------------------------------------------------ lookups
    @property
    def districts(self) -> list[str]:
        return self.categories["district"]

    @property
    def building_types(self) -> list[str]:
        return self.categories["btype"]

    def roads_in(self, district: str) -> list[str]:
        r = self.roads.loc[self.roads["district"] == district, "road"].dropna()
        return sorted(r.unique().tolist())

    def default_land_area(self, district: str, btype: str) -> float:
        d = self.defaults[(self.defaults["district"] == district) & (self.defaults["btype"] == btype)]
        if len(d) and pd.notna(d["land_area_ping"].iloc[0]):
            return float(d["land_area_ping"].iloc[0])
        return 10.0

    # ----------------------------------------------------------------- features
    def build_row(self, h: HouseInput) -> tuple[dict, dict]:
        """Return (model feature row, notes about defaults that were filled in)."""
        if h.district not in self.districts:
            raise ValueError(f"unknown district: {h.district}")
        if h.btype not in self.building_types:
            raise ValueError(f"unknown building type: {h.btype}")

        notes: dict = {}
        if h.road and (h.district, h.road) in self._road_lookup:
            road_te, road_n = self._road_lookup[(h.district, h.road)]
            notes["road_used"] = h.road
        else:
            road_te = self.meta["district_mean_log_price"].get(h.district, self.meta["global_mean_log_price"])
            road_n = 0.0
            notes["road_used"] = None
            if h.road:
                notes["road_warning"] = f"{h.road} 不在 {h.district} 的訓練資料中，改用行政區平均行情"

        land = h.land_area_ping
        if land is None:
            land = self.default_land_area(h.district, h.btype)
            notes["land_area_ping_default"] = round(land, 2)

        is_house = h.btype == "透天厝"
        floor = np.nan if is_house or h.floor is None else float(h.floor)
        elevator = bool(h.elevator) if is_house else ELEVATOR_BY_TYPE.get(h.btype, bool(h.elevator))
        management = h.management if h.management is not None else h.btype in ("大樓", "華廈", "套房")

        row = {
            "district": h.district, "btype": h.btype,
            "log_area": float(np.log(h.area_ping)), "log_land_area": float(np.log1p(land)),
            "age_years": float(h.age_years),
            "floor": floor, "total_floors": float(h.total_floors),
            "floor_ratio": np.nan if is_house else floor / h.total_floors,
            "rooms": float(h.rooms), "halls": float(h.halls), "baths": float(h.baths),
            "elevator": float(elevator), "management": float(management), "partition": 1.0,
            "has_parking": float(h.parking != "none"),
            "parking_unpriced": float(h.parking == "included"),
            "has_addition": float(h.has_addition),
            "t": float(self.meta["t_latest"]), "road_te": float(road_te), "road_n": float(road_n),
        }
        return row, notes

    def _matrix(self, rows: list[dict]) -> xgb.DMatrix:
        X = pd.DataFrame(rows)[self.features]
        for c in ("district", "btype"):
            X[c] = pd.Categorical(X[c], categories=self.categories[c])
        for c in self.features:
            if c not in ("district", "btype"):
                X[c] = X[c].astype(float)
        return xgb.DMatrix(X, enable_categorical=True)

    # ------------------------------------------------------------------ predict
    def predict(self, houses: list[HouseInput], explain: bool = True, top_k: int = 8) -> list[dict]:
        built = [self.build_row(h) for h in houses]
        dm = self._matrix([row for row, _ in built])
        pred_log = self.booster.predict(dm)
        contribs = self.booster.predict(dm, pred_contribs=True) if explain else None
        q10, q90 = self.meta["interval_log_resid_q10_q90"]

        out = []
        for i, (h, (_, notes)) in enumerate(zip(houses, built)):
            p = float(pred_log[i])
            unit, lo, hi = np.exp(p), np.exp(p + q10), np.exp(p + q90)
            result = {
                "unit_price_10k_per_ping": round(float(unit), 2),
                "total_price_10k": round(float(unit * h.area_ping), 1),
                "interval_80": {
                    "unit_low": round(float(lo), 2), "unit_high": round(float(hi), 2),
                    "total_low": round(float(lo * h.area_ping), 1),
                    "total_high": round(float(hi * h.area_ping), 1),
                },
                **notes,
            }
            if contribs is not None:
                c = pd.Series(contribs[i][:-1], index=self.features).drop("t")
                top = c.reindex(c.abs().sort_values(ascending=False).index).head(top_k)
                result["baseline_unit_price"] = round(float(np.exp(contribs[i][-1])), 2)
                result["top_factors"] = [
                    {"feature": f, "effect_pct": round(float(100 * (np.exp(v) - 1)), 2)}
                    for f, v in top.items()
                ]
            out.append(result)
        return out
