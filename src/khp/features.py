"""Feature engineering shared by the models, the policy study and the app."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config

CAT_FEATURES = ["district", "btype"]
NUM_FEATURES = [
    "log_area", "log_land_area", "age_years", "floor", "total_floors", "floor_ratio",
    "rooms", "halls", "baths", "elevator", "management", "partition",
    "has_parking", "parking_unpriced", "t", "road_te", "road_n",
]
FEATURES = CAT_FEATURES + NUM_FEATURES

# readable names for charts
LABELS = {
    "log_area": "floor area (log 坪)", "log_land_area": "land share (log 坪)",
    "age_years": "building age", "floor": "floor", "total_floors": "total floors",
    "floor_ratio": "floor / total floors", "rooms": "bedrooms", "halls": "living rooms",
    "baths": "bathrooms", "elevator": "elevator", "management": "management committee",
    "partition": "partitioned", "has_parking": "has parking", "parking_unpriced": "parking not priced separately",
    "t": "time (month index)", "road_te": "road price level", "road_n": "road deal count",
    "district": "district", "btype": "building type",
}


def floor_category(row) -> str:
    if row["btype"] == "透天厝" or row.get("whole_building", 0) == 1:
        return "whole building / 透天"
    f = row["floor"]
    if pd.isna(f):
        return "unknown"
    if f <= 1:
        return "1F"
    if f <= 3:
        return "2-3F"
    if f <= 6:
        return "4-6F"
    if f <= 12:
        return "7-12F"
    if f <= 20:
        return "13-20F"
    return "21F+"


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_area"] = np.log(out["area_ex_park_ping"])
    out["log_land_area"] = np.log1p(out["land_area_ping"].fillna(0))
    out["rooms"] = out["rooms"].clip(upper=8)
    out["halls"] = out["halls"].clip(upper=5)
    out["baths"] = out["baths"].clip(upper=6)
    out["floor_cat"] = out.apply(floor_category, axis=1)
    out["elevator"] = out["elevator"].fillna(0)
    out["management"] = out["management"].fillna(0)
    out["partition"] = out["partition"].fillna(0)
    out["segment"] = np.select(
        [out["btype"].eq("套房") | out["rooms"].between(1, 2), out["rooms"] >= 3],
        ["small", "family"], default="other")
    return out


class RoadEncoder:
    """Smoothed (m-estimate) mean of log unit price per (district, road).

    Rare roads are shrunk toward their district mean; unseen roads fall back to
    the district mean.  ``fit_transform`` returns out-of-fold encodings for the
    training rows so the model never sees a row's own price through this feature.
    """

    def __init__(self, m: float = 20.0, n_splits: int = 5):
        self.m = m
        self.n_splits = n_splits

    def _table(self, df: pd.DataFrame):
        g = df.groupby("district")["log_unit_price"]
        district_mean = g.mean()
        road = (df.dropna(subset=["road"])
                  .groupby(["district", "road"])["log_unit_price"].agg(["mean", "count"]).reset_index())
        prior = road["district"].map(district_mean)
        road["te"] = (road["count"] * road["mean"] + self.m * prior) / (road["count"] + self.m)
        return district_mean, road.set_index(["district", "road"]), df["log_unit_price"].mean()

    @staticmethod
    def _apply(df, district_mean, road, global_mean):
        key = pd.MultiIndex.from_frame(df[["district", "road"]].fillna({"road": ""}))
        te = pd.Series(road["te"].reindex(key).to_numpy(), index=df.index)
        n = pd.Series(road["count"].reindex(key).to_numpy(), index=df.index).fillna(0)
        fallback = df["district"].map(district_mean).fillna(global_mean)
        return te.fillna(fallback), n

    def fit(self, df: pd.DataFrame) -> "RoadEncoder":
        self.district_mean_, self.road_, self.global_mean_ = self._table(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["road_te"], out["road_n"] = self._apply(out, self.district_mean_, self.road_, self.global_mean_)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["road_te"] = np.nan
        out["road_n"] = np.nan
        kf = KFold(self.n_splits, shuffle=True, random_state=config.RANDOM_STATE)
        for tr_idx, va_idx in kf.split(out):
            tr, va = out.iloc[tr_idx], out.iloc[va_idx]
            dm, road, gm = self._table(tr)
            te, n = self._apply(va, dm, road, gm)
            out.iloc[va_idx, out.columns.get_loc("road_te")] = te.to_numpy()
            # rescale fold counts so they are comparable to full-data counts used at predict time
            out.iloc[va_idx, out.columns.get_loc("road_n")] = n.to_numpy() * self.n_splits / (self.n_splits - 1)
        self.fit(df)
        return out


def to_model_matrix(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    for c in NUM_FEATURES:
        X[c] = X[c].astype(float)
    return X
