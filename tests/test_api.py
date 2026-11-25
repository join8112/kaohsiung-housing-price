"""Tests for the FastAPI valuation service (api/main.py)."""
import math

import pytest
from fastapi.testclient import TestClient

from api.main import app, predictor

client = TestClient(app)

CONDO = {
    "district": "鳳山區", "btype": "大樓", "area_ping": 35, "age_years": 10,
    "total_floors": 15, "floor": 7, "rooms": 3, "halls": 2, "baths": 2,
    "road": "中山東路", "parking": "separate",
}
HOUSE = {
    "district": "鳳山區", "btype": "透天厝", "area_ping": 50, "age_years": 25,
    "total_floors": 4, "rooms": 4, "halls": 2, "baths": 3,
}


def post(body, **params):
    return client.post("/predict", json=body, params=params)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_info_reports_test_metrics():
    info = client.get("/model/info").json()
    assert info["test_metrics"]["MdAPE_%"] > 0
    assert "road_te" in info["features"]


def test_districts_and_roads():
    d = client.get("/districts").json()
    assert "鳳山區" in d["districts"] and "透天厝" in d["building_types"]
    roads = client.get("/districts/鳳山區/roads").json()["roads"]
    assert "中山東路" in roads
    assert client.get("/districts/台北區/roads").status_code == 404


def test_predict_condo_is_in_plausible_range():
    r = post(CONDO)
    assert r.status_code == 200, r.text
    p = r.json()
    # Kaohsiung unit prices are mostly 15-45 (10k NTD / ping)
    assert 10 < p["unit_price_10k_per_ping"] < 80
    assert math.isclose(p["total_price_10k"], p["unit_price_10k_per_ping"] * 35, rel_tol=1e-3)
    iv = p["interval_80"]
    assert iv["unit_low"] < p["unit_price_10k_per_ping"] < iv["unit_high"]
    assert p["road_used"] == "中山東路"
    assert len(p["top_factors"]) == 8


def test_interval_uses_test_period_residual_quantiles():
    p = post(CONDO).json()
    q10, q90 = predictor.meta["interval_log_resid_q10_q90"]
    unit = p["unit_price_10k_per_ping"]
    assert math.isclose(p["interval_80"]["unit_low"], unit * math.exp(q10), rel_tol=1e-3)
    assert math.isclose(p["interval_80"]["unit_high"], unit * math.exp(q90), rel_tol=1e-3)


def test_house_without_floor():
    r = post(HOUSE)
    assert r.status_code == 200, r.text
    assert r.json()["road_used"] is None  # no road given -> district average


def test_unknown_road_falls_back_with_warning():
    p = post({**CONDO, "road": "不存在路"}).json()
    assert p["road_used"] is None and "不存在路" in p["road_warning"]
    no_road = post({k: v for k, v in CONDO.items() if k != "road"}).json()
    assert p["unit_price_10k_per_ping"] == no_road["unit_price_10k_per_ping"]


def test_newer_home_is_worth_more():
    new = post({**CONDO, "age_years": 2}).json()["unit_price_10k_per_ping"]
    old = post({**CONDO, "age_years": 40}).json()["unit_price_10k_per_ping"]
    assert new > old


def test_explain_false_omits_factors():
    p = post(CONDO, explain=False).json()
    assert p["top_factors"] is None


@pytest.mark.parametrize("bad", [
    {"district": "台北區"},
    {"btype": "別墅"},
    {"area_ping": -5},
    {"floor": None},
    {"floor": 20},             # above total_floors
    {"parking": "maybe"},
])
def test_invalid_input_returns_422(bad):
    assert post({**CONDO, **bad}).status_code == 422


def test_batch_matches_single():
    r = client.post("/predict/batch", json={"houses": [CONDO, HOUSE]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["unit_price_10k_per_ping"] == post(CONDO).json()["unit_price_10k_per_ping"]


def test_batch_limit():
    assert client.post("/predict/batch", json={"houses": []}).status_code == 422
    assert client.post("/predict/batch", json={"houses": [CONDO] * 101}).status_code == 422
