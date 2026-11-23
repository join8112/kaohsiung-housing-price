"""Hedonic regression (explanation) and XGBoost (prediction) for Kaohsiung unit prices.

Evaluation is out-of-time: the models are trained on older deals and scored on
the most recent ones, the way an appraisal model would be used in practice.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xgboost as xgb

from . import config
from .features import CAT_FEATURES, FEATURES, RoadEncoder, add_basic_features, to_model_matrix

VALID_START = pd.Timestamp("2025-07-01")
TEST_START = pd.Timestamp("2026-01-01")

HEDONIC_RHS = (
    "log_area + log_land_area + age_years + I(age_years**2) + C(btype) + C(floor_cat)"
    " + rooms + baths + elevator + management + has_parking + parking_unpriced + has_addition"
    " + C(district)"
)

XGB_PARAMS = dict(
    objective="reg:squarederror", tree_method="hist", enable_categorical=True,
    max_cat_to_onehot=1, random_state=config.RANDOM_STATE, n_jobs=-1,
)
PARAM_GRID = [
    dict(learning_rate=0.05, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
    dict(learning_rate=0.05, max_depth=8, min_child_weight=10, subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0),
    dict(learning_rate=0.05, max_depth=10, min_child_weight=20, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0),
    dict(learning_rate=0.05, max_depth=7, min_child_weight=3, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0),
]


# ---------------------------------------------------------------------------
def load_model_data() -> pd.DataFrame:
    df = pd.read_parquet(config.CLEAN_PARQUET)
    df = add_basic_features(df)
    ok = df["age_years"].notna() & df[["rooms", "halls", "baths"]].notna().all(axis=1)
    df = df[ok]
    n_deals = df.groupby("district")["district"].transform("size")
    return df[n_deals >= config.MIN_DISTRICT_DEALS].reset_index(drop=True)


def price_metrics(y_true_log_unit, y_pred_log_unit, area) -> dict:
    true_total = np.exp(y_true_log_unit) * area
    pred_total = np.exp(y_pred_log_unit) * area
    ape = np.abs(pred_total / true_total - 1)
    resid = y_true_log_unit - y_pred_log_unit
    r2 = 1 - np.sum(resid ** 2) / np.sum((y_true_log_unit - np.mean(y_true_log_unit)) ** 2)
    return {
        "MAPE_%": round(100 * float(np.mean(ape)), 2),
        "MdAPE_%": round(100 * float(np.median(ape)), 2),
        "within_10%": round(100 * float(np.mean(ape <= 0.10)), 1),
        "within_20%": round(100 * float(np.mean(ape <= 0.20)), 1),
        "R2_log_unit_price": round(float(r2), 3),
        "n": int(len(ape)),
    }


# ---------------------------------------------------------------------------
def fit_hedonic_index(df: pd.DataFrame):
    """Full-sample hedonic regression with month fixed effects (for explanation and the price index)."""
    formula = "log_unit_price ~ " + HEDONIC_RHS + " + C(ym)"
    model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["district"]})

    params, bse = model.params, model.bse
    coefs = pd.DataFrame({"coef": params, "se": bse, "p_value": model.pvalues})
    coefs["pct_effect"] = 100 * (np.exp(coefs["coef"]) - 1)
    keep = ~coefs.index.str.startswith(("C(ym)", "C(district)"))
    coefs = coefs[keep].round(4)

    months = sorted(df["ym"].unique())
    fe = [0.0] + [params.get(f"C(ym)[T.{m}]", np.nan) for m in months[1:]]
    index = pd.DataFrame({"ym": months, "log_fe": fe})
    index["hedonic_index"] = 100 * np.exp(index["log_fe"])
    counts = df.groupby("ym").size().rename("deals")
    raw_median = df.groupby("ym")["unit_price"].median().rename("median_unit_price")
    index = index.join(counts, on="ym").join(raw_median, on="ym")
    index["median_index"] = 100 * index["median_unit_price"] / index["median_unit_price"].iloc[0]
    return model, coefs, index


# ---------------------------------------------------------------------------
def time_split(df: pd.DataFrame):
    train = df[df["trade_date"] < VALID_START]
    valid = df[(df["trade_date"] >= VALID_START) & (df["trade_date"] < TEST_START)]
    test = df[df["trade_date"] >= TEST_START]
    return train.copy(), valid.copy(), test.copy()


def baseline_predict(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Median log unit price of the same district × building type over the last 12 months of training data."""
    recent = train[train["trade_date"] >= train["trade_date"].max() - pd.DateOffset(months=12)]
    cell = recent.groupby(["district", "btype"])["log_unit_price"].median()
    dist = recent.groupby("district")["log_unit_price"].median()
    key = pd.MultiIndex.from_frame(target[["district", "btype"]])
    pred = pd.Series(cell.reindex(key).to_numpy(), index=target.index)
    return pred.fillna(target["district"].map(dist)).fillna(recent["log_unit_price"].median()).to_numpy()


def fit_hedonic_predictive(train: pd.DataFrame):
    formula = "log_unit_price ~ " + HEDONIC_RHS + " + t + road_te"
    return smf.ols(formula, data=train).fit()


def fit_xgb(train_X, train_y, valid_X, valid_y, log=print):
    best = None
    for i, params in enumerate(PARAM_GRID, 1):
        model = xgb.XGBRegressor(n_estimators=4000, early_stopping_rounds=150, eval_metric="rmse",
                                 **XGB_PARAMS, **params)
        model.fit(train_X, train_y, eval_set=[(valid_X, valid_y)], verbose=False)
        score = model.best_score
        log(f"  config {i}: valid RMSE(log) = {score:.4f} at {model.best_iteration} trees")
        if best is None or score < best[0]:
            best = (score, params, model.best_iteration, model)
    return best


def run(log=print) -> dict:
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS.mkdir(exist_ok=True)
    df = load_model_data()

    # 1) explanatory hedonic model + quality-adjusted price index ---------------
    log("fitting full-sample hedonic regression …")
    hed_model, coefs, index = fit_hedonic_index(df)
    coefs.to_csv(config.REPORTS / "hedonic_coefficients.csv", encoding="utf-8-sig")
    index.to_csv(config.REPORTS / "hedonic_price_index.csv", index=False)
    log(f"  R² = {hed_model.rsquared:.3f}, n = {int(hed_model.nobs):,}")

    # 2) out-of-time prediction benchmark ----------------------------------------
    train, valid, test = time_split(df)
    log(f"split: train {len(train):,} | valid {len(valid):,} | test {len(test):,}")
    enc = RoadEncoder()
    train = enc.fit_transform(train)
    valid, test = enc.transform(valid), enc.transform(test)
    categories = {c: sorted(df[c].unique().tolist()) for c in CAT_FEATURES}

    results = {}
    results["Baseline: district × type median"] = price_metrics(
        test["log_unit_price"].to_numpy(), baseline_predict(train, test), test["area_ex_park_ping"].to_numpy())

    ols = fit_hedonic_predictive(pd.concat([train, valid]))
    results["Hedonic OLS (+ road level, trend)"] = price_metrics(
        test["log_unit_price"].to_numpy(), ols.predict(test).to_numpy(), test["area_ex_park_ping"].to_numpy())

    log("tuning XGBoost on the validation period …")
    Xtr, Xva, Xte = (to_model_matrix(d, categories) for d in (train, valid, test))
    _, best_params, best_iter, xgb_model = fit_xgb(Xtr, train["log_unit_price"], Xva, valid["log_unit_price"], log)
    # refit on train + valid with the chosen settings, then score the untouched test period
    n_trees = int(best_iter * 1.1) + 1
    enc_tv = RoadEncoder()
    train_valid = enc_tv.fit_transform(pd.concat([train, valid]))
    test_tv = enc_tv.transform(test)
    model_tv = xgb.XGBRegressor(n_estimators=n_trees, **XGB_PARAMS, **best_params)
    model_tv.fit(to_model_matrix(train_valid, categories), train_valid["log_unit_price"])
    test_pred = model_tv.predict(to_model_matrix(test_tv, categories))
    results["XGBoost"] = price_metrics(
        test["log_unit_price"].to_numpy(), test_pred, test["area_ex_park_ping"].to_numpy())

    metrics = pd.DataFrame(results).T
    metrics.index.name = "model"
    metrics.to_csv(config.REPORTS / "model_comparison.csv")
    log("\n" + metrics.to_string())

    # residual quantiles on the out-of-time test set -> prediction interval for the app
    resid = test["log_unit_price"].to_numpy() - test_pred
    q10, q90 = np.quantile(resid, [0.10, 0.90])
    test_out = test_tv[["serial", "trade_date", "district", "btype", "area_ex_park_ping", "unit_price"]].copy()
    test_out["pred_unit_price"] = np.exp(test_pred)
    test_out.to_parquet(config.REPORTS / "test_predictions.parquet", index=False)

    # 3) final model on all data, for the app -----------------------------------
    log("refitting final model on all data …")
    enc_all = RoadEncoder()
    full = enc_all.fit_transform(df)
    final = xgb.XGBRegressor(n_estimators=n_trees, **XGB_PARAMS, **best_params)
    final.fit(to_model_matrix(full, categories), full["log_unit_price"])
    final.save_model(config.MODELS_DIR / "xgb_unit_price.json")

    road_table = enc_all.road_.reset_index()[["district", "road", "te", "count"]]
    road_table.to_csv(config.MODELS_DIR / "road_encoding.csv", index=False, encoding="utf-8-sig")
    defaults = (full.groupby(["district", "btype"])
                    .agg(land_area_ping=("land_area_ping", "median"),
                         total_floors=("total_floors", "median"), deals=("serial", "size"))
                    .reset_index())
    defaults.to_csv(config.MODELS_DIR / "district_defaults.csv", index=False, encoding="utf-8-sig")

    meta = {
        "features": FEATURES,
        "categories": categories,
        "district_mean_log_price": enc_all.district_mean_.to_dict(),
        "global_mean_log_price": float(enc_all.global_mean_),
        "t_latest": int(df["t"].max()),
        "data_through": df["trade_date"].max().strftime("%Y-%m-%d"),
        "data_from": df["trade_date"].min().strftime("%Y-%m-%d"),
        "n_training_rows": int(len(df)),
        "xgb_params": best_params,
        "n_estimators": n_trees,
        "interval_log_resid_q10_q90": [float(q10), float(q90)],
        "test_metrics": results["XGBoost"],
        "test_period_start": TEST_START.strftime("%Y-%m-%d"),
    }
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"df": df, "test": test_tv, "categories": categories, "model_tv": model_tv,
            "metrics": metrics, "coefs": coefs, "index": index, "hedonic": hed_model}


if __name__ == "__main__":
    run()
