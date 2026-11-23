"""Did the 2024 credit controls hit investor-type units harder?

Design (exposure-based difference-in-differences / event study)
----------------------------------------------------------------
The 7th wave of the central bank's selective credit controls (effective
2024-09-20) cut loan-to-value limits for second homes, third-plus homes and
corporate buyers, and removed grace periods for owners buying again.  Buyers
of small units (studios, 1–2 bedrooms) are disproportionately investors and
second-home buyers, whereas 3+ bedroom flats are mostly bought by owner-
occupiers (first-time buyers, and 新青安 subsidised loans, were exempt).

So we compare small units (more exposed) with family-size units (less
exposed), within the same district and month, before and after 2024Q3:

* price:  quality-adjusted log unit price, hedonic controls + district FE +
  month FE + small × event-quarter dummies (reference quarter 2024Q2);
* volume: Poisson regression (PPML) of deal counts by district × segment ×
  quarter with district×quarter and district×segment fixed effects.

The window ends on 2026-03-19, the day before the first easing.  Standard
errors are clustered by district.  This identifies a *differential* effect
only: anything that hit both segments equally (e.g. the bank lending quota
from mid-August 2024) is absorbed by the fixed effects.

Inference with few clusters
---------------------------
There are only ~30 districts, so cluster-robust Wald tests can over-reject.
Two robustness checks are reported next to the usual cluster SEs:

* pre-trend tests use an F distribution with G − 1 denominator df;
* the price DiD p-value is also computed with a wild cluster restricted
  bootstrap (WCR, Webb six-point weights), the standard fix for few clusters
  (Cameron, Gelbach & Miller 2008; Roodman et al. 2019).

If small units were already gaining on family units before the controls (a
pre-trend), the plain DiD mixes that trend into the "effect".  A trend-
adjusted DiD (small × linear quarter trend) is reported as a further check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from . import config
from .features import add_basic_features

WINDOW_START = pd.Timestamp("2022-01-01")
WINDOW_END = config.POLICY_EVENTS["easing_2026_03"]  # exclusive
EVENT_Q = pd.Period("2024Q3")
REF = -1                       # 2024Q2
POST_START = pd.Timestamp("2024-10-01")

CONTROLS = ("log_area + age_years + I(age_years**2) + C(btype) + C(floor_cat)"
            " + baths + management + has_parking + parking_unpriced + has_addition")


def policy_sample() -> pd.DataFrame:
    df = add_basic_features(pd.read_parquet(config.CLEAN_PARQUET))
    df = df[df["btype"].isin(["大樓", "華廈", "公寓", "套房"])
            & df["segment"].isin(["small", "family"])
            & df["age_years"].notna()
            & (df["trade_date"] >= WINDOW_START) & (df["trade_date"] < WINDOW_END)].copy()
    df["small"] = (df["segment"] == "small").astype(int)
    q = df["trade_date"].dt.to_period("Q")
    df["rel"] = [(p - EVENT_Q).n for p in q]
    return df


def _rel_dummies(df: pd.DataFrame, prefix: str = "ev") -> list[str]:
    names = []
    for k in sorted(df["rel"].unique()):
        if k == REF:
            continue
        name = f"{prefix}_{'m' if k < 0 else 'p'}{abs(k)}"
        df[name] = ((df["rel"] == k) & (df["small"] == 1)).astype(int)
        names.append((k, name))
    return names


def price_event_study(df: pd.DataFrame):
    df = df.copy()
    dummies = _rel_dummies(df)
    formula = ("log_unit_price ~ small + " + " + ".join(n for _, n in dummies)
               + " + " + CONTROLS + " + C(district) + C(ym)")
    fit = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["district"]})
    rows = [{"rel": REF, "coef": 0.0, "se": 0.0}]
    rows += [{"rel": k, "coef": fit.params[n], "se": fit.bse[n]} for k, n in dummies]
    es = pd.DataFrame(rows).sort_values("rel")
    es["quarter"] = [str(EVENT_Q + int(k)) for k in es["rel"]]
    es["pct"] = 100 * (np.exp(es["coef"]) - 1)
    es["ci_low"] = 100 * (np.exp(es["coef"] - 1.96 * es["se"]) - 1)
    es["ci_high"] = 100 * (np.exp(es["coef"] + 1.96 * es["se"]) - 1)
    return es, fit


def _did_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the 2024Q3 transition quarter and build the DiD regressors."""
    d = df[(df["trade_date"] < pd.Timestamp("2024-07-01")) | (df["trade_date"] >= POST_START)].copy()
    d["post"] = (d["trade_date"] >= POST_START).astype(int)
    d["small_post"] = d["small"] * d["post"]
    d["small_trend"] = d["small"] * d["rel"]      # small × linear quarter trend
    return d


PRICE_DID_FORMULA = "log_unit_price ~ small + small_post + " + CONTROLS + " + C(district) + C(ym)"


def price_did(df: pd.DataFrame, trend: bool = False):
    d = _did_sample(df)
    formula = PRICE_DID_FORMULA + (" + small_trend" if trend else "")
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["district"]})


WEBB_WEIGHTS = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def wild_cluster_bootstrap(formula: str, data: pd.DataFrame, param: str, cluster: str,
                           reps: int = 999, seed: int = config.RANDOM_STATE) -> dict:
    """Wild cluster restricted bootstrap p-value for H0: param = 0.

    1. Fit the model with the null imposed (param dropped) -> fitted values and residuals.
    2. For each replication, flip every cluster's residuals by one Webb weight,
       rebuild y*, re-estimate the full model and its cluster-robust t-stat.
    3. p-value = share of bootstrap |t*| at least as large as the observed |t|.

    The full-model OLS is solved once with a pseudo-inverse, so each replication
    is just a matrix-vector product.
    """
    import patsy

    y, X = patsy.dmatrices(formula, data, return_type="dataframe")
    j = list(X.columns).index(param)
    groups = pd.factorize(data.loc[X.index, cluster])[0]
    G = groups.max() + 1
    X, y = X.to_numpy(), y.to_numpy().ravel()
    n, k = X.shape

    pinv = np.linalg.pinv(X)                      # beta = pinv @ y
    c = pinv[j]                                   # beta_j = c @ y
    scale = G / (G - 1) * (n - 1) / (n - k)       # CRV1 small-sample factor

    def t_stat(yv):
        beta = pinv @ yv
        e = yv - X @ beta
        score = np.bincount(groups, weights=c * e, minlength=G)
        return beta[j] / np.sqrt(scale * np.sum(score ** 2))

    t_obs = t_stat(y)
    X0 = np.delete(X, j, axis=1)                  # restricted model (H0 imposed)
    fitted0 = X0 @ (np.linalg.pinv(X0) @ y)
    u0 = y - fitted0

    rng = np.random.default_rng(seed)
    t_boot = np.empty(reps)
    for r in range(reps):
        w = rng.choice(WEBB_WEIGHTS, size=G)[groups]
        t_boot[r] = t_stat(fitted0 + u0 * w)
    p = (1 + np.sum(np.abs(t_boot) >= abs(t_obs))) / (reps + 1)
    return {"t_stat": round(float(t_obs), 3), "p_value": round(float(p), 4),
            "reps": reps, "clusters": int(G), "weights": "Webb"}


def volume_panel(df: pd.DataFrame, min_deals: int = 150) -> pd.DataFrame:
    big = df.groupby("district").size()
    big = big[big >= min_deals].index
    d = df[df["district"].isin(big)]
    quarters = sorted(d["trade_date"].dt.to_period("Q").unique())
    idx = pd.MultiIndex.from_product([big, ["small", "family"], quarters],
                                     names=["district", "segment", "q"])
    counts = (d.assign(q=d["trade_date"].dt.to_period("Q"))
                .groupby(["district", "segment", "q"]).size().reindex(idx, fill_value=0)
                .rename("deals").reset_index())
    counts["small"] = (counts["segment"] == "small").astype(int)
    counts["rel"] = [(p - EVENT_Q).n for p in counts["q"]]
    counts["dq"] = counts["district"] + "_" + counts["q"].astype(str)
    counts["ds"] = counts["district"] + "_" + counts["segment"]
    # 2026Q1 is cut at 03-19: fine, district×quarter FE absorb the shorter quarter for both segments
    return counts


def volume_event_study(panel: pd.DataFrame):
    p = panel.copy()
    dummies = _rel_dummies(p)
    formula = "deals ~ " + " + ".join(n for _, n in dummies) + " + C(dq) + C(ds)"
    fit = smf.glm(formula, data=p, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(p["district"])[0]})
    rows = [{"rel": REF, "coef": 0.0, "se": 0.0}]
    rows += [{"rel": k, "coef": fit.params[n], "se": fit.bse[n]} for k, n in dummies]
    es = pd.DataFrame(rows).sort_values("rel")
    es["quarter"] = [str(EVENT_Q + int(k)) for k in es["rel"]]
    es["pct"] = 100 * (np.exp(es["coef"]) - 1)
    es["ci_low"] = 100 * (np.exp(es["coef"] - 1.96 * es["se"]) - 1)
    es["ci_high"] = 100 * (np.exp(es["coef"] + 1.96 * es["se"]) - 1)
    return es, fit


def volume_did(panel: pd.DataFrame):
    p = panel[(panel["rel"] < 0) | (panel["rel"] >= 1)].copy()   # drop the 2024Q3 transition quarter
    p["small_post"] = ((p["rel"] >= 1) & (p["small"] == 1)).astype(int)
    fit = smf.glm("deals ~ small_post + C(dq) + C(ds)", data=p, family=sm.families.Poisson()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(p["district"])[0]})
    return fit


def pretrend_test(fit, dummies_prefix: str = "ev_m") -> dict:
    names = [n for n in fit.params.index if n.startswith(dummies_prefix)]
    if not names:
        return {}
    R = np.zeros((len(names), len(fit.params)))
    for i, n in enumerate(names):
        R[i, list(fit.params.index).index(n)] = 1
    chi2 = fit.wald_test(R, scalar=True)
    # F version: with G clusters the cluster-robust covariance has at most G - 1 degrees of freedom
    G = len(np.unique(fit.model.data.frame["district"])) if "district" in fit.model.data.frame else None
    q = len(names)
    out = {"pre_period_dummies": q, "wald_chi2": round(float(chi2.statistic), 2),
           "p_value_chi2": float(chi2.pvalue)}
    if G:
        from scipy import stats
        F = float(chi2.statistic) / q
        out.update({"F": round(F, 2), "df": [q, G - 1], "p_value_F": float(stats.f.sf(F, q, G - 1))})
    return out


def _summ(fit, name: str) -> dict:
    b, se = fit.params[name], fit.bse[name]
    return {"coef": round(float(b), 4), "se": round(float(se), 4), "p_value": round(float(fit.pvalues[name]), 4),
            "pct_effect": round(100 * (np.exp(b) - 1), 1),
            "ci95_pct": [round(100 * (np.exp(b - 1.96 * se) - 1), 1), round(100 * (np.exp(b + 1.96 * se) - 1), 1)]}


def run(log=print) -> dict:
    out_dir = config.REPORTS / "policy"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = policy_sample()
    log(f"policy sample: {len(df):,} apartment deals, {df['small'].mean():.1%} small units")

    es_price, fit_price = price_event_study(df)
    es_price.to_csv(out_dir / "event_study_price.csv", index=False)
    did_price = price_did(df)
    did_price_trend = price_did(df, trend=True)
    log("wild cluster bootstrap for the price DiD …")
    wcb = wild_cluster_bootstrap(PRICE_DID_FORMULA, _did_sample(df), "small_post", "district")

    panel = volume_panel(df)
    es_vol, fit_vol = volume_event_study(panel)
    es_vol.to_csv(out_dir / "event_study_volume.csv", index=False)
    did_vol = volume_did(panel)

    summary = {
        "window": [str(WINDOW_START.date()), str((WINDOW_END - pd.Timedelta(days=1)).date())],
        "n_deals": int(len(df)),
        "n_districts_volume": int(panel["district"].nunique()),
        "price_did_small_x_post": {**_summ(did_price, "small_post"), "wild_cluster_bootstrap": wcb},
        "price_did_trend_adjusted": _summ(did_price_trend, "small_post"),
        "volume_did_small_x_post": _summ(did_vol, "small_post"),
        "price_pretrend_test": pretrend_test(fit_price),
        "volume_pretrend_test": pretrend_test(fit_vol),
    }
    pd.Series(summary).to_json(out_dir / "did_summary.json", force_ascii=False, indent=2)
    log(pd.Series(summary).to_string())
    return {"sample": df, "es_price": es_price, "es_vol": es_vol, "panel": panel, "summary": summary}


if __name__ == "__main__":
    run()
