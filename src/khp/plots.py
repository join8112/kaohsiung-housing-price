"""All figures used in the README (saved to reports/figures/)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

from matplotlib import font_manager  # noqa: E402

_available = {f.name for f in font_manager.fontManager.ttflist}
_CJK = next((f for f in ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC"] if f in _available),
            "sans-serif")

plt.rcParams.update({
    "font.family": [_CJK, "sans-serif"],
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})
BLUE, ORANGE, GREY, RED = "#2563eb", "#ea580c", "#6b7280", "#b91c1c"

EVENT_LABELS = {
    "moral_suasion": "Aug 2024 bank lending quota",
    "wave7": "Sep 20, 2024 credit controls (wave 7)",
    "easing_2026_03": "Mar 20, 2026 first easing",
}


def _events(ax, keys=("moral_suasion", "wave7", "easing_2026_03"), date_axis=True, ymax_frac=0.98):
    for i, k in enumerate(keys):
        x = config.POLICY_EVENTS[k]
        ax.axvline(x, color=RED if k == "wave7" else GREY, ls="--", lw=1)
        ax.text(x, ax.get_ylim()[1] * (ymax_frac - 0.07 * i), " " + EVENT_LABELS[k], fontsize=8,
                color=RED if k == "wave7" else GREY, va="top")


def _save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / name)
    plt.close(fig)


def market_overview(index: pd.DataFrame):
    d = index.copy()
    d["date"] = pd.to_datetime(d["ym"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[1.3, 1])
    ax1.plot(d["date"], d["hedonic_index"], color=BLUE, lw=2, label="Quality-adjusted (hedonic) index")
    ax1.plot(d["date"], d["median_index"], color=GREY, lw=1.2, alpha=0.8, label="Raw median unit price")
    ax1.set_ylabel(f"Index ({d['ym'].iloc[0]} = 100)")
    ax1.set_title("Kaohsiung residential prices and deal volume")
    ax1.legend(loc="upper left", fontsize=9, frameon=False)
    ax2.bar(d["date"], d["deals"], width=25, color=BLUE, alpha=0.55)
    ax2.set_ylabel("Deals per month")
    _events(ax2)
    for ax in (ax1,):
        for k in ("moral_suasion", "wave7", "easing_2026_03"):
            ax.axvline(config.POLICY_EVENTS[k], color=RED if k == "wave7" else GREY, ls="--", lw=1)
    _save(fig, "01_market_overview.png")


def district_prices(summary: pd.DataFrame, top: int = 20):
    d = summary.sort_values("median_unit_price_last_12m", ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [ORANGE if (v is not None and v < 0) else BLUE for v in d["yoy_pct"].fillna(0)]
    ax.barh(d["district"], d["median_unit_price_last_12m"], color=colors, alpha=0.85)
    for y, (v, yoy) in enumerate(zip(d["median_unit_price_last_12m"], d["yoy_pct"])):
        label = f"{v:.1f}" + ("" if pd.isna(yoy) else f"  ({yoy:+.1f}% YoY)")
        ax.text(v + 0.3, y, label, va="center", fontsize=8)
    ax.set_xlabel("Median unit price, last 12 months (萬元 / 坪, excl. parking)")
    ax.set_title(f"Top {top} districts by price (orange = lower than a year earlier)")
    ax.set_xlim(0, d["median_unit_price_last_12m"].max() * 1.3)
    _save(fig, "02_district_prices.png")


def age_curve(age: pd.DataFrame):
    order = ["0-4", "5-9", "10-19", "20-29", "30-39", "40+"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for btype, g in age.groupby("btype"):
        g = g.set_index("age_bucket").reindex(order).dropna(subset=["median_unit_price"])
        ax.plot(g.index, g["median_unit_price"], marker="o", label=btype)
    ax.set_xlabel("Building age (years)")
    ax.set_ylabel("Median unit price (萬元 / 坪)")
    ax.set_title("Unit price by building age and type")
    ax.legend(frameon=False, ncol=5, fontsize=9)
    _save(fig, "03_age_curve.png")


def model_comparison(metrics: pd.DataFrame, test_pred: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    m = metrics["MdAPE_%"].astype(float)
    ax1.barh(m.index, m.values, color=[GREY, ORANGE, BLUE][: len(m)])
    for y, v in enumerate(m.values):
        ax1.text(v + 0.2, y, f"{v:.1f}%", va="center")
    ax1.set_xlabel("Median absolute % error on total price (lower is better)")
    ax1.set_title("Out-of-time test set")
    ax1.invert_yaxis()

    sample = test_pred.sample(min(4000, len(test_pred)), random_state=0)
    ax2.scatter(sample["unit_price"], sample["pred_unit_price"], s=4, alpha=0.25, color=BLUE)
    lim = np.quantile(np.r_[sample["unit_price"], sample["pred_unit_price"]], [0.005, 0.995])
    ax2.plot(lim, lim, color=RED, lw=1)
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("Actual unit price (萬元 / 坪)")
    ax2.set_ylabel("XGBoost prediction")
    ax2.set_title("XGBoost: predicted vs actual (test set)")
    _save(fig, "04_model_comparison.png")


def event_study(es_price: pd.DataFrame, es_vol: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, es, title in ((axes[0], es_price, "Quality-adjusted unit price"),
                          (axes[1], es_vol, "Number of deals (PPML)")):
        x = es["rel"].to_numpy()
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(-0.5, color=RED, ls="--", lw=1)
        ax.fill_between(x, es["ci_low"], es["ci_high"], color=BLUE, alpha=0.18, lw=0)
        ax.plot(x, es["pct"], marker="o", color=BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels(es["quarter"], rotation=60, fontsize=8)
        ax.set_title(f"Small vs family-size units: {title}")
        ax.set_ylabel("Difference vs 2024Q2 (%)")
    fig.suptitle("Event study around the Sept 2024 credit controls (95% CI, clustered by district)", y=1.02)
    _save(fig, "05_event_study.png")


def segment_volume(seg: pd.DataFrame):
    d = seg.copy()
    d["date"] = pd.to_datetime(d["ym"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for segment, g in d.groupby("segment"):
        g = g.sort_values("date")
        base = g[(g["date"] >= "2024-01-01") & (g["date"] < "2024-07-01")]["deals"].mean()
        y = (100 * g["deals"] / base).rolling(3, min_periods=1).mean()
        ax.plot(g["date"], y, lw=2, label=segment, color=ORANGE if segment.startswith("small") else BLUE)
    ax.set_ylabel("Deals, 3-month avg (2024H1 = 100)")
    ax.set_title("Deal volume by unit size")
    ax.legend(frameon=False)
    _events(ax)
    _save(fig, "06_segment_volume.png")


def shap_plots(model, X: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """SHAP values from XGBoost's built-in TreeSHAP (pred_contribs), plotted with the shap package."""
    import shap
    import xgboost as xgb

    sample = X.sample(min(3000, len(X)), random_state=0)
    contribs = model.get_booster().predict(xgb.DMatrix(sample, enable_categorical=True), pred_contribs=True)
    values, base = contribs[:, :-1], contribs[:, -1]
    numeric = sample.copy()
    for c in numeric.columns:
        if isinstance(numeric[c].dtype, pd.CategoricalDtype):
            numeric[c] = numeric[c].cat.codes  # colour only; categories have no natural order
    names = [labels.get(c, c) for c in sample.columns]
    sv = shap.Explanation(values=values, base_values=base, data=numeric.to_numpy(dtype=float), feature_names=names)

    plt.figure()
    shap.plots.beeswarm(sv, max_display=14, show=False)
    plt.title("What drives the predicted unit price (SHAP, test period)")
    _save(plt.gcf(), "07_shap_summary.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ["age_years", "log_area"]):
        i = list(sample.columns).index(col)
        xv = sample[col].to_numpy(dtype=float)
        if col == "log_area":
            xv = np.exp(xv)
        ax.scatter(xv, values[:, i], s=4, alpha=0.3, color=BLUE)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Building age (years)" if col == "age_years" else "Floor area excl. parking (坪)")
        ax.set_ylabel("SHAP value (log unit price)")
    axes[0].set_title("Age effect")
    axes[1].set_title("Size effect")
    _save(fig, "08_shap_dependence.png")

    importance = (pd.DataFrame({"feature": sample.columns, "label": names,
                                "mean_abs_shap": np.abs(values).mean(axis=0)})
                    .sort_values("mean_abs_shap", ascending=False))
    importance.to_csv(config.REPORTS / "shap_importance.csv", index=False)
    return importance
