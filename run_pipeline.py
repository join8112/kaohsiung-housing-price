"""Run the whole project end to end.

    python run_pipeline.py                 # all steps
    python run_pipeline.py --from model    # start from a later step

Steps: ingest -> clean -> sql -> model -> policy -> figures
Put the season zip files from https://plvr.land.moi.gov.tw/DownloadOpenData
(非本期下載, CSV) into data/raw/ first.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
# Windows consoles/pipes default to cp950, which cannot print characters such as "²" or "×"
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

from khp import clean, config, database, ingest, modeling, plots, policy  # noqa: E402
from khp.features import LABELS, to_model_matrix  # noqa: E402

STEPS = ["ingest", "clean", "sql", "model", "policy", "figures"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", choices=STEPS, default="ingest")
    args = parser.parse_args()
    todo = STEPS[STEPS.index(args.start):]
    state: dict = {}

    def banner(name):
        print(f"\n=== {name} " + "=" * (60 - len(name)))

    t0 = time.time()
    if "ingest" in todo:
        banner("ingest")
        ingest.main()
    if "clean" in todo:
        banner("clean")
        clean.main()
    if "sql" in todo:
        banner("sql")
        database.main()
    if "model" in todo:
        banner("model")
        state["model"] = modeling.run()
    if "policy" in todo:
        banner("policy")
        state["policy"] = policy.run()
    if "figures" in todo:
        banner("figures")
        sql_dir = config.REPORTS / "sql"
        plots.market_overview(pd.read_csv(config.REPORTS / "hedonic_price_index.csv"))
        plots.district_prices(pd.read_csv(sql_dir / "11_district_summary.csv"))
        plots.age_curve(pd.read_csv(sql_dir / "14_age_premium.csv"))
        plots.segment_volume(pd.read_csv(sql_dir / "15_segment_monthly.csv"))
        plots.model_comparison(pd.read_csv(config.REPORTS / "model_comparison.csv", index_col=0),
                               pd.read_parquet(config.REPORTS / "test_predictions.parquet"))
        pol = config.REPORTS / "policy"
        plots.event_study(pd.read_csv(pol / "event_study_price.csv"), pd.read_csv(pol / "event_study_volume.csv"))
        if "model" in state:
            m = state["model"]
            imp = plots.shap_plots(m["model_tv"], to_model_matrix(m["test"], m["categories"]), LABELS)
            print(imp.head(10).to_string(index=False))
        print("figures written to", config.FIGURES)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
