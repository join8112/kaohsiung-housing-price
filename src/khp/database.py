"""Load the cleaned data into SQLite and run the analysis queries in sql/."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import config

SQL_OUT = config.REPORTS / "sql"


def build_database(clean: pd.DataFrame, db_path: Path = config.DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    df = clean.copy()
    df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")
    with sqlite3.connect(db_path) as con:
        con.executescript((config.SQL_DIR / "00_schema.sql").read_text(encoding="utf-8"))
        cols = [r[1] for r in con.execute("PRAGMA table_info(transactions)")]
        df[cols].to_sql("transactions", con, if_exists="append", index=False)
        con.executescript((config.SQL_DIR / "01_indexes.sql").read_text(encoding="utf-8"))


def run_queries(db_path: Path = config.DB_PATH) -> dict[str, pd.DataFrame]:
    """Run every numbered analysis query (sql/1*.sql …) and save its result as CSV."""
    SQL_OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    with sqlite3.connect(db_path) as con:
        for path in sorted(config.SQL_DIR.glob("[1-9]*.sql")):
            df = pd.read_sql_query(path.read_text(encoding="utf-8"), con)
            df.to_csv(SQL_OUT / f"{path.stem}.csv", index=False, encoding="utf-8-sig")
            results[path.stem] = df
            print(f"{path.name:<40} -> {len(df):>6} rows")
    return results


def main() -> None:
    clean = pd.read_parquet(config.CLEAN_PARQUET)
    build_database(clean)
    run_queries()


if __name__ == "__main__":
    main()
