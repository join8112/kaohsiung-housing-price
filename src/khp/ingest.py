"""Read Kaohsiung records out of the MOI 實價登錄 season zip files.

Each season zip (e.g. downloaded from
https://plvr.land.moi.gov.tw/DownloadOpenData, 非本期下載, CSV) holds ~236 CSV
files for every city.  We only need two of them:

* ``E_LVR_LAND_A.CSV`` – Kaohsiung existing-home sales (不動產買賣)
* ``E_LVR_LAND_B.CSV`` – Kaohsiung pre-sale sales (預售屋買賣)

Quirks handled here:
* the server sometimes returns a small HTML error page with status 200 instead
  of a zip, so every file is checked for the ``PK`` zip signature;
* CSVs are UTF-8 with BOM and have two header rows (Chinese, then English);
* the same record never appears twice in theory, but we de-duplicate on the
  official serial number (編號) anyway.

Files are read straight from the zip in memory – nothing is extracted to disk.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from . import config

# Chinese header -> English column name.  Header names are normalised first
# (spaces and brackets removed) because they vary slightly between seasons.
COLUMN_MAP = {
    "鄉鎮市區": "district",
    "交易標的": "target",
    "土地位置建物門牌": "address",
    "土地移轉總面積平方公尺": "land_area_m2",
    "都市土地使用分區": "urban_zone",
    "非都市土地使用分區": "nonurban_zone",
    "非都市土地使用編定": "nonurban_use",
    "交易年月日": "trade_date_roc",
    "交易筆棟數": "trade_counts",
    "移轉層次": "floor_text",
    "總樓層數": "total_floors_text",
    "建物型態": "building_type_raw",
    "主要用途": "main_use",
    "主要建材": "material",
    "建築完成年月": "completion_roc",
    "建物移轉總面積平方公尺": "building_area_m2",
    "建物現況格局-房": "rooms",
    "建物現況格局-廳": "halls",
    "建物現況格局-衛": "baths",
    "建物現況格局-隔間": "partition",
    "有無管理組織": "management",
    "總價元": "total_price",
    "單價元平方公尺": "unit_price_m2",
    "車位類別": "parking_type",
    "車位移轉總面積平方公尺": "parking_area_m2",
    "車位總價元": "parking_price",
    "備註": "note",
    "編號": "serial",
    "主建物面積": "main_area_m2",
    "附屬建物面積": "aux_area_m2",
    "陽台面積": "balcony_area_m2",
    "電梯": "elevator",
    "移轉編號": "transfer_id",
    # pre-sale only
    "建案名稱": "project_name",
    "棟及號": "unit_no",
    "解約情形": "cancellation",
}


def _normalise_header(name: str) -> str:
    name = name.replace("﻿", "").strip()
    return re.sub(r"[\s()（）]", "", name)


def is_zip(path: Path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(2) == b"PK"


def _find_member(zf: zipfile.ZipFile, wanted: str) -> str | None:
    for name in zf.namelist():
        if Path(name).name.lower() == wanted.lower():
            return name
    return None


def read_csv_from_zip(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    raw = zf.read(member)
    for enc in ("utf-8-sig", "cp950"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - would mean an unknown encoding
        raise ValueError(f"cannot decode {member}")
    # first row = Chinese header, second row = English header -> skip it
    df = pd.read_csv(io.StringIO(text), dtype=str, skiprows=[1], keep_default_na=False)
    df.columns = [_normalise_header(c) for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    return df


def load_city_files(raw_dir: Path = config.DATA_RAW, city: str = config.CITY_CODE):
    """Return (existing-home sales, pre-sale sales, per-file log)."""
    sales, presale, log = [], [], []
    zips = sorted(p for p in Path(raw_dir).glob("*") if p.suffix.lower() == ".zip")
    for path in zips:
        entry = {"file": path.name, "bytes": path.stat().st_size}
        if not is_zip(path):
            entry["status"] = "not a zip (probably an HTML error page) – skipped"
            log.append(entry)
            continue
        with zipfile.ZipFile(path) as zf:
            for kind, bucket in (("A", sales), ("B", presale)):
                member = _find_member(zf, f"{city}_LVR_LAND_{kind}.CSV")
                if member is None:
                    entry[f"rows_{kind}"] = 0
                    continue
                df = read_csv_from_zip(zf, member)
                df["source_file"] = path.name
                bucket.append(df)
                entry[f"rows_{kind}"] = len(df)
        entry["status"] = "ok"
        log.append(entry)

    if not sales:
        raise FileNotFoundError(f"no usable season zip files found in {raw_dir}")

    def _combine(frames):
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        before = len(df)
        if "serial" in df.columns:
            df = df.drop_duplicates(subset="serial", keep="last")
        df.attrs["duplicates_dropped"] = before - len(df)
        return df

    sales_df, presale_df = _combine(sales), _combine(presale)
    log_df = pd.DataFrame(log)
    return sales_df, presale_df, log_df


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    sales, presale, log = load_city_files()
    sales.to_parquet(config.RAW_PARQUET, index=False)
    if len(presale):
        presale.to_parquet(config.PRESALE_PARQUET, index=False)
    log.to_csv(config.REPORTS / "ingest_log.csv", index=False)
    print(log.to_string(index=False))
    print(f"\nexisting-home rows: {len(sales):,} (duplicates dropped: {sales.attrs.get('duplicates_dropped', 0)})")
    print(f"pre-sale rows:      {len(presale):,}")


if __name__ == "__main__":
    main()
