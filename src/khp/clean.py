"""Turn raw 實價登錄 rows into an analysis-ready table of residential sales.

Every filtering step is logged so the README can show exactly how many
records were removed and why.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Small parsers (unit-tested in tests/test_clean.py)
# ---------------------------------------------------------------------------
_FULLWIDTH = str.maketrans("０１２３４５６７８９～－", "0123456789~-")
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def to_halfwidth(s: str) -> str:
    return s.translate(_FULLWIDTH) if isinstance(s, str) else s


def parse_roc_date(value) -> pd.Timestamp:
    """'1130921' -> 2024-09-21 ; '990105' -> 2010-01-05 ; day/month 00 -> 1."""
    if value is None:
        return pd.NaT
    s = re.sub(r"\D", "", str(value))
    if len(s) not in (6, 7):
        return pd.NaT
    year, month, day = int(s[:-4]) + 1911, int(s[-4:-2]), int(s[-2:])
    month = month or 1
    day = day or 1
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return pd.NaT


def chinese_to_int(text: str) -> int | None:
    """Convert Chinese numerals up to 999: '十二' -> 12, '二十三' -> 23, '一百零一' -> 101."""
    if not text:
        return None
    if text.isdigit():
        return int(text)
    total, current = 0, 0
    for ch in text:
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
        elif ch == "十":
            total += (current or 1) * 10
            current = 0
        elif ch == "百":
            total += (current or 1) * 100
            current = 0
        else:
            return None
    return total + current


def parse_floor(text) -> tuple[float, int, bool]:
    """Parse 移轉層次 -> (lowest above-ground floor, number of floors listed, whole-building flag).

    Examples: '五層' -> (5, 1, False); '十二層，十三層' -> (12, 2, False);
    '全' -> (nan, 0, True); '地下一層' -> (-1, 1, False).
    """
    if not isinstance(text, str) or not text.strip():
        return np.nan, 0, False
    t = to_halfwidth(text.strip())
    if t.startswith("全"):
        return np.nan, 0, True
    floors = []
    for part in re.split(r"[，,、]", t):
        part = part.strip()
        m = re.fullmatch(r"(地下)?([零〇一二兩三四五六七八九十百\d]+)層", part)
        if not m:
            continue
        n = chinese_to_int(m.group(2))
        if n is None:
            continue
        floors.append(-n if m.group(1) else n)
    if not floors:
        return np.nan, 0, False
    above = [f for f in floors if f > 0]
    return float(min(above) if above else min(floors)), len(floors), False


def parse_total_floors(text) -> float:
    if not isinstance(text, str):
        return np.nan
    m = re.search(r"([零〇一二兩三四五六七八九十百\d]+)層", to_halfwidth(text))
    if not m:
        return np.nan
    n = chinese_to_int(m.group(1))
    return float(n) if n is not None else np.nan


def parse_trade_counts(text) -> tuple[float, float, float]:
    """'土地2建物1車位1' -> (2, 1, 1)."""
    out = []
    for key in ("土地", "建物", "車位"):
        m = re.search(key + r"(\d+)", str(text))
        out.append(float(m.group(1)) if m else np.nan)
    return tuple(out)


_ROAD_RE = re.compile(r"區(.+?(?:大道|路|街))([一二三四五六七八九十]+段)?")


def extract_road(address) -> str | None:
    """'高雄市三民區建工路601巷1~30號' -> '建工路'; keeps 段 when present."""
    if not isinstance(address, str):
        return None
    m = _ROAD_RE.search(to_halfwidth(address))
    if not m:
        return None
    road = m.group(1)
    # drop village names that sometimes precede the road (e.g. '仁武里仁雄路')
    road = re.sub(r"^.+?[里村](?:\d+鄰)?(?=.+(?:大道|路|街)$)", "", road)
    road = re.sub(r"^\d+鄰(?=.+(?:大道|路|街)$)", "", road)
    return road + (m.group(2) or "")


def short_building_type(raw) -> str | None:
    if not isinstance(raw, str):
        return None
    for prefix, short in config.RESIDENTIAL_TYPES.items():
        if raw.startswith(prefix):
            return short
    return None


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


# ---------------------------------------------------------------------------
# Main cleaning routine
# ---------------------------------------------------------------------------
def clean_sales(raw: pd.DataFrame, start: str = "2022-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    log: list[dict] = []

    def step(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        prev = log[-1]["rows_after"] if log else len(raw)
        log.append({"step": name, "rows_after": len(frame), "removed": prev - len(frame)})
        return frame

    df = raw.copy()
    step("raw Kaohsiung rows (de-duplicated)", df)

    df = df[df["target"].str.contains("建物", na=False)]
    df = step("keep sales that include a building (drop land-only / parking-only)", df)

    df["btype"] = df["building_type_raw"].map(short_building_type)
    df = df[df["btype"].notna()]
    df = step("keep residential building types (大樓/華廈/公寓/透天厝/套房)", df)

    counts = df["trade_counts"].map(parse_trade_counts)
    df["n_land"] = [c[0] for c in counts]
    df["n_building"] = [c[1] for c in counts]
    df["n_parking"] = [c[2] for c in counts]
    df = df[df["n_building"] == 1]
    df = step("keep single-building deals (建物 = 1)", df)

    df["trade_date"] = df["trade_date_roc"].map(parse_roc_date)
    df = df[df["trade_date"] >= pd.Timestamp(start)]
    df = step(f"trade date on/after {start}", df)

    df["has_addition"] = df["note"].fillna("").str.contains("|".join(config.ADDITION_KEYWORDS)).astype(int)

    pattern = "|".join(config.SPECIAL_DEAL_KEYWORDS)
    special = df["note"].fillna("").str.contains(pattern)
    df = df[~special]
    df = step("drop special / non-arm's-length deals flagged in 備註", df)

    use = df["main_use"].fillna("")
    non_res_use = use.str.contains("商業|工業|辦公|店舖|廠房") & ~use.str.contains("住")
    df = df[~non_res_use]
    df = step("drop units whose main use is commercial / industrial", df)

    # ---- numeric fields -------------------------------------------------
    for col in ["land_area_m2", "building_area_m2", "total_price", "parking_area_m2",
                "parking_price", "rooms", "halls", "baths", "main_area_m2",
                "aux_area_m2", "balcony_area_m2"]:
        if col in df.columns:
            df[col] = _num(df[col])
        else:
            df[col] = np.nan
    df["parking_area_m2"] = df["parking_area_m2"].fillna(0)
    df["parking_price"] = df["parking_price"].fillna(0)

    df["area_ex_park_ping"] = (df["building_area_m2"] - df["parking_area_m2"]) * config.PING_PER_M2
    df["land_area_ping"] = df["land_area_m2"] * config.PING_PER_M2
    df["price_ex_park"] = df["total_price"] - df["parking_price"]
    df["has_parking"] = ((df["parking_area_m2"] > 0) | (df["n_parking"] > 0)
                         | df["parking_type"].fillna("").ne("")).astype(int)
    # parking included in the deal but not priced separately: price per ping is inflated
    df["parking_unpriced"] = ((df["has_parking"] == 1) & (df["parking_price"] <= 0)).astype(int)

    valid = (df["area_ex_park_ping"].between(5, 300) & (df["price_ex_park"] > 300_000))
    df = df[valid]
    df = step("valid area (5–300 坪 excl. parking) and price (> NT$300k)", df)

    df["unit_price"] = df["price_ex_park"] / df["area_ex_park_ping"] / 1e4  # 萬元 / 坪

    # ---- building attributes --------------------------------------------
    df["completion_date"] = df["completion_roc"].map(parse_roc_date)
    df["age_years"] = (df["trade_date"] - df["completion_date"]).dt.days / 365.25
    df.loc[df["age_years"].between(-3, 0), "age_years"] = 0.0  # handed over right after completion
    df.loc[df["age_years"] < -3, "age_years"] = np.nan

    floors = df["floor_text"].map(parse_floor)
    df["floor"] = [f[0] for f in floors]
    df["n_floors_transferred"] = [f[1] for f in floors]
    df["whole_building"] = [int(f[2]) for f in floors]
    df["total_floors"] = df["total_floors_text"].map(parse_total_floors)
    df["floor_ratio"] = df["floor"] / df["total_floors"]
    df.loc[~df["floor_ratio"].between(0, 1), "floor_ratio"] = np.nan

    df["elevator"] = df["elevator"].map({"有": 1, "無": 0}) if "elevator" in df else np.nan
    df.loc[df["btype"].isin(["大樓", "華廈"]) & df["elevator"].isna(), "elevator"] = 1
    df.loc[df["btype"].eq("公寓") & df["elevator"].isna(), "elevator"] = 0
    df["management"] = df["management"].map({"有": 1, "無": 0})
    df["partition"] = df["partition"].map({"有": 1, "無": 0})
    df["road"] = df["address"].map(extract_road)

    df["year"] = df["trade_date"].dt.year
    df["ym"] = df["trade_date"].dt.to_period("M").astype(str)
    df["quarter"] = df["trade_date"].dt.to_period("Q").astype(str)
    df["t"] = (df["trade_date"].dt.year - 2022) * 12 + df["trade_date"].dt.month - 1

    # ---- outliers: robust z-score of log unit price within district x type
    df["log_unit_price"] = np.log(df["unit_price"])
    grp = df.groupby(["district", "btype"])["log_unit_price"]
    med = grp.transform("median")
    mad = grp.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
    z = (df["log_unit_price"] - med) / mad.replace(0, np.nan)
    df = df[z.abs().fillna(0) <= 3.5]
    df = step("drop unit-price outliers (|robust z| > 3.5 within district × type)", df)

    keep = [
        "serial", "district", "road", "address", "trade_date", "ym", "quarter", "year", "t",
        "btype", "building_type_raw", "main_use", "urban_zone",
        "total_price", "price_ex_park", "unit_price", "log_unit_price",
        "area_ex_park_ping", "land_area_ping", "building_area_m2", "main_area_m2", "balcony_area_m2",
        "age_years", "floor", "total_floors", "floor_ratio", "n_floors_transferred", "whole_building",
        "rooms", "halls", "baths", "partition", "elevator", "management",
        "has_parking", "parking_unpriced", "has_addition", "parking_type", "parking_price", "n_parking",
        "note", "source_file",
    ]
    out = df[[c for c in keep if c in df.columns]].sort_values("trade_date").reset_index(drop=True)
    return out, pd.DataFrame(log)


def main() -> None:
    raw = pd.read_parquet(config.RAW_PARQUET)
    clean, log = clean_sales(raw)
    clean.to_parquet(config.CLEAN_PARQUET, index=False)
    log.to_csv(config.REPORTS / "cleaning_log.csv", index=False)
    print(log.to_string(index=False))
    print(f"\nclean rows: {len(clean):,}  ({clean['trade_date'].min():%Y-%m} – {clean['trade_date'].max():%Y-%m})")


if __name__ == "__main__":
    main()
