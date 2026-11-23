"""Project-wide paths and constants."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"          # put the season zip files here (not committed)
DATA_PROCESSED = ROOT / "data" / "processed"
DB_PATH = DATA_PROCESSED / "kaohsiung.db"
CLEAN_PARQUET = DATA_PROCESSED / "kaohsiung_clean.parquet"
RAW_PARQUET = DATA_PROCESSED / "kaohsiung_raw.parquet"
PRESALE_PARQUET = DATA_PROCESSED / "kaohsiung_presale_raw.parquet"
SQL_DIR = ROOT / "sql"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Kaohsiung City uses the letter E in the MOI open-data file names
# (E_LVR_LAND_A.CSV = existing-home sales, E_LVR_LAND_B.CSV = pre-sale sales).
CITY_CODE = "E"
CITY_NAME = "高雄市"

PING_PER_M2 = 1 / 3.30579  # 1 坪 = 3.30579 m²

# Residential building types kept for the analysis (建物型態 starts with one of these).
RESIDENTIAL_TYPES = {
    "住宅大樓": "大樓",
    "華廈": "華廈",
    "公寓": "公寓",
    "透天厝": "透天厝",
    "套房": "套房",
}

# Keywords in 備註 that mark non-arm's-length or otherwise non-comparable deals.
SPECIAL_DEAL_KEYWORDS = [
    "親友", "員工", "特殊關係", "二親等", "親屬", "債權債務", "抵債",
    "瑕疵", "凶宅", "非自然死亡",
    "法拍", "拍賣", "急買急賣", "政府機關", "公有", "地上權", "持分",
    "共有人", "毛胚", "含租約", "附租約", "建商與地主", "合建",
]

# Additions / rooftop extensions / unregistered space are common in Kaohsiung
# (about half of 透天厝 sales mention them) and are ordinary market deals, so they
# are kept and flagged with has_addition instead of being dropped.
ADDITION_KEYWORDS = ["增建", "頂加", "頂樓加蓋", "未登記"]

# Central bank (央行) selective credit control timeline used in the policy study.
POLICY_EVENTS = {
    "moral_suasion": pd.Timestamp("2024-08-15"),  # 央行道德勸說銀行自主控管不動產貸款總量（8 月中旬）
    "wave7": pd.Timestamp("2024-09-20"),          # 第七波選擇性信用管制生效
    "easing_2026_03": pd.Timestamp("2026-03-20"),  # 第 2 戶成數上限放寬（2026 年第 1 季理監事會）
    "easing_2026_09": pd.Timestamp("2026-09-18"),  # 第 2 戶成數上限 6 成 → 7 成
}

RANDOM_STATE = 42
# Remote mountain districts (e.g. 那瑪夏區, 桃源區) have only a handful of deals in
# four years: too few to estimate a district effect, so they are left out of modelling.
MIN_DISTRICT_DEALS = 10
