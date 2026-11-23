"""高雄房價估價小工具 (Streamlit).

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

LABELS_ZH = {
    "district": "行政區", "btype": "建物型態", "log_area": "坪數", "log_land_area": "土地持分",
    "age_years": "屋齡", "floor": "樓層", "total_floors": "總樓層", "floor_ratio": "樓層位置",
    "rooms": "房數", "halls": "廳數", "baths": "衛浴數", "elevator": "電梯", "management": "管理組織",
    "partition": "隔間", "has_parking": "車位", "parking_unpriced": "車位未分開計價", "t": "交易時間",
    "road_te": "路段行情", "road_n": "路段成交量",
}

st.set_page_config(page_title="高雄房價估價", page_icon="🏠", layout="wide")


@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model(MODELS / "xgb_unit_price.json")
    meta = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    roads = pd.read_csv(MODELS / "road_encoding.csv")
    defaults = pd.read_csv(MODELS / "district_defaults.csv")
    return booster, meta, roads, defaults


@st.cache_data
def load_reports():
    index = pd.read_csv(REPORTS / "hedonic_price_index.csv")
    quarterly = pd.read_csv(REPORTS / "sql" / "12_district_quarterly_yoy.csv")
    return index, quarterly


booster, meta, roads, defaults = load_model()
index, quarterly = load_reports()
cats = meta["categories"]

st.title("🏠 高雄市住宅估價")
st.caption(
    f"以內政部實價登錄 {meta['data_from'][:7]} 至 {meta['data_through'][:7]} 共 "
    f"{meta['n_training_rows']:,} 筆高雄住宅買賣訓練的 XGBoost 模型。"
    f"最新一段期間的樣本外測試：總價誤差中位數 {meta['test_metrics']['MdAPE_%']}%，"
    f"{meta['test_metrics']['within_10%']}% 的物件誤差在 10% 以內。僅供參考，不構成估價報告。"
)

tab_est, tab_market = st.tabs(["估價", "區域行情"])

with tab_est:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        district = st.selectbox("行政區", cats["district"],
                                index=cats["district"].index("鳳山區") if "鳳山區" in cats["district"] else 0)
        road_options = ["（不指定）"] + sorted(roads.loc[roads["district"] == district, "road"].dropna().unique())
        road = st.selectbox("路名", road_options, help="只列出訓練資料中出現過的路段；不指定則用行政區平均行情")
        btype = st.selectbox("建物型態", cats["btype"],
                             index=cats["btype"].index("大樓") if "大樓" in cats["btype"] else 0)
        c1, c2 = st.columns(2)
        area = c1.number_input("建坪（不含車位）", min_value=8.0, max_value=200.0, value=35.0, step=1.0)
        age = c2.number_input("屋齡（年）", min_value=0.0, max_value=60.0, value=10.0, step=1.0)
        if btype == "透天厝":
            total_floors = st.number_input("透天樓層數", 1, 7, 4)
            floor = np.nan
        else:
            c3, c4 = st.columns(2)
            total_floors = c3.number_input("總樓層", 1, 50, 15 if btype == "大樓" else 5)
            floor = c4.number_input("所在樓層", 1, int(total_floors), min(7, int(total_floors)))
        c5, c6, c7 = st.columns(3)
        rooms = c5.number_input("房", 0, 8, 1 if btype == "套房" else 3)
        halls = c6.number_input("廳", 0, 5, 1 if btype == "套房" else 2)
        baths = c7.number_input("衛", 0, 6, 1 if btype == "套房" else 2)
        parking = st.radio("車位", ["無", "有，分開計價", "有，含在總價內"], horizontal=True)
        management = st.checkbox("有管理組織", value=btype in ("大樓", "華廈", "套房"))
        elevator_default = btype in ("大樓", "華廈")
        elevator = st.checkbox("有電梯", value=elevator_default, disabled=btype in ("大樓", "華廈", "公寓"))
        with st.expander("進階：土地持分"):
            dflt = defaults[(defaults["district"] == district) & (defaults["btype"] == btype)]
            land_default = float(dflt["land_area_ping"].iloc[0]) if len(dflt) and pd.notna(dflt["land_area_ping"].iloc[0]) else 10.0
            land = st.number_input("土地持分（坪）", 0.0, 200.0, round(land_default, 1), step=0.5,
                                   help="預設為同行政區、同型態成交物件的中位數")

    # --- build the feature row exactly as in training ------------------------
    key = roads[(roads["district"] == district) & (roads["road"] == road)]
    if len(key):
        road_te, road_n = float(key["te"].iloc[0]), float(key["count"].iloc[0])
    else:
        road_te = meta["district_mean_log_price"].get(district, meta["global_mean_log_price"])
        road_n = 0.0
    row = {
        "district": district, "btype": btype,
        "log_area": np.log(area), "log_land_area": np.log1p(land), "age_years": age,
        "floor": floor, "total_floors": float(total_floors),
        "floor_ratio": (floor / total_floors) if btype != "透天厝" else np.nan,
        "rooms": rooms, "halls": halls, "baths": baths,
        "elevator": float(elevator if btype not in ("大樓", "華廈", "公寓") else elevator_default),
        "management": float(management), "partition": 1.0,
        "has_parking": float(parking != "無"), "parking_unpriced": float(parking == "有，含在總價內"),
        "t": float(meta["t_latest"]), "road_te": road_te, "road_n": road_n,
    }
    X = pd.DataFrame([row])[meta["features"]]
    for c in ("district", "btype"):
        X[c] = pd.Categorical(X[c], categories=cats[c])
    for c in meta["features"]:
        if c not in ("district", "btype"):
            X[c] = X[c].astype(float)
    dm = xgb.DMatrix(X, enable_categorical=True)
    pred_log = float(booster.predict(dm)[0])
    q10, q90 = meta["interval_log_resid_q10_q90"]
    unit, lo, hi = np.exp(pred_log), np.exp(pred_log + q10), np.exp(pred_log + q90)

    with right:
        st.subheader("估價結果")
        m1, m2 = st.columns(2)
        m1.metric("預估單價（萬元／坪）", f"{unit:,.1f}", help="不含車位")
        m2.metric("預估總價（萬元，不含車位）", f"{unit * area:,.0f}")
        st.write(f"80% 區間：單價 **{lo:,.1f} – {hi:,.1f}** 萬元／坪，總價 **{lo * area:,.0f} – {hi * area:,.0f}** 萬元")
        if parking == "有，含在總價內":
            st.info("車位沒有分開計價時，單價會包含車位價值，所以通常偏高。")
        if road != "（不指定）" and road_n < 20:
            st.warning(f"{road} 的成交筆數較少（{road_n:.0f} 筆），路段行情已向行政區平均收斂。")

        contribs = booster.predict(dm, pred_contribs=True)[0]
        contrib = pd.Series(contribs[:-1], index=meta["features"])
        contrib = contrib.drop("t").rename(LABELS_ZH)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
        st.markdown("**主要影響因素**（相對於全市平均物件，對單價的影響 %）")
        st.bar_chart(pd.DataFrame({"影響 %": 100 * (np.exp(top) - 1)}), horizontal=True)

with tab_market:
    st.subheader("全市行情（品質調整後的房價指數）")
    idx = index.assign(月份=pd.to_datetime(index["ym"])).set_index("月份")
    st.line_chart(idx[["hedonic_index", "median_index"]].rename(
        columns={"hedonic_index": "品質調整指數", "median_index": "單價中位數指數"}))
    st.caption("品質調整指數控制了坪數、屋齡、型態、樓層與行政區，比單純的中位數更能反映同一類房子的價格變化。")
    st.subheader("各行政區每季單價中位數")
    pick = st.multiselect("行政區", sorted(quarterly["district"].unique()),
                          default=[d for d in ["鳳山區", "左營區", "楠梓區", "苓雅區"] if d in set(quarterly["district"])])
    if pick:
        q = quarterly[quarterly["district"].isin(pick)].pivot(index="quarter", columns="district",
                                                              values="median_unit_price")
        st.line_chart(q)
