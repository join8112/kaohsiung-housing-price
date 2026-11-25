# CLAUDE.md：高雄房價分析與估價模型

這份文件是給 Claude Code 的交接說明：專案目的、目前進度、設計決策、下一步。
使用者以繁體中文溝通；回覆請用繁體中文。程式碼、註解、commit message 用英文。

## 專案目的

作品集專案（放在履歷與 GitHub），目標職缺是資料分析師／資料科學家與經濟研究。
要展示的能力：真實政府資料的取得與清理、SQL、計量經濟（特徵價格模型、DiD／事件研究）、
機器學習（XGBoost、樣本外驗證、SHAP）、以及可以實際操作的成品（Streamlit 估價網頁）。
使用者最熟悉 pandas 與 numpy；程式保持清楚易讀，面試時要能逐段解釋。

## 目前進度（2026-11-23）

- [x] 程式架構全部寫好，並用「模擬資料」把整條 pipeline 跑通（ingest → clean → sql → model → policy → figures），Streamlit app 也用 AppTest 測過
- [x] `pytest` 解析函式單元測試 7 個全過
- [x] 真實資料（111S1–115S3）下載並跑完整條 pipeline（2026-11-23），結果見下方「真實資料結果」
- [x] README.md（中文）＋ README.en.md（英文），內嵌 reports/figures 的圖（2026-11-23）
- [x] 部署 Streamlit Community Cloud（2026-11-23，使用者帳號）：https://kaohsiung-housing-price-gkrgupx6fvvypxjv8cxobe.streamlit.app/
- [x] notebooks/walkthrough.ipynb：執行過的導覽 notebook（2026-11-23；由 `jupyter nbconvert --execute` 重新產生輸出）
- [x] 估價 API：`api/main.py`（FastAPI）＋ `src/khp/predictor.py`（serving 層）＋ `Dockerfile` ＋ `.github/workflows/ci.yml`，`tests/test_api.py` 17 個測試（2026-11-25）

## 真實資料結果（2026-11-23）

- 清理後 108,063 筆（2022-01 至 2026-08）。路名解析率 98.9%、屋齡缺值 0.9%、單價中位數約 28.6 萬／坪。
- **增建／頂樓加蓋／未登記不再當特殊交易刪除**（原本會刪掉約一半的透天厝），改成 `has_addition` 特徵（使用者 2026-11-23 確認）。
- 成交少於 10 筆的山區行政區（那瑪夏、桃源、田寮、茂林）不進模型（`config.MIN_DISTRICT_DEALS`）。
- 測試期 MdAPE：baseline 18.9%、特徵價格 OLS 14.0%、XGBoost 8.9%（±10% 內 54.9%、±20% 內 80.3%）。
- 政策分析：價格 pre-trend 檢定拒絕（F 檢定 p≈0.0001），小坪數在管制前就相對走高。
  單純 DiD +3.2%（wild cluster bootstrap p=0.007），但加入小坪數線性趨勢後只剩 +1.7%（p≈0.10）。
  交易量沒有顯著差異。README 要說「沒有穩健證據顯示管制對小坪數價格有差異效果」。

## 下一步（照順序）

1. 下載資料：`python scripts/download_data.py`（111S1–115S3，共 19 季，每季約 15MB）。
   下載網址格式未經實測；若失敗，請使用者從 https://plvr.land.moi.gov.tw/DownloadOpenData
   「非本期下載」手動下載 CSV 版，放進 `data/raw/`（檔名不拘）。
2. `python run_pipeline.py`（可用 `--from model` 等從中間步驟重跑）。
3. **檢查真實資料**，模擬資料測不出這些：
   - `reports/ingest_log.csv`：每季都有讀到 `E_LVR_LAND_A.CSV`；欄名對應（`src/khp/ingest.py` 的 `COLUMN_MAP`）有沒有漏欄
   - `reports/cleaning_log.csv`：每一步刪除的筆數是否合理（特殊交易關鍵字是否誤刪太多）
   - `road` 欄位解析成功率（地址有遮蔽，目標 > 90%）
   - 車位：`parking_unpriced` 比例；單價分布是否合理（高雄多數落在 15–45 萬／坪）
   - 屋齡缺值比例（`completion_roc` 空白）
4. 看模型結果（`reports/model_comparison.csv`），XGBoost 應該贏過特徵價格 OLS 與 baseline；必要時調整特徵或 `PARAM_GRID`。
5. 政策分析（`reports/policy/`）：先看事件研究圖的前期係數（pre-trend）。結果要誠實陳述，不顯著就說不顯著。
   叢集標準誤只有約 30 多個行政區，pre-trend 的 Wald 檢定可能不穩，可改用 F 檢定或 wild cluster bootstrap 作為穩健性檢查。
6. 寫 README（見下方「README 要點」），commit 並 push。

## 專案結構

```
scripts/download_data.py   下載季檔
run_pipeline.py            一鍵執行全部步驟
src/khp/config.py          路徑、常數、政策日期
src/khp/ingest.py          從 zip 讀高雄檔（E_LVR_LAND_A / B），檢查 PK 檔頭、略過英文第二列表頭、依「編號」去重
src/khp/clean.py           解析民國日期、中文樓層、交易筆棟數、路名；過濾與離群值；cleaning log
src/khp/database.py        寫入 SQLite（data/processed/kaohsiung.db），執行 sql/1*.sql，結果存 reports/sql/
sql/                       schema、index、6 支分析查詢（CTE、window function 算中位數、LAG 算年增率、DENSE_RANK）
src/khp/features.py        特徵、路段 target encoding（out-of-fold）
src/khp/modeling.py        特徵價格模型（月份固定效果 → 品質調整房價指數）＋ baseline / OLS / XGBoost 樣本外比較，最終模型存 models/
src/khp/policy.py          信用管制事件研究與 DiD
src/khp/plots.py           README 用的圖（reports/figures/）
src/khp/predictor.py       估價 API 的 serving 層：載入 models/、組特徵（和訓練、Streamlit 完全一致）、SHAP
app/streamlit_app.py       估價網頁：streamlit run app/streamlit_app.py
api/main.py                FastAPI：uvicorn api.main:app --reload（/docs 有 Swagger UI）
Dockerfile、render.yaml     容器（python:3.13-slim、非 root、監聽 $PORT）與 Render Blueprint
.github/workflows/ci.yml   CI：Python 3.12/3.13 跑 pytest；另一 job 建映像、等 /health、打 /predict
tests/                     解析函式、wild cluster bootstrap 與 API 的單元測試：python -m pytest
```

## 重要設計決策

- **資料範圍**：2022-01 起的高雄市不動產買賣（成屋，A 檔）。只保留含建物、住宅型態（大樓／華廈／公寓／透天厝／套房）、單一建物的交易。
- **價格定義**：單價 = （總價 − 分開計價的車位價）÷（建物面積 − 車位面積），單位萬元／坪。
  車位有但沒有分開計價時，標記 `parking_unpriced = 1` 並當作特徵，而不是刪掉。
- **特殊交易**：`config.SPECIAL_DEAL_KEYWORDS` 比對備註（親友、瑕疵、法拍、持分……）。增建／頂樓加蓋／未登記（`config.ADDITION_KEYWORDS`）不刪除，改成 `has_addition` 特徵。
- **離群值**：行政區 × 型態內，log 單價的 robust z（MAD）超過 3.5 就刪除。
- **模型驗證採時間切分**：train < 2025-07、valid 2025-07～12（調參與 early stopping）、test ≥ 2026-01。
  報告的是 test 期的總價誤差（MdAPE、MAPE、誤差在 ±10%／±20% 內的比例）。
- **路段行情**：(行政區, 路名) 的 log 單價平滑平均（m-estimate，m=20），訓練資料用 5-fold out-of-fold 避免洩漏；沒看過的路名退回行政區平均。
- **app 的預測區間**：用 test 期 log 殘差的 10%／90% 分位數。

### 政策分析設計（src/khp/policy.py 的 docstring 有完整說明）

- 事件：2024-08 中旬央行道德勸說銀行控管不動產貸款總量；**2024-09-20 第七波選擇性信用管制生效**
  （第 2 戶貸款成數上限 6→5 成並擴及全國、名下有房者第 1 戶無寬限期、第 3 戶以上／法人／高價住宅 4→3 成）。
  **2026-03-20 第 2 戶成數放寬為 6 成**；2026-09-18 再放寬為 7 成。
- 設計：小坪數（套房或 1–2 房，投資客與第二屋比例較高）對照 3 房以上（自住為主，首購與新青安不受限）。
  只用公寓類（大樓／華廈／公寓／套房），樣本期間 2022-01-01 至 2026-03-19（第一次放寬前）。
- 價格：log 單價，控制特徵價格變數 ＋ 行政區 FE ＋ 月份 FE ＋ small × 事件季虛擬變數（基期 2024Q2）。
- 交易量：行政區 × 區隔 × 季的筆數，PPML（Poisson），行政區×季 FE 與 行政區×區隔 FE。
- 標準誤以行政區為 cluster。這只能估「差異效果」：兩組同時受到的衝擊（例如銀行房貸總量控管）會被 FE 吸收，README 要寫清楚這個限制。

## README 要點（之後要寫）

開頭三行講清楚：問題、方法、結果（附數字）。接著依序放：資料與清理流程（附 cleaning log 表）、SQL 分析、
房價指數圖、模型比較表與 SHAP 圖、政策事件研究圖與解讀（含限制）、app 截圖與連結、如何重現。
資料來源標註：內政部不動產交易實價查詢服務網，政府資料開放授權條款第 1 版。

## 慣例

- 使用者的電腦是 Windows；指令請用 `python`（不是 `python3`）。中文字型 `plots.py` 會自動找 Microsoft JhengHei。
- `data/raw/`、`*.db`、大型 parquet 不 commit（見 .gitignore）。`models/` 與 `reports/` 的小檔案要 commit，Streamlit Cloud 部署需要。
- 每完成一個階段就 commit。
- Streamlit Cloud 用 `app/requirements.txt`（只裝 app 需要的 4 個套件），xgboost 版本要和存 `models/xgb_unit_price.json` 時一致；換版本重訓後要一起更新。
- API 的容器用 `api/requirements.txt`（fastapi、uvicorn、xgboost、pandas、numpy），xgboost 版本同上，重訓換版本時三個 requirements 檔要一起改。
- 模型輸入格式改動時，`src/khp/features.py`、`src/khp/predictor.py`、`app/streamlit_app.py` 三處要同步（app 之後可考慮改成呼叫 predictor）。
