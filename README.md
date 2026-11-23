# 高雄房價分析與估價模型 Kaohsiung Housing Price Analysis & Valuation

> 🚧 進行中：程式已完成並以模擬資料測試，正在用實價登錄真實資料執行。完整結果與說明完成後更新。

以內政部實價登錄 2022–2026 年高雄市住宅買賣資料，完成：

1. **資料工程**：自動下載季檔、清理（民國日期、中文樓層、車位拆分、特殊交易剔除、離群值），寫入 SQLite
2. **SQL 分析**：行政區行情、年增率、路段排名、屋齡折價（CTE、window functions）
3. **特徵價格模型（hedonic regression）**：品質調整後的房價指數
4. **XGBoost 估價模型**：以時間切分做樣本外驗證，與特徵價格模型及 baseline 比較，SHAP 解釋
5. **政策分析**：央行 2024 年 9 月第七波選擇性信用管制對小坪數與家庭型住宅的差異影響（事件研究／DiD）
6. **Streamlit 估價網頁**

## 如何執行

```bash
pip install -r requirements.txt
python scripts/download_data.py      # 下載 111S1–115S3 季檔到 data/raw/
python run_pipeline.py               # ingest → clean → sql → model → policy → figures
streamlit run app/streamlit_app.py   # 估價網頁
python -m pytest                     # 單元測試
```

資料來源：內政部不動產交易實價查詢服務網（政府資料開放授權條款第 1 版）。
