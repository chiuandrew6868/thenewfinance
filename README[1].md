# Finance Strategy Backtester

這是一個 Streamlit 單頁網站。使用者選擇台泥、台積電或元大高股息 0056 後，系統會載入專案內建的 Google Trends CSV，並從 Yahoo Finance 取得股價資料進行回測。

## 功能

- 股票選擇：台泥 `1101.TW`、台積電 `2330.TW`、元大高股息 `0056.TW`
- Yahoo Finance 自動取得股價資料
- Google Trends 使用內建 CSV，不自動抓取 Google Trends，也不需要使用者上傳
- 可勾選使用 Google Trends、MACD、RSI
- 使用可重現的自動參數掃描，不依賴 Optuna
- 自動找出最佳、最差、理論常用三組參數
- 顯示策略報酬、買進持有報酬、年化報酬、最大回撤、Sharpe、交易次數
- 顯示價格訊號、資金曲線、MACD、RSI、Google Trends 圖表

## 參數組合

- 最佳：自動掃描中策略報酬最高的組合
- 最差：自動掃描中策略報酬最低的組合
- 理論常用：MACD `12/26/9`，RSI 週期 `14`，RSI 區間 `50-75`

## 執行方式

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Community Cloud

1. 將本資料夾上傳到 GitHub repository。
2. 到 Streamlit Community Cloud 建立新 app。
3. Repository 選擇你的 GitHub repo。
4. Main file path 填入 `app.py`。
5. Deploy。

## 內建 Google Trends 資料

三份 CSV 需隨 repo 一起部署：

- `data/taiwan_cement_trends.csv`
- `data/tsmc_trends.csv`
- `data/yuanta_high_dividend_0056_trends.csv`
