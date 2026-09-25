# trading-lab

每日交易日報：**[reports/latest.md](reports/latest.md)**　｜　回報問題：Issues → New issue → 回報問題

## 流程

1. 下載 0050 成分股股價、月營收、財報（損益、資產負債、現金流）、估值、三大法人、融資、外資持股
2. 找出交易員會出手的型態
   - 趨勢突破：趨勢模板 ≥ 7 項、放量突破 20 日高點（Minervini、Livermore、O'Neil）
   - 多頭回檔：站穩 200 日線、回測 50 日線止跌（Steve Burns）
3. 每筆交易先訂好停損、目標（3R）與移動停利，模擬實際出場
4. 判官（邏輯迴歸）從過去交易學習哪些基本面、籌碼、技術、大盤條件與獲利有關，只用當時已完成的交易訓練
5. 逐年驗收＋最近 6 個月期末考，只看沒訓練過的資料
6. 今日訊號通過判官才列入交易計畫，附進出場價、部位與理由，寫入模擬帳戶追蹤

## 執行

```bash
pip install -r requirements.txt
python -m trend.lab
```

FinMind 免費額度每小時約 300 次，設定 `FINMIND_TOKEN`（repo Settings → Secrets）可提高上限。
