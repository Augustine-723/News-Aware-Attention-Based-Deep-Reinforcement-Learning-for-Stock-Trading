# V8.24 樣本外成對差異與區間

使用測試期逐股權益資料作為輸入。兩組必須有相同股票、相同且嚴格遞增的交易日、有限正值的權益。首日權益標準化後，按等額初始資金彙總獨立股票帳戶，再計算每日組合報酬。兩組以同一組日期區塊重抽樣，保留同日市場衝擊及兩組的成對關係。環狀區塊預設五日、5000 次，以三日和十日做敏感度比較。

主要指標為 treatment 的成本後 Sharpe 減 control 的成本後 Sharpe。沿用 250 日年化和 1.5% 年無風險利率。若某組全期或任一重抽樣的波動為零，結果會記錄原因與有效重抽樣次數。若有未定義的抽樣，保留原始樣本可計的點估計，整個區間標記為未定義。結果只描述目前樣本和固定種子的政策。

## 用既有真實行情結果核對統計程式

先以 V8.22 的 Buy & Hold 為 control，M0 為 treatment。兩組比較覆蓋 33 個報酬日及 3 檔股票。

| M0 種子 | ΔSharpe | 五日區塊 95% 描述性區間 |
|---|---:|---|
| 42 | +0.56 | 部分重抽樣零波動，無區間 |
| 100 | +0.02 | [−7.60, +5.05] |
| 2024 | −2.52 | [−5.49, −0.42] |

[三種區塊長度、三組種子的逐筆結果](../results/v8_24_paired_performance.json)

## 用於 M3 對 M0

各組輸入為逐股 JSON，值含依序對齊的 session_dates 和 equity。每次比較一對同種子模型，control 給 M0，treatment 給 M3。外層實驗仍須核對資料版本、政策 seed、樣本切割、成本和事件的可驗證時間。統計 CLI 核對權益與日期的結構，只處理已生成的回測結果。

```bash
python -m stock_rl.evaluation.paired_performance m0.json m3.json --block-length 5
```

## 真實公告進度

從本工作階段讀取 MOPS 歷史頁返回錯誤轉址。FinLab 列有歷史逐筆公告時間、標題與全文，官方取得流程要求登入。該說明未建立原文版本與逐筆歷史送達的對應。若收到經授權匯出檔，仍需核對這項證據。原定 TWSE／證期局樣本請求見既有草稿。目前保留 1 筆 SEC 個案、0 筆已驗證 MOPS 原件與 0 筆歷史送達憑據。兩條可取得路徑如下。

- [MOPS 歷史重大訊息](https://mops.twse.com.tw/mops/web/t05st01)
- [FinLab 重訊公告產品及登入方式](https://finlab.finance/data/important-announcements)
- [TWSE／證期局樣本申請草稿](../inquiries/twse_historical_disclosure_inquiry.md)
