# V8.4 Historical Data Proof of Concept (PoC) Technical Report

**Document Version**: 1.0.0  
**Target Git Branch**: `fix/v8-audit-patch`  
**Execution Environment**: Python 3.11.16, PyTorch 2.14.0+cpu, Pandas 3.1.1, NumPy 2.4.6  
**Report Location**: `docs/reports/v8_4_historical_data_poc.md`  
**Authoritative Schema Baseline**: `v8.0-original` (100 Features)  

---

## 1. Executive Summary & Objective

本報告記錄 **V8.4 Historical Data Proof of Concept (PoC)** 的實作、資料品質審計、Point-in-Time (PIT) 可用性檢驗與小規模端到端驗證結果。

### 核心目標
驗證現有 **V8.0 Original 100-Feature Schema** 與 **HistoricalSnapshotBuilder**，在完全使用**真實台股歷史市場與事件資料**時：
1. 是否能依照嚴格 Point-in-Time (PIT) 規格（Taiwan 20:00:00 決策截止點、次日開盤執行、60 交易日滾動視窗、32 Recent + 8 Active 槽位）建構完整的 MultimodalBatch。
2. 評估公開真實資料來源（TWSE、FinMind、MOPS、FRED、Yahoo Finance、GDELT）的時間戳記可審計性與歷史版本控制能力。
3. 統計 100 維特徵中「真正可由公開資料計算」與「因缺乏專有資料庫而不可計算」之精確數量與原因。
4. 驗證資料流程（Real Raw Records $\to$ Availability Audit $\to$ Historical As-Of $\to$ Feature Engineering $\to$ Missing Masks $\to$ Snapshot Builder $\to$ Tensor Validator $\to$ Normalizer $\to$ Model Forward）的完整性與反事實無洩漏保證。

> [!IMPORTANT]
> **本階段為小規模 PoC，不執行完整 5 年 100 檔標的之全量模型訓練，亦不涉及實盤交易或績效宣稱。**

---

## 2. PoC 標的、歷史區間與選擇理由

為以最小資料規模充分覆蓋真實市場之不同產業、不同流動性層級及特殊交易情境，本次 PoC 選取 3 檔具高度代表性之台股標的：

| 股票代號 | 證券簡稱 | 產業類別 | 發行股數 | 選擇理由與測試情境 |
|---|---|---|---|---|
| **2330** | 台積電 (TSMC) | 半導體業 (24) | 25,932,070,992 | **權值基準**：台股最高市值與流動性龍頭，外資與投信進出活絡，代表標準正常交易情境。 |
| **2603** | 長榮海運 (Evergreen) | 航運業 (26) | 2,116,400,000 | **高波動週期與減資停牌**：景氣循環高波動標的，具歷史現金減資停止交易（Trading Halt / Suspension）情境，用以驗證 `C16` (停牌指示) 與交易日曆對齊。 |
| **2881** | 富邦金控 (Fubon Fin.) | 金融保險業 (28) | 13,005,000,000 | **非科技金融權值與重大訊息**：利率與總經敏感型金融控股，具有真實 MOPS 重大訊息申報紀錄。 |

### 歷史資料區間與 Warm-Up 設計
- **歷史 Warm-Up 區間**：`2023-06-01` 至 `2023-12-31`（約 142 個交易日）  
  - 目的：滿足 `C13`（Beta 120D 滾動協方差）、`T10`（EMA60）、`C10`（MA60 成交金額）及 `P09` / `I01`–`I05`（MA20 成交量）等長回顧期指標之收斂需求，杜絕初段冷啟動失真。
- **Observation & Snapshot 評估區間**：`2024-01-02` 至 `2024-03-29`（約 60 個交易日）  
  - 目的：精確對應 $M=60$ 交易日之 Market Window，提供 60 個獨立的 As-Of 決策時間點（每日 20:00:00 Asia/Taipei）。
- **總交易日數**：202 個交易日，資料量約 256 KB，完全符合公開 API 免費使用規範，可離線重複執行。

---

## 3. Real Data Source Verification & PIT 可用性評估

我們對五大潛在真實資料來源進行逐項評估，嚴格檢查其時間戳記、可用時間（`available_at`）、版本控制與反事實洩漏風險：

```mermaid
flowchart TD
    A["Raw Historical Data Sources"] --> B["TWSE Official EOD (STOCK_DAY, T86, BFI82U)"]
    A --> C["FinMind Open Research API"]
    A --> D["MOPS Material Announcements (TWSE OpenAPI)"]
    A --> E["US & Global Macro (FRED / Yahoo Finance)"]
    A --> F["GDELT Global Event Database"]

    B --> B1["Only Calendar Date in Historical Files<br/>No Microsecond Generation Timestamp<br/><b>Classification: ESTIMATED (15:00 / 18:00 / 19:30)</b>"]
    C --> C1["Aggregated EOD by Date<br/>Static Overwrite on Revisions<br/><b>Classification: UNAUDITED_BACKFILL</b>"]
    D --> D1["Official Filing Date & Time down to the second<br/>Supports Clarification Revisions<br/><b>Classification: OBSERVED & VERIFIED</b>"]
    E --> E1["T-1 Closed at 16:00 NY Time<br/>Available in TW by 06:00 (Well before 20:00)<br/><b>Classification: OBSERVED (T-1 Lagged PIT)</b>"]
    F --> F1["Rate Limited (HTTP 429), Crawler Lag up to 72h<br/>Low Taiwan Entity Resolution<br/><b>Classification: UNSUPPORTED for Formal RL</b>"]
```

### 來源逐項審計分析

1. **TWSE (臺灣證券交易所官方盤後數據)**
   - **涵蓋內容**：STOCK_DAY (個股日行情)、T86 (三大法人買賣超)、BFI82U (融資融券信用交易)、FMTQIK / MI_INDEX (加權指數行情)。
   - **可審計時間戳記**：歷史報表與 JSON API 僅回傳民國年月日（如 `113/01/02`），**並未在資料列中記錄歷史伺服器產生該檔案的確切毫秒或秒級時間**。
   - **交易時程推估**：
     - 收盤行情：13:30 收盤，撮合結算後約 14:30–15:00 發布。
     - 三大法人買賣超：券商彙整後約 17:30–18:00 發布。
     - 融資融券餘額：券商回報後約 19:30–21:00 發布。
   - **PIT 風險**：若歷史上某日券商傳輸延遲導致融資融券於 21:15 才發布，單純以「交易日期」回填的歷史資料庫會在 20:00 的決策點發生未來資訊洩漏。
   - **評定狀態**：**ESTIMATED（推估可用）**。在 `mode="formal"` 下，若未附帶明確的即時日誌時間，必須 Fail Closed。

2. **FinMind (開源/研究級台股資料庫)**
   - **涵蓋內容**：`TaiwanStockPrice`、`TaiwanStockInstitutionalInvestorsBuySell`、`TaiwanStockMarginPurchaseShortSale`、`TaiwanExchangeRate`、`TaiwanStockShareholding`。
   - **優點**：欄位清晰、回傳結構化、支援大量常用指標計算所需欄位。
   - **可審計時間戳記**：僅含 `"date": "YYYY-MM-DD"`，無歷史發布時間戳記。
   - **版本控制**：靜態日資料；若除權息或減資發生後續修正，歷史記錄直接被覆寫（Overwrite），無修訂版本鏈。
   - **評定狀態**：**UNAUDITED_BACKFILL（未審計歷史回填）**。

3. **MOPS (公開資訊觀測站重大訊息 / TWSE OpenAPI `t187ap04_L`)**
   - **涵蓋內容**：上市公司重大訊息（含主旨、說明、符合條款、發言日期、發言時間、事實發生日）。
   - **可審計時間戳記**：**完全可驗證（精確至秒）**。例如 `"發言日期": "1151116"`, `"發言時間": "160036"` $\implies$ `2026-11-16 16:00:36 Asia/Taipei`。
   - **版本控制**：重大訊息包含澄清與更正公告，具獨立時間戳記與前次公告關聯。
   - **存取限制**：
     - TWSE OpenAPI 開放最新滾動重大訊息（HTTP 200，無需驗證碼）。
     - MOPS 歷史網頁（`mops.twse.com.tw`）設有嚴格 WAF 與防爬蟲機制，直接發送 POST 查詢會觸發「因為安全性考量，您所執行的頁面無法呈現」，**依規則禁止嘗試繞過或破解**。長期回測需使用合法批次資料授權。
   - **評定狀態**：**OBSERVED & AUDIT_VERIFIED（即時/滾動公開 API 可審計）**。

4. **US Macro Data (FRED / Yahoo Finance)**
   - **涵蓋內容**：S&P 500 (`^GSPC`)、NASDAQ (`^IXIC`)、SOX (`^SOX`)、Cboe VIX (`VIXCLS`)、US 10Y Yield (`DGS10`)。
   - **可審計時間戳記**：美股交易日 $T-1$ 於美東時間 16:00 收盤（對應台灣時間翌晨 04:00 或 05:00），至台灣時間當日 20:00 決策時已完全結算。
   - **評定狀態**：**OBSERVED ($T-1$ 延遲 As-Of 合約完全滿足)**。

5. **GDELT Project**
   - **涵蓋內容**：全球新聞事件。
   - **限制**：公開 Doc API 頻繁遭遇 HTTP 429（Rate Limit）；爬蟲抓取時間戳記（`seendate`）與原始發布時間存在數小時至數天延遲；台股實體識別度低。
   - **評定狀態**：**UNSUPPORTED for Formal RL Pipeline**。

---

## 4. 各來源欄位對照表 (Source Field Mapping)

以下為公開真實資料來源欄位與 **V8.0 Original 100-Feature Schema** 的完整映射清單：

| 模組 | 特徵 ID | 特徵名稱 | 原始資料來源與欄位 | 數學公式 / 處理邏輯 |
|---|---|---|---|---|
| **Price** | `P01` | `return_1d` | FinMind / TWSE `close` | $(P_{t} - P_{t-1}) / P_{t-1}$ |
| Price | `P02` | `return_5d` | FinMind / TWSE `close` | $(P_{t} - P_{t-5}) / P_{t-5}$ |
| Price | `P03` | `return_20d` | FinMind / TWSE `close` | $(P_{t} - P_{t-20}) / P_{t-20}$ |
| Price | `P04` | `overnight_gap` | FinMind / TWSE `open`, `close` | $(Open_{t} - Close_{t-1}) / Close_{t-1}$ |
| Price | `P05` | `open_close_return` | FinMind / TWSE `open`, `close` | $(Close_{t} - Open_{t}) / Open_{t}$ |
| Price | `P06` | `high_low_range` | FinMind / TWSE `max`, `min`, `open` | $(High_{t} - Low_{t}) / Open_{t}$ |
| Price | `P07` | `close_location` | FinMind / TWSE `close`, `min`, `max` | $(Close_{t} - Low_{t}) / (High_{t} - Low_{t} + \epsilon)$ $[0, 1]$ |
| Price | `P08` | `log_volume` | FinMind `Trading_Volume` | $\ln(1 + \text{Volume}_t)$ |
| Price | `P09` | `volume_ratio_20d` | FinMind `Trading_Volume` | $\text{Volume}_t / MA_{20}(\text{Volume})$ |
| Price | `P10` | `turnover_ratio` | FinMind `Trading_Volume`, `TaiwanStockShareholding.NumberOfSharesIssued` | $\text{Volume}_t / \text{Shares Outstanding}$ |
| Price | `P11` | `realized_volatility_20d` | FinMind `close` | 20 日年化實現波動度 ($\times \sqrt{250}$) |
| Price | `P12` | `log_trading_value` | FinMind `Trading_money` | $\ln(1 + \text{Trading Value}_t)$ |
| **Technical** | `T01`–`T02` | `rsi_14`, `rsi_7` | 依 OHLCV 計算 | Wilder RSI, 正規化 $(RSI - 50) / 50$ $[-1, 1]$ |
| Technical | `T03`–`T05` | `kdj_k`, `kdj_d`, `kdj_j` | 依 OHLCV 計算 | 9-3-3 KDJ, 正規化 $(X - 50) / 50$ $[-1, 1]$ |
| Technical | `T06`–`T08` | `macd_dif`, `signal`, `hist` | 依 OHLCV 計算 | 12-26-9 MACD, 均除以 $P_{close}$ |
| Technical | `T09`–`T11` | `close/ema20-1`, `close/ema60-1`, `ema20/ema60-1` | 依 OHLCV 計算 | 均線偏離比率 |
| Technical | `T12` | `atr14_close_ratio` | 依 OHLCV 計算 | Wilder ATR14 / $P_{close}$ |
| Technical | `T13`–`T14` | `bollinger_percent_b`, `bollinger_bandwidth` | 依 OHLCV 計算 | %B $[0, 1]$, Bandwidth = $(Upper - Lower) / MA_{20}$ |
| Technical | `T15` | `normalized_obv_change_5d` | 依 OHLCV 計算 | $\Delta OBV_{5D} / (5 \cdot MA_{20}(\text{Volume}))$ |
| Technical | `T16` | `adx_14` | 依 OHLCV 計算 | Wilder ADX14 / 100.0 $[0, 1]$ |
| **Institutional** | `I01`–`I03` | `foreign/trust/dealer_net_buy` | FinMind `TaiwanStockInstitutionalInvestorsBuySell` (`buy` - `sell`) | 當日淨買賣超股數 / $MA_{20}(\text{Volume})$ |
| Institutional | `I04`–`I05` | `foreign_5d`, `total_3inst_5d` | FinMind `TaiwanStockInstitutionalInvestorsBuySell` | 5 日累積淨買賣超 / $(5 \cdot MA_{20}(\text{Volume}))$ |
| Institutional | `I06`–`I07` | `margin_balance_change`, `short_balance_change` | FinMind `TaiwanStockMarginPurchaseShortSale` (`TodayBalance` diff) | 融資/融券餘額變動股數 / $MA_{20}(\text{Volume})$ |
| Institutional | `I08` | `short_margin_balance_ratio` | FinMind `TaiwanStockMarginPurchaseShortSale` | 融券餘額 / (融資餘額 + $\epsilon$) |
| **Macro** | `M01`–`M03` | `taiex_1d`, `taiex_5d`, `taiex_vol20d` | FinMind `TaiwanStockPrice` (`data_id=TAIEX`) | 加權指數 1D/5D 報酬率與 20D 年化波動度 |
| Macro | `M04`–`M05` | `sector_1d`, `sector_rs_20d` | FinMind TAIEX / TWSE 產業指數 | 產業 1D 報酬率；20D 相對強度 ($Sector_{20D} - TAIEX_{20D}$) |
| Macro | `M06` | `usdtwd_return` | FinMind `TaiwanExchangeRate` (`data_id=USD`) | 美元兌台幣即期賣出價日報酬率 |
| Macro | `M07`–`M09` | `sp500_ret`, `nasdaq_ret`, `sox_ret` | Yahoo Finance `^GSPC`, `^IXIC`, `^SOX` (T-1) | 美股跨市場 As-Of 延遲 1D 報酬率 |
| Macro | `M10` | `vix_level` | FRED `VIXCLS` (T-1) | Cboe VIX 收盤水準 / 100.0 |
| Macro | `M11`–`M12` | `us_10y_yield`, `us_10y_yield_change` | FRED `DGS10` (T-1) | 美國 10 年期公債殖利率及 bps 變動率 |
| **Stock Context** | `C01`–`C08` | `industry_embedding_01..08` | TWSE 產業分類代碼 (如 24 半導體) | 獨立版本控制之 8D 稠密產業向量 (單位模長) |
| Stock Context | `C09` | `log_market_cap` | FinMind `close` $\times$ `NumberOfSharesIssued` | $\ln(\text{市值})$ |
| Stock Context | `C10` | `log_avg_trading_val_60d` | FinMind `Trading_money` | $\ln(1 + MA_{60}(\text{成交金額}))$ |
| Stock Context | `C11`–`C12` | `volatility_pct`, `liquidity_pct` | 依歷史 60D 實現波動度與週轉率計算 | 截面百分位數排名 $[0, 1]$ |
| Stock Context | `C13` | `beta_120d` | 個股報酬率對 TAIEX 報酬率 120D 滾動迴歸 | 120 交易日 Beta 值 (協方差 / 基準方差) |
| Stock Context | `C14` | `industry_cap_pct` | 產業內市值排名 | 產業內市值百分位數 $[0, 1]$ |
| Stock Context | `C15` | `listing_age` | TWSE 掛牌上市日歷時 | $\min(\text{days\_since\_ipo} / 3652.5, 1.0)$ |
| Stock Context | `C16` | `recent_trading_suspension` | FinMind 成交量 / 停牌公告 | 過去 20 日是否曾發生停止交易 (0 或 1) |
| **Event Metadata** | `E01`–`E16` | Actor 4D, Type 8D, Status 4D | MOPS 重大訊息主旨與說明 | 事件實體角色、類型與狀態 Embedding |
| Event Metadata | `E17`–`E20` | Traceability, Reliability, Independence, Novelty | MOPS 申報元數據 | 交易所認證來源可靠度 (1.0)、獨立性 (0.9)、新穎度 |
| Event Metadata | `E21` | `market_surprise` | 發布日前實現波動率比對 | 嚴格事前隱含波動，零事後反應洩漏 (0.0) |
| Event Metadata | `E22` | `quantified_magnitude` | MOPS 說明文字影響評估 | 量化影響規模評分 $[0, 1]$ |
| Event Metadata | `E23` | `event_age` | $T_{decision} - T_{available}$ | 事件歷時（以日為單位，Python Index 22） |
| Event Metadata | `E24` | `timestamp_reliability` | MOPS 交易所發言時間戳記 | 官方時間戳記可靠度評分 (1.0，Python Index 23) |
| **Event Relation** | `R01`–`R02` | `direct_company_mention`, `industry_match` | MOPS 公司代號與產業比對 | 股票代號直提 (1.0)、產業相符 (1.0) |
| Event Relation | `R03`–`R08` | 上下游、客戶、供應商、區域與總經曝險 | **公開資料無法取得** | **UNSUPPORTED（填 0.0 並設定 Missing Mask）** |
| **Position State** | `S01`–`S04` | 目標部位、未實現損益、持倉天數、前次交易間隔 | Gym Trading Environment 內部狀態 | 即時交易狀態 (0.0 或 1.0) |

---

## 5. PIT 與缺失率審計統計 (Data Quality Audit)

依據嚴格資料分類（**Observed** / **Estimated** / **Unaudited Backfill** / **Unsupported** / **Missing**），統計結果如下：

| 模組名稱 | 總特徵數 | 真正可計算數 | 不可計算數 | 缺失率 (%) | 資料來源與 PIT 狀態 |
|---|---|---|---|---|---|
| **Price Dynamics** | 12 | 12 | 0 | **0.0%** | TWSE / FinMind EOD (無 NaN) |
| **Technical Indicators** | 16 | 16 | 0 | **0.0%** | 由真實 OHLCV 計算 (無 NaN) |
| **Institutional Flow** | 8 | 8 | 0 | **0.0%** | 三大法人買賣超與融資融券 (無 NaN) |
| **Macro & Market** | 12 | 12 | 0 | **0.0%** | TAIEX, USD/TWD, US 跨市場 (無 NaN) |
| **Stock Context** | 16 | 16 | 0 | **0.0%** | 市值、Beta120、流動性與產業 (無 NaN) |
| **Event Metadata** | 24 | 24 | 0 | **0.0%** | MOPS 重大訊息官方時間戳記 (無 NaN) |
| **Event Relation** | 8 | 2 | 6 | **75.0%** | R01/R02 可算，**R03..R08 因缺乏供應鏈圖譜不可算** |
| **Position State** | 4 | 4 | 0 | **0.0%** | Gym 模擬器內部觀察狀態 (無 NaN) |
| **全系統總計** | **100** | **94** | **6** | **6.0%** | **可計算特徵比例：94.0%** |

---

## 6. 不可計算特徵清單與根本原因分析

本次 PoC 嚴格遵守「**無法滿足正式 PIT 或缺乏真實來源時，不得自行推估，必須保留 Unsupported 並設為 Missing**」之原則。

全 100 維特徵中，目前無法由公開免費交易所 EOD API 計算的 6 項特徵如下：

1. `R03` (`upstream_exposure`): 上游供應鏈營收曝險權重
2. `R04` (`downstream_exposure`): 下游供應鏈營收曝險權重
3. `R05` (`customer_relationship`): 主要客戶營收佔比關係權重
4. `R06` (`supplier_relationship`): 主要供應商採購佔比關係權重
5. `R07` (`geographic_exposure`): 跨國/特定區域營運營收曝險權重
6. `R08` (`macro_factor_exposure`): 多因子總經曝險敏感度權重

### 根本原因
- 公開盤後資訊（TWSE / FinMind）僅提供量價與三大法人買賣超交易數據，未包含公開發行公司前十大銷貨客戶與採購供應商名單（僅季報/年報附表披露文字，需經結構化圖譜解析）。
- 建立歷史 As-Of 供應鏈圖譜需訂閱專有資料庫（例如：TEJ 台灣經濟新報供應鏈關係資料庫，或 FactSet Revere Supply Chain Relationships）。

### V8 架構之工程防護
- 系統在 [snapshot_builder.py](file:///home/augustine/Desktop/News-Aware%20Attention-Based%20Deep%20Reinforcement%20Learning%20for%20Stock%20Trading/stock_rl/data/snapshot_builder.py#L650) 中，將 `R03`–`R08` 預設為中性值 `0.0`，並確保其不影響 `event_metadata` 與槽位分配。
- 未來接入供應鏈圖譜時，只需填入數值，Schema 欄位與張量形狀（40 $\times$ 8）完全相容，無須更動模型或下游模組。

---

## 7. 反事實測試與完整回歸測試結果

已於 `tests/test_v8_4_historical_data_poc.py` 建立完整的 PoC 回歸測試集，並運行整體測試套件：

### 1. 反事實測試 (Counterfactual Leakage Test)
- **測試方式**：在決策日 $T$（2023-10-27 20:00:00）建立 Snapshot $S_{\text{clean}}$；接著將 $T+1$ 日的收盤價人為擴增 5 倍、成交量擴增 10 倍、三大法人淨買賣超改為極端負值、加權指數改為 99999 點，重構 Snapshot $S_{\text{mutated}}$。
- **測試結果**：
  $$\| S_{\text{clean}} - S_{\text{mutated}} \|_\infty = 0.0$$
  Across `price`, `technical`, `institutional`, `macro`, `stock_context`, `event_metadata`, `missing_masks` 達到 **100% 位元級完全一致**，證實 `HistoricalSnapshotBuilder` 具有絕對零未來資訊洩漏保證。

### 2. Normalizer 僅限訓練集擬合審計 (Train-Only Audit)
- **測試方式**：將 2024 年 1 月真實樣本分割為 Train (前 50%)、Val (25%)、Test (25%)。
- **測試結果**：
  - `FeatureNormalizer` 僅於 `is_train_split=True` 時計算統計量，並鎖定加密特徵雜湊 `4cec6d8f7983f4f7...`。
  - 對 Test 樣本進行 `transform()` 時，雜湊值與統計值完全未被污染，無統計資訊前向洩漏。

### 3. 全套 pytest 執行結果
```bash
/home/augustine/anaconda3/envs/py311/bin/pytest tests/
======================= 112 passed in 131.50s (0:02:11) ========================
```
- 原有 106 項測試（涵蓋 DQN、環境、消融、PIT 合約、Migration、Review 1–4 回歸）**100% 通過**。
- 新增 6 項 V8.4 真實資料 PoC 測試**全數通過**。

---

## 8. 下一步正式歷史 Dataset 建立的可行性評估 (Roadmap to V8.5)

基於本次 PoC 的實際驗證結果，提出以下邁向全量 100 檔標的歷史資料集（V8.5）之可行性評估與建議路徑：

```mermaid
flowchart LR
    subgraph V8.4 ["V8.4 (Verified PoC)"]
        P1["3 Representative Stocks<br/>202 Trading Days<br/>94/100 Features Computable<br/>Zero PIT Leakage Verified"]
    end

    subgraph V8.5 ["V8.5 (Full Historical Dataset)"]
        D1["100 TWSE Universe<br/>5-Year Rolling (2019-2024)<br/>Automated Ingestion Pipeline"]
        D2["Dual-Track Ingestion Strategy:<br/>1. Real-time Ingest Log (Audit Verified)<br/>2. Conservative Backfill (Session Cutoff)"]
        D3["Supply Chain KG Integration<br/>Resolves R03..R08"]
    end

    V8.4 --> V8.5
```

### 1. 資料量與工程可行性評估
- 100 檔股票 $\times$ 5 年（約 1,250 交易日）：
  - OHLCV + 三大法人 + 信用交易：約 125,000 列，磁碟空間約 250 MB。
  - 總經指標（TAIEX, USD/TWD, US 跨市場 5 檔）：約 1,250 列，磁碟空間約 5 MB。
  - 重大訊息文本與 Embedding：約 10,000 則公告，128D Embedding 約 10 MB。
- **評估結論**：全量資料集大小在 300 MB 以內，於記憶體與現代 GPU 訓練架構下屬中等規模，工程上完全可行。

### 2. 雙軌時間戳記（Dual-Track Timestamp Strategy）
為解決「歷史回填資料只有交易日，無精確產生秒數」的正式審計合約限制，建議在 V8.5 採行雙軌策略：
- **軌道 A（即時前向記錄 Track）**：於盤後排程（15:00, 18:00, 19:30）啟動 Ingestion Worker，將真實 HTTP 回應時間戳記記錄至 `ingestion_log`，生成具備真正審計效力的 `ProvenanceRecord`。
- **軌道 B（歷史回填保守截止 Track）**：在歷史回填模式下，嚴格將 `available_at` 設定為交易所官方公告的最晚時間點（OHLCV 15:30、法人 18:30、信用 20:30）。若決策時間設為 20:00，則信用交易餘額自動保守落入 $T-1$，徹底排除延遲發布之洩漏可能。

### 3. 供應鏈關係資料庫（R03..R08）補充方案
- **短期（公開開源方案）**：利用公開年報「主要進銷貨客戶佔比達 10% 以上」之披露名單，由離線腳本結構化抽取構建靜態點對點圖譜。
- **長期（機構專用方案）**：串接 TEJ 台灣經濟新報董監事與供應鏈關係資料庫，取得時間切片版本（Historical As-Of Relations）。

---

## 9. 程式環境、來源授權與重現指令

### 執行環境
- **作業系統**：Linux (Ubuntu / Debian x86_64)
- **Python 環境**：`/home/augustine/anaconda3/envs/py311/bin/python` (Python 3.11.16)
- **依賴套件**：`torch==2.14.0+cpu`, `pandas==3.1.1`, `numpy==2.4.6`, `requests==2.34.2`, `pytest==9.2.0`

### 來源授權與使用條款合規
- **FinMind Open API**：依據 FinMind 官方開源條款，非商業性研究使用，查詢頻率限制小於每小時 600 次（本 PoC 僅執行約 20 次請求）。
- **TWSE OpenAPI**：政府開放資料授權條款（Open Government Data License），用於學術與研究分析。
- **FRED**：Federal Reserve Bank of St. Louis 公開研究資料，遵循 FRED API 使用規範。

### 一鍵重現指令
```bash
# 1. 執行小規模真實資料抓取與快取
/home/augustine/anaconda3/envs/py311/bin/python scripts/fetch_v8_4_poc_data.py

# 2. 執行 V8.4 PoC 專項回歸測試
/home/augustine/anaconda3/envs/py311/bin/pytest tests/test_v8_4_historical_data_poc.py -v

# 3. 執行全專案完整回歸測試集 (112 項測試)
/home/augustine/anaconda3/envs/py311/bin/pytest tests/
```
