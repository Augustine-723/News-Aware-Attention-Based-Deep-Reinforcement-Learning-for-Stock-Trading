# V8.14 — Official Disclosure Retrieval & Multimodal Integration Prototype Report

## Executive Summary

本階段（V8.14）聚焦於建立具備真實網路取得證據（Genuine HTTPS Retrieval Evidence）的官方重大訊息取得流程，封閉生產用檢索信任邊界（Retrieval Trust Boundary），建立原始回應重新解析與文本綁定機制（Raw Response Binding），並落實嚴格的時點可用性檢驗（Point-in-Time As-Of Availability Gate）於多模態決策管線槽位寫入之前。

依據獨立審查意見（Code Review 1），由於公開資訊觀測站（MOPS）設有反爬蟲安全防禦機制（FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED），且 TWSE OpenAPI 僅提供現時滾動重大訊息（Rolling Current Disclosures），無法回溯 2024 Q1 歷史公告，因此 **V8.14 正式定位為「官方公告取得服務與多模態管線整合原型」（Official Disclosure Retrieval & Multimodal Integration Prototype），而非完整歷史資料集**。

---

### 八項核心審查維度與驗收數據對照表 (Core Gate Matrix)

依據 Code Review 1 規定，核心數據嚴格區分如下，絕不混淆即時資料、歷史資料、存在性驗證與時間可用性驗證：

| 審查維度 | 實測數值 | 規格 / 判定狀態 | 審查依據與治理規範 |
| :--- | :---: | :---: | :--- |
| **1. 真實 HTTPS 取得證明 (Real HTTPS Retrieval Demonstrated)** | **2 筆** | **VERIFIED (真實 HTTPS 請求)** | 透過 `GenuineHttpsRetrievalService` 實際連線 TWSE OpenAPI 與 MOPS 接口，保存原始 Response Bytes 與 SHA-256 摘要。 |
| **2. 真實現時公告取得筆數 (Genuine Current Disclosure Obtained)** | **1 筆** | **SUCCESSFUL (`2881` 富邦金控)** | 由 TWSE OpenAPI 成功取得並解析富邦金控現時公告，出表時間 2026-11-16 16:00:36 Asia/Taipei。 |
| **3. 歷史公告取得筆數 (Historical Disclosure Retrieved)** | **0 筆** | **FAIL-CLOSED (誠實申報 0 筆)** | MOPS 歷史查詢受反爬防禦阻擋，OpenAPI 無回溯參數。2024 Q1 歷史公告實際取得數嚴格為 **0 筆**。 |
| **4. 來源存在性驗證通過數 (Source Existence Verified)** | **1 筆** | **SOURCE_EXISTENCE_VERIFIED** | 透過 `EvidenceManifest.verify_entry_integrity()` 重新解析原始 OpenAPI JSON，確認公告存在於原始位元組。 |
| **5. 歷史供應商可用時間驗證數 (Historical Provider Availability Verified)** | **0 筆** | **UNVERIFIED (缺少 Vendor Feed Log)** | 官方公告僅有出表時間，缺乏第三方資訊商（如彭博/路透/TEJ）Drop-Copy Feed 送達終端時間戳記。 |
| **6. 正式 PIT 符合筆數 (Formal PIT Eligible)** | **0 筆** | **ESTIMATED_PIT_ONLY (維持隔離)** | 無供應商送達日誌，依法定規則維持 `ESTIMATED_PIT_ONLY`，禁止自封或自動提升為 `FORMAL_PIT_ELIGIBLE`。 |
| **7. 張量形狀與嵌入前向傳遞 (Tensor Shape / Forward Pass)** | **100% 通過** | **INTERFACE_SPECIFICATION_ONLY** | BGE 512D $\to$ 128D $\to$ 24D $\to$ 8D $\to$ (40, 160) $\to$ (60, 100) 前向輸出無誤，標記為**介面契約驗證**。 |
| **8. 歷史 PIT 決策狀態整合 (Historical PIT State Integration)** | **100% 驗證** | **PIT As-Of Gate Enforced** | 槽位寫入前執行 As-Of 檢驗；盤後 20:30、未來 $T+1$、公休日、2026 擷取時間等事件**一律拒絕寫入槽位**。 |

---

## 1. P0-1 — Close the Genuine Retrieval Trust Boundary

### 1.1 生產寫入路徑與測試工廠徹底分離
在前版中，`RetrievalSessionRegistry.register_genuine_session()` 允許呼叫端直接提供 `service_provenance` 或傳入 `allow_legacy_fixture=True` 註冊 Session，存在測試資料滲透至生產認證的漏洞。

V8.14 實施雙重路徑隔離：
1. **生產專屬註冊路徑 (`register_production_session`)**：
   - 僅能由 `GenuineHttpsRetrievalService` 在實際完成 HTTPS 網路請求且 HTTP 狀態碼為 200 時呼叫。
   - 必須傳入內部服務密鑰 `service_auth_key == RetrievalSessionRegistry._AUTHENTIC_SERVICE_AUTH_KEY`，否則直接拋出 `PermissionError`。
   - 嚴格拒絕已知 Fixture 雜湊（`is_known_fixture_payload`），標記 `is_production_certified=True`、`is_simulated=False`、`execution_mode="AUTHENTIC"`。
2. **測試工廠路徑 (`register_simulated_session`)**：
   - 專供單元測試與離線模擬使用，強制標記 `is_simulated=True`、`is_production_certified=False`、`execution_mode="SIMULATED"`。
3. **一般呼叫端防禦 (`register_genuine_session`)**：
   - 一般呼叫端自行呼叫或使用 `allow_legacy_fixture=True` 時，系統自動將 Session 標記為 `is_production_certified=False` 與 `is_simulated=True`。
   - 當 `TrustedSourceCertifier.verify_evidence(enforce_production_boundary=True)` 審查生產邊界時，任何 `is_simulated=True` 或缺少生產認證的 Session 一律 Fail-Closed 判定為 `MOPS_CANDIDATE_UNVERIFIED`。

### 1.2 通過之反例拒絕測試 (Counterexample Tests)
- `test_unauthorized_production_registration_rejection`：非授權呼叫端嘗試呼叫 `register_production_session()` 立即拋出 `PermissionError`。
- `test_counterexample_forged_provenance_rejected_in_production_boundary`：自填 `service_provenance` 或 Session Token 於生產邊界審查時直接被拒絕。
- `test_counterexample_legacy_fixture_bypass_rejected`：使用 `allow_legacy_fixture=True` 註冊之 Session 明確標為 `SIMULATED`，禁止取得正式認證。
- `test_openapi_session_cannot_certify_mops_historical_disclosure`：TWSE OpenAPI 即時 Session 嚴格禁止認證歷史 MOPS 公告。

---

## 2. P0-2 — Bind Manifest Fields to Raw Response

### 2.1 原始回應重新解析與欄位綁定 (Content Binding Re-parsing)
前版 `EvidenceManifest.verify_entry_integrity()` 僅比對 Raw Bytes Hash 與 Parsed Content Hash，若惡意呼叫端將真實 Raw Bytes 配對虛構之公告標題與內文，仍可能通過驗證。

V8.14 全面重構 `verify_entry_integrity()`，執行**原始資料動態重新解析**：
1. **OpenAPI JSON 重新解析**：
   - 將 `entry.raw_response_bytes` 解碼為 JSON 物件陣列。
   - 逐筆比對 `公司代號`、`主旨`、`詳細內容/說明` 與 `出表日期`。
   - 若原始位元組中不存在與清冊完全吻合之股票代號、標題與正文，直接判定 `Content binding verification failed`。
2. **MOPS HTML/Text 重新解析**：
   - 解碼原始 HTML 文字，比對目標 `stock_id`、標題或內文是否確實存在於回應中。
3. **本地記錄識別碼標記 (`local_record_id`)**：
   - TWSE OpenAPI 缺乏官方分配之伺服器 UUID，系統明確指派 `local_record_id = LOCAL-OPENAPI-{stock}-{date}-{hash}`，並標記 `is_local_record_id=True`，明示為本地衍生識別碼。
4. **發布時間解析真實性檢驗**：
   - 若官方格式無法解析發布時間，標記 `publication_time_parsed=False`，`filing_datetime=None`。
   - 嚴格禁止解析失敗時靜默以 `retrieved_at` 偽裝為公告發布時間（違反時 `verify_entry_integrity` 直接報錯）。
5. **原始資料留存狀態標記**：
   - 保留 `raw_retention_status = "FULL_RAW_BYTES_RETAINED"`。若日後涉及版權無法保存原始內文，標明 `INDEPENDENT_REPRODUCTION_PENDING`，誠實揭露無法獨立重算 Raw Hash。

### 2.2 通過之防偽反例測試
- `test_counterexample_manifest_tampered_content_rejection`：傳入真實 OpenAPI Raw Bytes，但清冊中置換為虛構之併購標題與內文，重新解析機制立即攔截並報錯。
- `test_counterexample_manifest_unparsed_time_rejection`：解析失敗卻以擷取時間冒充發布時間，清冊完整性檢查直接失敗。

---

## 3. P0-3 — Enforce PIT Before Event Slot Ingestion

### 3.1 時間戳記三權分立與假設標記
在 `execute_minimal_pipeline_integration` 中，徹底拆分三類時間戳記：
1. `observed_retrieved_at`：實際自網路下載之觀測時間戳記（如 2026-11-18）。
2. `estimated_provider_available_at`：公告發布時間 $+5\text{ minutes}$，**明確標記 `is_estimated_hypothesis = True`，僅為估計假設，非歷史事實**。
3. `verified_provider_available_at`：歷史第三方資訊商日誌時間戳記（目前為 `None`）。

### 3.2 決策槽位寫入前之 As-Of Availability Check
在特徵向量寫入 `recent_slots[0]` 之前，強制執行 As-Of 時點可用性檢查：
- **交易日檢驗**：決策日期必須存在於官方交易日曆中。
- **發布時間檢驗**：公告發布時間不得晚於決策時點（$20:00:00$ Asia/Taipei）。
- **可用性時間檢驗**：有效可用時間（`effective_available_at`）必須小於或等於決策時點。
- **防未來資訊外洩（Lookahead Bias Prevention）**：
  - 若事件晚於 $20:00$（例如 $20:30$ 公告）：**拒絕寫入當日決策槽位**，槽位維持全零 Zero-padded。
  - 若事件屬於次日（$T+1$ 未來事件）：**拒絕寫入當日決策槽位**。
  - 若事件發生於非交易日（如週日）：**拒絕形成該非交易日之決策 State**。
  - 若事件僅於 2026 年擷取且無歷史資訊商日誌：**拒絕進入 2024 年歷史決策 State**。

### 3.3 市場特徵佔位標記與研究宣稱邊界
- 市場特徵矩陣 `(60, 100)` 明確標記 `market_context_status = "SYNTHETIC_MARKET_PLACEHOLDER"`。
- 返回字典明確標註 `pipeline_verification_level = "INTERFACE_SPECIFICATION_VERIFICATION_ONLY"` 與 `empirical_drl_validated = False`。
- **嚴格宣告：成功輸出 `(40, 160)` 與 `(60, 100)` 僅代表多模態特徵管線架構與介面契約吻合，絕不代表已在真實資料上證明 DRL 交易具備超額收益或語意價值。**

---

## 4. P1 — Evidence Reproducibility & Source Governance

### 4.1 查詢結果細緻分類統計 (WAF vs Parsed)
在 `OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics()` 中，將查詢失敗細緻解耦：
- `total_queries_executed`：實際發送之真實網路請求總數。
- `successful_parsed_responses`：HTTP 200 且未受安全機制阻擋之正常解析回應數。
- `waf_access_blocked_responses`：觸發「安全性考量 / FOR SECURITY REASONS」或 HTTP 403/429 阻擋之請求數。

### 4.2 原始資料快照與清冊留存
依據政府資料開放授權條款（Open Government Data License Version 1.0），本專案將無版權爭議之公開資料快照保存於可重現路徑：
- 資料快照：`data/official_disclosures/twse_openapi_snapshot_2026.json`
- 證據清冊：`data/official_disclosures/evidence_manifest_v8_14.json`
- 授權狀態：`LICENSE_VERIFIED`（允許研究與註明出處之統計分析；禁止商業轉售）。
- 智慧財產與反爬遵循：不爬取或儲存未獲授權之商業新聞全文，嚴守 GitHub 開源治理標準。

### 4.3 即時資料與 2024 Q1 歷史資料庫嚴格隔離
- OpenAPI 樣本週期明確標記為 `SAMPLE_PERIOD = "2026-CURRENT-ROLLING"`。
- 2024 Q1 計畫之 168 個 Stock-Day 單位中，真實歷史公告取得筆數**持續保持為 0 筆**，絕不將現時重大訊息充作歷史回測資料。

---

## 5. 最小 Pipeline 整合實測規格 (Multimodal Contract)

真實取得之官方重大訊息（富邦金控現時公告）通過 `execute_minimal_pipeline_integration` 前向驗證：

```
Official OpenAPI JSON Response
       │  (90,709 Bytes, SHA-256 Verified)
       ▼
EvidenceManifest Entry
       │  (Re-parsed & Bound: 2881, 1151116, LOCAL-OPENAPI-...)
       ▼
Pretrained Semantic Encoder (BAAI/bge-small-zh-v1.5)
       │  CLS 512D Raw Embedding
       ▼
Deterministic QR Orthogonal Projection Matrix
       │  128D Projected Semantic Vector (L2 Norm = 1.0000)
       ▼
Multimodal Feature Concatenation
       │  128D Semantic + 24D Event Metadata + 8D Relation Vector
       ▼
Combined Event Representation (160D)
       │
       ├── PIT As-Of Availability Gate (Cutoff: 20:00:00 Asia/Taipei)
       │   └── Status: INGESTED (Effective availability <= Cutoff)
       ▼
Dual-Slot News Tensor: (40, 160)
       ├── Recent News Slots: (32, 160)  [Slot 0 filled, 1-31 Zero-padded]
       └── Active News Slots:  (8, 160)  [0-7 Zero-padded]
       ▼
Market Context Tensor: (60, 100)
       └── SYNTHETIC_MARKET_PLACEHOLDER
```

### 數值檢驗規格一覽表

| 管線特徵元件 | 輸出維度 / 形狀 | 檢驗指標 | 審查合規判定 |
| :--- | :---: | :---: | :--- |
| **原始語意嵌入 (Raw CLS Embedding)** | `(512,)` | BGE 官方預訓練權重 | 吻合 V8.8 規範 |
| **降維投影向量 (Projected Semantic Vector)** | `(128,)` | L2 Norm = 1.0000 | 嚴格正交歸一化 |
| **事件權威特徵 (Event Metadata Vector)** | `(24,)` | E01–E24 數值特徵 | 吻合 V8.0 契約 |
| **股票關聯特徵 (Event-Stock Relation Vector)** | `(8,)` | R01=1.0（富邦金精準匹配） | 吻合關係結構 |
| **融合事件向量 (Combined Event Vector)** | `(160,)` | 128D + 24D + 8D | 向量維度精準對齊 |
| **近期槽位 (Recent News Slots)** | `(32, 160)` | Slot 0 填入事件特徵 | 吻合雙槽位架構 |
| **活躍槽位 (Active News Slots)** | `(8, 160)` | 預設保留槽位 (Zero-padded) | 吻合雙槽位架構 |
| **決策輸入新聞張量 (News Slots)** | `(40, 160)` | 32 + 8 槽位拼接 | 吻合 V8.0 輸入契約 |
| **市場行情佔位 (Market Context)** | `(60, 100)` | 標記為 Synthetic Placeholder | 隔離未驗證行情 |
| **時點可用性檢驗 (PIT As-Of Gate)** | **PASSED** | 盤後/未來/公休日事件全數阻擋 | 杜絕 Lookahead 偏差 |

---

## 6. 完整測試套件執行結果 (Pytest Suite)

在標準 Conda Python 3.11 環境下執行全量 pytest 測試套件：

```
============================= test session starts ==============================
rootdir: /home/augustine/Desktop/News-Aware Attention-Based Deep Reinforcement Learning for Stock Trading
configfile: pyproject.toml
plugins: anyio-4.16.0
collected 232 items

........................................................................ [ 31%]
........................................................................ [ 62%]
........................................................................ [ 93%]
................                                                         [100%]
============================= 232 passed in 238.45s (0:03:58) =============================
```

**測試結論**：
- 全體 232 項測試（包含 V8.14 新增之 9 項邊界拒絕測試、反例防禦測試與 PIT As-Of 閘門測試）**100% 通過**。
- 無任何回歸錯誤，既有 DQN 訓練、Walk-Forward 交叉驗證、配對 Bootstrap 及多模態特徵抽取全部保持綠燈。

---

## 7. 現存研究限制與後續路徑 (Outstanding Limitations & Path Forward)

### 7.1 現存限制誠實宣告
1. **官方歷史資料接口缺乏自動化通路**：
   MOPS 設有反爬蟲安全屏障，不開放未授權之自動化爬取；TWSE OpenAPI 僅提供滾動現時重大訊息。在不違反系統安全規則之前提下，無法以爬蟲補齊 2024 Q1 歷史全量公告。
2. **缺乏歷史資訊商下發時間戳記 (Vendor Feed Logs)**：
   官方公告僅記載出表時間，無法證明歷史各交易決策當下該訊息送達下游交易終端之精確毫秒時點。現有所有公告均維持 `ESTIMATED_PIT_ONLY`。
3. **原型驗證不構成實證超額收益之證明**：
   前向管線通過僅證明技術架構上能正確處理官方公告並對齊形狀契約，不宣稱 News-Aware DRL 策略已在台股市場獲取超額報酬。

### 7.2 後續工程與學術治理路徑
1. **依合規途徑洽詢官方或授權學術資料庫**：
   循正式管道向台灣證券交易所申請歷史 Open Data 批次檔案，或透過 TEJ / 證基會學術資料庫取得 2024 Q1 具備資訊商下發日誌之歷史重大訊息。
2. **保持嚴格 Fail-Closed 原則**：
   在取得授權歷史資料集前，2024 Q1 歷史認證公告數持續維持 0 筆，禁止人工偽造或回填未經驗證之公告。
3. **維持既有 RL 模型架構安全隔離**：
   在取得足夠統計檢定力之可信歷史資料集前，不進行大型強化學習模型重訓，以保護核心算法架構免受未驗證樣本污染。
