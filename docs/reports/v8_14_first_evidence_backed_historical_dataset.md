# V8.14 — Official Disclosure Retrieval & Multimodal Integration Prototype Report

## Executive Summary

本階段（V8.14）聚焦於建立具備真實網路取得證據（Genuine HTTPS Retrieval Evidence）的官方重大訊息取得流程，封閉檢索信任邊界（Retrieval Trust Boundary），建立原始回應重新解析與精準文本綁定機制（Exact Content Binding Re-parsing），嚴格區分原始快照（Raw Response Snapshot）與解析摘要（Parsed Summary），並於多模態決策管線槽位寫入前實施「推估回放」（`ESTIMATED_REPLAY`）與「時間點檢核」（`EVIDENCE_VERIFIED_ASOF`）的雙模式 As-Of 時點可用性隔離。

依據獨立審查意見（Final Evidence Contract Patch），由於公開資訊觀測站（MOPS）設有反爬蟲安全防禦機制（`FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED`），且 TWSE OpenAPI 僅提供現時滾動重大訊息（Rolling Current Disclosures），無法回溯 2024 Q1 歷史公告，因此 **V8.14 正式定位為「官方公告取得服務與多模態管線整合原型」（Official Disclosure Retrieval & Multimodal Integration Prototype），而非已完成的歷史新聞資料集**。正式之歷史已認證來源數量（Historical Verified Sources）**嚴格保持為 0 筆**。

---

### 九大核心審查維度與驗收數據對照表 (Nine Core Evaluation Dimensions)

依據 Final Evidence Contract Patch 規定，核心數據嚴格區分如下，絕不混淆即時資料、歷史資料、存在性驗證與時間可用性驗證：

| 審查維度 | 實測數值 / 狀態 | 規格判定 | 審查依據與治理規範 |
| :--- | :---: | :---: | :--- |
| **1. 評估範圍與樣本庫 (Scope & Corpus)** | 2024 Q1 (56 交易日, 168 Stock-Day) + 2026 現時快照 | **ISOLATED** | 2024 Q1 涵蓋 2330、2603、2881 三檔標的；2026 OpenAPI 現時滾動樣本與歷史樣本嚴格隔離。 |
| **2. 檢索信任邊界 (Retrieval Trust Boundary)** | 單一程序工程控制邊界 | **ENGINEERING_TRUST_BOUNDARY_ONLY** | 移除原始碼固定授權字串；以程序內部控制（`PROCESS_INTERNAL_CONTROL_AUDIT`）限制生產 Session 登記，呼叫端自填一律 Fail Closed。 |
| **3. 官方來源可查詢性與防禦 (Official Queryability & Barriers)** | OpenAPI 200 (155,496 Bytes) / MOPS 安全阻擋 | **FAIL-CLOSED (0 違規繞過)** | TWSE OpenAPI 成功取得 155 KB 原始 JSON；MOPS 遭遇反爬屏障誠實記錄，無偽造繞過。歷史取得數嚴格為 **0 筆**。 |
| **4. 原始快照可重現性 (Raw Snapshot Reproducibility)** | 155,496 Bytes, SHA-256: `949d1b51...` | **FULL_RAW_BYTES_RETAINED** | 原始未壓縮回應儲存為 `twse_openapi_raw_response_snapshot.json`；解析摘要另存為 `parsed_disclosure_summary.json`。 |
| **5. 精確內容綁定與時間解析 (Exact Binding & Datetime Parsing)** | `發言日期`+`發言時間` 精確解析 | **EXACT_MATCH_VERIFIED** | 解析民國年日期時間（1151117 154758）；落實 `stock_id`、`headline`、`raw_content` 精準比對，全面禁止寬鬆子字串比對。 |
| **6. As-Of 時點可用性雙模式隔離 (PIT As-Of Segregation)** | `ESTIMATED_REPLAY` vs `EVIDENCE_VERIFIED_ASOF` | **DUAL_MODE_ENFORCED** | `ESTIMATED_REPLAY` 採 +5 分鐘假設（標記非正式 PIT）；`EVIDENCE_VERIFIED_ASOF` 強制要求 Provider Log 與交易日曆，缺少時一律拒絕寫入。 |
| **7. 多模態前向管線整合 (Multimodal Pipeline Forward Pass)** | BGE 512D $\to$ 128D $\to$ 160D $\to$ (40, 160) | **INTERFACE_SPECIFICATION_ONLY** | 通過 BGE 語意抽取、正交投影與雙槽位佈局（32 Recent + 8 Active）及行情佔位 `(60, 100)` 前向輸出。 |
| **8. 授權條款與資料治理 (Legal Terms & Governance)** | 政府資料開放授權 1.0 / MOPS 法律檢視 | **NEEDS_LEGAL_REVIEW** | OpenAPI 依政府資料開放條款合規使用；MOPS 公共資訊維持 `NEEDS_LEGAL_REVIEW`，禁止未授權之商業模型訓練與內文重分發。 |
| **9. 強化學習實證宣稱邊界 (Empirical DRL Validation Boundary)** | 介面規範驗證 | **EXPLORATORY_ONLY (未宣稱實證價值)** | 嚴正宣告：管線輸出吻合僅代表特徵架構介面契約通過，絕不宣稱 News-Aware DRL 交易策略已在台股市場獲取超額報酬。 |

---

## 1. P0-1 — Remove Source-Code-Based Production Certification

### 1.1 移除原始碼固定字串授權機制
在前版架構中，`RetrievalSessionRegistry` 依賴公開之字串常數 `_AUTHENTIC_SERVICE_AUTH_KEY` 作為驗證凭證，呼叫端可藉由讀取該變數或自填字串取得生產認證。

本輪修正徹底清除此項缺陷：
1. **完全移除 `_AUTHENTIC_SERVICE_AUTH_KEY`**：不再以原始碼中可讀取的靜態 Token 宣稱密碼學防偽。
2. **誠實標記信任邊界等級**：
   - `RetrievalSessionRegistry.TRUST_BOUNDARY_LEVEL = "ENGINEERING_TRUST_BOUNDARY_ONLY"`
   - `RetrievalSessionRegistry.AUDIT_MODEL = "PROCESS_INTERNAL_CONTROL_AUDIT"`
   - 坦誠揭露在單一 Python 程序執行模型下，此邊界為軟體工程程序控制，而非零信任密碼學證明。
3. **生產註冊路徑權限受控**：
   - `register_production_session()` 透過執行堆疊檢查（Call-Stack Internal Control）確認呼叫者確實源自 `GenuineHttpsRetrievalService` 於真實網路取得之執行上下文。
   - 一般呼叫端傳入任意授權字串或直接調用 `register_production_session()`，一律拋出 `PermissionError`。
4. **測試工廠與正式來源紀錄分離**：
   - 一般呼叫端調用 `register_genuine_session()` 時，系統強制設定 `is_production_certified=False` 與 `is_simulated=True`。
   - 在 `TrustedSourceCertifier.verify_evidence(enforce_production_boundary=True)` 審查下，任何 `is_simulated=True` 或非生產認證之 Session 一律 Fail Closed，判定為 `MOPS_CANDIDATE_UNVERIFIED`。
5. **歷史認證公告數保持為 0**：
   - 正式歷史重大訊息認證筆數維持為 **0 筆**，杜絕以即時探測冒充歷史認證。

---

## 2. P0-2 — Separate Estimated Replay from Verified As-Of

### 2.1 雙模式執行機制 (`execute_minimal_pipeline_integration`)
針對先前版本中混合「歷史回放推估（Filing + 5 分鐘）」與「As-Of 歷史時點檢核」之問題，重構管線整合函數，提供兩種嚴格隔離之模式：

```python
class PITExecutionMode(str, Enum):
    ESTIMATED_REPLAY = "ESTIMATED_REPLAY"
    EVIDENCE_VERIFIED_ASOF = "EVIDENCE_VERIFIED_ASOF"
```

#### Mode A: `EVIDENCE_VERIFIED_ASOF`（正式 As-Of 驗證模式）
- **強制要求歷史資訊商可用日誌**：必須具備可驗證之第三方資訊商送達時間（`entry.verified_provider_available_at`），**嚴格禁止以 +5 分鐘推估時間替代**。若缺少日誌，立即 Fail Closed（`is_available_as_of=False`, `slot_ingested=False`, `is_formal_pit_validated=False`）。
- **強制要求官方交易日曆**：若 `trading_calendar is None`，**一律 Fail Closed 拒絕通過**，杜絕日曆缺失時預設放行的漏洞。
- **時點可用性檢核**：有效可用時間必須 $\le$ 當日決策截止時點（$20:00:00$ Asia/Taipei）。

#### Mode B: `ESTIMATED_REPLAY`（推估回放研究模式）
- **保留推估假設**：允許在探索性實驗中採用公告發布時間 $+5\text{ minutes}$ 之推估時間（`estimated_provider_available_at`）。
- **保留不確定性標記**：明確標記 `effective_availability_source = "ESTIMATED_HYPOTHESIS_PLUS_5MIN"` 與 `pit_evidence_level = "ESTIMATED_HYPOTHESIS_UNVERIFIED"`。
- **禁止標記正式 PIT**：此模式下輸出的 `is_formal_pit_validated` **永遠為 `False`**，禁止流入正式回測決策。

#### 共通防外洩規則（兩模式皆嚴格阻擋）
1. **盤後公告阻擋**：公告發布時間晚於 $20:00:00$ 者（例如 $20:15:00$），兩模式下均拒絕進入當日狀態（`slot_ingested=False`）。
2. **次日未來公告阻擋**：屬於次日（$T+1$）之公告，兩模式下均拒絕進入當日狀態。
3. **非交易日公告阻擋**：假日（如週日）之公告，兩模式下均拒絕進入非交易日狀態。

### 2.2 統一標準化輸出結構
```python
{
    "effective_availability_source": "VERIFIED_PROVIDER_FEED" | "ESTIMATED_HYPOTHESIS_PLUS_5MIN" | ...,
    "pit_mode": "EVIDENCE_VERIFIED_ASOF" | "ESTIMATED_REPLAY",
    "pit_evidence_level": "FORMAL_AS_OF_VERIFIED" | "ESTIMATED_HYPOTHESIS_UNVERIFIED" | "EVIDENCE_MISSING_UNVERIFIED",
    "slot_ingested": bool,
    "is_formal_pit_validated": bool,
    "rejection_reason": str,
    # Pipeline tensor dimensions
    "text_raw_dim": 512,
    "text_proj_dim": 128,
    "metadata_dim": 24,
    "relation_dim": 8,
    "combined_dim": 160,
    "recent_slots_shape": [32, 160],
    "active_slots_shape": [8, 160],
    "news_slots_shape": [40, 160],
    "market_context_shape": [60, 100],
    "market_context_status": "SYNTHETIC_MARKET_PLACEHOLDER",
}
```

---

## 3. P1-3 — Raw Snapshot Reproducibility

### 3.1 原始位元組快照與解析摘要嚴格分離
1. **原始回應快照檔案**：
   - 檔名：`data/official_disclosures/twse_openapi_raw_response_snapshot.json`
   - 大小：**155,496 Bytes**（完整、未壓縮、伺服器原始回應 Array）
   - SHA-256 雜湊：`949d1b51b851737030c47c6bfe97af6f6143d34247ab45ac543ae9f78e83394d`
   - 包含當日全部重大訊息（含富邦金控 4 筆公告）。
2. **解析後摘要檔案**：
   - 檔名：`data/official_disclosures/parsed_disclosure_summary.json`
   - 類型：`PARSED_DISCLOSURE_SUMMARY`
   - 明確標示關聯之原始快照檔名與原始雜湊，不混淆為原始資料。
3. **原始位元組留存狀態管理**：
   - 具備完整 raw bytes 時標記 `raw_retention_status = "FULL_RAW_BYTES_RETAINED"`。
   - 若未來受限於版權而無法留存 raw bytes，必須標記為 `INDEPENDENT_REPRODUCTION_PENDING`，此時 `verify_entry_integrity()` 立即拒絕認證，誠實揭露無法獨立重算 Hash。

---

## 4. P1-4 — Strengthen Manifest Content Binding

### 4.1 原始回應重新解析與精準欄位比對
在 `EvidenceManifest.verify_entry_integrity()` 中實施全面重構：
1. **精準欄位匹配（禁止寬鬆子字串匹配）**：
   - 比對 `stock_id == item_stock`。
   - 比對 `entry.headline.strip() == item_sub`。
   - 比對 `entry.raw_content.strip() == item_content`。
   - 徹底杜絕如「2月營收」模糊匹配到「12月營收」之假陽性。
2. **重新解析正式發布日期與時間**：
   - 實作 `_parse_raw_record_publication_datetime`，解析官方民國年字串（如 `'1151117'` + `'154758'` $\to$ `2026-11-17 15:47:58 Asia/Taipei`）。
   - 優先讀取 `'發言日期'` 與 `'發言時間'`，無紀錄時降級讀取 `'出表日期'` 與 `'出表時間'`。
   - 比對解析出之時間與 `entry.filing_datetime` 是否精準一致。
3. **禁止以擷取時間冒充發布時間**：
   - 當 `publication_time_parsed = False` 時，`filing_datetime` 必須為 `None`；若自填 `retrieved_at` 冒充發布時間，完整性檢查立即失敗。
4. **本地記錄識別碼（`local_record_id`）**：
   - 針對官方未發行 UUID 之公告，保持指派 `LOCAL-OPENAPI-{stock}-{date}-{hash}`，並標記 `is_local_record_id=True`。
5. **授權狀態保守標記**：
   - 對缺少個別公開授權證明的來源，維持 `LICENSE_UNVERIFIED` 或 `NEEDS_LEGAL_REVIEW`。

---

## 5. 測試套件執行結果 (Pytest Suite Verification)

在 Python 3.11 環境下執行全量 pytest 測試套件：

```
============================= test session starts ==============================
rootdir: /home/augustine/Desktop/News-Aware Attention-Based Deep Reinforcement Learning for Stock Trading
configfile: pyproject.toml
plugins: anyio-4.16.0
collected 239 items

........................................................................ [ 30%]
........................................................................ [ 60%]
........................................................................ [ 90%]
.......................                                                  [100%]
============================= 239 passed in 232.06s (0:03:52) =============================
```

### 本次新增與通過之反例與契約測試
- `test_p0_1_production_session_requires_genuine_service_context`：自填字串 Token 呼叫生產 Session 註冊立即被 `PermissionError` 攔截；一般註冊強制為 `is_simulated=True`。
- `test_p0_1_historical_verified_source_count_strictly_zero`：確認正式歷史已認證來源數量嚴格為 0 筆。
- `test_p0_2_pit_modes_separation_estimated_replay_vs_evidence_verified_asof`：
  - 缺少 Provider Log 時：`EVIDENCE_VERIFIED_ASOF` 模式立即拒絕；`ESTIMATED_REPLAY` 標記非正式 PIT。
  - 盤後 20:00 後公告：兩模式下皆阻擋。
  - 次日（$T+1$）公告：兩模式下皆阻擋。
  - 缺失官方交易日曆：`EVIDENCE_VERIFIED_ASOF` 模式 Fail Closed 拒絕。
- `test_p1_3_raw_snapshot_reproducibility_and_parsed_summary`：驗證 155,496 Bytes 原始快照之 SHA-256 吻合，且 `INDEPENDENT_REPRODUCTION_PENDING` 狀態被拒絕。
- `test_p1_4_publication_time_reparsing_and_exact_field_matching`：驗證民國年日期時間精確解析、發布時間不一致拒絕、相似標題不一致拒絕、必填欄位缺失拒絕。

---

## 6. 研究限制與治理聲明

1. **原型定位與歷史資料庫缺失**：
   本原型已證明技術管線能合規取得官方資料、精確重算雜湊並以正交雙槽位格式注入強化學習狀態；但因官方歷史資料接口設有反爬機制，**2024 Q1 歷史認證公告數嚴格為 0**。
2. **禁止推估時間冒充歷史時點**：
   在取得具備彭博/路透/TEJ 下發時間戳記的正式資料前，任何 $+5\text{ 分鐘}$ 推估僅限於 `ESTIMATED_REPLAY` 模式，不得宣稱具備 Point-in-Time 免疫性。
3. **強化學習演算法隔離**：
   在取得統計顯著之可信歷史資料集前，保持核心 DQN/Double DQN/Dueling DQN 模型架構凍結，不以未驗證樣本進行模型重訓。
