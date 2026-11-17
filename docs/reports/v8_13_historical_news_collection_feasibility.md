# V8.13 Deliverable Report: Systematic Historical News Collection Feasibility, Evidence Partitioning & Statistical Calibration Gate

## 1. Executive Summary & V8.13 Scope

The **V8.13** milestone establishes the engineering feasibility, data provenance infrastructure, legal boundary audit, and econometric power calibration required for systematic historical financial news collection for Taiwan Stock Exchange (TWSE) liquid constituents. Following the V8.13 Code Review 1 gate, all empirical claims of collection coverage, source certification, and statistical power are grounded strictly in auditable evidence, eliminating fabricated coverage claims and self-attested certifications.

### Seven Rigorously Partitioned Evidence Levels
In accordance with the V8.13 Code Review 1 governance gate, the milestone distinctly partitions and documents seven progressive evidence levels:
1. **Fixture Inventory Built**: 12 local candidate corporate disclosures (`LOCAL_Q1_2024_CANDIDATE_FIXTURE`) compiled across TSMC (2330), Evergreen (2603), and Fubon Financial (2881) for 2024 Q1, cleanly segregated from the 50 legacy internal pilot fixtures (`PilotSourceManifest`).
2. **Query Interface Investigated**: Comprehensive survey of TWSE MOPS HTML Archive, TWSE OpenAPI (`t187ap04_L`), and Commercial Vendor Feeds across rate limits, historical range, and licensing terms.
3. **Real HTTP Query Executed**: Genuine, lawful network query executed against TWSE OpenAPI (`https://openapi.twse.com.tw/v1/opendata/t187ap04_L`), recording real query start/end timestamps and network status without fabrication.
4. **Actual Raw Response Retrieved**: 168,879 raw HTTP response bytes retrieved from TWSE OpenAPI with verified SHA-256 digest (`39e5f05528250f864a1fd6d76d5043de541cb01ec0ffffded4113f13e0db4d79`).
5. **Historical Source Existence Verified**: Currently **0** formal verifications. All 12 candidate records remain `MOPS_CANDIDATE_UNVERIFIED` pending live exchange retrieval receipts.
6. **Historical Provider Availability Verified**: Currently **0** formal verifications. MOPS filings record company upload times but lack downstream quantitative vendor ingestion receipts (Estimated-PIT boundary strictly preserved).
7. **Simulation Calibration Only**: Empirical Type-I error and power simulations explicitly marked as **`ASSUMPTION_BASED`**, with `power_status: POWER_NOT_ESTABLISHED` and 250–300 designated strictly as `PROVISIONAL_PLANNING_TARGET_ONLY`.

---

## 2. P0-1 — Auditable Collection Tracking & Dynamic Coverage Resolution

### Elimination of Fabricated Coverage Claims
Previously, collection coverage reported hardcoded values (`sessions_queried=56`, `confirmed_zero_sessions=47`, `failed_queries=0`). Under V8.13 Review 1:
- All coverage metrics are **dynamically computed** from auditable `QueryAttemptRecord` instances.
- Each query attempt tracks: `target_date`, `stock_id`, `source_channel`, `endpoint`, `query_start_time`, `query_end_time`, `http_status`, `response_status`, `actual_raw_response_hash`, `result_count`, `error_info`, `retry_info`, and `raw_bytes_len`.
- In default offline prototype execution without network queries, coverage defaults honestly to:
  * `sessions_queried = 0`
  * `collection_coverage_ratio = 0.0`
  * `collection_status = "NOT_EXECUTED_OFFLINE_PROTOTYPE"`
  * `three_way_session_resolution`: `NOT_ATTEMPTED = 56`, `QUERY_COMPLETED_ZERO_RESULTS = 0`, `AUTHENTIC_NO_ANNOUNCEMENT_CONFIRMED = 0`, `ANNOUNCEMENTS_RETRIEVED = 0`

```
+----------------------------------------------------------------------------------------+
+ 2024 Q1 Collection & Coverage Metrics (Audited Dynamic Resolution)                     |
+--------------------------------------------------+------------------+------------------+
| Metric Name                                      | Evaluated Value  | Status           |
+--------------------------------------------------+------------------+------------------+
| Planned Trading Sessions in 2024 Q1              | 56 sessions      | Authentic TWSE   |
| Executed Query Sessions (Offline Prototype Mode) | 0 sessions       | NOT_ATTEMPTED: 56|
| Collection Coverage Ratio                        | 0 / 56 (0.0%)    | NOT_EXECUTED     |
| Candidate Fixture Disclosures                    | 12 disclosures   | UNVERIFIED       |
| Sessions with Candidate Disclosures              | 9 sessions       | 16.07%           |
| Live OpenAPI Query Probe HTTP Status             | 200 OK           | 168,879 Bytes    |
| Live OpenAPI Query Probe Raw Payload Hash        | 39e5f055...      | VERIFIED         |
| Three-Way Resolution: NOT_ATTEMPTED              | 56 sessions      | 100.0%           |
| Three-Way Resolution: ZERO_RESULTS               | 0 sessions       | (Requires Live)  |
| Three-Way Resolution: AUTHENTIC_CONFIRMED_ZERO   | 0 sessions       | (Requires Live)  |
| Event Clusters in Candidate Fixture              | 10 clusters      | Deduplication    |
| Rule 51 Clarification Lineage                    | 1 disclosure     | Lineage Handled  |
| Historical Provider Availability Coverage        | 0 / 12 (0.0%)    | Lacks Feed Logs  |
| Formally Verified Sources (Formal PIT Verified)  | 0 / 12 (0.0%)    | 0 Formal PIT     |
+--------------------------------------------------+------------------+------------------+
```

---

## 3. P0-2 — Segregation of Candidate Fixture Inventory from Raw Official Data

### Candidate Fixture vs. Raw Official Response
The 2024 Q1 announcement catalog is formally designated as `LOCAL_Q1_2024_CANDIDATE_FIXTURE`. In accordance with P0-2:
1. `candidate_fixture_id` replaces announcement ID (e.g., `CANDIDATE-Q1-2024-2330-01` through `CANDIDATE-Q1-2024-2881-03`).
2. `announcement_id` is set strictly to `None` pending live exchange retrieval.
3. `retrieved_at` is set strictly to `None` (eliminating fake synthetic timestamps).
4. `raw_official_response_bytes` and `raw_response_hash` are set strictly to `None`.
5. `candidate_fixture_hash` records the SHA-256 of local candidate body text.
6. `normalized_content_hash` records the normalized headline + body text hash for model evaluation.
7. Verification status is locked to `SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED`.
8. Complete segregation is maintained between the 50 legacy internal pilot fixtures and the 12 Q1 candidate fixtures.

| Candidate Fixture ID | Stock | Filing Date / Time | Headline | Category | Cluster ID | Rule 51 Revision |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| `CANDIDATE-Q1-2024-2330-01` | 2330 | 2024-01-18 14:30 | 台積電說明112年第四季財務報告及113年第一季業績展望 | earnings | `CLU-2330-Q4-2023` | No |
| `CANDIDATE-Q1-2024-2330-02` | 2330 | 2024-02-05 17:15 | 台積電董事會決議核准配發112年第四季每股現金股利3.5元及核准資本預算 | dividend | `CLU-2330-DIV-CAPEX-Q4` | No |
| `CANDIDATE-Q1-2024-2330-03` | 2330 | 2024-02-15 14:00 | 台積電公告113年1月合併營業收入淨額2157億8500萬元月增逾兩成 | earnings | `CLU-2330-REV-JAN24` | No |
| `CANDIDATE-Q1-2024-2330-04` | 2330 | 2024-03-08 14:00 | 台積電公告113年2月合併營業收入淨額1816億4800萬元 | earnings | `CLU-2330-REV-FEB24` | No |
| `CANDIDATE-Q1-2024-2330-05` | 2330 | 2024-03-18 15:30 | 台積電核准於嘉義科學園區建置先進封裝廠 | capex | `CLU-2330-CHIAYI-COWOS` | No |
| `CANDIDATE-Q1-2024-2330-06` | 2330 | 2024-03-18 17:00 | 澄清媒體報導台積電嘉義建廠投資規模相關訊息 | clarification | `CLU-2330-CHIAYI-COWOS` | **Yes (Rule 51)** |
| `CANDIDATE-Q1-2024-2603-01` | 2603 | 2024-01-15 17:45 | 長榮海運公告因應紅海航線受阻加徵海運附加費相關說明 | crisis | `CLU-2603-RED-SEA-SURCHARGE` | No |
| `CANDIDATE-Q1-2024-2603-02` | 2603 | 2024-02-15 15:00 | 長榮公告113年1月合併營業收入新台幣283億5200萬元月增8% | earnings | `CLU-2603-REV-JAN24` | No |
| `CANDIDATE-Q1-2024-2603-03` | 2603 | 2024-03-14 18:20 | 長榮董事會決議通過112年度財務報告及擬配發現金股利每股10元 | earnings | `CLU-2603-EARNINGS-FY23` | No |
| `CANDIDATE-Q1-2024-2881-01` | 2881 | 2024-01-10 16:30 | 富邦金控公告112年12月自結合併稅前淨利及累計每股盈餘4.81元 | earnings | `CLU-2881-FULLYEAR-2023` | No |
| `CANDIDATE-Q1-2024-2881-02` | 2881 | 2024-02-15 16:30 | 富邦金控公告113年1月自結合併稅後盈餘140.7億元EPS 1.08元 | earnings | `CLU-2881-REV-JAN24` | No |
| `CANDIDATE-Q1-2024-2881-03` | 2881 | 2024-03-15 17:30 | 富邦金控董事會通過112年度財務報告及承認營業報告書 | earnings | `CLU-2881-FULLYEAR-2023` | No |

---

## 4. P0-3 — Authentic Receipt Requirements & Fail-Closed Gate

### Elimination of Caller Self-Attestation
In accordance with P0-3, callers cannot obtain formal certification solely by passing flags such as `is_authentic=True` or `receipt_mode`:
- All receipts produced by standard test factories (`create_adapter_receipt`, `certify_with_adapter_receipt`) are strictly `SIMULATED_ADAPTER_RECEIPT` (`is_simulated = True`, `is_verified_live_retrieval = False`).
- Passing `is_authentic=True` without verified live network retrieval **fails closed** in `TrustedSourceCertifier.verify_evidence`, returning `MOPS_CANDIDATE_UNVERIFIED`.
- Formal elevation to `VERIFIED_SOURCE` strictly requires:
  1. Actual HTTP 200 status code.
  2. Non-empty `raw_http_response_bytes` with verified SHA-256 payload digest.
  3. Active, auditable `retrieval_session_id`.
  4. Execution through dedicated `issue_receipt_from_live_retrieval` / `certify_with_live_retrieval_evidence`.

```
+----------------------------------------------------------------------------------------+
| TrustedSourceCertifier Verification Flow                                               |
|----------------------------------------------------------------------------------------|
| [1] Adapter Receipt Present?                       --> If None: FAIL CLOSED            |
| [2] Certifier Authority Recognized?                --> If untrusted: FAIL CLOSED       |
| [3] Reference URL Scheme HTTPS & Domain Whitelist? --> If invalid: FAIL CLOSED         |
| [4] URL Matches Adapter Receipt Endpoint?          --> If mismatch: FAIL CLOSED        |
| [5] Announcement ID in Candidate Catalog?          --> If unknown: FAIL CLOSED         |
| [6] Cross-Announcement Reuse Detected?             --> If reused: FAIL CLOSED          |
| [7] Company ID & Publication Datetime Match?       --> If mismatch: FAIL CLOSED        |
| [8] Payload Hash Matches Content?                  --> If mismatch: FAIL CLOSED        |
| [9] Receipt is SIMULATED_ADAPTER_RECEIPT?          --> If True: FAIL CLOSED            |
| [10] is_verified_live_retrieval == False?          --> If False: FAIL CLOSED (P0-3)    |
| [11] retrieval_session_id is Missing / Empty?      --> If missing: FAIL CLOSED (P0-3)  |
| [12] raw_http_status != 200?                       --> If != 200: FAIL CLOSED (P0-3)   |
| [13] raw_bytes_len <= 0?                           --> If <= 0: FAIL CLOSED (P0-3)     |
+----------------------------------------------------------------------------------------+
                                            |
                                            v (All 13 Checks Satisfied)
                               +-------------------------+
                               |     VERIFIED_SOURCE     |
                               +-------------------------+
```

---

## 5. P1 — Statistical Calibration: Oracle Simulation vs. Implemented Test

### Dual-Calibration Architecture
P1 explicitly distinguishes **Oracle Simulation Calibration** from **Implemented Statistical Test Calibration**:
- **Oracle Simulation Calibration**: Evaluates test statistic $\Delta \hat{\rho}$ using the oracle standard error $SE_{\text{oracle}} = \text{std}(\Delta \hat{\rho})$ computed across all Monte Carlo universes.
- **Implemented Statistical Test Calibration**: For each simulated dataset in the Monte Carlo run, executes the actual within-dataset paired Cluster Bootstrap hypothesis test ($B=40$ cluster resamplings), computes $SE_{\text{boot}} = \text{std}(\Delta \hat{\rho}^*)$, evaluates studentized ratio $t = \Delta \hat{\rho} / SE_{\text{boot}}$, and measures the empirical rejection fraction ($|t| > 1.96$) across independent datasets.

```
+----------------------------------------------------------------------------------------+
| Dual-Calibration Empirical Type-I Error Results (H0: delta_rho = 0.0, alpha = 0.05)     |
+----------------------+--------------------+--------------------+-----------------------+
| Test Sample Size (N) | Oracle Simulation  | Implemented Test   | Calibration Status    |
|                      | False Rejection    | (Cluster Bootstrap)|                       |
+----------------------+--------------------+--------------------+-----------------------+
| N = 25               | 0.0483 (~4.8%)     | 0.0650 (~6.5%)     | Calibrated (nominal)  |
| N = 50               | 0.0467 (~4.7%)     | 0.0583 (~5.8%)     | Calibrated (nominal)  |
| N = 100              | 0.0483 (~4.8%)     | 0.0517 (~5.2%)     | Calibrated (nominal)  |
| Overall Mean         | 0.0478 (~4.8%)     | 0.0583 (~5.8%)     | ASSUMPTION_BASED      |
+----------------------+--------------------+--------------------+-----------------------+
```

### Statistical Governance Disclosures
1. All power simulation results are classified strictly as **`ASSUMPTION_BASED`**.
2. Statistical power status remains locked at **`POWER_NOT_ESTABLISHED`** under the current pilot size ($N_{\text{test}} = 11$).
3. The sample size range of 250–300 is designated strictly as **`PROVISIONAL_PLANNING_TARGET_ONLY`**. Proposing a proven minimum sample size is prohibited prior to empirical parameter calibration on real TWSE constituent data.
4. Multi-dimensional sensitivity analysis across ICC $\in [0.05, 0.15, 0.30]$, valid label ratio $\in [0.80, 0.90, 1.00]$, and purging loss $\in [0.05, 0.15, 0.25]$ confirms that intra-cluster dependency drastically reduces effective degrees of freedom (power at $N=100$ drops from 61.2% to 22.4%).

---

## 6. Official Data Channel Feasibility & Access Terms

```
+--------------------------------+--------------------+---------------------+----------------------+
| Dimension                      | TWSE MOPS Archive  | TWSE OpenAPI        | Commercial Vendor    |
+--------------------------------+--------------------+---------------------+----------------------+
| Endpoint URL                   | mops.twse.com.tw   | openapi.twse.com.tw | Proprietary APIs     |
| Queryable Historical Range     | 1999 to Present    | Rolling 30 Days     | 20+ Years            |
| Query Method                   | HTTP POST/GET Form | RESTful JSON API    | SDK / FTP / REST     |
| Access Legality                | Public Inspection  | Public Open Data    | Commercial License   |
| Rate Limits / Anti-Crawler     | Strict Throttling  | Standard API Limits | Dedicated Bandwidth  |
| Downstream Vendor Feed Logs    | NO (Filing Time)   | NO (Live Stream)    | YES (Millisecond)    |
| Research Licensing Status      | NEEDS_LEGAL_REVIEW | NEEDS_LEGAL_REVIEW  | Commercial Contract  |
| Commercial Redistribution      | Prohibited         | Prohibited          | Strictly Prohibited  |
| Feasibility Verdict            | Feasible for Audit | Rolling Probe Only  | Formal PIT Capable   |
+--------------------------------+--------------------+---------------------+----------------------+
```

---

## 7. Regression Test Suite & Verification Summary

All unit and integration tests across the entire repository execute with 100% pass rate:
- `tests/test_v8_13_historical_collection.py`: 5 passed
- `tests/test_v8_12_dataset_expansion_feasibility.py`: 5 passed
- `tests/test_v8_11_evidence_and_learning.py`: 5 passed
- All existing V8.0–V8.10 regression suites: passed
- **Total Repository Test Pass**: 100% (209 tests passed)
- **Git Branch**: `fix/v8-audit-patch`
- **Formal Verified Sources Count**: Strictly `0` (authentic baseline preserved)
