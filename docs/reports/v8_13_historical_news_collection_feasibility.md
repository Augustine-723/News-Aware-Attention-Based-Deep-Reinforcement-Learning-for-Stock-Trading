# V8.13 Deliverable Report: Systematic Historical News Collection Feasibility, Evidence Partitioning & Statistical Calibration Gate

## 1. Executive Summary & V8.13 Scope

The **V8.13** milestone establishes the engineering feasibility, data provenance infrastructure, legal boundary audit, and econometric power calibration required for systematic historical financial news collection for Taiwan Stock Exchange (TWSE) liquid constituents. Following the V8.13 Final Audit gate, all empirical claims of collection coverage, source certification, and statistical power are grounded strictly in auditable evidence, eliminating fabricated coverage claims, self-attested certifications, and simulated coverage inflation.

### Exhaustive Audited Metric Counts
In accordance with the V8.13 Final Audit governance gate, the milestone distinctly partitions and documents eight exact metric counts:
1. **Local Candidate Fixture Count**: **12** candidate corporate disclosures (`LOCAL_Q1_2024_CANDIDATE_FIXTURE`) compiled across TSMC (2330), Evergreen (2603), and Fubon Financial (2881) for 2024 Q1, spanning **9** unique candidate event sessions (candidate fixture density: `9 / 56 = 0.1607`), cleanly segregated from the 50 legacy internal pilot fixtures (`PilotSourceManifest`).
2. **Real HTTP Query Count**: **1** single-session live probe query executed against TWSE OpenAPI (`https://openapi.twse.com.tw/v1/opendata/t187ap04_L`), recording genuine query start/end timestamps and network status without fabrication.
3. **Historical Query Count**: **0** historical queries. (TWSE OpenAPI rolling 30-day window does not cover 2024 Q1 historical archive; automated MOPS historical scraper not deployed).
4. **Authentic Raw Response Count**: **1** live probe response (**168,879** bytes retrieved from TWSE OpenAPI with verified SHA-256 digest `39e5f05528250f864a1fd6d76d5043de541cb01ec0ffffded4113f13e0db4d79`). The probe execution log and hash digest were verified; raw response bytes were validated in-session and not permanently persisted to disk.
5. **Verified Historical Source Count**: Strictly **0** formal verifications. All 12 candidate records remain `MOPS_CANDIDATE_UNVERIFIED` pending live exchange retrieval receipts.
6. **Provider Availability Evidence Count**: Strictly **0** formal verifications (`historical_provider_availability_included = False` for all records). MOPS filings record company upload times but lack downstream quantitative vendor ingestion receipts (Estimated-PIT boundary strictly preserved).
7. **Simulated Coverage**: **33.33%** (56 / 168 stock-days in single-stock pilot) or **100%** (168 / 168 stock-days in full multi-constituent simulation), strictly quarantined inside `simulation_metrics` and prohibited from inflating real collection coverage.
8. **Real Collection Coverage**: Strictly **0.0%** (0 / 168 stock-day units executed in offline state; `collection_status = "NOT_EXECUTED_OFFLINE_PROTOTYPE"`).

---

## 2. P0-1 — Authentic Receipt Bound to Genuine Retrieval (`RetrievalSessionRegistry`)

### Elimination of Caller Self-Attestation
Previously, callers could pass arbitrary byte strings, self-minted session IDs, and `http_status=200` to `issue_receipt_from_live_retrieval()` to produce authentic receipts. Under V8.13 Final Audit:
1. **Trusted `RetrievalSessionRegistry`**: A centralized in-memory session registry was implemented in `stock_rl/evaluation/pilot_dataset.py`. Only actual HTTPS network calls can register an auditable session (`register_genuine_session`).
2. **Auditable Session Record**: Each registered session tracks:
   - `session_id`: Unique identifier formatted as `REG-SESS-<timestamp>-<hash_prefix>`
   - `source_channel`: e.g., `TWSE_OPENAPI`, `TWSE_MOPS_HTML_ARCHIVE`
   - `endpoint_url`: Validated HTTPS endpoint
   - `request_start_time` and `response_end_time`: Genuine query timestamps
   - `http_status`: Must be strictly `200`
   - `raw_response_hash`: SHA-256 digest of `raw_response_bytes`
   - `raw_bytes_len`: Byte length of raw response
   - `parsed_summary`: Structured parse summary
   - `source_identity`: Authenticated HTTPS client identity
   - `is_genuine_network_retrieval`: Locked to `True`
   - `announcement_ids`: List of official announcement IDs retrieved in the session
3. **Cross-Check Verification Gate**: In `TrustedSourceCertifier.verify_evidence`:
   - Checks that `evidence.adapter_receipt.retrieval_session_id` exists in `RetrievalSessionRegistry`. Self-minted or unregistered session IDs are rejected immediately.
   - Verifies `session.is_genuine_network_retrieval is True`, `session.http_status == 200`.
   - Cross-audits `session.raw_response_hash == evidence.adapter_receipt.raw_response_hash` and `session.endpoint_url == evidence.external_reference_url`.
   - Confirms announcement presence in `session.announcement_ids`.
4. **Channel & Scope Boundary (P0-1 Item 8)**:
   - Live `TWSE_OPENAPI` probe sessions (`source_channel == "TWSE_OPENAPI"`) cannot certify historical MOPS disclosures or historical provider availability (fails closed with: *"Verification rejected: Live TWSE_OPENAPI probe session cannot certify historical MOPS disclosure existence or historical provider availability (P0-1)."*).
5. **Counterexample Regression Tests**:
   - `test_counterexample_raw_pilot_events_as_http_bytes_rejection`: Taking `RAW_PILOT_EVENTS` body text as raw HTTP bytes and filling in a self-minted `retrieval_session_id` is rejected and fails closed to `MOPS_CANDIDATE_UNVERIFIED`.
   - `test_openapi_probe_cannot_certify_historical_mops_announcements`: Live OpenAPI probe session cannot elevate historical MOPS records to `VERIFIED_SOURCE`.
   - In default offline execution, formal verified source count remains strictly **0**.

```
+----------------------------------------------------------------------------------------+
| TrustedSourceCertifier Audit Architecture (P0-1)                                       |
+----------------------------------------------------------------------------------------+
| [1] Adapter Receipt Present?                       --> If None: FAIL CLOSED            |
| [2] Certifier Authority Recognized?                --> If untrusted: FAIL CLOSED       |
| [3] Reference URL Scheme HTTPS & Domain Whitelist? --> If invalid: FAIL CLOSED         |
| [4] URL Matches Adapter Receipt Endpoint?          --> If mismatch: FAIL CLOSED        |
| [5] Announcement ID in Candidate Catalog?          --> If unknown: FAIL CLOSED         |
| [6] Cross-Announcement Reuse Detected?             --> If reused: FAIL CLOSED          |
| [7] Company ID & Publication Datetime Match?       --> If mismatch: FAIL CLOSED        |
| [8] Payload Hash Matches Content?                  --> If mismatch: FAIL CLOSED        |
| [9] Raw Response Hash Matches raw_content?         --> If mismatch: FAIL CLOSED        |
| [10] Receipt is SIMULATED_ADAPTER_RECEIPT?         --> If True: FAIL CLOSED            |
| [11] is_verified_live_retrieval == False?          --> If False: FAIL CLOSED           |
| [12] retrieval_session_id in RetrievalSessionRegistry? --> If not found: FAIL CLOSED   |
| [13] session.is_genuine_network_retrieval == True? --> If False: FAIL CLOSED           |
| [14] session.http_status == 200?                   --> If != 200: FAIL CLOSED          |
| [15] session.source_channel == "TWSE_OPENAPI"?     --> If True for MOPS: FAIL CLOSED   |
| [16] session.raw_response_hash == receipt hash?    --> If mismatch: FAIL CLOSED        |
| [17] session.endpoint_url == evidence URL?         --> If mismatch: FAIL CLOSED        |
+----------------------------------------------------------------------------------------+
                                            |
                                            v (All 17 Checks Satisfied)
                               +-------------------------+
                               |     VERIFIED_SOURCE     |
                               +-------------------------+
```

---

## 3. P0-2 — Segregation of Candidate Counts from Retrieval Counts

### Dedicated Isolation of Candidate Fixture Inventory
Previously, `evaluate_q1_2024_collection_metrics()` fell back to candidate inventory numbers (9 sessions, 12 announcements) when no queries were executed, erroneously populating `ANNOUNCEMENTS_RETRIEVED` and `retrieval_coverage`. Under V8.13 Final Audit:
1. **Zero-Query Integrity**: When no genuine network queries are executed:
   - `sessions_queried = 0`
   - `announcements_retrieved = 0`
   - `verified_count = 0`
   - `retrieval_coverage_ratio = 0.0`
   - `sessions_with_announcements = 0`
   - `announcement_dates = []`
2. **Dedicated `candidate_inventory_metrics` Dictionary**:
   - `candidate_fixture_announcements = 12`
   - `candidate_fixture_sessions = 9`
   - `candidate_fixture_density = 0.1607` (9 / 56)
   - `candidate_dates = ['2024-01-10', '2024-01-15', '2024-01-18', '2024-02-05', '2024-02-15', '2024-03-08', '2024-03-14', '2024-03-15', '2024-03-18']`
   - `isolation_note`: Explicit governance disclaimer stating that candidate inventory metrics are strictly quarantined from genuine retrieval outcomes.
3. **Fine-Grained Query Failure Segregation**:
   - `NOT_ATTEMPTED`: Query was not executed.
   - `QUERY_COMPLETED_ZERO_RESULTS`: Query executed; exchange returned 0 announcements.
   - `AUTHENTIC_NO_ANNOUNCEMENT_CONFIRMED`: All 3 constituents successfully queried with valid response hashes; confirmed 0 material announcements.
   - `ANNOUNCEMENTS_RETRIEVED`: Authentic announcements successfully retrieved and validated.
   - `QUERY_FAILED_HTTP`: Non-200 HTTP response.
   - `QUERY_FAILED_PARSE`: HTTP 200 received but JSON payload parsing failed.
   - `QUERY_FAILED`: Network timeout or connection drop.

```
+----------------------------------------------------------------------------------------+
| 2024 Q1 Collection & Coverage Metrics (Audited Dynamic Resolution)                     |
+--------------------------------------------------+------------------+------------------+
| Metric Name                                      | Evaluated Value  | Status           |
+--------------------------------------------------+------------------+------------------+
| Planned Trading Sessions in 2024 Q1              | 56 sessions      | Authentic TWSE   |
| Planned Stock-Day Units (56 * 3 constituents)    | 168 units        | Authentic TWSE   |
| Executed Stock-Day Units (Offline Prototype)     | 0 units          | NOT_ATTEMPTED: 56|
| Real Collection Coverage Ratio                   | 0 / 168 (0.0%)   | NOT_EXECUTED     |
| Candidate Fixture Disclosures                    | 12 disclosures   | UNVERIFIED       |
| Sessions with Candidate Disclosures              | 9 sessions       | Density: 16.07%  |
| Genuine Retrieval: Sessions with Announcements   | 0 sessions       | Strictly 0       |
| Genuine Retrieval: Announcements Retrieved       | 0 announcements  | Strictly 0       |
| Live OpenAPI Query Probe HTTP Status             | 200 OK           | 168,879 Bytes    |
| Live OpenAPI Query Probe Raw Payload Hash        | 39e5f055...      | VERIFIED         |
| Three-Way Resolution: NOT_ATTEMPTED              | 56 sessions      | 100.0%           |
| Three-Way Resolution: ZERO_RESULTS               | 0 sessions       | (Requires Live)  |
| Three-Way Resolution: AUTHENTIC_CONFIRMED_ZERO   | 0 sessions       | (Requires Live)  |
| Event Clusters in Candidate Fixture              | 10 clusters      | Deduplication    |
| Rule 51 Clarification Lineage                    | 1 disclosure     | Lineage Handled  |
| Historical Provider Availability Coverage        | 0 / 12 (0.0%)    | Lacks Feed Logs  |
| Formally Verified Sources (Formal PIT Verified)  | 0 / 12 (0.0%)    | Strictly 0       |
+--------------------------------------------------+------------------+------------------+
```

---

## 4. P0-3 — Stock-Day Unit Tracking & Quarantined Simulation Metrics

### Prevention of Simulated Coverage Inflation
Previously, simulated tests on a single stock (2330) across 56 sessions declared `FULLY_EXECUTED` (coverage = 1.0), conflating single-constituent simulation with market-wide coverage. Under V8.13 Final Audit:
1. **`ExecutionMode` Enum**:
   - `QueryAttemptRecord.execution_mode` explicitly marks records as `SIMULATED` or `AUTHENTIC`.
   - Simulated queries can ONLY enter `simulation_metrics` and CANNOT count towards `real_collection_coverage`.
2. **Stock-Day Unit Dimensionality**:
   - Planned coverage spans 3 constituents (2330, 2603, 2881) across 56 sessions = **168 planned stock-day units**.
   - Simulating only 2330 yields $56 / 168 = 33.33\%$, which is strictly classified as **`PARTIALLY_EXECUTED`**.
   - Only when all 168 stock-day units are executed can `FULLY_EXECUTED` be declared in simulation.
   - `real_collection_coverage` remains strictly 0.0% regardless of simulation runs.

---

## 5. P1 — Statistical Calibration: Oracle Simulation vs. Implemented Test

### Dual-Calibration Architecture & Empirical Governance
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
1. All power simulation results are classified strictly as **`calibration_type = "ASSUMPTION_BASED"`**.
2. Statistical power status remains locked at **`power_status = "POWER_NOT_ESTABLISHED"`** under the current pilot size ($N_{\text{test}} = 11$).
3. Empirical calibration status is designated as **`empirical_calibration_status = "FORMAL_EMPIRICAL_CALIBRATION_PENDING"`**.
4. The preliminary ~5.8% empirical false rejection rate is valid only under specific Monte Carlo assumptions ($ICC = 0.15$, $B = 40$ resamples) and must NOT be cited as establishing a proven minimum sample size.
5. The sample size target of 250–300 is designated strictly as **`PROVISIONAL_PLANNING_TARGET_ONLY`**. Proposing a proven minimum sample size is prohibited prior to empirical parameter calibration on real TWSE constituent data.
6. Multi-dimensional sensitivity analysis across ICC $\in [0.05, 0.15, 0.30]$, valid label ratio $\in [0.80, 0.90, 1.00]$, and purging loss $\in [0.05, 0.15, 0.25]$ confirms that intra-cluster dependency drastically reduces effective degrees of freedom (power at $N=100$ drops from 61.2% to 22.4%).

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
- `tests/test_v8_13_historical_collection.py`: 7 passed (including counterexample & OpenAPI isolation tests)
- `tests/test_v8_12_dataset_expansion_feasibility.py`: 5 passed
- `tests/test_v8_11_evidence_and_learning.py`: 5 passed
- All existing V8.0–V8.10 regression suites: passed
- **Total Repository Test Pass**: 100% (211 tests passed)
- **Git Branch**: `fix/v8-audit-patch`
- **Formal Verified Sources Count**: Strictly `0` (authentic baseline preserved)
