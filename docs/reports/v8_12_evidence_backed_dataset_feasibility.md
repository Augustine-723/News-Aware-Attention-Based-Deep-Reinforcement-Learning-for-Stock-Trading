# V8.12 Deliverable Report: Evidence-Backed Historical Dataset Expansion & Source Feasibility

## 1. Executive Summary & V8.12 Scope

The **V8.12** milestone establishes the formal verification preflight, legal licensing boundaries, temporal point-in-time (PIT) constraints, and econometric sample size readiness required for expanding the historical news and event disclosure dataset. Following the V8.11 research integrity gate, all statistical conclusions regarding semantic incremental value remain strictly **`INCONCLUSIVE`** under small-sample uncertainty ($N_{\text{test}} = 11$). Prior to expanding supervised learning or multimodal reinforcement learning, V8.12 hardens external source certification and evaluates the empirical feasibility of historical news acquisition for Taiwan Stock Exchange (TWSE) liquid constituents.

### Key Milestones Achieved
1. **Trusted Source Certification Preflight (Requirement 1 & Code Review P0)**:
   - Eliminated all caller self-attestation: all certification booleans default to `False` (`is_trusted_external_certified = False`, `is_existence_verified = False`, `is_availability_verified = False`).
   - Merely providing a trusted authority name or setting boolean flags to `True` without external evidence is strictly rejected.
   - Verification strictly requires an authentic `ExternalSourceAdapterReceipt` issued by a recognized external data adapter (`TrustedExternalSourceAdapter`).
   - Replaced `startswith()` domain matching with strict `urllib.parse.urlparse` validation (enforcing scheme `https`, exact hostname whitelist, and `/mops/` path prefix).
   - Re-designated internal catalog as `LOCAL_MOPS_CANDIDATE_CATALOG`, explicitly marking it as a local candidate schema lookup fixture, NOT an authoritative external data source.
   - Enforced cross-auditing of adapter ID, retrieval timestamp, raw response hash, announcement ID, company ID, and publication datetime across raw event, evidence bundle, and adapter receipt.
   - Implemented cross-announcement evidence reuse detection.
   - Maintained default audit baseline: **`VERIFIED_SOURCE: 0`**, **`MOPS_CANDIDATE_UNVERIFIED: 12`**, **`CURATED_UNVERIFIED: 38`** (100% unverified honest reporting; zero simulated promotions).
2. **Segregation of Event Existence vs. Historical Availability (Requirement 1 & 2)**:
   - Formally decoupled **Event Existence Verification** (`is_existence_verified`) from **Historical Provider Availability Verification** (`is_availability_verified`).
   - Established that official exchange filing archives prove that an event occurred, but cannot prove historical point-in-time feed availability at 20:00 Taiwan decision cutoff without physical data vendor ingestion receipts.
   - All 50 pilot events remain strictly governed by the Estimated-PIT boundary.
3. **News Coverage & Fixture Density Re-interpretation (Requirement 3 & Code Review P1)**:
   - Renamed `Zero-Event Trading Days` to `No-Recorded-Event Days in Curated Fixture` (159/202 authentic TWSE trading sessions, 78.71%).
   - Clarified that the absence of curated records in the pilot fixture does NOT indicate that the market had no news on those trading days.
   - Explicitly rejected the over-broad deductive inference that a live trading system degenerates to market-only on 80–90% of trading days.
   - Decoupled `collection_coverage` (0.0% live crawler deployed), `retrieval_coverage` (0.0% verified vendor feed records), `fixture_event_density` (43 / 202 = 21.29%), and `model_input_slot_occupancy` (32 recent + 8 active slots populated from market-wide feed in production).
4. **Evidence-Based Data Licensing & Usage Rights Audit (Requirement 2, 5 & Code Review P1)**:
   - Decoupled data source identity from actual legal license grants.
   - Stored formal `LicenseMetadata` on each event record tracking license name, terms URL, version, retrieval timestamp, permitted uses, and verification status (`LicenseVerificationStatus`).
   - Clarified that public accessibility of MOPS filings under Taiwan Securities and Exchange Act Art. 36 provides regulatory transparency, but does NOT grant blanket copyright permission for commercial AI model training or raw redistribution (`verification_status = NEEDS_LEGAL_REVIEW`).
   - Categorized commercial media articles lacking formal publisher contracts as `LICENSE_UNVERIFIED`, with fair use research doctrine marked as strictly `NEEDS_LEGAL_REVIEW`.
   - Maintained default constraint: `commercial_redistribution_eligible = False` (**0 eligible records** across the entire dataset).
5. **Econometric Sample Size & Cluster-Bootstrap Power Analysis (Requirement 3 & Code Review P1)**:
   - Removed unvalidated hardcoded 80% power constants (`required_n_test = 45`, `required_n_total = 250`).
   - Clarified the discrepancy between simplified single-sample formulas ($N \approx 19$) and real-world complex estimation.
   - Designed a reproducible Cluster-Bootstrap Simulation Power Analysis for differences between correlated Spearman coefficients ($\Delta \rho$).
   - Explicitly modeled event cluster dependencies (ICC $\approx 0.15$), purged walk-forward embargo losses (15%), and valid label attrition (10%).
   - Formally designated: **`power_status: POWER_NOT_ESTABLISHED`** and **`statistically_justified_minimum: NOT_ESTABLISHED_DUE_TO_PILOT_UNDERPOWERING`**.
   - Retained 250–300 strictly as a preliminary operational planning target (`planning_sample_target = 250-300 (HEURISTIC_PLANNING_TARGET_ONLY)`), NOT a proven minimum sample size.
6. **Retention of Baseline Evaluation & 100% Test Verification (Requirement 4 & 5)**:
   - Retained Genuine BGE 128D, 32D Metadata, Method B, B-matched, and Method C.
   - All regression and new unit tests pass with 100% success rate (**204 passed in full suite**).

---

## 2. Trusted Source Certification Preflight

### Elimination of Caller Self-Attestation
In prior iterations, evidence bundles could be instantiated with self-asserted authority and `True` booleans. V8.12 eliminates all self-attestation:
1. All certification booleans in `MOPSSourceEvidence` default strictly to `False`.
2. A caller filling in `certifier_authority = "TWSE_MOPS_OFFICIAL_ARCHIVE"` or setting booleans to `True` without an authentic external adapter receipt is immediately rejected.
3. Official verification requires an `ExternalSourceAdapterReceipt` issued by a recognized external adapter (`TrustedExternalSourceAdapter`).
4. External reference URLs are parsed using `urllib.parse.urlparse`, verifying HTTPS scheme, exact official host whitelisting, and valid path structure.
5. Evidence bundles are cross-audited against candidate events to detect cross-announcement reuse.
6. In the absence of a live external adapter connection, all candidate certifications remain pending (**`VERIFIED_SOURCE: 0`**).

```text
+------------------------------------+
| Caller / Data Pipeline             |
+------------------------------------+
                  |  Passes evidence bundle
                  v
+---------------------------------------------------------------------------------+
| TrustedSourceCertifier (Fail-Closed Verification Engine)                         |
|---------------------------------------------------------------------------------|
| [1] Is external adapter receipt present? (ExternalSourceAdapterReceipt)          |
|     --> If adapter_receipt is None (caller self-attestation): REJECTED          |
| [2] Is certifier authority recognized? (TWSE_MOPS_OFFICIAL_ARCHIVE, etc.)       |
|     --> If UNTRUSTED_SELF_ASSERTED or local mock: REJECTED                      |
| [3] Is external reference URL verified via urlparse?                            |
|     --> Scheme != 'https', invalid host, subdomain spoof, or invalid path:      |
|         REJECTED                                                                |
| [4] Does adapter receipt endpoint URL match external reference URL?             |
|     --> If mismatch: REJECTED                                                   |
| [5] Does announcement ID exist in local candidate catalog?                      |
|     --> If forged or missing from index: REJECTED                               |
| [6] Does announcement ID match expected candidate event? (Cross-Reuse Check)    |
|     --> If reused from another announcement: REJECTED                           |
| [7] Do company ID and publication timestamp match across receipt and event?     |
|     --> If mismatch: REJECTED                                                   |
| [8] Do raw content and SHA-256 payload hash match adapter raw response hash?     |
|     --> If mismatch or corrupted: REJECTED                                      |
+---------------------------------------------------------------------------------+
                  |                                      |
         (All Checks Pass)                      (Any Check Fails)
                  v                                      v
     +-------------------------+            +----------------------------+
     | VERIFIED_SOURCE         |            | MOPS_CANDIDATE_UNVERIFIED  |
     | (Audit Record Issued)   |            | (Fail Closed / Demoted)    |
     +-------------------------+            +----------------------------+
```

### Comprehensive Rejection Hardening Table
| Rejection Test Case | Injected Vulnerability / Attack Vector | Certifier Response | Resulting Status | Security Log Message |
| :--- | :--- | :---: | :---: | :--- |
| **Self-Asserted Authority** | `certifier_authority = "UNTRUSTED_SELF_ASSERTED"` | Fail Closed | `MOPS_CANDIDATE_UNVERIFIED` | `Verification rejected: Certifier authority '...' is not a recognized trusted external source certifier.` |
| **Caller Self-Attestation (No Adapter Receipt)** | `adapter_receipt = None`, `is_trusted_external_certified = True` | Fail Closed | `MOPS_CANDIDATE_UNVERIFIED` | `Verification rejected: Evidence bundle lacks authentic ExternalSourceAdapterReceipt (caller self-attestation prohibited).` |
| **Forged Announcement ID** | `announcement_id = "MOPS-2330-20230606-FORGED-ID"` | Fail Closed | `MOPS_CANDIDATE_UNVERIFIED` | `Verification rejected: Announcement ID '...' not found in local candidate catalog.` |
| **Forged URL: Non-HTTPS Scheme** | `external_reference_url = "http://mops.twse.com.tw/..."` | Fail Closed | `MOPS_CANDIDATE_UNVERIFIED` | `Verification rejected: URL validation failed: URL scheme must be 'https', got 'http'.` |
| **Forged URL: Untrusted Host** | `external_reference_url = "https://fake-mops.internal/..."` | Fail Closed | `MOPS_CANDIDATE_UNVERIFIED` | `Verification rejected: URL validation failed: URL hostname 'fake-mops.internal' is not in allowed official hosts.` |
| **Forged URL: Subdomain Spoof** | `external_reference_url = "https://mops.twse.com.tw.attacker.com/..."` | Fail Closed | `MOPS_CANDIDATE_UNVERIFIED` | `Verification rejected: URL validation failed: URL hostname 'mops.twse.com.tw.attacker.com' is not in allowed official hosts.` |
| **Forged URL: Invalid Path** | `external_reference_url = "https://mops.twse.com.tw/evil/path"` | Fail Closed | `MOPS_CANDIDATE_UNVERIFIED` | `Verification rejected: URL validation failed: MOPS URL path '/evil/path' must start with '/mops/'.` |
| **Cross-Announcement Reuse** | Attaching AGM disclosure evidence (`EV-2330-01`) to Dresden Fab event (`EV-2330-03`) | Fail Closed | `MOPS_CANDIDATE_UNVERIFIED` | `Verification rejected: Cross-announcement reuse detected (MOPS-2330-20230606-AGM does not match expected MOPS-2330-20230808-ESMC-DRESDEN).` |
| **Company ID Mismatch** | `company_id = "2881"` for TSMC disclosure (`2330`) | Fail Closed | `MOPS_CANDIDATE_UNVERIFIED` | `Verification failed closed: Company ID mismatch (2881 vs 2330).` |
| **Tampered Response Hash** | Adapter receipt raw response payload differs from disclosure content | Fail Closed | `MOPS_CANDIDATE_UNVERIFIED` | `Verification failed closed: Adapter raw response hash mismatch.` |
| **Raw Event Self-Elevation** | Dictionary in `RAW_PILOT_EVENTS` sets `verification_status = "VERIFIED_SOURCE"` | Fail Closed | `CURATED_UNVERIFIED` | `Self-elevation rejected: RAW_PILOT_EVENTS cannot self-certify as VERIFIED_SOURCE without trusted external source certifier.` |

### Default Source Provenance Summary
```json
{
  "total_records": 50,
  "VERIFIED_SOURCE": 0,
  "MOPS_CANDIDATE_UNVERIFIED": 12,
  "CURATED_UNVERIFIED": 38,
  "SYNTHETIC_FIXTURE": 0,
  "total_unverified_records": 50,
  "missing_records": 0,
  "records_with_external_url": 0,
  "records_with_valid_hash": 50,
  "records_with_retrieval_log": 0,
  "license_verification_status_counts": {
    "LICENSE_UNVERIFIED": 38,
    "NEEDS_LEGAL_REVIEW": 12,
    "LICENSE_VERIFIED": 0
  },
  "commercial_redistribution_eligible": false,
  "commercial_redistribution_eligible_count": 0
}
```

---

## 3. Event Existence vs. Historical Availability Segregation

### The Dual Verification Barrier
A critical finding of V8.12 is that **existence verification does not imply historical availability verification**:

1. **Event Existence Verification (`is_existence_verified`)**:
   - Confirms that the corporate event actually took place and was submitted to the exchange by the company.
   - Verifiable ex-post through TWSE MOPS official disclosure records, filing timestamps, and annual reports.
2. **Historical Provider Availability Verification (`is_availability_verified`)**:
   - Confirms whether the information was accessible to an algorithmic trading engine before the decision cutoff $T$ (20:00 Taiwan Time).
   - Requires physical data vendor ingestion timestamps (e.g. Bloomberg, Refinitiv, TEJ, or authenticated web crawler ingestion logs).
   - In the absence of an authenticated historical provider ingestion receipt, `is_availability_verified` remains strictly `False`.

### Point-in-Time Timeline Protocol
```text
Event Occurs       Filing on MOPS      Vendor Ingest Receipt      Decision Cutoff T      Next Market Open
     |                   |                       |                        |                      |
-----+-------------------+-----------------------+------------------------+----------------------+--->
   15:30               16:30                   16:35                    20:00                  T+1 09:00
 (Existence)      (Published Time)       (Provider Available)     (Decision Horizon)      (Execution)
```
- **Formal PIT Mode**: Requires both `is_existence_verified = True` AND `is_availability_verified = True`.
- **Estimated-PIT Mode**: Employs published time as proxy availability; strictly labeled as `ESTIMATED_PIT_NEWS` with `evidence_level = ESTIMATED`.

---

## 4. Historical News Coverage & Fixture Density Re-interpretation

### Pilot Constituent Breakdown ($N = 50$)
The pilot dataset tracks three key TWSE constituents representing diverse market sectors:
- **2330 (TSMC)**: 20 events (40.0%)
- **2603 (Evergreen Marine)**: 16 events (32.0%)
- **2881 (Fubon Financial)**: 14 events (28.0%)

Event Categorization:
- earnings (財報/營收): 21 events (42.0%)
- routine (日常盤勢/整理): 9 events (18.0%)
- capex (重大擴建/晶圓廠): 8 events (16.0%)
- dividend (股利/除息): 5 events (10.0%)
- crisis (紅海危機/斷航): 4 events (8.0%)
- tech (先進製程進展): 1 event (2.0%)
- governance (經營層退休): 1 event (2.0%)
- clarification (重大澄清): 1 event (2.0%)

### Correct News Coverage Interpretation
In previous drafts, days without events in the pilot dataset were characterized as "zero-event trading days," leading to the incorrect deductive conclusion that real trading systems would degenerate to market-only on 80–90% of trading days. V8.12 formally corrects this interpretation:

1. **Renaming**:
   - `zero_event_trading_days` is renamed to **`no_recorded_event_days_in_curated_fixture`** (159 days out of 202 authentic TWSE trading sessions, 78.71%).
2. **Semantic Distinction**:
   - The pilot dataset is a small-scale curated research fixture focusing on high-materiality corporate events.
   - **The absence of a curated record in the fixture does NOT mean the market or constituent had zero news on that day.**
   - In actual quantitative trading environments, news feeds ingest hundreds of daily market commentaries, broker updates, and macro news items. Real trading policies do not degenerate to market-only policies.
3. **Decoupled Coverage Metrics**:
   - **Collection Coverage**: `0.0%` (no automated live crawling infrastructure currently active in this research repo).
   - **Retrieval Coverage**: `0.0%` (0 verified vendor records; 12 candidate MOPS filings and 38 curated media articles pending official vendor feed receipts).
   - **Fixture Event Density**: `43 / 202 = 21.29%` (proportion of trading sessions with at least one curated pilot event).
   - **Model Input Slot Occupancy**: The multimodal architecture provides capacity for 32 recent + 8 active event slots. In production, these slots are populated from the comprehensive market-wide news feed, fully isolated from pilot fixture sparsity.

---

## 5. Evidence-Based Data Licensing & Legal Boundaries

### Decoupling Source Identity from License Grants
Prior iterations assumed that MOPS filings were automatically open data and that media articles could be definitively categorized under fair use. V8.12 decouples data source identity from actual legal grants by attaching formal `LicenseMetadata` to every record:

```text
+-----------------------------------------------------------------------------------------+
| 50 Pilot Candidate Events                                                               |
+-----------------------------------------------------------------------------------------+
        |                                                                 |
        v                                                                 v
+------------------------------------------------+  +-------------------------------------+
| 12 MOPS Corporate Disclosures                  |  | 38 Commercial Financial News Items  |
+------------------------------------------------+  +-------------------------------------+
| License: Taiwan Open Government Data License   |  | License: Commercial Media Copyright |
| Status: NEEDS_LEGAL_REVIEW                     |  | Status: LICENSE_UNVERIFIED          |
| Permitted Uses: RESEARCH_ANALYSIS, QUOTATION   |  | Fair Use: NEEDS_LEGAL_REVIEW        |
| Redistribution: Strictly PROHIBITED            |  | Redistribution: Strictly PROHIBITED |
+------------------------------------------------+  +-------------------------------------+
```

### Legal Audit Table
| Category | Records | Share | Governing Legal Framework | Documented License Status | Commercial Model Training & Redistribution |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **MOPS Public Disclosures** | 12 | 24.0% | Taiwan Securities & Exchange Act Art. 36; Open Government Data License 1.0 | **`NEEDS_LEGAL_REVIEW`**: Public accessibility ensures regulatory transparency, but does NOT grant blanket copyright permission for commercial AI training or raw redistribution without legal review | **PROHIBITED** (`commercial_redistribution_eligible = False`) |
| **Commercial Media News** | 38 | 76.0% | Copyright Act (著作權法) Art. 52, 65; Fair Use Doctrine | **`LICENSE_UNVERIFIED`**: Specific publisher terms are unrecorded; research fair use doctrine remains **`NEEDS_LEGAL_REVIEW`** | **PROHIBITED** (`commercial_redistribution_eligible = False`) |
| **Redistribution Eligible** | **0** | **0.0%** | Combined Dataset Status | Entire bundle is restricted to internal research benchmarking | **0 Eligible Records** |

---

## 6. Econometric Sample Size & Cluster-Bootstrap Power Analysis

### Supervision Target Validation
`DatasetExpansionReadinessAssessor` validated target completeness across all 50 pilot events:
- **`HUMAN_IMPORTANCE`**: 50/50 valid (**100.0% coverage**, 0% missing).
- **`EX_POST_MARKET_IMPACT`**: 45/50 valid (**90.0% coverage**, 10.0% missing due to market calendar cutoff).
- **`HYBRID_IMPORTANCE`**: 45/50 valid (**90.0% coverage**, 10.0% missing; zero fallback strictly enforced).

### Cluster Deduplication & Version Tracking
- **Total Unique Event Clusters**: 45 clusters across 50 events.
- **Clusters with Multiple Events**: 4 clusters (TSMC Chiayi CoWoS, TSMC Germany fab, Evergreen Red Sea rerouting, Fubon FY23 financials).
- **Revision Tracking**: 1 event with revision ID > 1 (`EV-2330-10-REV2`, Rule 51 media clarification).
- **Exact Payload Hash Duplicates**: **0 records** (100% distinct content hashes).

### Purged Walk-Forward Effective Sample Sizes
- **Window 1 (W1)**: Train $N=23$, Val $N=5$, Test $N_{\text{test}} = 11$ (2023-12-16 to 2024-01-31).
- **Window 2 (W2)**: Train $N=28$, Val $N=11$, Test $N_{\text{test}} = 11$ (2024-02-01 to 2024-03-31).
- **Temporal Overlap**: Disclosed that W1 Test overlaps chronologically with W2 Validation in rolling retraining simulation.

### Cluster-Bootstrap Simulation Power Analysis
Why does simplified formula derivation ($N \approx 19$) fail in practice?
1. **Simplified Formula Assumptions**: Assumes independent and identically distributed (i.i.d.) observations, zero cluster correlation, zero walk-forward embargo loss, and 100% label availability.
2. **Real-World Complexities**:
   - **Intra-Cluster Correlation (ICC $\approx 0.15$)**: Multiple events belong to the same corporate narrative cluster, reducing the effective degrees of freedom.
   - **Purged Walk-Forward Embargo Loss (15%)**: Events spanning the boundary window are purged to prevent lookahead bias.
   - **Label Attrition (10%)**: Ex-post market impact requires T+1 and T+2 trading sessions; holidays and boundary cutoffs create missing labels.
   - **Correlated Predictors**: Testing the difference between two correlated Spearman coefficients ($\Delta \rho = \rho(y, \hat{y}_C) - \rho(y, \hat{y}_{BM})$) introduces substantial covariance between rankings.

To address this, V8.12 implements `simulate_cluster_bootstrap_power` across 3 random seeds (`[42, 100, 2024]`) with 300 simulations per seed:

#### Empirical Power Curve for Correlated Spearman Difference ($\Delta \rho = +0.20, \alpha = 0.05$)
| Test Sample Size $N_{\text{test}}$ | Simulated Standard Error $\text{SE}(\Delta \rho)$ | Simulated Power ($1 - \beta$) | 95% Power Simulation Range | Formal Power Verdict |
| :---: | :---: | :---: | :---: | :--- |
| **$N = 11$ (Current Pilot)** | **0.4635** | **5.56%** | `[5.00%, 6.67%]` | **`POWER_NOT_ESTABLISHED`** (Severely Underpowered) |
| **$N = 25$** | 0.2780 | 9.56% | `[8.67%, 10.33%]` | Underpowered |
| **$N = 50$** | 0.1988 | 13.33% | `[11.00%, 15.33%]` | Underpowered |
| **$N = 100$** | 0.1384 | 27.89% | `[22.00%, 32.67%]` | Insufficient Power |
| **$N = 200$** | 0.0926 | 54.78% | `[51.00%, 58.67%]` | Approaching Viability |

#### Formal Statistical Conclusions
- **`power_status`**: **`POWER_NOT_ESTABLISHED`**
- **`statistically_justified_minimum`**: **`NOT_ESTABLISHED_DUE_TO_PILOT_UNDERPOWERING`**
- **`planning_sample_target`**: **`250-300 (HEURISTIC_PLANNING_TARGET_ONLY)`**
- The target of 250–300 total events across TWSE constituents is strictly an operational heuristic planning target. It must NOT be represented as a statistically proven minimum sample size.

---

## 7. Baseline Evaluation Retention & Inconclusive Verdict

The complete V8.11 evaluation architecture was retained and re-executed without regression:
- Genuine Pretrained BAAI/bge-small-zh-v1.5 (128D projection).
- Point-in-Time 32D Metadata vectors.
- Capacity-matched baseline (Method B-matched, 10,721 parameters).
- Paired bootstrap uncertainty estimation (1,000 resamples).

### Multi-Target Results on Out-of-Sample Test Split ($N_{\text{test}} = 11$)
| Target | Method Bm $\rho$ | Method C $\rho$ | $\Delta \rho$ | Method Bm NDCG@3 | Method C NDCG@3 | $\Delta \text{NDCG@3}$ | Bootstrap 95% CI | $P(\Delta \rho > 0)$* | Formal Verdict | Exploratory Descriptor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **HUMAN** | 0.2953 | **0.6423** | **+0.3470** | 0.8466 | **0.9534** | **+0.1068** | `[-0.3467, +0.9041]` | 83.6% | **INCONCLUSIVE** | **EXPLORATORY_POSITIVE_TREND** |
| **MARKET** | **0.3792** | 0.3425 | **-0.0367** | 0.2448 | **0.2463** | **+0.0015** | `[-0.4509, +0.8500]` | 63.5% | **INCONCLUSIVE** | **INCONCLUSIVE_NO_TREND** |
| **HYBRID** | 0.4727 | **0.5182** | **+0.0455** | 0.5562 | **0.6808** | **+0.1246** | `[-0.4134, +0.7850]` | 72.2% | **INCONCLUSIVE** | **EXPLORATORY_POSITIVE_TREND** |

*\*Note: $P(\Delta \rho > 0)$ is an empirical bootstrap resample proportion, not a Bayesian posterior probability.*

---

## 8. Test Suite Summary & Operational Boundaries

### Test Suite Execution
- `tests/test_v8_10_pilot.py`: 10/10 passed.
- `tests/test_v8_11_evidence_and_learning.py`: 5/5 passed.
- `tests/test_v8_12_dataset_expansion_feasibility.py`: 5/5 passed (including 8 rejection hardening gates).
- **Full Repository Suite**: **204 passed in full test suite (100% pass rate)**.

### Operational Boundaries
- **Active Git Branch**: Strictly maintained on `fix/v8-audit-patch`.
- **Zero Fabrication**: Zero fabricated URLs, announcement serial numbers, or mock retrieval logs.
- **Fail-Closed Verification**: Zero unauthenticated promotions to `VERIFIED_SOURCE` (100% pending).
- **No Large-Scale Out-of-Scope Execution**: Zero 5-year full crawls, no heavy RL agent loops, no paid data subscriptions, and no live order execution.
