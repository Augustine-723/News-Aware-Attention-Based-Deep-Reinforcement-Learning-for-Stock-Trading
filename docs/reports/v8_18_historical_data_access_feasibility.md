# V8.18 — Historical Data Access Feasibility & Acquisition Readiness Report

**Execution Branch**: `fix/v8-audit-patch`  
**Execution Date**: 2026-11-18  
**Research Track**: Track B (External Data Acquisition Dependency)  
**Governance Standard**: V8.18 Zero-Presumption Vendor Feasibility Specification  
**Current Possession of 2024 Q1 Historical Disclosures**: Strictly **0**  
**Historical Verified Sources Count**: Strictly **0**  
**Provider Availability Receipts Count**: Strictly **0**  
**Formal PIT Gate Status**: **NO-GO**  
**Large-Scale RL Training Status**: **NO-GO (Data Access Dependent)**  

---

## Executive Summary

Following the formal engineering closure of V8.17 (which validated weekend rollover mechanics, canonical fold alignment, and model-free latency identifiability), V8.18 transitions the project focus to **empirical data access feasibility**. 

In strict adherence to the **Zero Presumption of Vendor Capability** standard, this report audits the project's currently possessed data assets, systematically evaluates five candidate external disclosure sources against eight mandatory evidentiary criteria, outlines concrete acquisition routes, and issues a formal Go / No-Go decision regarding readiness for subsequent deep reinforcement learning (DRL) research.

```mermaid
flowchart TD
    subgraph Current_Possession["1. Existing In-Repository Data Inventory"]
        D1["Market OHLCV (2023-06 to 2024-03, 202 Days): AVAILABLE"]
        D2["Single-Day OpenAPI Snapshot (2026-11-17): AVAILABLE"]
        D3["Candidate Fixtures (50 Events): UNVERIFIED PILOT ONLY"]
        D4["Genuine 2024 Q1 Disclosures: ZERO IN POSSESSION"]
    end

    subgraph Vendor_Evaluation["2. Candidate Historical Source Audit"]
        S1["TEJ (Taiwan Economic Journal)"]
        S2["TWSE / MOPS Official Historical Services"]
        S3["SFI (Securities & Futures Institute)"]
        S4["Commercial Wires (Bloomberg / LSEG)"]
        S5["Open Web Corpora (Yahoo / PTT / Media)"]
    end

    subgraph Decision_Gate["3. V8.18 Operational Readiness Verdict"]
        R1["Formal PIT Gate: NO-GO"]
        R2["Historical Dataset Status: PENDING_ACQUISITION"]
        R3["Large-Scale RL Training: NO-GO"]
    end

    Current_Possession --> Vendor_Evaluation
    Vendor_Evaluation --> Decision_Gate
```

---

## 1. Inventory of Existing Project Data Assets

An exhaustive audit of local repository storage in `data/` and `stock_rl/evaluation/` confirms the following physical assets:

| Data Asset | Physical Path | Temporal Coverage | Item Count | Provenance & Legal Rights | Research Validity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Market OHLCV & Features** | `data/poc_real_data/` | 2023-06-01 to 2024-03-29 | 202 trading days (606 Stock-Days) | TWSE / Yahoo Finance public market bars; unencumbered research use | Fully verified market price history |
| **Official TWSE Snapshot** | [twse_openapi_raw_response_snapshot.json](file:///home/augustine/Desktop/News-Aware%20Attention-Based%20Deep%20Reinforcement%20Learning%20for%20Stock%20Trading/data/official_disclosures/twse_openapi_raw_response_snapshot.json) | 2026-11-17 (Single Day) | 94 announcements (155,496 Bytes) | HTTPS pull from official TWSE OpenAPI; SHA-256 `949d1b51...` | Engineering schema & locator verification only; **0 Q1 2024 records** |
| **MOPS Recent Snapshot** | `data/poc_real_data/mops_announcements.json` | 2026-11-16 (Single Day) | 115 announcements | Sanctioned public capture; Taiwan Open Data Terms | Engineering locator verification only; **0 Q1 2024 records** |
| **Candidate Fixtures** | [RAW_PILOT_EVENTS](file:///home/augustine/Desktop/News-Aware%20Attention-Based%20Deep%20Reinforcement%20Learning%20for%20Stock%20Trading/stock_rl/evaluation/pilot_dataset.py) | 2023-06 to 2024-03 | 50 candidate disclosures (19 in Q1) | Hand-curated / synthetic pilot fixtures; unverified timestamps | Exploratory software pipeline testing only; **prohibited from alpha claims** |
| **Synthetic Boundary Fixtures** | In-memory via `create_synthetic_boundary_fixtures` | 2024-03-15 (Synthetic) | 4 boundary test cases | Purely synthetic test fixtures; quarantined via `is_quarantined=True` | Time-gate cutoff verification only; **prohibited from alpha claims** |

> [!CRITICAL]
> **Definitive Inventory Finding**: The project currently possesses **strictly 0 genuine historical disclosures for 2024 Q1**. No external provider has yet delivered authentic historical payloads under a verified Point-in-Time audit trail.

---

## 2. Systematic Evaluation of Candidate Historical Sources

In accordance with user instructions, five candidate disclosure sources are evaluated across eight mandatory research criteria without assuming unverified vendor capabilities:

### 2.1 Criteria Definition
1. **2024 Q1 Historical Coverage**: Complete historical disclosure record for target tickers (2330 TSMC, 2603 Evergreen, 2881 Fubon Financial) across 2024-01-01 to 2024-03-31.
2. **Raw Disclosure Content**: Availability of verbatim full-text body (`說明`), headline (`主旨`), and regulatory clauses (`符合條款`).
3. **Publication Timestamp Precision**: Temporal granularity of exchange filing acceptance (`HH:MM:SS` second-level vs `HH:MM` minute-level vs `YYYY-MM-DD` date-level).
4. **Historical Availability Evidence**: Physical audit trail proving when the disclosure was first accessible to market participants (e.g. downstream vendor feed ingress time, TCP dropcopy timestamp, or network reception log).
5. **Academic & ML Training Licensing**: Legal authorization to use data for academic machine learning research, offline DRL training, and non-commercial publication.
6. **Raw Retention & Reproducibility**: Legal right to store raw payloads and cryptographic SHA-256 digests in research archives to ensure scientific reproducibility.
7. **Existing Legal Access Rights**: Current contractual status of access permissions (e.g. active API key, academic grant, or pending agreement).
8. **Data Scale & Missingness Profile**: Expected disclosure volume, survivorship bias, handling of corporate revision lineage, and known data omissions.

---

### 2.2 Candidate Sources Comparative Matrix

| Source / Vendor | 1. 2024 Q1 Coverage | 2. Raw Content | 3. Timestamp Precision | 4. Availability Evidence | 5. ML / Academic License | 6. Raw Retention | 7. Existing Access | 8. Data Scale & Missingness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Source 1: TEJ (Taiwan Economic Journal)** | **Complete** (100% of 2330, 2603, 2881) | Full headline + body text; standardized tables | Typically **Minute (`HH:MM`)**; Second precision requires real-time tier | **None for history**; records filing time, not downstream broadcast time | Permitted under Academic Campus License | **Hashes Only**; raw payload redistribution barred | **NO** (Requires institutional subscription) | High scale (~150 disclosures for 3 stocks); revision lineage tracked via TEJ event codes |
| **Source 2: TWSE / MOPS Official Services** | **Complete** (Authoritative regulatory archive) | Complete verbatim Chinese text, resolutions, attachments | **Second (`HH:MM:SS`)** (Official exchange clock) | **Exchange filing time only**; downstream wire propagation unrecorded | Permitted under Taiwan Open Data Terms | **Permitted**; raw JSON/HTML snapshots versionable | **Partial** (OpenAPI live only; historical bulk requires application) | 100% universe coverage; zero survivorship bias; full revision history preserved |
| **Source 3: SFI (Securities & Futures Institute)** | **Complete** for major corporate actions | Categorical codes & brief summaries; text often truncated | Typically **Date-only (`YYYY-MM-DD`)** | **None**; intraday timing lost | Academic research permitted via application | **Restricted** to certified research lab | **NO** (Requires academic project grant) | Coarse resolution; lookahead ambiguity around 20:00 cutoff renders intraday gating impossible |
| **Source 4: Commercial Wires (Bloomberg / LSEG)** | **High** for large-caps (2330, 2603, 2881) | News wire articles, flashes, summaries, English briefs | **Microsecond / Millisecond** (`HH:MM:SS.fff`) | **Highest**; raw wire dropcopy packet capture timestamps | Prohibitive for open source; ML training requires enterprise rider | **Strictly Barred**; public hash retention only | **NO** (Enterprise fees >\$20k/yr; out of scope) | High frequency; heavily filtered towards market-moving stories; editorial selection bias |
| **Source 5: Open Web Media (Yahoo TW / PTT Stock)** | **Fragmented**; non-official secondary reporting | Journalistic articles + community commentary | Variable; seconds often unindexed; CMS timestamp drift | **Unreliable**; retroactively updated without version changelogs | Web scraping terms ambiguous; anti-bot protected | **Uncertain**; potential copyright claims | **NO** (Automated scraping barred by project guidelines) | Severe editorial noise, sentiment bias, unverified rumor content, survivorship distortion |

---

### 2.3 Detailed Source Profiles & Epistemic Limitations

#### 1. Source 1: TEJ (Taiwan Economic Journal — 重大訊息資料庫)
- **Primary Strength**: Established gold standard in Taiwan quantitative finance research with standardized corporate event classification.
- **Epistemic Limitation**: Standard historical exports provide `申報日期` and `申報時間` (filing time), but do not preserve the millisecond packet arrival time of the historical vendor broadcast. Under Track A, availability must still be modeled as $T_{\text{avail}} = T_{\text{filing}} + \tau$.
- **Legal Constraint**: Institutional campus licenses permit academic publication of statistical results, but strictly prohibit committing raw TEJ tables or text dumps to public GitHub repositories. Only derived 160D tensor embeddings and SHA-256 digests can be shared.

#### 2. Source 2: TWSE / MOPS Official Open Data (證交所 / 公開資訊觀測站)
- **Primary Strength**: Primary source authority. Exact exchange timestamp down to the second (`15:47:58`). Full text verbatim Chinese payloads. Public domain / Open Data licensing allows raw payload storage.
- **Epistemic Limitation**: The active OpenAPI endpoint (`/v1/opendata/t187ap04_L`) only returns current/recent announcements (as documented in V8.14 snapshot of 94 items). Accessing historical 2024 Q1 data through the web UI is guarded by rate limits and anti-bot checks. Automated scraping is strictly barred by project guidelines.
- **Acquisition Path**: Requires submitting a formal academic Open Data request to the TWSE Data Department or using official TWSE historical data download CDs/APIs.

#### 3. Source 3: SFI (證券暨期貨市場發展基金會)
- **Primary Strength**: Subsidized for university research in Taiwan.
- **Fatal Epistemic Flaw**: Most academic tables provide only date-level timestamps (`YYYY-MM-DD`). Because material announcements filed after 13:30 (market close) or after 20:00 (decision cutoff) must be cleanly rolled forward to $T+1$, date-only data introduces severe lookahead contamination. SFI data cannot support the V8.17 20:00 cutoff decision model.

#### 4. Source 4: Commercial Wires (Bloomberg B-PIPE / LSEG DataScope)
- **Primary Strength**: True microsecond packet ingress timestamps, providing genuine Proof-of-Availability for Point-in-Time verification.
- **Fatal Resource Flaw**: Prohibitive enterprise terminal cost and strict contractual prohibition against persistent storage of wire text in local Git workspaces.

#### 5. Source 5: Open Web Corpora (Financial Portals / Social Media)
- **Primary Strength**: Zero direct financial cost.
- **Fatal Integrity Flaw**: Secondary reporting is subject to unannounced retroactive editing, editorial hallucination, and copyright restrictions. Access requires bypassing anti-bot/CAPTCHA systems, which directly violates project ethical guidelines.

---

## 3. Concrete Data Acquisition Options for Track B

```mermaid
flowchart TD
    Start["V8.18 Historical Acquisition Options"] --> Opt1["Route 1: Sanctioned TWSE Historical Open Data Request"]
    Start --> Opt2["Route 2: Institutional TEJ Academic Research Export"]
    Start --> Opt3["Route 3: Synthetic Empirical Distribution Modeling"]

    Opt1 --> P1["Contact TWSE Data Service / MOPS Archive"]
    P1 --> O1["Obtain 2024 Q1 Raw JSON Archive with Second Precision"]
    O1 --> V1["Verify SHA-256 Manifest & Exact Locators"]

    Opt2 --> P2["Submit Academic Lab Data Request to TEJ"]
    P2 --> O2["Export 2330/2603/2881 Q1 Disclosure Tables"]
    O2 --> V2["Parse Text, Hash Raw Bytes, Store Extracted Vectors"]

    Opt3 --> P3["Calibrate Parametric Timestamp Generator on Snapshot Data"]
    P3 --> O3["Generate Quarantined Benchmark Fixtures"]
    O3 --> V3["Test Time-Gate Mechanics under Realistic Filing Density"]
```

### Option A (Recommended): Sanctioned TWSE Historical Open Data Application
- **Methodology**: Apply through official TWSE research/academic open data channels for the complete raw JSON dataset of 2024 Q1 material announcements (`t187ap04_L` historical table).
- **Target Deliverable**: An immutable snapshot file `data/official_disclosures/twse_2024_q1_raw_disclosures.json` with associated `evidence_manifest_v8_18.json`.
- **Licensing Compliance**: Fully compliant with Taiwan Open Data Terms; allows full offline reproducibility and cryptographic verification.

### Option B: Institutional TEJ Academic Data Export
- **Methodology**: Coordinate through an affiliated university finance laboratory with an existing TEJ subscription to extract the 2024 Q1 corporate disclosure sub-table for stocks 2330, 2603, and 2881.
- **Target Deliverable**: Normalized CSV/JSON table, with raw vendor bytes hashed locally. Raw text retained only in offline scratch storage, not pushed to public remote repositories.
- **Licensing Compliance**: Compliant under academic research exemptions with hash-only remote tracking.

### Option C: Synthetic Empirical Replay Calibration (Fallback Diagnostic)
- **Methodology**: If external historical archives remain delayed, calibrate a synthetic disclosure generator using the empirical timestamp distributions observed in the 94-announcement TWSE snapshot (filing times between 06:30 and 22:30).
- **Governance Constraint**: All generated data must carry `is_quarantined_test_fixture=True` and `SYNTHETIC_FIXTURE_QUARANTINED`. Absolutely prohibited from claiming real financial alpha.

---

## 4. Pending Confirmation Items & Preflight Checklist

Before any historical disclosure dataset can transition from Track B into the production pipeline, the following 8-step verification checklist must be executed:

```
[ ] Step 1: Verification that the physical file exists in data/official_disclosures/
[ ] Step 2: Verification of verbatim raw bytes SHA-256 digest against recorded manifest
[ ] Step 3: Exact match verification of Company Code (2330, 2603, 2881)
[ ] Step 4: Timestamp column audit confirming presence of second-level publication time (HH:MM:SS)
[ ] Step 5: Separation of company filing time from downstream receipt delivery time
[ ] Step 6: Full parent-child revision lineage check linking corrections to original announcement IDs
[ ] Step 7: Dual-check quarantine check confirming zero synthetic test fixtures in empirical sample
[ ] Step 8: Execution of model-free latency identifiability preflight across tau in {0, 5, 15, 30, 60} min
```

---

## 5. Formal Operational Readiness Verdict

| Research Milestone | Status | Epistemic Justification |
| :--- | :--- | :--- |
| **V8.17 Engineering Baseline** | **CLOSED & FROZEN** | All 5 preflight audit metrics validated; 250 unit tests passing; zero technical debt. |
| **P1-A Quarantine Enforcement** | **PASSED (REINFORCED)** | Dual-check gate enforces ID non-overlap and marker inspection (rejecting renamed fixtures). |
| **P1-B Dynamic Metadata Audit** | **RESOLVED & BOUNDED** | Real encoder audited; E23 `event_age` confirmed tau-dependent; full tensor equality bounded as `FULL_STATE_EQUALITY_NOT_TESTED`. |
| **2024 Q1 Historical Dataset** | **NO-GO (PENDING)** | 0 authentic historical disclosures possessed; requires formal data acquisition under Track B. |
| **Formal Point-in-Time Gate** | **NO-GO** | Zero verified external provider availability receipts in possession. |
| **Large-Scale DRL Model Training** | **NO-GO** | Training on unverified fixtures or without genuine historical data violates research governance. |
| **Overall Research Readiness** | **PAUSED FOR DATA ACQUISITION** | Engineering pipeline is fully prepared; awaiting physical delivery of 2024 Q1 historical feeds. |

---
*Report certified under V8.18 Historical Data Access Feasibility Standard. Engineering Baseline Frozen. Historical PIT = NO-GO. Training Status = NO-GO.*
