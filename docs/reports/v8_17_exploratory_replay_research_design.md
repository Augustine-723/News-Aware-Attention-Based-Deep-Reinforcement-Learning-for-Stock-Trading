# V8.17 — Exploratory Replay Research Design & Baseline Freeze Specification

**Execution Branch**: `fix/v8-audit-patch`  
**Base Commit**: `9cf110b484cb1d91dedb47721133899514dea2f1`  
**Execution Date**: 2026-11-18  
**Audit Standard**: V8.17 Engineering Baseline Freeze & Exploratory Research Governance  
**Operational Status**: Research Design Phase (Zero Model Training / Purely Analytical)  
**Formal PIT Status**: **NO-GO**  
**Historical Verified Sources**: Strictly **0**  
**Provider Availability Verified**: Strictly **0**  

---

## Executive Summary

Following the formal closure of V8.16, this document establishes the **V8.16 Engineering Baseline Freeze** and delineates the complete research methodology for **Track A (Exploratory Replay Research)** and **Track B (External Data Acquisition Dependency)**. 

In strict adherence to research governance principles:
1. **No Large-Scale RL Training**: The engineering architecture is frozen; no DQN policy refactoring, attention head additions, or model hyperparameter retraining are conducted in this phase.
2. **Preservation of the Claim Boundary**: Without authentic third-party vendor streams, cryptographic zero-trust proofs cannot be claimed. Local test fixtures remain strictly `SIMULATED_CONTRACT_TEST`.
3. **Exploratory Scope Quarantine**: Research under `ESTIMATED_REPLAY` is classified strictly as exploratory feasibility research. Unverified disclosures cannot be used to claim proven financial alpha or semantic representation superiority.

---

## 1. Preservation of the Formal PIT Claim Boundary

Prior to outlining experimental protocols, the epistemic boundaries of the system's verification mechanisms must be explicitly established:

```mermaid
flowchart TD
    subgraph Local_Boundary["Local Engineering Trust Boundary (Single Process)"]
        A["raw_receipt_bytes"] --> B["SHA-256 Digest Recomputation"]
        B --> C{"Matches receipt_hash?"}
        C -->|Yes| D["Receipt Byte Integrity Verified (SIMULATED_CONTRACT_TEST)"]
        C -->|No| E["Rejection (Corrupted/Forged String)"]
    end

    subgraph Missing_External_Boundary["External Provider Verification (Currently Missing)"]
        F["Vendor TLS Session"] -.-> G["Cryptographic Digital Signature"]
        G -.-> H["Independent Dropcopy Audit Log"]
        H -.-> I["Authenticity Verified (Formal PIT)"]
    end

    D -.->|CANNOT Prove| I
```

### 1.1 Five Epistemic Guardrails
1. **SHA-256 Digest Limitation**: SHA-256 consistency merely verifies that `raw_receipt_bytes` matches the recorded `receipt_hash`. It provides **zero mathematical proof** that the bytes originated from an authorized external vendor or market feed.
2. **Prohibition of "Authenticity Verified" Claims**: In the absence of an authenticated TLS session, asymmetric cryptographic signature (e.g., RSA/ECDSA from an exchange/vendor PKI), or immutable external audit trail, the system makes **zero claim of Authenticity Verified**.
3. **Receipt Content Binding Boundary**: Without a trusted external adapter that actively parses `raw_receipt_bytes` to cross-validate `provider_id`, `announcement_id`, `content_payload_hash`, and millisecond `observed_available_at`, **Receipt Content Binding is strictly unverified**.
4. **Local Positive Tests Quarantine**: All unit and integration test fixtures constructed locally are formally designated as `SIMULATED_CONTRACT_TEST`. They confirm schema contract compliance only, and are strictly barred from granting formal Point-in-Time status.
5. **Formal PIT Status**: In the absence of an operational, externally certified provider adapter, **Formal PIT remains strictly NO-GO**.

---

## 2. V8.16 Engineering Baseline Freeze

The engineering baseline achieved at Commit `9cf110b484cb1d91dedb47721133899514dea2f1` is formally **frozen**. The architecture is structured as follows:

| Subsystem Component | Frozen Specification | Invariance Constraint |
| :--- | :--- | :--- |
| **Source Evidence Manifest** | [EvidenceManifestEntry](file:///home/augustine/Desktop/News-Aware%20Attention-Based%20Deep%20Reinforcement%20Learning%20for%20Stock%20Trading/stock_rl/evaluation/official_source_collector.py#L221-L254) with 7-step integrity audit | No relaxation of required fields, hashes, or locators |
| **Retrieval Session Registry** | Strict isolation between Production and Simulated sessions | Production sessions require genuine HTTPS retrieval provenance |
| **Exact Source Locator** | Structured locator: `JSON_ARRAY_RECORD[stock_id,raw_array_index,matched_stock_index]` | Bidirectional 1-to-1 index consistency against snapshot |
| **Snapshot Hash Verification** | TWSE OpenAPI snapshot (155,496 Bytes; SHA-256: `949d1b51...`) | Snapshot file immutable; exact byte re-parsing |
| **Point-in-Time Modes** | `EVIDENCE_VERIFIED_ASOF` (Formal Gate) vs `ESTIMATED_REPLAY` (Exploratory) | Formal gate strictly fail-closed; Replay strictly exploratory |
| **Decision Cutoff Rule** | 20:00:00 Asia/Taipei Trading Day Cutoff | Disclosures filed after 20:00 queued for next trading day |
| **Semantic Text Encoder** | BGE-base-zh-v1.5 (512D) $\to$ Fixed Linear Projection (128D) | Fixed projection weights; frozen language model |
| **Metadata & Relation Vectors** | Metadata 24D + Stock/Event Relation 8D (Total 160D per disclosure) | Preserves categorical and relational event semantics |
| **Attention Working Memory** | Recent Slots (32) + Active Slots (8) | Memory slot dimensions and attention pooling frozen |
| **RL Policy Architecture** | Dual-Stream Q-Network (Market-State Stream + Attention-News Stream) | No DQN architectural redesign; no additional attention heads |

> [!IMPORTANT]
> Any future refactoring of DQN architectures, loss functions, reward clipping, or attention pooling mechanisms is deferred until data availability is resolved under Track B or formally simulated under Track A.

---

## 3. Track A: Exploratory Replay Research Design

Track A outlines a controlled academic investigation into the behavior of news-aware deep reinforcement learning agents under explicit Point-in-Time uncertainty.

### 3.1 Research Hypotheses
- **Null Hypothesis ($H_0$)**: Under strict chronological evaluation and market friction, incorporating unverified news disclosures under estimated availability ($T_{\text{filing}} + 5\text{ min}$) yields no statistically significant improvement in risk-adjusted returns (Sharpe ratio, Calmar ratio) over a pure Market-Only technical baseline.
- **Alternative Hypothesis ($H_1$)**: The structured multimodal attention mechanism extracts genuine semantic signals from material disclosures that improve downside risk protection (Maximum Drawdown reduction), exhibiting stability across moderate transmission latencies ($\tau \le 30\text{ min}$).

---

### 3.2 Controlled Ablation Architecture

To rigorously evaluate the incremental value of news features without conflating market trend momentum with semantic awareness, the experimental framework defines five strictly controlled systems:

```mermaid
flowchart TD
    subgraph Inputs["Market & Disclosure Stream (2024 Q1)"]
        P["Daily OHLCV + Technical Factors"]
        N["Material Disclosures (50 Candidate Fixtures)"]
    end

    subgraph Systems["Controlled Experimental Arms"]
        S1["Arm 1: Market-Only Baseline (Zero News Slots)"]
        S2["Arm 2: News-Aware (Estimated Replay, tau = 5 min)"]
        S3["Arm 3: News-Aware (Zero Latency Check, tau = 0 min)"]
        S4["Arm 4: News-Aware (Latency Perturbed, tau in {15, 30, 60} min)"]
        S5["Arm 5: Shuffled / Permuted Semantic Control"]
    end

    P --> S1
    P & N --> S2
    P & N --> S3
    P & N --> S4
    P & N --> S5

    subgraph Evaluation["Evaluation & Governance Audit"]
        E1["Performance Metrics (Sharpe, Return, MDD)"]
        E2["Latency Fragility Check"]
        E3["Semantic Gain Verification"]
    end

    S1 & S2 & S3 & S4 & S5 --> Evaluation
```

1. **Arm 1: Market-Only Baseline ($S_{\text{market}}$)**:
   - State representation: Technical indicators (RSI, MACD, Bollinger Bands, ATR, Price Returns) + Portfolio State.
   - News slots: Disabled (zeroed out or bypassed).
   - Serves as the primary benchmark to determine whether news awareness delivers true incremental value.
2. **Arm 2: News-Aware Canonical Replay ($S_{\text{replay-5m}}$)**:
   - Full multimodal pipeline with estimated availability: $T_{\text{available}} = T_{\text{filing}} + 5\text{ min}$.
   - Cutoff: 20:00:00 Asia/Taipei.
3. **Arm 3: Zero-Latency Diagnostic ($S_{\text{zero-leak}}$)**:
   - Disclosures ingested instantaneously at $T_{\text{available}} = T_{\text{filing}}$.
   - Serves strictly as a diagnostic upper bound to quantify lookahead sensitivity.
4. **Arm 4: Latency Perturbed Replay ($S_{\text{perturbed}}$)**:
   - Disclosures delayed by discrete intervals $\tau \in \{15, 30, 60\}\text{ min}$.
5. **Arm 5: Permuted Semantic Control ($S_{\text{shuffled}}$)**:
   - News text embeddings are randomly permuted across trading days while preserving announcement volume and timing.
   - Tests whether performance gains originate from semantic content or merely from announcement frequency acting as an activity proxy.

---

### 3.3 Latency Sensitivity Protocol ($\tau$-Sweep)

To detect whether a model is overfit to instantaneous information arrival or exploits hidden lookahead bias, the research protocol evaluates a discrete latency sweep:

$$\tau \in \{0, 5, 15, 30, 60\} \text{ minutes}$$

```
Decision Cutoff (20:00)
       |
---[Filing: 19:40]------------------------------------> Time
       |---> tau = 0m:  Available at 19:40 <= 20:00 (Ingested Today)
       |---> tau = 5m:  Available at 19:45 <= 20:00 (Ingested Today)
       |---> tau = 15m: Available at 19:55 <= 20:00 (Ingested Today)
       |---> tau = 30m: Available at 20:10 >  20:00 (Queued to Tomorrow)
       |---> tau = 60m: Available at 20:40 >  20:00 (Queued to Tomorrow)
```

#### Fragility and Overfitting Criteria
1. **Critical Boundary Shifts**: Announcements filed between 19:30 and 20:00 shift from same-day execution to next-day execution as $\tau$ increases from $5\text{ min}$ to $30\text{ min}$.
2. **Fragility Index ($FI$)**:
   $$FI = \frac{\text{Sharpe}(\tau = 5) - \text{Sharpe}(\tau = 15)}{\text{Sharpe}(\tau = 5)}$$
   - If $FI > 0.35$ (more than $35\%$ performance drop under a $10$-minute latency increase), the policy is classified as **Lookahead-Fragile** and rejected from academic claims of robustness.
   - If performance exhibits monotonic, graceful degradation ($FI \le 0.15$), the policy demonstrates structural information absorption.

---

### 3.4 Event Revision Lineage & Timestamp Leakage Prevention

In corporate disclosure datasets, corrections and clarifications frequently occur hours or days after an initial filing.

```
Session T:    [Initial Filing: Dividend = $3.00] ----------> Ingested at T
Session T+1:  [Correction: Dividend revised to $2.50] ------> Ingested at T+1
```

#### Leakage Prevention Rules
1. **Cluster-Based Lineage Tracking**: Every entry carries `cluster_id`, `is_revision`, and `revised_announcement_id`.
2. **Strict Chronological Masking**:
   - At decision time $T$, the agent **must never** have access to text, hashes, or revision flags of disclosures filed at $T+k$.
   - Revisions must not back-propagate or retroactively overwrite earlier records in the Recent/Active memory slots.
3. **Purged Cluster Cross-Validation**:
   - In cross-validation or train/test splitting, entire event clusters must reside on one side of the split to prevent textual similarity leakage across folds.

---

### 3.5 Chronological Evaluation & Cluster Purging

Standard $K$-Fold cross-validation is strictly prohibited in financial time-series due to serial correlation and textual overlap. Track A enforces **Purged and Embargoed Walk-Forward Cross-Validation**:

```
[--- Train Fold 1 ---][Purge Window][--- Val Fold 1 ---][Embargo][--- Test Fold 1 ---]
                          10 Days                          5 Days
```

1. **Purge Window**: A $10$-day buffer between training and validation folds eliminates autoregressive feature overlap.
2. **Embargo Window**: A $5$-day buffer post-validation prevents performance metric leakage into out-of-sample test splits.
3. **Strict Chronological Ordering**: Time flows forward only; future data is never used in feature normalization, PCA/projection fitting, or vocabulary statistics.

---

### 3.6 Bootstrap Uncertainty & Statistical Power Limits

With $N=50$ curated pilot disclosures across 56 trading sessions (168 Stock-Days for 2330, 2603, 2881), the sample size is constrained. Statistical reporting must adhere to the following feasibility matrix:

| Performance / Statistical Metric | Feasibility Status | Methodological Limitation & Handling |
| :--- | :--- | :--- |
| **Cumulative Return & Annualized Return** | `ESTIMABLE` | Point estimates computed; subject to high variance on 56 sessions. |
| **Annualized Sharpe Ratio** | `ESTIMABLE` | Requires Newey-West standard error correction for serial correlation. |
| **Maximum Drawdown (MDD)** | `ESTIMABLE` | Observed trajectory MDD reportable with confidence intervals. |
| **Latency Sensitivity Decay Curve** | `ESTIMABLE` | Relative Sharpe ratio decay across $\tau \in \{0, 5, 15, 30, 60\}$. |
| **Permutation Test ($p$-value)** | `ESTIMABLE` | 1,000 bootstrap shuffles of announcement-return pairs. |
| **High-Frequency Alpha Decay ($< 1\text{ min}$)** | `NOT_ESTIMABLE` | Daily bars cannot resolve sub-minute execution alpha. |
| **Cross-Industry Factor Generalizability** | `NOT_ESTIMABLE` | Only 3 tickers (Semiconductor, Shipping, Financial); insufficient for cross-sectional factor models. |
| **Tail-Risk Event VaR / CVaR ($99\%$)** | `NOT_ESTIMABLE` | 56 trading days provide insufficient sample density for extreme tail estimation. |
| **Market Impact & Execution Slippage** | `NOT_ESTIMABLE` | Order book depth is not modeled in offline prototype. |

> [!WARNING]
> Because candidate fixtures represent unverified pilot samples, any positive alpha or Sharpe improvement observed in Track A must be explicitly reported as **"Preliminary hypothesis-generating prototype evidence under Estimated Replay"**, and **never** claimed as proven production alpha.

---

## 4. Track B: External Data Acquisition Dependency Specification

Track B remains an independent, non-blocking data acquisition track. In accordance with audit instructions, the project maintains **Zero Presumption of Vendor Capability**: without physical data inspection, contract confirmation, and independent schema audits, no provider is presumed to possess compliant Point-in-Time feeds.

### 4.1 Data Vendor Preflight Audit Checklist

| Partner / Vendor Candidate | Target Historical Coverage | Required Publication Timestamp | Required Downstream Delivery Timestamp | Required Raw Retention Rights | Licensing Terms Needed | Target Verification Deliverable |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TEJ (Taiwan Economic Journal)** | 2024 Q1 (100% of 2330, 2603, 2881 disclosures) | `filing_datetime` exact to second | Millisecond Vendor Ingestion / Push Timestamp (`transfer_time`) | Right to retain and hash raw TCP/API response bytes | Academic Research & Laboratory License | Signed technical schema document & sample dropcopy audit log |
| **Securities & Futures Institute (SFI)** | 2024 Q1 Historical Disclosure Archive | Official exchange filing timestamp | Downstream release timestamp (if archived) | Right to store raw CSV/XML exports for audit reproducibility | Academic Data Grant Agreement | Dataset column specification confirming exact second timestamps |
| **Bloomberg / LSEG** | Enterprise Historical News Feed (B-PIPE / DataScope) | Millisecond wire publication time | Microsecond gateway ingest timestamp | Enterprise research license permitting persistent hash storage | Commercial Terminal / Feed Agreement | Sample PCAP or feed dropcopy with cryptographic verification |

### 4.2 Minimum Contractual Requirements for Formal PIT Transition
Before Track B can transition from **NO-GO** to **GO**, an external data provider must deliver:
1. **Unambiguous Timestamp Column Separation**: Clear distinction between `company_filing_time` (exchange submission) and `feed_publish_time` (vendor broadcast).
2. **Immutable Raw Payload Exports**: Provision of verbatim raw payloads allowing SHA-256 hash recomputation.
3. **Full Revision History**: Complete audit trail linking correction announcements to original filings with timestamps for both events.

---

## 5. Experimental Protocol & Governance Guardrails

To prevent premature claims or non-compliant actions during subsequent research phases:

1. **No Live Trading**: All experiments are conducted offline on historical bars. The software environment has no network connectivity to brokerage execution APIs.
2. **Strict Branch Quarantine**: All commits, documentation, and exploratory code must remain on `fix/v8-audit-patch`. Merging to `main` is strictly prohibited until formal empirical peer review and external data acquisition are completed.
3. **No Unsanctioned Scraping**: Automated requests to MOPS or other anti-bot-protected sites remain strictly disabled.
4. **Execution Decision Gate**: Before any RL training script is executed in future phases, the experimenter must verify:
   - `pit_mode == PITExecutionMode.ESTIMATED_REPLAY`
   - Explicit disclaimer printed in run logs: *"EXPLORATORY_PROTOTYPE_ONLY: Unverified pilot news fixtures in use; results cannot certify production alpha."*

---

## 6. Summary of Audit Metrics & Deliverables

- **Verified Historical Disclosures Count**: **0**
- **Verified Provider Availability Receipts Count**: **0**
- **Formal Point-in-Time Status**: **NO-GO**
- **2024 Q1 Historical Dataset Status**: **NO-GO**
- **Track A Status**: **DESIGN_FROZEN_READY_FOR_OFFLINE_EXPLORATION**
- **Track B Status**: **PENDING_EXTERNAL_PROVIDER_CONFIRMATION**
- **Primary Deliverables**:
  - [v8_16_historical_source_access_decision.md](file:///home/augustine/Desktop/News-Aware%20Attention-Based%20Deep%20Reinforcement%20Learning%20for%20Stock%20Trading/docs/reports/v8_16_historical_source_access_decision.md) (Updated with Section 1.3 Claim Boundary)
  - [v8_17_exploratory_replay_research_design.md](file:///home/augustine/Desktop/News-Aware%20Attention-Based%20Deep%20Reinforcement%20Learning%20for%20Stock%20Trading/docs/reports/v8_17_exploratory_replay_research_design.md) (Complete Research Design & Feasibility Matrix)

---
*Report certified under V8.17 Research Governance Baseline. Engineering Baseline Frozen. Formal PIT = NO-GO.*
