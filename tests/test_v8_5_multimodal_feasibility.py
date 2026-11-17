"""
V8.5 Historical News & Multimodal Dataset Feasibility Test Suite
Validates:
1. Deterministic Stable Event ID generation and cross-platform reproducibility.
2. PIT Timezone Trap Detection (UTC vs Asia/Taipei 08:00:00 post-market news artifact).
3. Revision Chain Resolution (Parent/Child linkage, As-Of version selection).
4. Dual-Slot Allocation (Recent 32 vs Active 8, at most 1 slot per event entity).
5. Counterfactual Zero Leakage (future news & late revisions have 0 effect on past decision snapshots).
6. Four-Layer Coverage Accounting (Strict separation of Schema, Tensor, Slot Occupancy, and Observability).
7. Formal Mode Fail-Closed Enforcement.
"""

import copy
import hashlib
import numpy as np
import pandas as pd
import pytest

from stock_rl.config.schema import (
    RECENT_EVENT_SLOTS,
    ACTIVE_EVENT_SLOTS,
    TOTAL_EVENT_SLOTS,
)
from stock_rl.data.pit_engine import (
    ProvenanceRecord,
    MissingStatus,
)
from stock_rl.data.multimodal_pipeline import (
    NewsSourceType,
    EventLifecycleStatus,
    RawNewsArticle,
    normalize_news_headline,
    compute_stable_event_id,
    detect_timezone_trap,
    HistoricalMultimodalPipeline,
    MultimodalCoverageAccountant,
)


# ==============================================================================
# 1. Deterministic Stable Event ID Generation & Normalization
# ==============================================================================

def test_stable_event_id_determinism_and_normalization():
    """
    Verifies that compute_stable_event_id generates deterministic SHA-256 IDs,
    normalizes headline media suffixes, and handles timezone conversions to UTC.
    """
    ts1 = pd.Timestamp("2023-11-06 16:00:00", tz="Asia/Taipei")
    ts2 = pd.Timestamp("2023-11-06 08:00:00", tz="UTC")  # Exactly equivalent to ts1

    h1 = "【台股盤中】台積電及AI族群強彈 台股中場漲逾300點 - Yahoo奇摩股市"
    h2 = "台積電及ai族群強彈 台股中場漲逾300點 - 聯合新聞網"
    h3 = "台積電及ai族群強彈 台股中場漲逾300點"

    # All three headlines normalize to identical canonical string
    assert normalize_news_headline(h1) == normalize_news_headline(h2)
    assert normalize_news_headline(h1) == normalize_news_headline(h3)

    id1 = compute_stable_event_id("Yahoo奇摩股市", "2330", ts1, h1)
    id2 = compute_stable_event_id("Yahoo奇摩股市", "2330", ts2, h2)

    assert id1 == id2, f"Event IDs must match across equivalent timezones and syndicated sources: {id1} vs {id2}"
    assert id1.startswith("ev_")
    assert len(id1) == 16


# ==============================================================================
# 2. Timezone Trap & PIT Anomaly Detection
# ==============================================================================

def test_timezone_trap_detection():
    """
    Audits the fatal PIT lookahead bug discovered during real data investigations:
    After-market foreign buy/sell news erroneously stamped at 08:00:00 Asia/Taipei
    (due to 08:00:00 UTC being naively treated as local morning time).
    """
    # Case 1: Trapped article (after-market news stamped at 08:00:00 Taipei)
    ts_trapped = pd.Timestamp("2023-11-06 08:00:00", tz="Asia/Taipei")
    headline_trapped = "《台北股市》外資06日買超股：台積電、玉山金、台新金 - Yahoo奇摩股市"
    is_trapped, reason = detect_timezone_trap(ts_trapped, headline_trapped, "Yahoo奇摩股市")
    assert is_trapped is True
    assert "FATAL TIMEZONE TRAP" in reason

    # Case 2: Legitimate morning news (macro commentary, US market overnight summary)
    ts_morning = pd.Timestamp("2023-11-06 07:30:00", tz="Asia/Taipei")
    headline_morning = "美股道瓊大漲逾200點 台積電ADR上揚1.5%"
    is_trapped_m, reason_m = detect_timezone_trap(ts_morning, headline_morning, "經濟日報")
    assert is_trapped_m is False

    # Case 3: Legitimate post-market news with verified post-market timestamp
    ts_afternoon = pd.Timestamp("2023-11-06 17:30:00", tz="Asia/Taipei")
    is_trapped_a, _ = detect_timezone_trap(ts_afternoon, headline_trapped, "Yahoo奇摩股市")
    assert is_trapped_a is False


# ==============================================================================
# 3. Revision Chain & Clarification As-Of Resolution
# ==============================================================================

def test_revision_chain_and_asof_resolution():
    """
    Verifies that:
    - At T1, snapshot uses initial filing (revision 1).
    - At T2, snapshot uses regulatory clarification (revision 2) with updated content.
    - Future revisions at T3 cannot leak into T2.
    """
    pipe = HistoricalMultimodalPipeline(mode="estimated")

    t1 = pd.Timestamp("2023-11-10 10:00:00", tz="Asia/Taipei")
    t2 = pd.Timestamp("2023-11-10 15:30:00", tz="Asia/Taipei")
    t3 = pd.Timestamp("2023-11-11 11:00:00", tz="Asia/Taipei")

    # Initial announcement
    art1 = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="媒體報導台積電先進封裝產能預測",
        body_text="報導稱產能將擴充 50%",
        source_published_at=t1,
        provider_available_at=t1,
        revision_id=1,
        lifecycle_status=EventLifecycleStatus.PUBLISHED,
        is_active_candidate=True,
        is_audit_verified=False,
    )
    group_id = pipe.ingest_article(art1)

    # Clarification announcement (MOPS Rule 51)
    art2 = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="說明媒體報導先進封裝產能",
        body_text="澄清媒體報導係自行推估，公司未提供財務或產能預測",
        source_published_at=t2,
        provider_available_at=t2,
        revision_id=2,
        parent_event_id=group_id,
        lifecycle_status=EventLifecycleStatus.CLARIFIED,
        is_active_candidate=True,
        is_audit_verified=False,
    )
    pipe.ingest_article(art2)

    # Subsequent resolution (Event resolved and closed)
    art3 = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="先進封裝擴產案主管機關備查完成",
        body_text="備查程序完結",
        source_published_at=t3,
        provider_available_at=t3,
        revision_id=3,
        parent_event_id=group_id,
        lifecycle_status=EventLifecycleStatus.RESOLVED,
        is_active_candidate=False,
        is_audit_verified=False,
    )
    pipe.ingest_article(art3)

    # As-Of Snapshot at T1 + 1 hour (11:00:00): must see Revision 1
    recent_t1, active_t1 = pipe.resolve_asof_events("2330", pd.Timestamp("2023-11-10 11:00:00", tz="Asia/Taipei"))
    assert len(active_t1) == 1
    assert active_t1[0]["revision_id"] == 1
    assert active_t1[0]["lifecycle_status"] == EventLifecycleStatus.PUBLISHED.value
    assert "產能將擴充 50%" in active_t1[0]["body"]

    # As-Of Snapshot at T2 + 1 hour (16:30:00): must see Revision 2 (Clarification)
    recent_t2, active_t2 = pipe.resolve_asof_events("2330", pd.Timestamp("2023-11-10 16:30:00", tz="Asia/Taipei"))
    assert len(active_t2) == 1
    assert active_t2[0]["revision_id"] == 2
    assert active_t2[0]["lifecycle_status"] == EventLifecycleStatus.CLARIFIED.value
    assert "澄清媒體報導" in active_t2[0]["body"]
    assert len(recent_t2) == 0  # Deduplicated: exactly 1 slot across active and recent!

    # As-Of Snapshot at T3 + 1 hour (12:00:00): Event resolved, transitions out of active pool
    recent_t3, active_t3 = pipe.resolve_asof_events("2330", pd.Timestamp("2023-11-11 12:00:00", tz="Asia/Taipei"))
    assert len(active_t3) == 0  # No longer active
    assert len(recent_t3) == 1  # Transitioned to recent historical pool
    assert recent_t3[0]["revision_id"] == 3


# ==============================================================================
# 4. Counterfactual Zero Leakage Test
# ==============================================================================

def test_counterfactual_future_news_zero_leakage():
    """
    Counterfactual Test:
    Adding new events or modifying articles at T+1 must yield identical
    event representations at decision cutoff T.
    """
    pipe = HistoricalMultimodalPipeline(mode="estimated")

    t_dec = pd.Timestamp("2023-11-10 20:00:00", tz="Asia/Taipei")

    # Ingest past news available before cutoff
    past_art = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="中央社",
        stock_id="2330",
        headline="台積電法說會前展望樂觀",
        source_published_at=pd.Timestamp("2023-11-10 14:00:00", tz="Asia/Taipei"),
        provider_available_at=pd.Timestamp("2023-11-10 14:15:00", tz="Asia/Taipei"),
        revision_id=1,
    )
    pipe.ingest_article(past_art)

    # Capture snapshot before injecting future news
    recent_before, active_before = pipe.resolve_asof_events("2330", t_dec)

    # Inject multiple future news items at T+1 (e.g. 2023-11-11)
    future_art1 = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="經濟日報",
        stock_id="2330",
        headline="未來新聞：台積電宣布最新資本支出",
        source_published_at=pd.Timestamp("2023-11-11 09:30:00", tz="Asia/Taipei"),
        provider_available_at=pd.Timestamp("2023-11-11 09:40:00", tz="Asia/Taipei"),
        revision_id=1,
    )
    future_art2 = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="未來重訊：台積電董事會重大決議",
        source_published_at=pd.Timestamp("2023-11-11 17:00:00", tz="Asia/Taipei"),
        provider_available_at=pd.Timestamp("2023-11-11 17:00:00", tz="Asia/Taipei"),
        revision_id=1,
        is_active_candidate=True,
    )
    pipe.ingest_article(future_art1)
    pipe.ingest_article(future_art2)

    # Capture snapshot after injecting future news
    recent_after, active_after = pipe.resolve_asof_events("2330", t_dec)

    # Verify zero leakage: exactly identical
    assert len(recent_before) == len(recent_after)
    assert len(active_before) == len(active_after)
    assert recent_before == recent_after
    assert active_before == active_after


# ==============================================================================
# 5. Formal Mode Fail-Closed Enforcement
# ==============================================================================

def test_formal_mode_fail_closed_enforcement():
    """
    Verifies that HistoricalMultimodalPipeline in formal mode strictly rejects:
    - Articles with is_audit_verified=False.
    - Articles triggering timezone traps.
    """
    pipe_formal = HistoricalMultimodalPipeline(mode="formal")

    # 1. Unverified article rejected
    art_unverified = RawNewsArticle(
        source_type=NewsSourceType.FINMIND_RSS,
        source_name="Yahoo奇摩股市",
        stock_id="2330",
        headline="半導體市況展望良好",
        source_published_at=pd.Timestamp("2023-11-06 14:00:00", tz="Asia/Taipei"),
        is_audit_verified=False,
    )
    with pytest.raises(ValueError, match="Cannot ingest unverified article"):
        pipe_formal.ingest_article(art_unverified)

    # 2. Timezone trapped article rejected
    art_trapped = RawNewsArticle(
        source_type=NewsSourceType.FINMIND_RSS,
        source_name="Yahoo奇摩股市",
        stock_id="2330",
        headline="《台北股市》外資06日買超股：台積電、玉山金",
        source_published_at=pd.Timestamp("2023-11-06 08:00:00", tz="Asia/Taipei"),
        is_audit_verified=True,  # Even if mistakenly marked verified, timezone trap must catch it!
    )
    with pytest.raises(ValueError, match="Strict PIT Rejection: FATAL TIMEZONE TRAP"):
        pipe_formal.ingest_article(art_trapped)


# ==============================================================================
# 6. Four-Layer Coverage Accounting Audit
# ==============================================================================

def test_four_layer_coverage_accounting():
    """
    Verifies MultimodalCoverageAccountant:
    Strictly separates 100% tensor validity from 0% event slot occupancy
    and granular PIT observability percentages.
    """
    report = MultimodalCoverageAccountant.compute_accounting_report(
        total_market_cells=487200,
        valid_market_cells=487200,
        imputed_market_cells=0,
        occupied_event_slots=0,
        total_event_slots=6720,
        verified_pit_cells=672,
        estimated_cells=464520,
        placeholder_cells=1848,
    )

    # Layer 1
    assert report["layer_1_schema_contract_validity"] == 100.0

    # Layer 2: Tensor validity is 100%, imputation is 0%
    l2 = report["layer_2_numerical_tensor_validity"]
    assert l2["validity_pct"] == 100.0
    assert l2["imputation_rate_pct"] == 0.0

    # Layer 3: Slot occupancy is 0% (NOT 100%)
    l3 = report["layer_3_event_slot_occupancy"]
    assert l3["occupied_slots"] == 0
    assert l3["occupancy_pct"] == 0.0

    # Layer 4: Granular breakdown
    l4 = report["layer_4_pit_observability"]
    assert l4["verified_pit_pct"] > 0.1 and l4["verified_pit_pct"] < 0.2
    assert l4["estimated_pct"] > 95.0
    assert l4["placeholder_pct"] < 1.0

    # Anti-fraud statement
    assert "ZERO-LEAKAGE MANDATE" in report["anti_fraud_declaration"]
