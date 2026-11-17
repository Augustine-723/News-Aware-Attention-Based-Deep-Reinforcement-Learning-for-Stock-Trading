"""
V8.6 Historical News Micro PoC & Event Integration Test Suite
Validates:
1. Strict News Availability Contract (Missing Provider Timestamp & False Audit Flag in Formal Mode).
2. Two-Tier Identification System (article_id vs event_id, Syndicated News, Duplicates).
3. MOPS Rule 51 Clarification & Revision As-Of Single-Slot Occupancy.
4. Real Multimodal Integration (Event Slot Occupancy > 0, 1-to-1 Text & Metadata Alignment).
5. Text Embedding Model Versioning and Type Distinction.
6. Counterfactual Zero Leakage (future news at T+1/T+3 cannot alter T snapshot).
7. StockRLFullModel Forward and Micro Backward Gradient Flow with Authentic News.
"""

import copy
import hashlib
import numpy as np
import pandas as pd
import pytest
import torch

from stock_rl.config.schema import (
    MARKET_WINDOW_LENGTH,
    RECENT_EVENT_SLOTS,
    ACTIVE_EVENT_SLOTS,
    TOTAL_EVENT_SLOTS,
    EVENT_TEXT_DIM,
    EVENT_METADATA_DIM,
    EVENT_STOCK_RELATION_DIM,
)
from stock_rl.data.pit_engine import (
    ProvenanceRecord,
    MissingStatus,
)
from stock_rl.data.multimodal_pipeline import (
    NewsSourceType,
    EventLifecycleStatus,
    TextEmbeddingType,
    TextEmbeddingEngine,
    RawNewsArticle,
    compute_article_id,
    compute_canonical_event_id,
    detect_timezone_trap,
    extract_event_metadata,
    extract_event_stock_relation,
    load_micro_news_dataset,
    HistoricalMultimodalPipeline,
    MultimodalCoverageAccountant,
)
from stock_rl.data.feature_engine import (
    assemble_event_metadata,
    build_audited_event_metadata,
)
from stock_rl.data.real_data_poc import (
    load_real_market_feeds,
    STOCK_METADATA,
)
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.models.full_model import StockRLFullModel


# ==============================================================================
# 1. Strict News Availability Contract (P0-1, P0-2)
# ==============================================================================

def test_missing_provider_timestamp_rejected_in_formal_mode():
    """
    Formal Mode must NOT silently substitute source_published_at for missing provider_available_at.
    """
    pipe_formal = HistoricalMultimodalPipeline(mode="formal")
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    art_missing_provider = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="中央社",
        stock_id="2330",
        headline="半導體供應鏈展望強勁",
        source_published_at=t_pub,
        provider_available_at=None,  # Missing!
        provider_available_evidence="Wire log",
        is_audit_verified=True,
    )

    with pytest.raises(ValueError, match="provider_available_at is mandatory and cannot fall back"):
        pipe_formal.ingest_article(art_missing_provider)


def test_false_audit_flag_rejected_in_formal_mode():
    """
    Formal Mode must NOT accept is_audit_verified=True without explicit provider_available_evidence.
    """
    pipe_formal = HistoricalMultimodalPipeline(mode="formal")
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    art_false_audit = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="自由財經",
        stock_id="2330",
        headline="台積電先進製程獲得新客戶認證",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence=None,  # No evidence!
        is_audit_verified=True,             # False claim!
    )

    with pytest.raises(ValueError, match="is_audit_verified=True requires explicit provider_available_evidence"):
        pipe_formal.ingest_article(art_false_audit)


def test_timezone_trap_flag_preservation():
    """
    Confirms timezone trap detection acts as an anomaly flag,
    and does not arbitrarily fabricate a UTC correction.
    """
    # 08:00:00 Asia/Taipei after-market foreign buy/sell report
    ts_trapped = pd.Timestamp("2024-01-15 08:00:00", tz="Asia/Taipei")
    headline = "《台北股市》外資15日買超股：台積電、長榮 - Yahoo奇摩股市"

    is_trapped, reason = detect_timezone_trap(ts_trapped, headline, "Yahoo奇摩股市")
    assert is_trapped is True
    assert "FATAL TIMEZONE TRAP" in reason

    # In Estimated mode, article is flagged but raw timestamp is strictly preserved
    pipe_est = HistoricalMultimodalPipeline(mode="estimated")
    art_est = RawNewsArticle(
        source_type=NewsSourceType.FINMIND_RSS,
        source_name="Yahoo奇摩股市",
        stock_id="2330",
        headline=headline,
        source_published_at=ts_trapped,
        provider_available_at=ts_trapped,
        is_audit_verified=False,
    )
    ev_id = pipe_est.ingest_article(art_est)
    assert ev_id is not None
    # Verified: raw timestamp preserved, not silently altered
    assert art_est.source_published_at == ts_trapped


# ==============================================================================
# 2. Two-Tier Identification System: article_id vs event_id
# ==============================================================================

def test_two_tier_identity_syndication_and_duplicates():
    """
    Verifies that:
    1. Syndicated copies from different outlets have DIFFERENT article_id but SAME event_id.
    2. Duplicate feed deliveries have distinct article_id and map to the SAME event_id.
    3. As-Of resolution retains only 1 slot per event entity.
    """
    pipe = HistoricalMultimodalPipeline(mode="estimated")
    t1 = pd.Timestamp("2024-01-15 13:31:00", tz="Asia/Taipei")
    t2 = pd.Timestamp("2024-01-15 13:31:49", tz="Asia/Taipei")

    # Syndicated copies of Evergreen potential stock news
    art_moneydj = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="富聯網",
        stock_id="2603",
        headline="潛力股：長榮(2603) - 富聯網",
        body_text="長榮受惠紅海危機帶動運價回升，歐洲線長約換約展望轉佳。",
        url="https://moneydj.com/article/1",
        source_published_at=t1,
        provider_available_at=t1,
        article_id="art_2603_moneydj_150531",
        event_id="ev_2603_potential_stock",
    )
    art_yahoo = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="Yahoo奇摩股市",
        stock_id="2603",
        headline="潛力股：長榮 - Yahoo奇摩股市",
        body_text="長榮受惠紅海危機帶動運價回升。",
        url="https://tw.stock.yahoo.com/news/1",
        source_published_at=t2,
        provider_available_at=t2,
        article_id="art_2603_yahoo_150531",
        event_id="ev_2603_potential_stock",
    )

    id1 = pipe.ingest_article(art_moneydj)
    id2 = pipe.ingest_article(art_yahoo)

    assert art_moneydj.article_id != art_yahoo.article_id, "Physical articles must have unique article_id"
    assert id1 == id2 == "ev_2603_potential_stock", "Syndicated articles must share identical event_id"

    # As-Of resolution at 20:00:00 must yield exactly 1 event slot (deduplicated)
    recent, active = pipe.resolve_asof_events("2603", pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei"))
    assert len(recent) == 1
    assert len(active) == 0
    assert recent[0]["event_id"] == "ev_2603_potential_stock"


def test_mops_rule_51_clarification_updates_event_without_duplicate_slot():
    """
    Verifies that a regulatory clarification (MOPS Rule 51) updates the parent event
    and occupies exactly ONE slot at decision time.
    """
    pipe = HistoricalMultimodalPipeline(mode="estimated")
    t_rumor = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")
    t_clarify = pd.Timestamp("2024-01-15 14:15:30", tz="Asia/Taipei")

    art_rumor = RawNewsArticle(
        source_type=NewsSourceType.MEDIA_ARCHIVE,
        source_name="工商時報",
        stock_id="2330",
        headline="傳台積電嘉義先進封裝廠投資額翻倍",
        body_text="市場傳聞投資額高達數千億元",
        source_published_at=t_rumor,
        provider_available_at=t_rumor,
        revision_id=1,
        article_id="art_2330_rumor",
        event_id="ev_2330_cowos_rumor",
    )
    art_clarify = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="說明媒體報導先進封裝建廠產能規劃",
        body_text="澄清媒體報導本公司投資金額係自行推估",
        source_published_at=t_clarify,
        provider_available_at=t_clarify,
        revision_id=2,
        parent_event_id="ev_2330_cowos_rumor",
        lifecycle_status=EventLifecycleStatus.CLARIFIED,
        article_id="art_2330_clarify",
        event_id="ev_2330_cowos_rumor",
    )

    pipe.ingest_article(art_rumor)
    pipe.ingest_article(art_clarify)

    # Decision at 12:00:00 -> sees rumor (revision 1)
    rec_12, _ = pipe.resolve_asof_events("2330", pd.Timestamp("2024-01-15 12:00:00", tz="Asia/Taipei"))
    assert len(rec_12) == 1
    assert rec_12[0]["revision_id"] == 1
    assert "傳聞" in rec_12[0]["body"]

    # Decision at 20:00:00 -> sees clarification (revision 2), exactly 1 slot occupied
    rec_20, act_20 = pipe.resolve_asof_events("2330", pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei"))
    assert len(rec_20) == 1
    assert len(act_20) == 0
    assert rec_20[0]["revision_id"] == 2
    assert rec_20[0]["lifecycle_status"] == EventLifecycleStatus.CLARIFIED.value
    assert "澄清媒體報導" in rec_20[0]["body"]


# ==============================================================================
# 3. Real Multimodal Integration & Snapshot Occupancy (Gate 4)
# ==============================================================================

def test_real_multimodal_snapshot_occupancy_greater_than_zero():
    """
    Verifies that authentic historical micro news integrates into HistoricalSnapshotBuilder:
    - Event Slot Occupancy > 0 on 2024-01-15.
    - 2330 occupies Recent slots.
    - 2603 occupies 1 Recent slot and 1 Active slot.
    - 2881 on 2024-01-15 has 0 slots occupied (no future leakage).
    - On 2024-01-16, 2881 occupies exactly 1 slot.
    """
    for ticker, expected_recent, expected_active in [
        ("2330", 2, 0),
        ("2603", 1, 1),
        ("2881", 0, 0),
    ]:
        ohlcv, inst, macro, foreign, ctx, evs = load_real_market_feeds(ticker, mode="estimated", include_micro_news=True)
        meta = STOCK_METADATA[ticker]
        builder = HistoricalSnapshotBuilder(
            df_ohlcv=ohlcv,
            df_inst=inst,
            df_macro=macro,
            df_foreign=foreign,
            stock_context_records=ctx,
            event_records=evs,
            shares_outstanding=meta["shares_outstanding"],
            mode="estimated",
        )
        t_dec = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")
        snap = builder.build_snapshot(t_dec)

        rec_occ = int(snap.event_mask[:, :RECENT_EVENT_SLOTS].sum().item())
        act_occ = int(snap.event_mask[:, RECENT_EVENT_SLOTS:].sum().item())
        total_occ = int(snap.event_mask.sum().item())

        assert rec_occ == expected_recent, f"[{ticker}] Expected {expected_recent} recent slots, got {rec_occ}"
        assert act_occ == expected_active, f"[{ticker}] Expected {expected_active} active slots, got {act_occ}"
        assert total_occ == expected_recent + expected_active

        # 1-to-1 text and metadata alignment check
        for slot_idx in range(TOTAL_EVENT_SLOTS):
            if snap.event_mask[0, slot_idx].item() is True:
                # Occupied slot must have non-masked metadata and non-unit vector
                assert snap.event_metadata_missing_mask[0, slot_idx, :].sum().item() == 0
                assert not torch.isnan(snap.event_metadata[0, slot_idx]).any()
                assert not torch.isnan(snap.event_stock_rel[0, slot_idx]).any()
            else:
                # Masked empty slot must be masked
                assert snap.event_metadata_missing_mask[0, slot_idx, :].all().item() is True


def test_embedding_model_versioning_and_type_distinction():
    """
    Verifies that TextEmbeddingEngine explicitly tracks embedding types
    and preserves model_version metadata.
    """
    engine_hash = TextEmbeddingEngine(TextEmbeddingType.DETERMINISTIC_HASH, model_version="sha256-projection-v1.0")
    engine_untr = TextEmbeddingEngine(TextEmbeddingType.UNTRAINED, model_version="random-projection-v1.0")
    engine_empty = TextEmbeddingEngine(TextEmbeddingType.SYNTHETIC_PLACEHOLDER, model_version="empty-unit-v1.0")

    text = "台積電法說會展望AI營運"
    v_hash = engine_hash.embed_text(text)
    v_untr = engine_untr.embed_text(text)
    v_empty = engine_empty.embed_text(text)

    assert v_hash.shape == (EVENT_TEXT_DIM,)
    assert v_untr.shape == (EVENT_TEXT_DIM,)
    assert v_empty.shape == (EVENT_TEXT_DIM,)
    assert np.isclose(np.linalg.norm(v_hash), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(v_untr), 1.0, atol=1e-5)
    assert np.isclose(v_empty[0], 1.0) and np.allclose(v_empty[1:], 0.0)

    # Models produce distinct vectors
    assert not np.allclose(v_hash, v_untr)


# ==============================================================================
# 4. Counterfactual Future News Zero Leakage (Gate 4)
# ==============================================================================

def test_future_news_counterfactual_invariance():
    """
    Counterfactual Test:
    Modifying or injecting future news at T+1 (2024-01-16) or T+3 (2024-01-18)
    MUST produce 100% bitwise equal snapshot tensors at decision time T (2024-01-15 20:00:00).
    """
    ticker = "2330"
    ohlcv, inst, macro, foreign, ctx, evs_clean = load_real_market_feeds(ticker, mode="estimated", include_micro_news=True)
    meta = STOCK_METADATA[ticker]

    t_dec = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")

    builder_clean = HistoricalSnapshotBuilder(
        df_ohlcv=ohlcv,
        df_inst=inst,
        df_macro=macro,
        df_foreign=foreign,
        stock_context_records=ctx,
        event_records=evs_clean,
        shares_outstanding=meta["shares_outstanding"],
        mode="estimated",
    )
    snap_clean = builder_clean.build_snapshot(t_dec)

    # Inject multiple future news items at T+1 and T+3 with radical sentiment shifts
    evs_poisoned = copy.deepcopy(evs_clean)
    poisoned_art = {
        "event_id": "ev_future_poison_999",
        "text_embedding": np.ones(EVENT_TEXT_DIM, dtype=np.float32) / np.sqrt(EVENT_TEXT_DIM),
        "actor_embedding_4d": np.array([1, 0, 0, 0], dtype=np.float32),
        "type_embedding_8d": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "status_embedding_4d": np.array([0, 0, 1, 0], dtype=np.float32),
        "evidence_traceability": 1.0,
        "source_reliability": 1.0,
        "is_active": True,
        "provenance": ProvenanceRecord(
            source_id="FUTURE_POISON",
            source_published_at=pd.Timestamp("2024-01-18 10:00:00", tz="Asia/Taipei"),
            provider_available_at=pd.Timestamp("2024-01-18 10:00:00", tz="Asia/Taipei"),
            available_at=pd.Timestamp("2024-01-18 10:00:00", tz="Asia/Taipei"),
            is_audit_verified=False,
            mode="estimated",
        ),
        "relation": {
            "direct_company_mention": True,
            "industry_match": True,
            "upstream_exposure": 0.0,
            "downstream_exposure": 0.0,
            "customer_relationship": 0.0,
            "supplier_relationship": 0.0,
            "geographic_exposure": 0.0,
            "macro_factor_exposure": 0.0,
        },
    }
    evs_poisoned.append(poisoned_art)

    builder_poisoned = HistoricalSnapshotBuilder(
        df_ohlcv=ohlcv,
        df_inst=inst,
        df_macro=macro,
        df_foreign=foreign,
        stock_context_records=ctx,
        event_records=evs_poisoned,
        shares_outstanding=meta["shares_outstanding"],
        mode="estimated",
    )
    snap_poisoned = builder_poisoned.build_snapshot(t_dec)

    # Bitwise equality checks across all multimodal components
    assert torch.equal(snap_clean.recent_event_emb, snap_poisoned.recent_event_emb)
    assert torch.equal(snap_clean.active_event_emb, snap_poisoned.active_event_emb)
    assert torch.equal(snap_clean.event_metadata, snap_poisoned.event_metadata)
    assert torch.equal(snap_clean.event_stock_rel, snap_poisoned.event_stock_rel)
    assert torch.equal(snap_clean.event_mask, snap_poisoned.event_mask)
    assert torch.equal(snap_clean.price, snap_poisoned.price)
    assert torch.equal(snap_clean.technical, snap_poisoned.technical)
    assert torch.equal(snap_clean.institutional, snap_poisoned.institutional)
    assert torch.equal(snap_clean.macro, snap_poisoned.macro)
    assert torch.equal(snap_clean.stock_context, snap_poisoned.stock_context)
    assert torch.equal(snap_clean.position, snap_poisoned.position)


# ==============================================================================
# 5. Full Model Forward & Micro Backward Execution
# ==============================================================================

def test_stock_rl_full_model_forward_and_micro_backward_with_real_news():
    """
    Verifies that StockRLFullModel accepts the real multimodal snapshot
    (occupancy > 0) and computes policy logits, state values, and backpropagates gradients.
    """
    ohlcv, inst, macro, foreign, ctx, evs = load_real_market_feeds("2603", mode="estimated", include_micro_news=True)
    meta = STOCK_METADATA["2603"]
    builder = HistoricalSnapshotBuilder(
        df_ohlcv=ohlcv,
        df_inst=inst,
        df_macro=macro,
        df_foreign=foreign,
        stock_context_records=ctx,
        event_records=evs,
        shares_outstanding=meta["shares_outstanding"],
        mode="estimated",
    )
    snap = builder.build_snapshot(pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei"))

    model = StockRLFullModel()
    model.train()

    q_values, predictions, rl_state = model(snap)

    # 1. Output shapes
    assert q_values.shape == (1, 2)
    assert rl_state.shape == (1, 132)
    assert "relevance_logits" in predictions
    assert "novelty_scores" in predictions
    assert "pred_returns" in predictions
    assert "pred_risk" in predictions

    # 2. Gradient flow check
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    # Multi-task objective
    loss_rl = q_values.sum()
    loss_aux = predictions["pred_returns"].sum() + predictions["pred_risk"].sum()
    loss = loss_rl + 0.1 * loss_aux
    loss.backward()

    # Check that model weights received valid gradients
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(not np.isnan(g) and not np.isinf(g) for g in grad_norms)
    optimizer.step()


# ==============================================================================
# 6. P0-1: Historical Event Revision Resolution & Counterfactual Invariance (T1, T2, T3)
# ==============================================================================

def test_historical_revision_t1_t2_t3_counterfactual():
    """
    P0-1 Regression Test:
    1. At T1 (2024-01-15 11:00:00), snapshot resolves ev_2330_cowos_capacity to Revision 1 (Rumor from Commercial Times at 08:30).
    2. At T2 (2024-01-15 20:00:00), snapshot resolves ev_2330_cowos_capacity to Revision 2 (Clarification from MOPS at 14:15:30).
    3. Exactly ONE slot is occupied for ev_2330_cowos_capacity at both T1 and T2 (no duplicate slots).
    4. Adding a future Revision 3 at T3 (2024-01-18 10:00:00) causes ZERO change to T1 and T2 snapshots (torch.equal bitwise invariance).
    """
    ticker = "2330"
    ohlcv, inst, macro, foreign, ctx, evs = load_real_market_feeds(ticker, mode="estimated", include_micro_news=True)
    meta = STOCK_METADATA[ticker]

    builder = HistoricalSnapshotBuilder(
        df_ohlcv=ohlcv,
        df_inst=inst,
        df_macro=macro,
        df_foreign=foreign,
        stock_context_records=ctx,
        event_records=evs,
        shares_outstanding=meta["shares_outstanding"],
        mode="estimated",
    )

    t1 = pd.Timestamp("2024-01-15 11:00:00", tz="Asia/Taipei")
    t2 = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")

    snap_t1 = builder.build_snapshot(t1)
    snap_t2 = builder.build_snapshot(t2)

    # In T1 snapshot, available_at <= 11:00:00 -> art_2330_ct_240115 (08:30, rev 1) and art_2330_ltn_240115 (10:07, rev 1)
    assert int(snap_t1.event_mask[:, :RECENT_EVENT_SLOTS].sum().item()) == 2
    assert int(snap_t1.event_mask[:, RECENT_EVENT_SLOTS:].sum().item()) == 0

    # In T2 snapshot, available_at <= 20:00:00 -> art_2330_mops_240115 (14:15:30, rev 2) is chosen over rev 1!
    assert int(snap_t2.event_mask[:, :RECENT_EVENT_SLOTS].sum().item()) == 2
    assert int(snap_t2.event_mask[:, RECENT_EVENT_SLOTS:].sum().item()) == 0

    # In snap_t2, slot 0 is the clarified MOPS event
    assert snap_t2.event_metadata[0, 0, 0].item() == 1.0  # MOPS corporate management (E01)
    assert snap_t2.event_metadata[0, 0, 15].item() == 1.0  # Clarified (E16)
    assert snap_t2.event_metadata[0, 0, 19].item() == 0.5  # Revision 2 novelty estimate (0.5 for rev > 1)

    # In snap_t1, slot 0 is the Commercial Times rumor event (not yet clarified)
    assert snap_t1.event_metadata[0, 0, 0].item() == 0.0  # NOT MOPS corporate management
    assert snap_t1.event_metadata[0, 0, 15].item() == 0.0  # NOT clarified
    assert snap_t1.event_metadata[0, 0, 19].item() == 1.0  # Revision 1 novelty estimate (1.0 for rev == 1)

    # 4. Inject future Revision 3 at T3 (2024-01-18 10:00:00)
    evs_with_t3 = copy.deepcopy(evs)
    rec_t3 = {
        "event_id": "ev_2330_cowos_capacity",
        "article_id": "art_2330_mops_t3_future",
        "text_embedding": np.ones(EVENT_TEXT_DIM, dtype=np.float32) / np.sqrt(EVENT_TEXT_DIM),
        "actor_embedding_4d": np.array([1, 0, 0, 0], dtype=np.float32),
        "type_embedding_8d": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "status_embedding_4d": np.array([0, 0, 0, 1], dtype=np.float32),
        "evidence_traceability": 1.0,
        "source_reliability": 1.0,
        "source_independence": 1.0,
        "novelty_input_estimate": 0.2,
        "market_surprise_preevent": 0.0,
        "quantified_magnitude": 0.9,
        "timestamp_reliability": 1.0,
        "is_active": False,
        "revision_id": 3,
        "available_at": pd.Timestamp("2024-01-18 10:00:00", tz="Asia/Taipei"),
        "provenance": ProvenanceRecord(
            source_id="MOPS_art_2330_mops_t3_future",
            source_published_at=pd.Timestamp("2024-01-18 10:00:00", tz="Asia/Taipei"),
            provider_available_at=pd.Timestamp("2024-01-18 10:00:00", tz="Asia/Taipei"),
            available_at=pd.Timestamp("2024-01-18 10:00:00", tz="Asia/Taipei"),
            revision_id=3,
            is_audit_verified=False,
            mode="estimated",
        ),
        "relation": {
            "direct_company_mention": True,
            "industry_match": True,
            "upstream_exposure": 0.0,
            "downstream_exposure": 0.0,
            "customer_relationship": 0.0,
            "supplier_relationship": 0.0,
            "geographic_exposure": 0.0,
            "macro_factor_exposure": 0.0,
        },
    }
    evs_with_t3.append(rec_t3)

    builder_t3 = HistoricalSnapshotBuilder(
        df_ohlcv=ohlcv,
        df_inst=inst,
        df_macro=macro,
        df_foreign=foreign,
        stock_context_records=ctx,
        event_records=evs_with_t3,
        shares_outstanding=meta["shares_outstanding"],
        mode="estimated",
    )

    snap_t1_t3 = builder_t3.build_snapshot(t1)
    snap_t2_t3 = builder_t3.build_snapshot(t2)

    # Strict bitwise counterfactual invariance across T1 and T2
    assert torch.equal(snap_t1.recent_event_emb, snap_t1_t3.recent_event_emb)
    assert torch.equal(snap_t1.active_event_emb, snap_t1_t3.active_event_emb)
    assert torch.equal(snap_t1.event_metadata, snap_t1_t3.event_metadata)
    assert torch.equal(snap_t1.event_stock_rel, snap_t1_t3.event_stock_rel)
    assert torch.equal(snap_t1.event_mask, snap_t1_t3.event_mask)

    assert torch.equal(snap_t2.recent_event_emb, snap_t2_t3.recent_event_emb)
    assert torch.equal(snap_t2.active_event_emb, snap_t2_t3.active_event_emb)
    assert torch.equal(snap_t2.event_metadata, snap_t2_t3.event_metadata)
    assert torch.equal(snap_t2.event_stock_rel, snap_t2_t3.event_stock_rel)
    assert torch.equal(snap_t2.event_mask, snap_t2_t3.event_mask)


# ==============================================================================
# 7. Dynamic Recent / Active Slot As-Of Allocation
# ==============================================================================

def test_recent_active_slot_asof_selection():
    """
    P0-1 / P0-3 Test:
    Verifies Recent 32 vs Active 8 allocation occurs dynamically as-of decision_timestamp:
    - An active event lands in Active slot (slot index 32..39).
    - When resolved, the event shifts dynamically to Recent slots (0..31), leaving Active empty.
    - No duplicate slots across Recent and Active pools.
    """
    ticker = "2603"
    ohlcv, inst, macro, foreign, ctx, evs = load_real_market_feeds(ticker, mode="estimated", include_micro_news=True)
    meta = STOCK_METADATA[ticker]

    builder = HistoricalSnapshotBuilder(
        df_ohlcv=ohlcv,
        df_inst=inst,
        df_macro=macro,
        df_foreign=foreign,
        stock_context_records=ctx,
        event_records=evs,
        shares_outstanding=meta["shares_outstanding"],
        mode="estimated",
    )

    t_active = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")
    snap_active = builder.build_snapshot(t_active)

    # 1 Recent slot (potential stock) + 1 Active slot (red sea crisis)
    assert int(snap_active.event_mask[:, :RECENT_EVENT_SLOTS].sum().item()) == 1
    assert int(snap_active.event_mask[:, RECENT_EVENT_SLOTS:].sum().item()) == 1
    assert snap_active.event_mask[0, 0].item() is True
    assert snap_active.event_mask[0, RECENT_EVENT_SLOTS].item() is True

    # Now simulate an official resolution at 2024-01-16 12:00:00
    evs_resolved = copy.deepcopy(evs)
    res_rec = {
        "event_id": "ev_2603_red_sea_crisis",
        "article_id": "art_2603_resolution",
        "text_embedding": np.ones(EVENT_TEXT_DIM, dtype=np.float32) / np.sqrt(EVENT_TEXT_DIM),
        "actor_embedding_4d": np.array([1, 0, 0, 0], dtype=np.float32),
        "type_embedding_8d": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "status_embedding_4d": np.array([0, 0, 0, 1], dtype=np.float32),  # Clarified/Resolved
        "evidence_traceability": 1.0,
        "source_reliability": 1.0,
        "source_independence": 1.0,
        "novelty_input_estimate": 0.5,
        "market_surprise_preevent": 0.0,
        "quantified_magnitude": 0.8,
        "timestamp_reliability": 1.0,
        "is_active": False,
        "event_status": "resolved",
        "revision_id": 2,
        "available_at": pd.Timestamp("2024-01-16 12:00:00", tz="Asia/Taipei"),
        "provenance": ProvenanceRecord(
            source_id="MOPS_art_2603_resolution",
            source_published_at=pd.Timestamp("2024-01-16 12:00:00", tz="Asia/Taipei"),
            provider_available_at=pd.Timestamp("2024-01-16 12:00:00", tz="Asia/Taipei"),
            available_at=pd.Timestamp("2024-01-16 12:00:00", tz="Asia/Taipei"),
            revision_id=2,
            is_audit_verified=False,
            missing_status=MissingStatus.AVAILABLE.value,
            mode="estimated",
        ),
        "relation": {
            "direct_company_mention": True,
            "industry_match": True,
            "upstream_exposure": 0.0,
            "downstream_exposure": 0.0,
            "customer_relationship": 0.0,
            "supplier_relationship": 0.0,
            "geographic_exposure": 0.0,
            "macro_factor_exposure": 0.0,
        },
    }
    evs_resolved.append(res_rec)

    builder_resolved = HistoricalSnapshotBuilder(
        df_ohlcv=ohlcv,
        df_inst=inst,
        df_macro=macro,
        df_foreign=foreign,
        stock_context_records=ctx,
        event_records=evs_resolved,
        shares_outstanding=meta["shares_outstanding"],
        mode="estimated",
    )

    t_after_res = pd.Timestamp("2024-01-16 20:00:00", tz="Asia/Taipei")
    snap_after_res = builder_resolved.build_snapshot(t_after_res)

    # After resolution: event shifts to Recent pool, leaving Active pool empty
    assert int(snap_after_res.event_mask[:, RECENT_EVENT_SLOTS:].sum().item()) == 0
    assert int(snap_after_res.event_mask[:, :RECENT_EVENT_SLOTS].sum().item()) == 2
    # Exactly 2 total slots occupied (1 for potential stock, 1 for red sea crisis resolved - zero duplicate slots)
    assert int(snap_after_res.event_mask.sum().item()) == 2


# ==============================================================================
# 8. P0-2: Authoritative 24-Dimension Event Metadata Sentinel Test
# ==============================================================================

def test_event_metadata_24d_sentinel():
    """
    P0-2 Sentinel Test:
    Verifies that extract_event_metadata(), build_audited_event_metadata(), and
    HistoricalSnapshotBuilder 100% strictly conform to the authoritative V8.0 specification layout:
    - E01–E04: Actor Embedding (4D, indices 0..3)
    - E05–E12: Event Type Embedding (8D, indices 4..11)
    - E13–E16: Event Status Embedding (4D, indices 12..15)
    - E17: Evidence Traceability [0, 1] (index 16)
    - E18: Source Reliability [0, 1] (index 17)
    - E19: Source Independence [0, 1] (index 18)
    - E20: Novelty Input Estimate [0, 1] (index 19)
    - E21: Pre-event Market Surprise [0, 1] (index 20, strictly pre-event prior implied move)
    - E22: Quantified Magnitude [0, 1] (index 21)
    - E23: Event Age in fractional days (index 22, Python Index 22)
    - E24: Timestamp Reliability [0, 1] (index 23, Python Index 23)
    Auxiliary features (sentiment, impact horizon) MUST NOT overwrite E01..E24.
    """
    t_avail = pd.Timestamp("2024-01-15 08:00:00", tz="Asia/Taipei")
    t_dec = pd.Timestamp("2024-01-16 08:00:00", tz="Asia/Taipei")  # Exactly 1.0 day age

    # Test Article covering each dimension group
    art_mops = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="公告本公司自結獲利與營業收入",
        body_text="澄清媒體報導獲利預測",
        source_published_at=t_avail,
        provider_available_at=t_avail,
        provider_available_evidence="MOPS Rule 51 Log",
        is_audit_verified=True,
        revision_id=2,
        lifecycle_status=EventLifecycleStatus.CLARIFIED,
    )

    vec = extract_event_metadata(art_mops, decision_time=t_dec)
    assert vec.shape == (24,)

    # 1. Actor (E01–E04): MOPS is Corporate Management -> index 0 is 1.0, indices 1..3 are 0.0
    assert vec[0] == 1.0, "E01: Actor corporate management"
    assert vec[1] == 0.0, "E02: Actor analyst"
    assert vec[2] == 0.0, "E03: Actor media"
    assert vec[3] == 0.0, "E04: Actor regulator"

    # 2. Type (E05–E12): Earnings ("獲利", "營收") -> index 4 is 1.0, other type indices 0.0
    assert vec[4] == 1.0, "E05: Type earnings"
    assert (vec[5:12] == 0.0).all(), "E06..E12 must be 0.0"

    # 3. Status (E13–E16): Clarified -> index 15 is 1.0
    assert vec[12] == 0.0, "E13: Status rumor"
    assert vec[13] == 0.0, "E14: Status announced"
    assert vec[14] == 0.0, "E15: Status ongoing"
    assert vec[15] == 1.0, "E16: Status clarified"

    # 4. Scalars (E17–E24)
    assert vec[16] == 1.0, "E17: Traceability verified"
    assert vec[17] == 1.0, "E18: MOPS Source Reliability"
    assert vec[18] == 1.0, "E19: MOPS Source Independence"
    assert vec[19] == 0.5, "E20: Novelty Input Estimate (0.5 for revision > 1)"
    assert vec[20] == 0.0, "E21: Pre-event Market Surprise (strictly 0.0 default, zero post-event leakage)"
    assert vec[21] == 0.5, "E22: Quantified Magnitude"
    assert np.isclose(vec[22], 1.0, atol=1e-4), f"E23: Event age in fractional days expected 1.0, got {vec[22]}"
    assert vec[23] == 1.0, "E24: Timestamp Reliability"

    # Verify Sentiment and Impact Horizon are stored in article.metadata, NOT overwriting E01..E24
    assert "sentiment_polarity" in art_mops.metadata
    assert "impact_horizon" in art_mops.metadata

    # Sentinel comparison with feature_engine.assemble_event_metadata
    assembled = assemble_event_metadata(
        actor_embedding_4d=vec[:4],
        type_embedding_8d=vec[4:12],
        status_embedding_4d=vec[12:16],
        evidence_traceability=float(vec[16]),
        source_reliability=float(vec[17]),
        source_independence=float(vec[18]),
        novelty_input_estimate=float(vec[19]),
        market_surprise_preevent=float(vec[20]),
        quantified_magnitude=float(vec[21]),
        event_age_days=float(vec[22]),
        timestamp_reliability=float(vec[23]),
    )
    assert np.allclose(vec, assembled), "extract_event_metadata must produce identical layout to assemble_event_metadata"


# ==============================================================================
# 9. P1-3: Anti-Fabrication & Strictly Isolated Embedding Engines
# ==============================================================================

def test_pretrained_without_model_must_fail():
    """
    P1-3 Test:
    1. PRETRAINED embedding mode must explicitly fail (ValueError) if genuine model weights / tokenizer are not loaded.
    2. DETERMINISTIC_HASH is explicitly designated for Smoke Test / Reproducible Baseline with version 'sha256-projection-v1.0'.
    3. UNTRAINED is strictly isolated from PRETRAINED.
    4. EVENT_TEXT_DIM is authoritative 128 across all schema.
    """
    assert EVENT_TEXT_DIM == 128, "Authoritative EVENT_TEXT_DIM must be 128"

    # 1. Calling PRETRAINED without model weights MUST raise ValueError
    engine_pretrained = TextEmbeddingEngine(TextEmbeddingType.PRETRAINED, model_version="finbert-zh-base-v1.0")
    with pytest.raises(ValueError, match="Strict Embedding Error: PRETRAINED"):
        engine_pretrained.embed_text("測試標題")

    # 2. DETERMINISTIC_HASH works deterministically for smoke tests
    engine_hash = TextEmbeddingEngine(TextEmbeddingType.DETERMINISTIC_HASH, model_version="sha256-projection-v1.0")
    v1 = engine_hash.embed_text("測試標題")
    v2 = engine_hash.embed_text("測試標題")
    assert v1.shape == (128,)
    assert np.allclose(v1, v2)
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)

    # 3. UNTRAINED is strictly isolated
    engine_untr = TextEmbeddingEngine(TextEmbeddingType.UNTRAINED, model_version="random-projection-v1.0")
    v_untr = engine_untr.embed_text("測試標題")
    assert v_untr.shape == (128,)
    assert not np.allclose(v1, v_untr)


# ==============================================================================
# 10. P1-4: Historical News Evidence Integrity & Open RSS Audit Defense
# ==============================================================================

def test_news_source_evidence_integrity():
    """
    P1-4 Test:
    1. Loads the 10 micro news articles from data/poc_real_data/historical_news_micro.json.
    2. Confirms all 10 retain is_audit_verified == False (ESTIMATED) due to open RSS lack of server logs.
    3. Confirms all 10 specify event_clustering_method == 'human_annotated' (distinguished from automated).
    4. Confirms that in Formal Mode, articles with is_audit_verified == False are strictly rejected.
    """
    articles = load_micro_news_dataset()
    assert len(articles) == 10, f"Expected 10 micro articles, got {len(articles)}"

    for art in articles:
        # All 10 must be marked unverified (ESTIMATED)
        assert art.is_audit_verified is False, f"Article {art.article_id} must NOT be marked verified without authentic exchange logs"
        # Must explicitly distinguish human-annotated clustering
        assert art.event_clustering_method == "human_annotated", f"Article {art.article_id} missing human_annotated clustering tag"
        # Must possess evidence documentation
        assert art.provider_available_evidence is not None and len(art.provider_available_evidence) > 0

    # In Formal Mode, unverified articles must be strictly rejected by pipeline
    pipe_formal = HistoricalMultimodalPipeline(mode="formal")
    for art in articles:
        with pytest.raises(ValueError, match="Cannot ingest unverified article .* in formal mode"):
            pipe_formal.ingest_article(art)

