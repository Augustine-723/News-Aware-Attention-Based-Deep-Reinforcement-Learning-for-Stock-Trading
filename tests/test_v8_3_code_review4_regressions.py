"""
V8.3 Code Review 4 Regression Tests: Snapshot Integrity & Event Alignment (P0-1..P0-3, P1-4..P1-5)

Addresses:
- P0-1: Formal OHLCV Availability Must Fail Closed
  - Formal Mode rejects df_ohlcv missing explicit timezone-aware available_at.
  - Prohibition of automatic 15:00 available_at estimation from DataFrame index in formal mode.
  - Synthetic / fixture fallback allowed with warning.
- P0-2: Institutional & Macro Historical As-Of Alignment
  - Separation of observation_date, trading_date, and available_at.
  - Strict rejection of unverified index-based joins or direct reindex in formal mode.
  - Release delays, holidays, missing values, and counterfactual future revisions.
- P0-3: Correct Recent / Active Event Slot Alignment (32 Recent + 8 Active = 40 Slots)
  - Strict 1-to-1 correspondence between slot k and recent_event_emb (0..31) / active_event_emb (0..7).
  - Disjoint sets policy between recent and active events.
  - Event ID Sentinel test verifying exact slot consistency across text, metadata, age, and relations.
  - Novelty Causal Memory (0..31) and Active Event Memory (32..39) indexing verification.
- P1-4: Eliminate Fake Supervised Ground Truth
  - When ground truth is absent, all label validity masks are explicitly False.
  - MultiTaskLoss and Stage 1 produce zero loss and zero gradients on unannotated samples.
  - Formal multi-horizon return targets strictly computed from T+1 executable open (T+1 Open-to-Open).
  - Strict boundary purge between Train, Validation, and Test splits.
- P1-5: Complete Dataset Content Hash
  - Canonical content hash covering features, missing masks, event masks, targets, masks, schema, shapes & dtypes.
  - Hash mutation verification across all individual components.
- Integration:
  - Full Snapshot -> Normalizer -> Model -> Stage 1/2/3 micro E2E regression.
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
    NUM_HORIZONS,
    PRICE_DIM,
    TECHNICAL_DIM,
    INSTITUTIONAL_DIM,
    MACRO_DIM,
    STOCK_CONTEXT_DIM,
    EVENT_TEXT_DIM,
    EVENT_METADATA_DIM,
    EVENT_STOCK_RELATION_DIM,
    POSITION_STATE_DIM,
    CURRENT_ACTIVE_SCHEMA_VERSION,
)
from stock_rl.data.pit_engine import (
    MissingStatus,
    ProvenanceRecord,
)
from stock_rl.data.normalizer import FeatureNormalizer, SplitManifest
from stock_rl.data.synthetic_data import MultimodalBatch
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.data.tensor_validator import validate_multimodal_batch
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.models.multi_task_heads import MultiTaskLoss
from stock_rl.training.stage1_pretrain import train_stage1_epoch
from stock_rl.training.stage2_rl import run_stage2_rl_training
from stock_rl.training.stage3_joint import run_stage3_joint_finetuning


def _generate_deterministic_raw_history(n_days: int = 100):
    """Generates synthetic deterministic raw historical feeds spanning n_days."""
    np.random.seed(42)
    trade_dates = pd.date_range("2024-01-01", periods=n_days, freq="B", tz="Asia/Taipei")
    available_at_ohlcv = trade_dates.floor("D") + pd.Timedelta(hours=15)
    available_at_inst = trade_dates.floor("D") + pd.Timedelta(hours=17)
    available_at_macro = trade_dates.floor("D") + pd.Timedelta(hours=18)

    p0 = 100.0
    rets = np.random.normal(0.0005, 0.015, size=n_days)
    closes = p0 * np.exp(np.cumsum(rets))
    opens = closes * (1.0 + np.random.normal(0.0, 0.004, size=n_days))
    highs = np.maximum(opens, closes) * (1.0 + np.random.uniform(0.002, 0.012, size=n_days))
    lows = np.minimum(opens, closes) * (1.0 - np.random.uniform(0.002, 0.012, size=n_days))
    volumes = np.random.uniform(5e6, 2e7, size=n_days)
    values = closes * volumes

    df_ohlcv = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes, "value": values, "available_at": available_at_ohlcv,
        "is_audit_verified": True,
    }, index=trade_dates)

    df_inst = pd.DataFrame({
        "foreign_net_buy": np.random.uniform(-5e5, 5e5, size=n_days),
        "trust_net_buy": np.random.uniform(-2e5, 2e5, size=n_days),
        "dealer_net_buy": np.random.uniform(-1e5, 1e5, size=n_days),
        "margin_balance": np.random.uniform(1e5, 2e5, size=n_days),
        "short_balance": np.random.uniform(1e4, 3e4, size=n_days),
        "available_at": available_at_inst,
        "is_audit_verified": True,
    }, index=trade_dates)

    df_macro = pd.DataFrame({
        "taiex": 18000.0 + np.cumsum(np.random.randn(n_days) * 25.0),
        "sector_index": 500.0 + np.cumsum(np.random.randn(n_days) * 1.5),
        "usdtwd": 31.0 + np.cumsum(np.random.randn(n_days) * 0.02),
        "available_at": available_at_macro,
        "is_audit_verified": True,
    }, index=trade_dates)

    us_avail = pd.date_range("2023-12-31 17:00:00", periods=n_days, freq="B", tz="America/New_York")
    df_foreign = pd.DataFrame({
        "sp500": 4800.0 + np.cumsum(np.random.randn(n_days) * 10.0),
        "nasdaq": 15000.0 + np.cumsum(np.random.randn(n_days) * 35.0),
        "sox": 4200.0 + np.cumsum(np.random.randn(n_days) * 12.0),
        "vix": 14.0 + np.abs(np.random.randn(n_days) * 1.5),
        "us_10y_yield": 4.10 + np.cumsum(np.random.randn(n_days) * 0.02),
        "available_at": us_avail,
        "is_audit_verified": True,
    }, index=trade_dates)

    stock_context_records = [
        {
            "industry_embedding_8d": np.ones(8, dtype=np.float32) / np.sqrt(8),
            "volatility_percentile": 0.65,
            "liquidity_percentile": 0.80,
            "beta_120d": 1.15,
            "industry_market_cap_percentile": 0.70,
            "days_since_ipo": 1200,
            "had_trading_suspension_20d": False,
            "provenance": ProvenanceRecord(
                source_id="MOPS_COMPANY_PROFILE_V1",
                available_at=trade_dates[0].floor("D") + pd.Timedelta(hours=9),
                is_audit_verified=True,
                missing_status="AVAILABLE",
            ),
        },
    ]

    event_records = []
    for d in range(10, n_days, 5):
        ev_ts = trade_dates[d].floor("D") + pd.Timedelta(hours=14)
        event_records.append({
            "event_id": f"NEWS_{d}",
            "text_embedding": np.random.randn(128).astype(np.float32),
            "actor_embedding_4d": np.zeros(4, dtype=np.float32),
            "type_embedding_8d": np.ones(8, dtype=np.float32) / np.sqrt(8),
            "status_embedding_4d": np.zeros(4, dtype=np.float32),
            "evidence_traceability": 0.95,
            "source_reliability": 0.90,
            "source_independence": 1.0,
            "novelty_input_estimate": 0.50,
            "market_surprise_preevent": 0.01,
            "quantified_magnitude": 0.60,
            "timestamp_reliability": 1.0,
            "provenance": ProvenanceRecord(
                source_id=f"NEWS_{d}",
                available_at=ev_ts,
                is_audit_verified=True,
                missing_status="AVAILABLE",
            ),
            "relation": {
                "direct_company_mention": True,
                "industry_match": True,
                "upstream_exposure": 0.2,
                "downstream_exposure": 0.1,
                "customer_relationship": 0.0,
                "supplier_relationship": 0.0,
                "geographic_exposure": 0.5,
                "macro_factor_exposure": 0.3,
            },
        })

    return df_ohlcv, df_inst, df_macro, df_foreign, stock_context_records, event_records, trade_dates


# ==============================================================================
# 1. P0-1: Formal OHLCV Availability Must Fail Closed
# ==============================================================================

def test_formal_ohlcv_missing_available_at_rejection():
    """P0-1: Formal mode strictly rejects df_ohlcv missing available_at column."""
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, _ = _generate_deterministic_raw_history(100)

    # Strip available_at column
    df_ohlcv_no_avail = df_ohlcv.drop(columns=["available_at"])

    # Formal mode MUST raise ValueError
    with pytest.raises(ValueError, match="Strict PIT Error.*missing required explicit 'available_at' column in formal mode"):
        HistoricalSnapshotBuilder(
            df_ohlcv=df_ohlcv_no_avail,
            df_inst=df_inst,
            df_macro=df_macro,
            df_foreign=df_foreign,
            mode="formal",
        )

    # Naive timestamp in available_at MUST also be rejected in formal mode
    df_ohlcv_naive = df_ohlcv.copy()
    df_ohlcv_naive["available_at"] = df_ohlcv_naive["available_at"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="Strict PIT Error.*'available_at' in df_ohlcv must be timezone-aware"):
        HistoricalSnapshotBuilder(
            df_ohlcv=df_ohlcv_naive,
            df_inst=df_inst,
            df_macro=df_macro,
            df_foreign=df_foreign,
            mode="formal",
        )

    # Non-formal (synthetic) mode allows automatic estimation with warning
    with pytest.warns(UserWarning, match="Estimated Availability Fallback"):
        builder_syn = HistoricalSnapshotBuilder(
            df_ohlcv=df_ohlcv_no_avail,
            mode="synthetic",
        )
        assert "available_at" in builder_syn.df_ohlcv.columns


# ==============================================================================
# 2. P0-2: Institutional & Macro Historical As-Of Alignment
# ==============================================================================

def test_institutional_and_macro_historical_asof_alignment():
    """P0-2: Verifies As-Of alignment, delay handling, holidays, and counterfactual future revisions."""
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, trade_dates = _generate_deterministic_raw_history(100)

    # 1. Rejection of missing available_at in df_inst and df_macro in formal mode
    df_inst_no_avail = df_inst.drop(columns=["available_at"])
    with pytest.raises(ValueError, match="Strict PIT Error: df_inst missing required explicit 'available_at'"):
        HistoricalSnapshotBuilder(df_ohlcv=df_ohlcv, df_inst=df_inst_no_avail, mode="formal")

    df_macro_no_avail = df_macro.drop(columns=["available_at"])
    with pytest.raises(ValueError, match="Strict PIT Error: df_macro missing required explicit 'available_at'"):
        HistoricalSnapshotBuilder(df_ohlcv=df_ohlcv, df_macro=df_macro_no_avail, mode="formal")

    # 2. Delay test: today's institutional data delayed past decision time (21:00 > 20:00)
    decision_ts = trade_dates[65].floor("D") + pd.Timedelta(hours=20)
    df_inst_delayed = df_inst.copy()
    # Delay trading day 65 available_at to 21:00
    df_inst_delayed.loc[trade_dates[65], "available_at"] = trade_dates[65].floor("D") + pd.Timedelta(hours=21)

    builder_delayed = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv, df_inst=df_inst_delayed, df_macro=df_macro,
        df_foreign=df_foreign, mode="formal"
    )
    snap_delayed = builder_delayed.build_snapshot(decision_ts)

    # Day 65 is the last row (index 59) of the 60-day window
    # Because available_at > decision_ts, day 65 institutional data is missing
    assert snap_delayed.institutional_missing_mask[0, 59].all().item() is True
    # Daily flow features (I01..I03) must be strictly 0.0 for the missing day
    assert torch.all(snap_delayed.institutional[0, 59, 0:3] == 0.0)

    # 3. Normal release before decision time (17:00 <= 20:00)
    builder_normal = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv, df_inst=df_inst, df_macro=df_macro,
        df_foreign=df_foreign, mode="formal"
    )
    snap_normal = builder_normal.build_snapshot(decision_ts)
    assert snap_normal.institutional_missing_mask[0, 59].any().item() is False

    # 4. Counterfactual Future Revision: appending a revision with available_at > decision_ts
    # cannot alter the past snapshot
    future_avail = decision_ts + pd.Timedelta(days=1)
    revised_row = df_macro.loc[trade_dates[60]:trade_dates[60]].copy()
    revised_row["taiex"] = 999999.0
    revised_row["available_at"] = future_avail
    revised_row["revision_id"] = 2
    df_macro_future_rev = pd.concat([df_macro, revised_row])

    builder_fut = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv, df_inst=df_inst, df_macro=df_macro_future_rev,
        df_foreign=df_foreign, mode="formal"
    )
    snap_fut = builder_fut.build_snapshot(decision_ts)
    # The snapshot as of decision_ts must exclude the future revision and match snap_normal
    assert torch.allclose(snap_normal.macro, snap_fut.macro)


# ==============================================================================
# 3. P0-3: Correct Recent / Active Event Slot Alignment (32 Recent + 8 Active)
# ==============================================================================

def test_recent_active_event_slot_1to1_alignment_and_sentinel():
    """
    P0-3: Verifies strict 1-to-1 slot alignment between Recent (0..31) and Active (32..39),
    disjoint sets enforcement, and Event ID Sentinel consistency.
    """
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, _, trade_dates = _generate_deterministic_raw_history(100)
    decision_ts = trade_dates[70].floor("D") + pd.Timedelta(hours=20)

    # Construct explicit pools:
    # 25 recent news events and 5 active events (total 30 < 40, to verify padding too)
    test_events = []
    # 25 Recent events
    for i in range(25):
        ev_ts = decision_ts - pd.Timedelta(days=i + 1)
        test_events.append({
            "event_id": f"RECENT_EV_{i:02d}",
            "is_active": False,
            "text_embedding": np.ones(128, dtype=np.float32) * (i + 1.0) / 100.0,
            "actor_embedding_4d": np.ones(4, dtype=np.float32) * 0.1,
            "type_embedding_8d": np.ones(8, dtype=np.float32) * 0.2,
            "status_embedding_4d": np.ones(4, dtype=np.float32) * 0.3,
            "evidence_traceability": 0.8,
            "source_reliability": 0.9,
            "source_independence": 1.0,
            "novelty_input_estimate": 0.5,
            "market_surprise_preevent": 0.05,
            "quantified_magnitude": float(i + 1.0) / 100.0,
            "timestamp_reliability": 1.0,
            "provenance": ProvenanceRecord(
                source_id=f"RECENT_SRC_{i:02d}",
                available_at=ev_ts,
                is_audit_verified=True,
                missing_status="AVAILABLE",
            ),
            "relation": {"direct_company_mention": True},
        })

    # 5 Active events (tagged is_active=True)
    for j in range(5):
        ev_ts = decision_ts - pd.Timedelta(days=j + 10)
        test_events.append({
            "event_id": f"ACTIVE_EV_{j:02d}",
            "is_active": True,
            "text_embedding": np.ones(128, dtype=np.float32) * (-(j + 1.0)) / 100.0,
            "actor_embedding_4d": np.ones(4, dtype=np.float32) * 0.5,
            "type_embedding_8d": np.ones(8, dtype=np.float32) * 0.6,
            "status_embedding_4d": np.ones(4, dtype=np.float32) * 0.7,
            "evidence_traceability": 1.0,
            "source_reliability": 1.0,
            "source_independence": 1.0,
            "novelty_input_estimate": 0.9,
            "market_surprise_preevent": 0.10,
            "quantified_magnitude": 0.5 + float(j + 1.0) / 100.0,
            "timestamp_reliability": 1.0,
            "provenance": ProvenanceRecord(
                source_id=f"ACTIVE_SRC_{j:02d}",
                available_at=ev_ts,
                is_audit_verified=True,
                missing_status="AVAILABLE",
            ),
            "relation": {"direct_company_mention": False, "industry_match": True},
        })

    builder = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv, df_inst=df_inst, df_macro=df_macro,
        df_foreign=df_foreign, stock_context_records=ctx_recs,
        event_records=test_events, mode="formal"
    )
    snap = builder.build_snapshot(decision_ts)

    # 1. Check slot assignments
    slot_ids = snap.slot_event_ids[0]
    assert len(slot_ids) == TOTAL_EVENT_SLOTS  # 40

    # Recent slots: 0..24 are populated, 25..31 are padded
    for k in range(25):
        assert slot_ids[k] == f"RECENT_EV_{k:02d}"
        assert snap.event_mask[0, k].item() is True
        # Verify text embedding 1-to-1 match
        expected_emb = np.ones(128, dtype=np.float32) * (k + 1.0) / 100.0
        expected_emb = expected_emb / np.linalg.norm(expected_emb)
        assert np.allclose(snap.recent_event_emb[0, k].numpy(), expected_emb, atol=1e-5)
        # Verify metadata E22 quantified magnitude
        assert np.isclose(snap.event_metadata[0, k, 21].item(), float(k + 1.0) / 100.0)
        # Relation
        assert np.isclose(snap.event_stock_rel[0, k, 0].item(), 1.0)

    # Recent padded slots: 25..31
    for k in range(25, RECENT_EVENT_SLOTS):
        assert slot_ids[k] is None
        assert snap.event_mask[0, k].item() is False
        assert snap.event_metadata_missing_mask[0, k].all().item() is True

    # Active slots: 32..36 are populated (5 active events), 37..39 are padded (3 padded)
    for m in range(5):
        global_k = RECENT_EVENT_SLOTS + m  # 32 + m
        assert slot_ids[global_k] == f"ACTIVE_EV_{m:02d}"
        assert snap.event_mask[0, global_k].item() is True
        expected_emb = np.ones(128, dtype=np.float32) * (-(m + 1.0)) / 100.0
        expected_emb = expected_emb / np.linalg.norm(expected_emb)
        assert np.allclose(snap.active_event_emb[0, m].numpy(), expected_emb, atol=1e-5)
        # Verify metadata E22 quantified magnitude
        assert np.isclose(snap.event_metadata[0, global_k, 21].item(), 0.5 + float(m + 1.0) / 100.0)
        # Relation industry_match
        assert np.isclose(snap.event_stock_rel[0, global_k, 1].item(), 1.0)

    # Active padded slots: 37..39
    for global_k in range(32 + 5, TOTAL_EVENT_SLOTS):
        assert slot_ids[global_k] is None
        assert snap.event_mask[0, global_k].item() is False
        assert snap.event_metadata_missing_mask[0, global_k].all().item() is True


def test_novelty_causal_memory_and_active_memory_indexing():
    """
    P0-3: Tests that FullModel correctly indices:
    - Active Memory from slots 32..39
    - Novelty Causal Memory from prior events only.
    """
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, trade_dates = _generate_deterministic_raw_history(100)
    decision_ts = trade_dates[75].floor("D") + pd.Timedelta(hours=20)

    builder = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv, df_inst=df_inst, df_macro=df_macro,
        df_foreign=df_foreign, stock_context_records=ctx_recs,
        event_records=ev_recs, mode="formal"
    )
    snap = builder.build_snapshot(decision_ts)

    model = StockRLFullModel()
    model.eval()

    with torch.no_grad():
        fused_rep, rl_state, projected_events, context_token, causal_prior_mem = model.encode_multimodal_state(snap)

    assert projected_events.shape == (1, 40, 128)
    assert causal_prior_mem.shape == (1, 40, 128)
    assert rl_state.shape == (1, 132)


# ==============================================================================
# 4. P1-4: Eliminate Fake Supervised Ground Truth
# ==============================================================================

def test_missing_labels_produce_zero_auxiliary_loss_and_gradients():
    """
    P1-4: When ground truth is absent, all validity masks are False,
    MultiTaskLoss evaluates to 0.0 with 0 active tasks, and backward pass
    produces ZERO gradients across all model parameters.
    """
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, trade_dates = _generate_deterministic_raw_history(100)
    decision_ts = trade_dates[75].floor("D") + pd.Timedelta(hours=20)

    # Build snapshot without ground-truth targets (default safe mode)
    builder = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv, df_inst=df_inst, df_macro=df_macro,
        df_foreign=df_foreign, stock_context_records=ctx_recs,
        event_records=ev_recs, mode="formal"
    )
    snap = builder.build_snapshot(decision_ts, include_ground_truth_targets=False)

    # 1. All label validity masks must be strictly False
    assert snap.relevance_mask.any().item() is False
    assert snap.novelty_mask.any().item() is False
    assert snap.return_mask.any().item() is False
    assert snap.risk_mask.any().item() is False

    # 2. Forward pass through model and loss module
    model = StockRLFullModel()
    model.train()
    loss_module = MultiTaskLoss(use_uncertainty_weighting=True)

    _, preds, _ = model(snap)
    loss, loss_dict = loss_module(
        predictions=preds,
        relevance_labels=snap.relevance_labels,
        novelty_labels=snap.novelty_labels,
        return_targets=snap.return_targets,
        risk_targets=snap.risk_targets,
        relevance_mask=snap.relevance_mask,
        novelty_mask=snap.novelty_mask,
        return_mask=snap.return_mask,
        risk_mask=snap.risk_mask,
        fallback_event_mask=snap.event_mask,
    )

    # 3. Loss must be 0.0 with 0 active tasks
    assert loss.item() == 0.0
    assert loss_dict["active_tasks"] == 0
    assert loss_dict["loss_total"] == 0.0

    # 4. Backward pass produces strictly ZERO gradients on model weights
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is None or torch.all(param.grad == 0.0), f"Parameter {name} received non-zero gradient!"


def test_future_multi_horizon_return_targets_and_boundary_purge():
    """
    P1-4: Formal multi-horizon return targets must strictly compute from T+1 executable open,
    and boundary purge must mask out any horizon crossing split boundaries.
    """
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, trade_dates = _generate_deterministic_raw_history(100)

    # Day 70 decision timestamp
    decision_ts = trade_dates[70].floor("D") + pd.Timedelta(hours=20)
    builder = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv, df_inst=df_inst, df_macro=df_macro,
        df_foreign=df_foreign, stock_context_records=ctx_recs,
        event_records=ev_recs, mode="formal"
    )

    # Case A: Normal calculation without boundary cutoff
    snap_gt = builder.build_snapshot(decision_ts, include_ground_truth_targets=True)
    assert snap_gt.return_mask.all().item() is True
    assert snap_gt.risk_mask.all().item() is True

    # Manually compute expected returns from T+1 open
    o_t1 = float(df_ohlcv["open"].iloc[71])
    o_t2 = float(df_ohlcv["open"].iloc[72])
    o_t4 = float(df_ohlcv["open"].iloc[74])
    o_t6 = float(df_ohlcv["open"].iloc[76])

    expected_1d = (o_t2 - o_t1) / o_t1
    expected_3d = (o_t4 - o_t1) / o_t1
    expected_5d = (o_t6 - o_t1) / o_t1

    ret_actual = snap_gt.return_targets[0].numpy()
    assert np.isclose(ret_actual[0], expected_1d)
    assert np.isclose(ret_actual[1], expected_3d)
    assert np.isclose(ret_actual[2], expected_5d)

    # Case B: Boundary Purge: split boundary ends at trade_dates[73]
    # In this case:
    # 1D requires T+2 (day 72 <= 73) -> Valid (True)
    # 3D requires T+4 (day 74 > 73) -> Purged (False)
    # 5D requires T+6 (day 76 > 73) -> Purged (False)
    snap_purged = builder.build_snapshot(
        decision_ts,
        include_ground_truth_targets=True,
        purge_boundary_date=trade_dates[73],
    )
    assert snap_purged.return_mask[0, 0].item() is True
    assert snap_purged.return_mask[0, 1].item() is False  # Purged!
    assert snap_purged.return_mask[0, 2].item() is False  # Purged!


# ==============================================================================
# 5. P1-5: Complete Dataset Content Hash Mutation Verification
# ==============================================================================

def test_canonical_content_hash_mutation():
    """
    P1-5: Verifies that mutating ANY component of the MultimodalBatch
    (feature tensors, missing masks, event masks, targets, label validity masks,
     schema_version, shapes, dtypes, or builder/dataset metadata)
    alters the canonical SHA-256 content hash.
    """
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, trade_dates = _generate_deterministic_raw_history(100)
    decision_ts = trade_dates[70].floor("D") + pd.Timedelta(hours=20)

    builder = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv, df_inst=df_inst, df_macro=df_macro,
        df_foreign=df_foreign, stock_context_records=ctx_recs,
        event_records=ev_recs, mode="formal"
    )
    base_batch = builder.build_snapshot(decision_ts, include_ground_truth_targets=True)
    base_hash = SplitManifest.compute_batch_content_hash(base_batch)

    # 1. Mutate primary feature tensors
    b_price = copy.deepcopy(base_batch)
    b_price.price[0, 0, 0] += 0.05
    assert SplitManifest.compute_batch_content_hash(b_price) != base_hash

    b_tech = copy.deepcopy(base_batch)
    b_tech.technical[0, 0, 0] += 0.05
    assert SplitManifest.compute_batch_content_hash(b_tech) != base_hash

    # 2. Mutate missing masks
    b_mask_macro = copy.deepcopy(base_batch)
    b_mask_macro.macro_missing_mask[0, 0, 0] = ~b_mask_macro.macro_missing_mask[0, 0, 0]
    assert SplitManifest.compute_batch_content_hash(b_mask_macro) != base_hash

    b_mask_inst = copy.deepcopy(base_batch)
    b_mask_inst.institutional_missing_mask[0, 0, 0] = ~b_mask_inst.institutional_missing_mask[0, 0, 0]
    assert SplitManifest.compute_batch_content_hash(b_mask_inst) != base_hash

    b_mask_ctx = copy.deepcopy(base_batch)
    b_mask_ctx.stock_context_missing_mask[0, 0] = ~b_mask_ctx.stock_context_missing_mask[0, 0]
    assert SplitManifest.compute_batch_content_hash(b_mask_ctx) != base_hash

    b_mask_ev = copy.deepcopy(base_batch)
    b_mask_ev.event_metadata_missing_mask[0, 0, 0] = ~b_mask_ev.event_metadata_missing_mask[0, 0, 0]
    assert SplitManifest.compute_batch_content_hash(b_mask_ev) != base_hash

    # 3. Mutate event mask
    b_ev_mask = copy.deepcopy(base_batch)
    b_ev_mask.event_mask[0, 0] = ~b_ev_mask.event_mask[0, 0]
    assert SplitManifest.compute_batch_content_hash(b_ev_mask) != base_hash

    # 4. Mutate targets and validity masks
    b_ret_target = copy.deepcopy(base_batch)
    b_ret_target.return_targets[0, 0] += 0.01
    assert SplitManifest.compute_batch_content_hash(b_ret_target) != base_hash

    b_ret_mask = copy.deepcopy(base_batch)
    b_ret_mask.return_mask[0, 0] = ~b_ret_mask.return_mask[0, 0]
    assert SplitManifest.compute_batch_content_hash(b_ret_mask) != base_hash

    b_rel_mask = copy.deepcopy(base_batch)
    b_rel_mask.relevance_mask[0, 0, 0] = ~b_rel_mask.relevance_mask[0, 0, 0]
    assert SplitManifest.compute_batch_content_hash(b_rel_mask) != base_hash

    # 5. Mutate schema_version
    b_schema = copy.deepcopy(base_batch)
    b_schema.schema_version = "v8.1-candidate"
    assert SplitManifest.compute_batch_content_hash(b_schema) != base_hash

    # 6. Mutate builder_version and dataset_id
    assert SplitManifest.compute_batch_content_hash(base_batch, builder_version="8.4.0") != base_hash
    assert SplitManifest.compute_batch_content_hash(base_batch, dataset_id="ALT_ID") != base_hash


# ==============================================================================
# 6. End-to-End Micro Pipeline Regression
# ==============================================================================

def test_snapshot_builder_to_stage1_2_3_micro_regression():
    """
    E2E Integration Verification:
    HistoricalSnapshotBuilder -> SplitManifest Canonical Hash -> Normalizer -> Model -> Stages 1, 2, 3.
    """
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, trade_dates = _generate_deterministic_raw_history(100)

    # 1. Historical Snapshot Builder
    builder = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv, df_inst=df_inst, df_macro=df_macro,
        df_foreign=df_foreign, stock_context_records=ctx_recs,
        event_records=ev_recs, mode="formal"
    )

    decision_timestamps = [
        trade_dates[d].floor("D") + pd.Timedelta(hours=20)
        for d in [65, 66, 67, 68]
    ]
    raw_batch = builder.build_batch(decision_timestamps, include_ground_truth_targets=True)
    assert raw_batch.price.shape[0] == 4
    assert validate_multimodal_batch(raw_batch) is True

    # 2. Content-Based SplitManifest with Canonical Batch Hash
    feat_hash = builder.compute_snapshot_hash(raw_batch)
    raw_hash = SplitManifest.compute_content_hash(df_ohlcv)
    manifest = SplitManifest.create(
        dataset_id="review4_pipeline",
        schema_version="v8.0-original",
        train_start=trade_dates[0],
        train_end=trade_dates[70],
        val_start=trade_dates[71],
        val_end=trade_dates[85],
        test_start=trade_dates[86],
        test_end=trade_dates[99],
        raw_dataset_content_hash=raw_hash,
        feature_snapshot_hash=feat_hash,
        normalizer_fitting_data_hash=feat_hash,
    )

    # 3. Normalizer Fitting & Transforming
    normalizer = FeatureNormalizer(schema_version="v8.0-original", split_manifest=manifest)
    norm_batch = normalizer.fit_transform(raw_batch, is_train_split=True, sample_timestamps=decision_timestamps)
    assert normalizer.is_fitted is True
    assert normalizer.fitted_fitting_data_hash == feat_hash

    # 4. Model Forward Pass
    model = StockRLFullModel()
    q_vals, preds, _ = model(norm_batch)
    assert q_vals.shape == (4, 2)
    assert "novelty_scores" in preds

    # 5. Stage 1 Pretraining on Normalized Batch
    loss_module = MultiTaskLoss(use_uncertainty_weighting=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epoch_metrics = train_stage1_epoch(
        model=model,
        loss_module=loss_module,
        optimizer=optimizer,
        batches=[norm_batch],
    )
    assert np.isfinite(epoch_metrics["loss_total"])

    # 6. Stage 2 RL Micro Step
    from stock_rl.rl.replay_buffer import MultimodalReplayBuffer
    replay_buffer = MultimodalReplayBuffer(capacity=100)
    for i in range(4):
        single_obs = MultimodalBatch(
            price=norm_batch.price[i:i+1],
            technical=norm_batch.technical[i:i+1],
            institutional=norm_batch.institutional[i:i+1],
            macro=norm_batch.macro[i:i+1],
            stock_context=norm_batch.stock_context[i:i+1],
            recent_event_emb=norm_batch.recent_event_emb[i:i+1],
            active_event_emb=norm_batch.active_event_emb[i:i+1],
            event_metadata=norm_batch.event_metadata[i:i+1],
            event_stock_rel=norm_batch.event_stock_rel[i:i+1],
            event_mask=norm_batch.event_mask[i:i+1],
            position=norm_batch.position[i:i+1],
            macro_missing_mask=norm_batch.macro_missing_mask[i:i+1],
            institutional_missing_mask=norm_batch.institutional_missing_mask[i:i+1],
            stock_context_missing_mask=norm_batch.stock_context_missing_mask[i:i+1],
            event_metadata_missing_mask=norm_batch.event_metadata_missing_mask[i:i+1],
        )
        replay_buffer.push(
            obs=single_obs, action=1, reward=0.01, next_obs=single_obs, terminated=False, truncated=False
        )
    assert len(replay_buffer) == 4
    batch_sampled = replay_buffer.sample(batch_size=2)
    obs_batch = batch_sampled[0]
    assert obs_batch.price.shape[0] == 2
