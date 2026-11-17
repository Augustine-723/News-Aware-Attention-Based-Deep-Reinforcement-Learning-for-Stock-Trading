"""
Regression Test Suite for V8.3 Code Review 3 — Final Historical PIT Integrity
Strictly verifies:
- P0-1: Decision-Aligned Historical Snapshot Builder
  - Unified HistoricalSnapshotBuilder constructing pointwise As-Of snapshots
  - Strict 60-day window ending on or before decision timestamp
  - Point-in-time dynamic Stock Context (market cap and trading value As-Of slice)
  - Counterfactual test: modifying future raw data cannot alter past snapshots
- P0-2: ProvenanceRecord Fail-Closed Defaults
  - is_audit_verified defaults to False
  - missing_status defaults to UNKNOWN
  - Naive timestamps rejected with ValueError in formal mode
  - HistoricalVersionSelector chooses authoritative revision As-Of decision time
- P0-3: Mandatory Foreign Source Audit
  - Rejection of foreign market data lacking explicit 'is_audit_verified' column
  - Exclusion of unverified foreign observations in formal mode
  - Unreleased revision cannot overwrite past values
- P1-4: Preserve Missing & Unknown Semantics
  - Formal distinction of Actual Zero vs Missing vs Unknown vs Not Yet Published
  - Missing masks in MultimodalBatch preserved through Normalizer
  - Counterfactual invariant: future data does not alter past missing masks
- P1-5: Content-Based Dataset Provenance
  - SplitManifest definition hash vs dataset content hash distinction
  - Content change alters content hash while definition hash remains stable
  - Normalizer fitting data hash verification and load_state_dict protection
- Integration:
  - Real Snapshot -> Normalizer -> Model -> Stage 1/2/3 micro dataflow
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
    HistoricalVersionSelector,
    asof_join_cross_market,
)
from stock_rl.data.feature_engine import (
    assemble_event_stock_relation,
    build_audited_event_metadata,
    build_audited_stock_context,
)
from stock_rl.data.normalizer import FeatureNormalizer, SplitManifest
from stock_rl.data.synthetic_data import MultimodalBatch
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.data.tensor_validator import validate_multimodal_batch, TensorSchemaError
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.training.stage1_pretrain import run_stage1_pretraining
from stock_rl.training.stage2_rl import run_stage2_rl_training
from stock_rl.training.stage3_joint import run_stage3_joint_finetuning


# ==============================================================================
# P0-1: Decision-Aligned Historical Snapshot Builder
# ==============================================================================

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
    }, index=trade_dates)

    df_inst = pd.DataFrame({
        "foreign_net_buy": np.random.uniform(-5e5, 5e5, size=n_days),
        "trust_net_buy": np.random.uniform(-2e5, 2e5, size=n_days),
        "dealer_net_buy": np.random.uniform(-1e5, 1e5, size=n_days),
        "margin_balance": np.random.uniform(1e5, 2e5, size=n_days),
        "short_balance": np.random.uniform(1e4, 3e4, size=n_days),
        "available_at": available_at_inst,
    }, index=trade_dates)

    df_macro = pd.DataFrame({
        "taiex": 18000.0 + np.cumsum(np.random.randn(n_days) * 25.0),
        "sector_index": 500.0 + np.cumsum(np.random.randn(n_days) * 1.5),
        "usdtwd": 31.0 + np.cumsum(np.random.randn(n_days) * 0.02),
        "available_at": available_at_macro,
    }, index=trade_dates)

    # US Foreign cross-market feed: T-1 16:00 EST available at 17:00 EST
    us_avail = pd.date_range("2023-12-31 17:00:00", periods=n_days, freq="B", tz="America/New_York")
    df_foreign = pd.DataFrame({
        "sp500": 4800.0 + np.cumsum(np.random.randn(n_days) * 10.0),
        "nasdaq": 15000.0 + np.cumsum(np.random.randn(n_days) * 35.0),
        "sox": 4200.0 + np.cumsum(np.random.randn(n_days) * 12.0),
        "vix": np.random.uniform(13.0, 24.0, size=n_days),
        "us_10y_yield": np.random.uniform(3.9, 4.4, size=n_days),
        "available_at": us_avail,
        "is_audit_verified": True,
    })

    # Stock context records published quarterly
    stock_context_records = [
        {
            "industry_embedding_8d": np.ones(8, dtype=np.float32) / np.sqrt(8),
            "volatility_percentile": 0.55,
            "liquidity_percentile": 0.70,
            "beta_120d": 1.10,
            "industry_market_cap_percentile": 0.80,
            "days_since_ipo": 1000,
            "had_trading_suspension_20d": False,
            "provenance": ProvenanceRecord(
                source_id="MOPS_Q4",
                available_at=pd.Timestamp("2024-01-01 09:00:00", tz="Asia/Taipei"),
                is_audit_verified=True,
                missing_status="AVAILABLE",
            ),
        },
        {
            "industry_embedding_8d": np.ones(8, dtype=np.float32) / np.sqrt(8),
            "volatility_percentile": 0.60,
            "liquidity_percentile": 0.75,
            "beta_120d": 1.15,
            "industry_market_cap_percentile": 0.85,
            "days_since_ipo": 1060,
            "had_trading_suspension_20d": False,
            "provenance": ProvenanceRecord(
                source_id="MOPS_Q1",
                available_at=pd.Timestamp("2024-03-31 09:00:00", tz="Asia/Taipei"),
                is_audit_verified=True,
                missing_status="AVAILABLE",
            ),
        },
    ]

    # Event records with verified timestamps
    event_records = []
    for d in range(10, n_days, 5):
        ev_ts = trade_dates[d].floor("D") + pd.Timedelta(hours=14)
        event_records.append({
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


def test_snapshot_builder_discrete_decision_alignment():
    """P0-1: Verify HistoricalSnapshotBuilder builds single-observation snapshot for exact decision timestamp."""
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, trade_dates = _generate_deterministic_raw_history(100)

    builder = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv,
        df_inst=df_inst,
        df_macro=df_macro,
        df_foreign=df_foreign,
        stock_context_records=ctx_recs,
        event_records=ev_recs,
        shares_outstanding=1e8,
        mode="formal",
    )

    dec_time = trade_dates[75].floor("D") + pd.Timedelta(hours=20)
    snap = builder.build_snapshot(decision_timestamp=dec_time)

    assert isinstance(snap, MultimodalBatch)
    assert snap.price.shape == (1, 60, 12)
    assert snap.technical.shape == (1, 60, 16)
    assert snap.institutional.shape == (1, 60, 8)
    assert snap.macro.shape == (1, 60, 12)
    assert snap.stock_context.shape == (1, 16)
    assert snap.recent_event_emb.shape == (1, 32, 128)
    assert snap.active_event_emb.shape == (1, 8, 128)
    assert snap.event_metadata.shape == (1, 40, 24)
    assert snap.event_stock_rel.shape == (1, 40, 8)
    assert snap.event_mask.shape == (1, 40)
    assert validate_multimodal_batch(snap) is True


def test_snapshot_builder_dynamic_point_in_time_stock_context():
    """
    P0-1 requirement 4: Stock Context must NOT use the last price/value of the full dataset.
    It must dynamically compute market cap and 60D MA volume using strictly the As-Of sliced window.
    """
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, trade_dates = _generate_deterministic_raw_history(100)

    builder = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv,
        df_inst=df_inst,
        df_macro=df_macro,
        df_foreign=df_foreign,
        stock_context_records=ctx_recs,
        event_records=ev_recs,
        shares_outstanding=1e8,
        mode="formal",
    )

    # Day 65 decision snapshot vs Day 90 decision snapshot
    dec_65 = trade_dates[65].floor("D") + pd.Timedelta(hours=20)
    dec_90 = trade_dates[90].floor("D") + pd.Timedelta(hours=20)

    snap_65 = builder.build_snapshot(decision_timestamp=dec_65)
    snap_90 = builder.build_snapshot(decision_timestamp=dec_90)

    # C09 is Log Market Capitalization (index 8)
    # C10 is Log Average Trading Value 60D (index 9)
    # They must differ because prices and 60D trading volumes evolved between day 65 and day 90
    c09_65 = snap_65.stock_context[0, 8].item()
    c09_90 = snap_90.stock_context[0, 8].item()
    assert c09_65 != c09_90

    # Specifically verify snap_65 used day 65's close price, NOT day 99's close price
    expected_mc_65 = np.log(float(df_ohlcv.loc[trade_dates[65], "close"]) * 1e8)
    assert abs(c09_65 - expected_mc_65) < 1e-4

    full_dataset_last_close = float(df_ohlcv["close"].iloc[-1])
    full_dataset_mc = np.log(full_dataset_last_close * 1e8)
    # Snapshot at day 65 must NOT match full dataset end market cap
    assert abs(c09_65 - full_dataset_mc) > 1e-3


def test_snapshot_builder_counterfactual_future_raw_modification():
    """
    P0-1 Counterfactual Test:
    Modifying raw records strictly after decision_timestamp must produce
    BIT-FOR-BIT IDENTICAL snapshots at decision_timestamp.
    """
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, trade_dates = _generate_deterministic_raw_history(100)

    builder_base = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv,
        df_inst=df_inst,
        df_macro=df_macro,
        df_foreign=df_foreign,
        stock_context_records=ctx_recs,
        event_records=ev_recs,
        shares_outstanding=1e8,
        mode="formal",
    )

    dec_time_70 = trade_dates[70].floor("D") + pd.Timedelta(hours=20)
    snap_base_70 = builder_base.build_snapshot(decision_timestamp=dec_time_70)

    # Now create perturbed raw datasets strictly after Day 70 (Day 71..99)
    df_ohlcv_pert = df_ohlcv.copy()
    df_ohlcv_pert.loc[trade_dates[71]:, "close"] *= 5.0
    df_ohlcv_pert.loc[trade_dates[71]:, "volume"] *= 10.0

    df_inst_pert = df_inst.copy()
    df_inst_pert.loc[trade_dates[71]:, "foreign_net_buy"] = 9999999.0

    df_macro_pert = df_macro.copy()
    df_macro_pert.loc[trade_dates[71]:, "taiex"] = 99999.0

    df_foreign_pert = df_foreign.copy()
    df_foreign_pert.loc[71:, "sp500"] = 99999.0

    ev_recs_pert = copy.deepcopy(ev_recs)
    # Add a massive future event on day 80
    ev_recs_pert.append({
        "text_embedding": np.ones(128, dtype=np.float32),
        "actor_embedding_4d": np.ones(4, dtype=np.float32),
        "type_embedding_8d": np.ones(8, dtype=np.float32),
        "status_embedding_4d": np.ones(4, dtype=np.float32),
        "evidence_traceability": 1.0,
        "source_reliability": 1.0,
        "source_independence": 1.0,
        "novelty_input_estimate": 1.0,
        "market_surprise_preevent": 0.5,
        "quantified_magnitude": 1.0,
        "timestamp_reliability": 1.0,
        "provenance": ProvenanceRecord(
            source_id="MASSIVE_FUTURE_NEWS",
            available_at=trade_dates[80].floor("D") + pd.Timedelta(hours=12),
            is_audit_verified=True,
            missing_status="AVAILABLE",
        ),
        "relation": {"direct_company_mention": True, "industry_match": True},
    })

    builder_pert = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv_pert,
        df_inst=df_inst_pert,
        df_macro=df_macro_pert,
        df_foreign=df_foreign_pert,
        stock_context_records=ctx_recs,
        event_records=ev_recs_pert,
        shares_outstanding=1e8,
        mode="formal",
    )

    snap_pert_70 = builder_pert.build_snapshot(decision_timestamp=dec_time_70)

    # 1. Snapshots at Day 70 MUST be bit-for-bit identical!
    assert torch.equal(snap_base_70.price, snap_pert_70.price)
    assert torch.equal(snap_base_70.technical, snap_pert_70.technical)
    assert torch.equal(snap_base_70.institutional, snap_pert_70.institutional)
    assert torch.equal(snap_base_70.macro, snap_pert_70.macro)
    assert torch.equal(snap_base_70.stock_context, snap_pert_70.stock_context)
    assert torch.equal(snap_base_70.event_metadata, snap_pert_70.event_metadata)
    assert torch.equal(snap_base_70.event_mask, snap_pert_70.event_mask)

    # 2. However, a snapshot at Day 85 MUST reflect the perturbed changes!
    dec_time_85 = trade_dates[85].floor("D") + pd.Timedelta(hours=20)
    snap_pert_85 = builder_pert.build_snapshot(decision_timestamp=dec_time_85)
    snap_base_85 = builder_base.build_snapshot(decision_timestamp=dec_time_85)
    assert not torch.equal(snap_base_85.price, snap_pert_85.price)
    assert not torch.equal(snap_base_85.macro, snap_pert_85.macro)


# ==============================================================================
# P0-2: ProvenanceRecord Fail-Closed Defaults
# ==============================================================================

def test_provenance_record_fail_closed_defaults():
    """P0-2: Verify ProvenanceRecord defaults to is_audit_verified=False and missing_status=UNKNOWN."""
    rec = ProvenanceRecord(source_id="UNAUDITED_FEED")
    assert rec.is_audit_verified is False
    assert rec.missing_status == MissingStatus.UNKNOWN.value

    dec_time = pd.Timestamp("2024-01-10 20:00:00", tz="Asia/Taipei")
    # Default unverified record must return False
    assert rec.is_available_as_of(dec_time) is False


def test_provenance_record_naive_timestamp_rejection():
    """P0-2: Verify that naive timestamps raise ValueError in formal mode and are NOT silently converted to UTC."""
    naive_available = pd.Timestamp("2024-01-10 10:00:00")  # Naive!
    aware_dec = pd.Timestamp("2024-01-10 20:00:00", tz="Asia/Taipei")

    rec = ProvenanceRecord(
        source_id="NAIVE_SOURCE",
        available_at=naive_available,
        is_audit_verified=True,
        missing_status="AVAILABLE",
    )

    with pytest.raises(ValueError, match="Strict PIT Error.*must be timezone-aware"):
        rec.is_available_as_of(aware_dec, mode="formal")


def test_historical_version_selector_asof_selection():
    """P0-2: Verify HistoricalVersionSelector selects the latest verified revision available As-Of decision time."""
    t1_avail = pd.Timestamp("2024-01-10 10:00:00", tz="Asia/Taipei")
    t2_avail = pd.Timestamp("2024-01-12 14:00:00", tz="Asia/Taipei")

    rev1 = ProvenanceRecord(
        source_id="FIN_REPORT_V1",
        available_at=t1_avail,
        revision_id=1,
        is_audit_verified=True,
        missing_status="AVAILABLE",
    )
    rev2 = ProvenanceRecord(
        source_id="FIN_REPORT_V2",
        available_at=t2_avail,
        revision_id=2,
        is_audit_verified=True,
        missing_status="AVAILABLE",
    )

    # As of Day 11 (between T1 and T2): must select Revision 1
    dec_11 = pd.Timestamp("2024-01-11 20:00:00", tz="Asia/Taipei")
    selected_11 = HistoricalVersionSelector.select_as_of([rev1, rev2], decision_time=dec_11)
    assert selected_11 is rev1

    # As of Day 13 (after T2): must select Revision 2
    dec_13 = pd.Timestamp("2024-01-13 20:00:00", tz="Asia/Taipei")
    selected_13 = HistoricalVersionSelector.select_as_of([rev1, rev2], decision_time=dec_13)
    assert selected_13 is rev2

    # As of Day 9 (before T1): none available
    dec_09 = pd.Timestamp("2024-01-09 20:00:00", tz="Asia/Taipei")
    selected_09 = HistoricalVersionSelector.select_as_of([rev1, rev2], decision_time=dec_09)
    assert selected_09 is None


# ==============================================================================
# P0-3: Mandatory Foreign Source Audit
# ==============================================================================

def test_asof_join_foreign_missing_audit_col_rejected_in_formal_mode():
    """P0-3: Foreign DataFrame with available_at but lacking 'is_audit_verified' must be rejected in formal mode."""
    tw_dates = pd.date_range("2024-01-10 20:00:00", periods=3, freq="D", tz="Asia/Taipei")
    us_dates = pd.date_range("2024-01-09 17:00:00", periods=3, freq="D", tz="America/New_York")

    # Has available_at, but missing 'is_audit_verified'
    df_foreign_no_audit = pd.DataFrame({
        "available_at": us_dates,
        "sp500": [4800.0, 4820.0, 4850.0],
    })

    with pytest.raises(ValueError, match="Strict PIT Error.*missing required audit verification column 'is_audit_verified'"):
        asof_join_cross_market(tw_dates, df_foreign_no_audit, mode="formal")


def test_asof_join_foreign_unverified_records_excluded():
    """P0-3: Unverified foreign market records (is_audit_verified=False) must be excluded in formal mode."""
    tw_dates = pd.date_range("2024-01-10 20:00:00", periods=3, freq="D", tz="Asia/Taipei")
    us_dates = pd.date_range("2024-01-09 17:00:00", periods=3, freq="D", tz="America/New_York")

    df_foreign = pd.DataFrame({
        "available_at": us_dates,
        "sp500": [4800.0, 4820.0, 4850.0],
        "is_audit_verified": [True, False, True],  # Second record is unverified!
    })

    joined = asof_join_cross_market(tw_dates, df_foreign, missing_policy="none", mode="formal")
    # Day 11 decision time would have matched the second record (4820.0) if verified,
    # but since it's unverified, backward join matches the first record (4800.0)
    assert joined.iloc[1]["sp500"] == 4800.0


def test_asof_join_foreign_unreleased_revision_cannot_overwrite_history():
    """P0-3: A revision published after Taiwan decision time cannot overwrite historical As-Of value."""
    tw_dates = pd.date_range("2024-01-10 20:00:00", periods=3, freq="D", tz="Asia/Taipei")

    # Day 9 data published on Day 9 at 17:00 NY (= Day 10 06:00 Taipei)
    # Day 9 revised data published on Day 11 at 17:00 NY (= Day 12 06:00 Taipei)
    t_orig = pd.Timestamp("2024-01-09 17:00:00", tz="America/New_York")
    t_rev = pd.Timestamp("2024-01-11 17:00:00", tz="America/New_York")

    df_foreign_rev = pd.DataFrame({
        "available_at": [t_orig, t_rev],
        "sp500": [4800.0, 9999.0],  # 9999.0 is the revised value
        "is_audit_verified": [True, True],
    })

    joined = asof_join_cross_market(tw_dates, df_foreign_rev, missing_policy="none", mode="formal")

    # Decision on 2024-01-10 20:00 Taipei: strictly sees 4800.0, NOT 9999.0
    assert joined.iloc[0]["sp500"] == 4800.0
    # Decision on 2024-01-11 20:00 Taipei: t_rev is 2024-01-11 17:00 NY (= 2024-01-12 06:00 Taipei),
    # which is strictly AFTER 2024-01-11 20:00 Taipei! So it still sees 4800.0!
    assert joined.iloc[1]["sp500"] == 4800.0
    # Decision on 2024-01-12 20:00 Taipei: now t_rev has occurred, so it sees 9999.0!
    assert joined.iloc[2]["sp500"] == 9999.0


# ==============================================================================
# P1-4: Preserve Missing & Unknown Semantics
# ==============================================================================

def test_missing_status_distinguishes_actual_zero_vs_missing_vs_unknown():
    """P1-4: Verify MissingStatus cleanly separates Actual Zero from Missing and Unknown."""
    dec_time = pd.Timestamp("2024-01-10 20:00:00", tz="Asia/Taipei")
    t_avail = pd.Timestamp("2024-01-10 15:00:00", tz="Asia/Taipei")

    # 1. Actual Zero: verified observation where value is truly zero (e.g. 0 net buys)
    rec_actual_zero = ProvenanceRecord(
        source_id="TWSE",
        available_at=t_avail,
        is_audit_verified=True,
        missing_status=MissingStatus.ACTUAL_ZERO.value,
    )
    assert rec_actual_zero.is_available_as_of(dec_time) is True

    # 2. Missing: expected data was delayed or absent
    rec_missing = ProvenanceRecord(
        source_id="TWSE",
        available_at=t_avail,
        is_audit_verified=True,
        missing_status=MissingStatus.MISSING.value,
    )
    assert rec_missing.is_available_as_of(dec_time) is False

    # 3. Unknown: provenance unverified
    rec_unknown = ProvenanceRecord(
        source_id="UNKNOWN_FEED",
        available_at=t_avail,
        is_audit_verified=True,
        missing_status=MissingStatus.UNKNOWN.value,
    )
    assert rec_unknown.is_available_as_of(dec_time) is False


def test_missing_masks_preservation_and_validation():
    """P1-4: Verify missing masks are validated by TensorValidator and preserved across Normalizer.transform."""
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, trade_dates = _generate_deterministic_raw_history(100)

    # Intentionally provide no institutional data to trigger institutional missing mask
    builder = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv,
        df_inst=None,  # Missing!
        df_macro=df_macro,
        df_foreign=df_foreign,
        stock_context_records=ctx_recs,
        event_records=ev_recs,
        shares_outstanding=1e8,
        mode="formal",
    )

    dec_time = trade_dates[70].floor("D") + pd.Timedelta(hours=20)
    snap = builder.build_snapshot(decision_timestamp=dec_time)

    # Verify institutional missing mask is True everywhere
    assert snap.institutional_missing_mask is not None
    assert snap.institutional_missing_mask.all()
    assert validate_multimodal_batch(snap) is True

    # Test through FeatureNormalizer
    manifest = SplitManifest.create(
        dataset_id="test_missing_mask",
        schema_version="v8.0-original",
        train_start=trade_dates[0],
        train_end=trade_dates[80],
        val_start=trade_dates[81],
        val_end=trade_dates[90],
        test_start=trade_dates[91],
        test_end=trade_dates[99],
    )
    normalizer = FeatureNormalizer(schema_version="v8.0-original", split_manifest=manifest)
    norm_snap = normalizer.fit_transform(snap, is_train_split=True, sample_timestamps=[dec_time])

    # Missing mask must be preserved identically after normalization
    assert norm_snap.institutional_missing_mask is not None
    assert torch.equal(snap.institutional_missing_mask, norm_snap.institutional_missing_mask)
    assert validate_multimodal_batch(norm_snap, is_normalized=True) is True


# ==============================================================================
# P1-5: Content-Based Dataset Provenance
# ==============================================================================

def test_split_manifest_definition_hash_vs_content_hash():
    """P1-5: Verify SplitManifest distinguishes Definition Hash from Content Hash."""
    df_ohlcv, _, _, _, _, _, trade_dates = _generate_deterministic_raw_history(100)

    # 1. Compute raw content hash
    raw_hash_1 = SplitManifest.compute_content_hash(df_ohlcv)

    manifest_1 = SplitManifest.create(
        dataset_id="twse_dataset_v8",
        schema_version="v8.0-original",
        train_start=trade_dates[0],
        train_end=trade_dates[70],
        val_start=trade_dates[71],
        val_end=trade_dates[85],
        test_start=trade_dates[86],
        test_end=trade_dates[99],
        raw_dataset_content_hash=raw_hash_1,
    )

    # Modify raw content
    df_ohlcv_modified = df_ohlcv.copy()
    df_ohlcv_modified.iloc[10, 0] += 1.0  # Change open price by 1.0
    raw_hash_2 = SplitManifest.compute_content_hash(df_ohlcv_modified)

    # Content hash must change!
    assert raw_hash_1 != raw_hash_2

    # Manifest definition hash must be IDENTICAL because date boundaries and schema didn't change
    manifest_2 = SplitManifest.create(
        dataset_id="twse_dataset_v8",
        schema_version="v8.0-original",
        train_start=trade_dates[0],
        train_end=trade_dates[70],
        val_start=trade_dates[71],
        val_end=trade_dates[85],
        test_start=trade_dates[86],
        test_end=trade_dates[99],
        raw_dataset_content_hash=raw_hash_2,
    )
    assert manifest_1.manifest_definition_hash == manifest_2.manifest_definition_hash
    assert manifest_1.raw_dataset_content_hash != manifest_2.raw_dataset_content_hash


def test_normalizer_fitting_data_hash_verification():
    """P1-5: Normalizer rejects fitting if training batch content differs from manifest contract."""
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, trade_dates = _generate_deterministic_raw_history(100)
    builder = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv, df_inst=df_inst, df_macro=df_macro, df_foreign=df_foreign,
        stock_context_records=ctx_recs, event_records=ev_recs, shares_outstanding=1e8, mode="formal"
    )

    dec_time = trade_dates[70].floor("D") + pd.Timedelta(hours=20)
    train_batch = builder.build_snapshot(decision_timestamp=dec_time)

    # Compute expected fitting hash
    expected_fit_hash = SplitManifest.compute_batch_content_hash(train_batch)

    # Manifest with contract specifying fitting data hash
    manifest_contract = SplitManifest.create(
        dataset_id="audit_contract",
        schema_version="v8.0-original",
        train_start=trade_dates[0],
        train_end=trade_dates[75],
        val_start=trade_dates[76],
        val_end=trade_dates[85],
        test_start=trade_dates[86],
        test_end=trade_dates[99],
        normalizer_fitting_data_hash=expected_fit_hash,
    )

    normalizer = FeatureNormalizer(schema_version="v8.0-original", split_manifest=manifest_contract)
    # Fitting on the exact batch succeeds
    normalizer.fit(train_batch, is_train_split=True, sample_timestamps=[dec_time])
    assert normalizer.fitted_fitting_data_hash == expected_fit_hash

    # Now tamper with training batch data
    tampered_batch = train_batch.clone()
    tampered_batch.price[0, 0, 0] += 999.0

    normalizer_tampered = FeatureNormalizer(schema_version="v8.0-original", split_manifest=manifest_contract)
    # Tampered batch MUST raise ValueError due to content hash mismatch!
    with pytest.raises(ValueError, match="Dataset Content Hash Mismatch.*Normalizer fitting data hash"):
        normalizer_tampered.fit(tampered_batch, is_train_split=True, sample_timestamps=[dec_time])


# ==============================================================================
# Full Integration: True As-Of Snapshot -> Normalizer -> Model -> Stage 1/2/3
# ==============================================================================

def test_true_snapshot_e2e_pipeline():
    """
    True End-to-End Pipeline Verification with HistoricalSnapshotBuilder:
    Raw Data -> As-Of Snapshots -> Content Hashes -> Normalizer -> Model -> Stage 1/2/3.
    """
    df_ohlcv, df_inst, df_macro, df_foreign, ctx_recs, ev_recs, trade_dates = _generate_deterministic_raw_history(100)

    # Step 1: Historical Snapshot Builder
    builder = HistoricalSnapshotBuilder(
        df_ohlcv=df_ohlcv,
        df_inst=df_inst,
        df_macro=df_macro,
        df_foreign=df_foreign,
        stock_context_records=ctx_recs,
        event_records=ev_recs,
        shares_outstanding=1e8,
        mode="formal",
    )

    decision_timestamps = [
        trade_dates[d].floor("D") + pd.Timedelta(hours=20)
        for d in [65, 66, 67, 68]
    ]
    raw_batch = builder.build_batch(decision_timestamps)
    assert raw_batch.price.shape[0] == 4
    assert validate_multimodal_batch(raw_batch) is True

    # Step 2: Content-Based SplitManifest
    raw_hash = SplitManifest.compute_content_hash(df_ohlcv)
    feat_hash = SplitManifest.compute_batch_content_hash(raw_batch)
    manifest = SplitManifest.create(
        dataset_id="e2e_review3_pipeline",
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

    # Step 3: FeatureNormalizer with Content Hash Verification
    normalizer = FeatureNormalizer(schema_version="v8.0-original", split_manifest=manifest)
    norm_batch = normalizer.fit_transform(
        raw_batch,
        is_train_split=True,
        sample_timestamps=decision_timestamps,
    )
    assert normalizer.is_fitted is True
    assert normalizer.fitted_fitting_data_hash == feat_hash
    assert validate_multimodal_batch(norm_batch, is_normalized=True) is True

    # Step 4: Model Forward Pass
    model = StockRLFullModel()
    q_vals, preds, mem = model(norm_batch)
    assert q_vals.shape == (4, 2)
    assert "novelty_scores" in preds

    # Step 5: Stage 1 Training on Exact Normalized Batch
    stage1_hist = run_stage1_pretraining(
        model=model,
        epochs=1,
        batches=[norm_batch],
    )
    assert len(stage1_hist) == 1
    assert np.isfinite(stage1_hist[0]["loss_total"])

    # Step 6: Stage 2 RL Training
    def _slice_batch(b: MultimodalBatch, idx: int) -> MultimodalBatch:
        return MultimodalBatch(
            price=b.price[idx:idx+1],
            technical=b.technical[idx:idx+1],
            institutional=b.institutional[idx:idx+1],
            macro=b.macro[idx:idx+1],
            stock_context=b.stock_context[idx:idx+1],
            recent_event_emb=b.recent_event_emb[idx:idx+1],
            active_event_emb=b.active_event_emb[idx:idx+1],
            event_metadata=b.event_metadata[idx:idx+1],
            event_stock_rel=b.event_stock_rel[idx:idx+1],
            event_mask=b.event_mask[idx:idx+1],
            position=b.position[idx:idx+1],
            relevance_labels=b.relevance_labels[idx:idx+1] if b.relevance_labels is not None else None,
            novelty_labels=b.novelty_labels[idx:idx+1] if b.novelty_labels is not None else None,
            return_targets=b.return_targets[idx:idx+1] if b.return_targets is not None else None,
            risk_targets=b.risk_targets[idx:idx+1] if b.risk_targets is not None else None,
            relevance_mask=b.relevance_mask[idx:idx+1] if b.relevance_mask is not None else None,
            novelty_mask=b.novelty_mask[idx:idx+1] if b.novelty_mask is not None else None,
            return_mask=b.return_mask[idx:idx+1] if b.return_mask is not None else None,
            risk_mask=b.risk_mask[idx:idx+1] if b.risk_mask is not None else None,
            macro_missing_mask=b.macro_missing_mask[idx:idx+1] if b.macro_missing_mask is not None else None,
            institutional_missing_mask=b.institutional_missing_mask[idx:idx+1] if b.institutional_missing_mask is not None else None,
            stock_context_missing_mask=b.stock_context_missing_mask[idx:idx+1] if b.stock_context_missing_mask is not None else None,
            event_metadata_missing_mask=b.event_metadata_missing_mask[idx:idx+1] if b.event_metadata_missing_mask is not None else None,
            schema_version=b.schema_version,
        )

    single_batches = [_slice_batch(norm_batch, i) for i in range(4)]
    prices_fixture = df_ohlcv[["open", "high", "low", "close"]].loc[[trade_dates[d] for d in [65, 66, 67, 68]]].values

    env = PointInTimeStockEnv(
        prices_df=prices_fixture,
        multimodal_batches=single_batches,
        initial_capital=1_000_000.0,
        schema_version="v8.0-original",
    )
    agent_stage2, buffer_stage2, stage2_hist = run_stage2_rl_training(
        model=model,
        env=env,
        episodes=1,
        warmup_steps=1,
        train_steps_per_env_step=1,
        batch_size=1,
        schema_version="v8.0-original",
    )
    assert len(stage2_hist) == 1
    assert len(buffer_stage2) >= 1

    # Step 7: Stage 3 Joint Fine-Tuning
    agent_stage3, buffer_stage3, stage3_hist = run_stage3_joint_finetuning(
        model=model,
        env=env,
        episodes=1,
        aux_weight=0.2,
        batch_size=1,
        min_buffer_size=1,
        agent=agent_stage2,
        buffer=buffer_stage2,
        schema_version="v8.0-original",
    )
    assert len(stage3_hist) == 1
    assert agent_stage3 is agent_stage2
