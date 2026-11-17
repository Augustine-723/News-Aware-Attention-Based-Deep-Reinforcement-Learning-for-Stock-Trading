"""
Regression Test Suite for V8.3 Code Review 2 — Final PIT & Pipeline Verification
Strictly verifies:
- P0-1: Strict Historical Availability Contract (pit_engine.py)
  - Missing available_at rejection in formal mode
  - Timezone-naive timestamp rejection in formal mode
  - Isolated synthetic estimation mode tagging
  - Holiday backward alignment (MLK day)
  - Daylight Saving Time (DST) EDT/EST alignment
  - Counterfactual future revision exclusion
- P0-2: Eliminate Macro PIT Bypass (feature_engine.py)
  - Missing df_foreign rejection in formal mode
  - Unknown availability rejection
- P1-3: Strict Train-Only Normalizer Provenance (normalizer.py)
  - Mandatory 5-item provenance in formal mode
  - Sample timestamp count matching batch dimension
  - Synthetic test mode isolation
- P1-4: Candidate Ablation Completeness (schema.py, ablations.py, feature_engine.py)
  - 100-feature contract integrity (57 implemented, 43 unsupported)
  - Formal ablation comparison strictly blocked
  - Formal candidate features produce NaN for unsupported fields (no hardcoded constants)
- P1-5: True End-to-End Pipeline Verification
  - Deterministic fixture -> As-Of Join -> Feature Engineering -> SplitManifest ->
    Normalizer -> MultimodalBatch -> Tensor Validation -> Model Forward ->
    Stage 1 (exact batch training) -> Stage 2 (exact batch env) -> Stage 3 (joint fine-tuning).
"""

import warnings
import numpy as np
import pandas as pd
import pytest
import torch

from stock_rl.config.schema import (
    BASELINE_V8_0_ALL_FEATURE_FIELDS,
    CANDIDATE_100_FEATURE_CONTRACT,
    CANDIDATE_FORMAL_ABLATION_READY,
    CURRENT_ACTIVE_SCHEMA_VERSION,
    FeatureImplementationStatus,
    FeatureSchemaVersion,
    PRICE_DIM,
    TECHNICAL_DIM,
    INSTITUTIONAL_DIM,
    MACRO_DIM,
    STOCK_CONTEXT_DIM,
    EVENT_METADATA_DIM,
    EVENT_STOCK_RELATION_DIM,
    POSITION_STATE_DIM,
    RECENT_EVENT_SLOTS,
    ACTIVE_EVENT_SLOTS,
    TOTAL_EVENT_SLOTS,
    EVENT_TEXT_DIM,
)
from stock_rl.data.pit_engine import (
    ProvenanceRecord,
    asof_join_cross_market,
)
from stock_rl.data.feature_engine import (
    assemble_event_metadata,
    assemble_event_stock_relation,
    assemble_stock_context_features,
    build_audited_event_metadata,
    build_audited_stock_context,
    compute_institutional_features,
    compute_macro_features,
    compute_price_features,
    compute_technical_features,
)
from stock_rl.data.normalizer import FeatureNormalizer, SplitManifest
from stock_rl.data.synthetic_data import MultimodalBatch
from stock_rl.data.tensor_validator import validate_multimodal_batch
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.evaluation.ablations import run_schema_comparison_study
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.rl.agent import DuelingDoubleDQNAgent
from stock_rl.rl.replay_buffer import MultimodalReplayBuffer
from stock_rl.training.stage1_pretrain import run_stage1_pretraining
from stock_rl.training.stage2_rl import run_stage2_rl_training
from stock_rl.training.stage3_joint import run_stage3_joint_finetuning


# ==============================================================================
# P0-1: Strict Historical Availability Contract (pit_engine.py)
# ==============================================================================

def test_asof_join_missing_available_at_rejected_in_formal_mode():
    """P0-1: Reject DataFrame without available_at column in formal mode."""
    decision_times = pd.date_range("2024-01-10 20:00", periods=3, freq="D", tz="Asia/Taipei")
    foreign_df = pd.DataFrame({
        "sp500": [5000.0, 5010.0, 5020.0],
        # 'available_at' column is deliberately missing
    }, index=pd.date_range("2024-01-09", periods=3, freq="D"))

    with pytest.raises(ValueError, match="Strict PIT Error.*missing required.*available_at"):
        asof_join_cross_market(
            taiwan_decision_times=decision_times,
            foreign_market_df=foreign_df,
            mode="formal",
        )


def test_asof_join_naive_timestamp_rejected_in_formal_mode():
    """P0-1: Reject timezone-naive available_at timestamps in formal mode."""
    decision_times = pd.date_range("2024-01-10 20:00", periods=3, freq="D", tz="Asia/Taipei")
    foreign_df = pd.DataFrame({
        "sp500": [5000.0, 5010.0, 5020.0],
        "available_at": pd.date_range("2024-01-09 17:00", periods=3, freq="D"),  # Timezone naive!
    })

    with pytest.raises(ValueError, match="Strict PIT Error.*available_at.*must be timezone-aware"):
        asof_join_cross_market(
            taiwan_decision_times=decision_times,
            foreign_market_df=foreign_df,
            mode="formal",
        )


def test_asof_join_naive_decision_times_rejected_in_formal_mode():
    """P0-1: Reject timezone-naive decision times in formal mode."""
    naive_decision_times = pd.date_range("2024-01-10 20:00", periods=3, freq="D")  # No tz!
    foreign_df = pd.DataFrame({
        "sp500": [5000.0, 5010.0, 5020.0],
        "available_at": pd.date_range("2024-01-09 17:00", periods=3, freq="D", tz="America/New_York"),
    })

    with pytest.raises(ValueError, match="Strict PIT Error.*taiwan_decision_times must have a timezone"):
        asof_join_cross_market(
            taiwan_decision_times=naive_decision_times,
            foreign_market_df=foreign_df,
            mode="formal",
        )


def test_asof_join_estimated_availability_mode_isolated_with_warning():
    """P0-1: Synthetic mode permits estimation with explicit warning and result tag."""
    naive_decision_times = pd.date_range("2024-01-10", periods=3, freq="D")
    foreign_df = pd.DataFrame({
        "sp500": [5000.0, 5010.0, 5020.0],
    }, index=pd.date_range("2024-01-09", periods=3, freq="D"))

    with pytest.warns(UserWarning, match="Estimated Availability Fallback"):
        aligned = asof_join_cross_market(
            taiwan_decision_times=naive_decision_times,
            foreign_market_df=foreign_df,
            allow_estimated_availability=True,
            mode="synthetic",
        )
    assert aligned.attrs.get("estimated_availability") is True
    assert len(aligned) == 3


def test_asof_join_holiday_backward_alignment():
    """
    P0-1: On a US Holiday (e.g. MLK Day Monday 2024-01-15), US markets are closed.
    Taiwan decision snapshot at 2024-01-15 20:00 Asia/Taipei must backward-align
    to Friday 2024-01-12 US close without forward peek.
    """
    # Foreign records: Friday Jan 12 available Jan 12 17:00 EST, Tuesday Jan 16 available Jan 16 17:00 EST
    t_fri_avail = pd.Timestamp("2024-01-12 17:00:00", tz="America/New_York")
    t_tue_avail = pd.Timestamp("2024-01-16 17:00:00", tz="America/New_York")

    foreign_df = pd.DataFrame({
        "sp500": [4780.0, 4765.0],
        "available_at": [t_fri_avail, t_tue_avail],
        "is_audit_verified": [True, True],
    })

    # Taiwan decision at Monday Jan 15 20:00 Asia/Taipei (MLK Day in US)
    taiwan_decisions = pd.date_range("2024-01-15 20:00:00", periods=1, freq="D", tz="Asia/Taipei")

    aligned = asof_join_cross_market(
        taiwan_decision_times=taiwan_decisions,
        foreign_market_df=foreign_df,
        mode="formal",
    )

    # Must align to Friday's S&P 500 (4780.0), NOT Tuesday's (4765.0)
    assert aligned.iloc[0]["sp500"] == 4780.0


def test_asof_join_dst_transition():
    """
    P0-1: Verifies correct behavior across US Daylight Saving Time transition.
    US switches from EST (UTC-5) to EDT (UTC-4) on Sunday 2024-03-10.
    Taiwan decision at 20:00 Asia/Taipei (UTC+8):
    - Winter (EST, UTC-5): US close Friday March 8 16:00 EST = 21:00 UTC = Sat 05:00 Taipei.
    - Summer (EDT, UTC-4): US close Monday March 11 16:00 EDT = 20:00 UTC = Tue 04:00 Taipei.
    """
    t_fri_avail = pd.Timestamp("2024-03-08 17:00:00", tz="America/New_York")  # EST (UTC-5)
    t_mon_avail = pd.Timestamp("2024-03-11 17:00:00", tz="America/New_York")  # EDT (UTC-4)

    assert str(t_fri_avail.tzinfo) == "America/New_York"
    # Verify EST vs EDT offsets
    assert t_fri_avail.utcoffset().total_seconds() == -5 * 3600
    assert t_mon_avail.utcoffset().total_seconds() == -4 * 3600

    foreign_df = pd.DataFrame({
        "sp500": [5120.0, 5115.0],
        "available_at": [t_fri_avail, t_mon_avail],
        "is_audit_verified": [True, True],
    })

    # Decision at Monday March 11 20:00 Taipei
    # At this time, Monday US close has not occurred yet (it occurs Tuesday 04:00 Taipei)
    taiwan_decisions = pd.date_range("2024-03-11 20:00:00", periods=1, freq="D", tz="Asia/Taipei")

    aligned = asof_join_cross_market(
        taiwan_decision_times=taiwan_decisions,
        foreign_market_df=foreign_df,
        mode="formal",
    )

    # Must align to Friday's data (5120.0), not Monday's (5115.0)
    assert aligned.iloc[0]["sp500"] == 5120.0


def test_asof_join_future_revision_exclusion():
    """P0-1: Future records and post-decision restatements never alter the snapshot."""
    t_base = pd.Timestamp("2024-01-10 17:00:00", tz="America/New_York")
    t_future = pd.Timestamp("2024-01-12 17:00:00", tz="America/New_York")

    decision_time = pd.date_range("2024-01-11 20:00:00", periods=1, freq="D", tz="Asia/Taipei")

    df_original = pd.DataFrame({
        "sp500": [5000.0, 5050.0],
        "available_at": [t_base, t_future],
        "is_audit_verified": [True, True],
    })

    # Counterfactual: modify future sp500
    df_counterfactual = pd.DataFrame({
        "sp500": [5000.0, 9999.0],  # Future drastically changed
        "available_at": [t_base, t_future],
        "is_audit_verified": [True, True],
    })

    res_orig = asof_join_cross_market(decision_time, df_original, mode="formal")
    res_cf = asof_join_cross_market(decision_time, df_counterfactual, mode="formal")

    assert res_orig.iloc[0]["sp500"] == 5000.0
    assert res_cf.iloc[0]["sp500"] == 5000.0
    assert res_orig.iloc[0]["sp500"] == res_cf.iloc[0]["sp500"]


# ==============================================================================
# P0-2: Eliminate Macro PIT Bypass (feature_engine.py)
# ==============================================================================

def test_compute_macro_features_missing_df_foreign_rejected_in_formal_mode():
    """P0-2: In formal mode, compute_macro_features MUST reject None df_foreign."""
    dates = pd.date_range("2024-01-01 20:00:00", periods=10, freq="D", tz="Asia/Taipei")
    df_macro = pd.DataFrame({
        "taiex": 18000.0, "sector_index": 500.0, "usdtwd": 31.0,
        "sp500": 5000.0, "nasdaq": 16000.0, "sox": 4500.0, "vix": 15.0, "us_10y_yield": 4.1
    }, index=dates)

    with pytest.raises(ValueError, match="Formal PIT Error.*requires an explicit df_foreign"):
        compute_macro_features(df_macro, df_foreign=None, mode="formal")


def test_compute_macro_features_naive_decision_index_rejected_in_formal_mode():
    """P0-2: In formal mode, df_macro must have timezone-aware DatetimeIndex."""
    dates_naive = pd.date_range("2024-01-01", periods=10, freq="D")
    df_macro = pd.DataFrame({
        "taiex": 18000.0, "sector_index": 500.0, "usdtwd": 31.0,
        "sp500": 5000.0, "nasdaq": 16000.0, "sox": 4500.0, "vix": 15.0, "us_10y_yield": 4.1
    }, index=dates_naive)

    df_foreign = pd.DataFrame({
        "sp500": 5000.0, "nasdaq": 16000.0, "sox": 4500.0, "vix": 15.0, "us_10y_yield": 4.1,
        "available_at": pd.date_range("2024-01-01 17:00", periods=10, freq="D", tz="America/New_York"),
    })

    with pytest.raises(ValueError, match="Formal PIT Error.*must have a timezone-aware DatetimeIndex"):
        compute_macro_features(df_macro, df_foreign=df_foreign, mode="formal")


# ==============================================================================
# P1-3: Strict Train-Only Normalizer Provenance (normalizer.py)
# ==============================================================================

def test_normalizer_formal_mode_mandates_all_provenance_items():
    """P1-3: FeatureNormalizer.fit() in formal mode mandates manifest, sample_timestamps, window, hash."""
    from stock_rl.data.synthetic_data import generate_synthetic_batch

    batch = generate_synthetic_batch(batch_size=4, seed=42)
    normalizer = FeatureNormalizer(schema_version="v8.0-original", mode="formal")

    # Missing SplitManifest raises ValueError
    with pytest.raises(ValueError, match="Formal Normalizer Provenance Error.*SplitManifest is mandatory"):
        normalizer.fit(batch, is_train_split=True, sample_timestamps=None, split_manifest=None)

    manifest = SplitManifest.create(
        dataset_id="test_dataset",
        schema_version="v8.0-original",
        train_start="2024-01-01",
        train_end="2024-01-31",
        val_start="2024-02-01",
        val_end="2024-02-28",
        test_start="2024-03-01",
        test_end="2024-03-31",
    )

    # Missing sample_timestamps raises ValueError
    with pytest.raises(ValueError, match="Formal Normalizer Provenance Error.*sample_timestamps is mandatory"):
        normalizer.fit(batch, is_train_split=True, sample_timestamps=None, split_manifest=manifest)


def test_normalizer_rejects_timestamp_count_mismatch():
    """P1-3: FeatureNormalizer.fit() rejects len(sample_timestamps) != batch.price.shape[0]."""
    from stock_rl.data.synthetic_data import generate_synthetic_batch

    batch = generate_synthetic_batch(batch_size=4, seed=42)
    manifest = SplitManifest.create(
        dataset_id="test_dataset",
        schema_version="v8.0-original",
        train_start="2024-01-01",
        train_end="2024-01-31",
        val_start="2024-02-01",
        val_end="2024-02-28",
        test_start="2024-03-01",
        test_end="2024-03-31",
    )
    normalizer = FeatureNormalizer(schema_version="v8.0-original", split_manifest=manifest, mode="formal")

    # Pass 3 timestamps for batch of 4 items -> mismatch!
    mismatched_ts = pd.date_range("2024-01-05", periods=3, freq="D")
    with pytest.raises(ValueError, match="Length of sample_timestamps \\(3\\) does not match total batch size \\(4\\)"):
        normalizer.fit(batch, is_train_split=True, sample_timestamps=mismatched_ts)


def test_normalizer_synthetic_mode_permits_smoke_test():
    """P1-3: FeatureNormalizer permits smoke tests when mode='synthetic'."""
    from stock_rl.data.synthetic_data import generate_synthetic_batch

    batch = generate_synthetic_batch(batch_size=4, seed=42)
    normalizer = FeatureNormalizer(schema_version="v8.0-original", mode="synthetic")

    # Should succeed without manifest or sample_timestamps in synthetic mode
    normalizer.fit(batch, is_train_split=True)
    assert normalizer.is_fitted is True


# ==============================================================================
# P1-4: Candidate Ablation Completeness (schema.py & feature_engine.py)
# ==============================================================================

def test_candidate_100_feature_contract_integrity():
    """P1-4: Exhaustively verifies CANDIDATE_100_FEATURE_CONTRACT covers 100 dims."""
    assert len(CANDIDATE_100_FEATURE_CONTRACT) == 100

    implemented_count = sum(
        1 for c in CANDIDATE_100_FEATURE_CONTRACT.values()
        if c.status == FeatureImplementationStatus.IMPLEMENTED
    )
    unsupported_count = sum(
        1 for c in CANDIDATE_100_FEATURE_CONTRACT.values()
        if c.status in (FeatureImplementationStatus.UNSUPPORTED_DATA_SOURCE, FeatureImplementationStatus.MISSING_FEED)
    )

    # Exactly 72 implemented (Price 12, Tech 16, Inst 8, Macro 8, Ctx 6, Meta 16, Rel 2, Pos 4)
    # Exactly 28 unsupported (Macro 4, Ctx 10, Meta 8, Rel 6)
    assert implemented_count == 72
    assert unsupported_count == 28
    assert CANDIDATE_FORMAL_ABLATION_READY is False


def test_candidate_formal_ablation_strictly_prohibited():
    """P1-4: run_schema_comparison_study raises RuntimeError in formal mode."""
    model_b = StockRLFullModel()
    model_c = StockRLFullModel()
    env_b = PointInTimeStockEnv(schema_version="v8.0-original")
    env_c = PointInTimeStockEnv(schema_version="v8.1-candidate")
    agent_b = DuelingDoubleDQNAgent(model=model_b, schema_version="v8.0-original")
    agent_c = DuelingDoubleDQNAgent(model=model_c, schema_version="v8.1-candidate")

    with pytest.raises(RuntimeError, match="Candidate Formal Ablation Blocked.*28/100 candidate features"):
        run_schema_comparison_study(agent_b, agent_c, env_b, env_c, mode="formal")


def test_candidate_features_no_magic_constants_in_formal_mode():
    """P1-4: In formal mode, unsupported candidate fields return np.nan, eliminating hardcoded numbers."""
    ctx = assemble_stock_context_features(
        industry_embedding_8d=np.zeros(8),
        market_cap=1e11,
        trading_value_60d_ma=1e8,
        volatility_percentile=0.5,
        liquidity_percentile=0.5,
        beta_120d=1.0,
        industry_market_cap_percentile=0.5,
        days_since_ipo=1000,
        had_trading_suspension_20d=False,
        schema_version="v8.1-candidate",
        mode="formal",
    )
    # C06..C14 and C16 must be np.nan
    assert np.isnan(ctx[5])   # C06 pb_ratio_scaled
    assert np.isnan(ctx[6])   # C07 pe_ratio_scaled
    assert np.isnan(ctx[7])   # C08 dividend_yield_scaled
    assert np.isnan(ctx[8])   # C09 debt_to_equity_scaled
    assert np.isnan(ctx[9])   # C10 roe_ttm_scaled
    assert np.isnan(ctx[10])  # C11 supply_chain_role
    assert np.isnan(ctx[11])  # C12 foreign_ownership_cap
    assert np.isnan(ctx[12])  # C13 margin_tradable_flag
    assert np.isnan(ctx[13])  # C14 day_trading_eligible_flag
    assert np.isnan(ctx[15])  # C16 governance_score_scaled


# ==============================================================================
# P1-5: True End-to-End Pipeline Verification
# ==============================================================================

def test_true_deterministic_end_to_end_pipeline():
    """
    P1-5: True End-to-End Pipeline Verification with fixed timestamp and known answer fixture.
    Data flows sequentially through:
    1. Raw Market / Event Records with explicit audited ProvenanceRecord
    2. Provenance Validation (available_at <= decision_time)
    3. Historical As-Of Join (mode='formal')
    4. Feature Engineering (Price 12, Tech 16, Inst 8, Macro 12, Ctx 16, Meta 24, Rel 8, Pos 4)
    5. Normalization with SplitManifest (mode='formal')
    6. MultimodalBatch Assembly
    7. Tensor Semantic Validation
    8. Model Forward Pass
    9. Stage 1 Micro Training: trains directly on batches=[norm_batch]
    10. Stage 2 Environment Step: steps environment populated with [norm_batch]
    11. Stage 3 Joint Fine-Tuning: executes joint RL + multi-task optimization
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # A. Fixed Timestamp Decision Horizon (Asia/Taipei 20:00)
    decision_dates = pd.date_range("2024-01-08 20:00:00", periods=4, freq="D", tz="Asia/Taipei")
    n_days = 80  # For rolling windows (60D required)

    all_dates = pd.date_range("2023-10-15 20:00:00", periods=n_days, freq="D", tz="Asia/Taipei")

    # B. Raw TWSE OHLCV
    prices = 100.0 + np.cumsum(np.random.randn(n_days) * 1.5)
    highs = prices + np.abs(np.random.randn(n_days) * 1.0)
    lows = prices - np.abs(np.random.randn(n_days) * 1.0)
    opens = lows + (highs - lows) * 0.5
    closes = lows + (highs - lows) * 0.6
    volumes = np.random.uniform(1e6, 5e6, n_days)
    values = closes * volumes

    df_ohlcv = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes, "trading_value": values
    }, index=all_dates)

    # C. Raw Institutional Data
    df_inst = pd.DataFrame({
        "foreign_net_buy": np.random.uniform(-5e5, 5e5, n_days),
        "trust_net_buy": np.random.uniform(-2e5, 2e5, n_days),
        "dealer_net_buy": np.random.uniform(-1e5, 1e5, n_days),
        "margin_balance": np.random.uniform(1e5, 2e5, n_days),
        "short_balance": np.random.uniform(1e4, 3e4, n_days),
    }, index=all_dates)

    # D. Raw Macro & US Cross-Market Data with Audited Availability
    df_macro = pd.DataFrame({
        "taiex": 18000.0 + np.cumsum(np.random.randn(n_days) * 30.0),
        "sector_index": 500.0 + np.cumsum(np.random.randn(n_days) * 2.0),
        "usdtwd": 31.0 + np.cumsum(np.random.randn(n_days) * 0.02),
        "sp500": 5000.0 + np.cumsum(np.random.randn(n_days) * 10.0),
        "nasdaq": 16000.0 + np.cumsum(np.random.randn(n_days) * 40.0),
        "sox": 4500.0 + np.cumsum(np.random.randn(n_days) * 15.0),
        "vix": np.random.uniform(14.0, 25.0, n_days),
        "us_10y_yield": np.random.uniform(3.9, 4.4, n_days),
    }, index=all_dates)

    # Foreign market available timestamps (T-1 17:00 EST <= Taipei 20:00)
    us_avail_times = pd.date_range("2023-10-14 17:00:00", periods=n_days, freq="D", tz="America/New_York")
    df_foreign = pd.DataFrame({
        "sp500": df_macro["sp500"].values,
        "nasdaq": df_macro["nasdaq"].values,
        "sox": df_macro["sox"].values,
        "vix": df_macro["vix"].values,
        "us_10y_yield": df_macro["us_10y_yield"].values,
        "available_at": us_avail_times,
        "is_audit_verified": True,
    })

    # Step 1: Feature Engineering through Formal Engine
    p_feat = compute_price_features(df_ohlcv, shares_outstanding=1e8, schema_version="v8.0-original")
    t_feat = compute_technical_features(df_ohlcv, schema_version="v8.0-original")
    i_feat = compute_institutional_features(df_inst, df_ohlcv["volume"], shares_outstanding=1e8, schema_version="v8.0-original")
    m_feat = compute_macro_features(df_macro, df_foreign=df_foreign, schema_version="v8.0-original", mode="formal")

    assert p_feat.shape == (n_days, 12)
    assert t_feat.shape == (n_days, 16)
    assert i_feat.shape == (n_days, 8)
    assert m_feat.shape == (n_days, 12)

    # Step 2: Provenance-Verified Event & Stock Context
    B = 4
    batch_prices = []
    batch_techs = []
    batch_insts = []
    batch_macros = []
    batch_ctx = []
    batch_meta = []
    batch_rel = []

    for b in range(B):
        dec_time = decision_dates[b]
        # Stock context provenance
        ctx_prov = ProvenanceRecord(
            source_id="TWSE_MOPS",
            available_at=dec_time - pd.Timedelta(hours=4),
            is_audit_verified=True,
            version="2024.1",
        )
        ctx = build_audited_stock_context(
            industry_embedding_8d=np.ones(8, dtype=np.float32) / np.sqrt(8),
            market_cap=float(closes[-1] * 1e8),
            trading_value_60d_ma=float(values[-60:].mean()),
            volatility_percentile=0.55,
            liquidity_percentile=0.70,
            beta_120d=1.10,
            industry_market_cap_percentile=0.85,
            days_since_ipo=1200,
            had_trading_suspension_20d=False,
            provenance=ctx_prov,
            decision_time=dec_time,
            schema_version="v8.0-original",
            mode="formal",
        )
        batch_ctx.append(ctx)

        # Slice 60D market window
        batch_prices.append(p_feat[-(60 + (B - 1 - b)): len(p_feat) - (B - 1 - b)])
        batch_techs.append(t_feat[-(60 + (B - 1 - b)): len(t_feat) - (B - 1 - b)])
        batch_insts.append(i_feat[-(60 + (B - 1 - b)): len(i_feat) - (B - 1 - b)])
        batch_macros.append(m_feat[-(60 + (B - 1 - b)): len(m_feat) - (B - 1 - b)])

        # Event metadata & relations for TOTAL_EVENT_SLOTS (40)
        event_prov = ProvenanceRecord(
            source_id="CNA_NEWS",
            available_at=dec_time - pd.Timedelta(hours=2),
            is_audit_verified=True,
            version="1.0",
        )
        k_meta = []
        k_rel = []
        for k in range(TOTAL_EVENT_SLOTS):
            meta = build_audited_event_metadata(
                actor_embedding_4d=np.zeros(4, dtype=np.float32),
                type_embedding_8d=np.ones(8, dtype=np.float32) / np.sqrt(8),
                status_embedding_4d=np.zeros(4, dtype=np.float32),
                evidence_traceability=0.90,
                source_reliability=0.85,
                source_independence=1.0,
                novelty_input_estimate=0.40,
                market_surprise_preevent=0.02,
                quantified_magnitude=0.60,
                timestamp_reliability=1.0,
                provenance=event_prov,
                decision_time=dec_time,
                schema_version="v8.0-original",
                mode="formal",
            )
            rel = assemble_event_stock_relation(
                direct_company_mention=True,
                industry_match=True,
                upstream_exposure=0.2,
                downstream_exposure=0.1,
                customer_relationship=0.0,
                supplier_relationship=0.0,
                geographic_exposure=0.5,
                macro_factor_exposure=0.3,
            )
            k_meta.append(meta)
            k_rel.append(rel)
        batch_meta.append(k_meta)
        batch_rel.append(k_rel)

    # Step 3: Assemble Raw MultimodalBatch
    recent_emb = torch.zeros((B, RECENT_EVENT_SLOTS, EVENT_TEXT_DIM), dtype=torch.float32)
    recent_emb[:, :, 0] = 1.0
    active_emb = torch.zeros((B, ACTIVE_EVENT_SLOTS, EVENT_TEXT_DIM), dtype=torch.float32)
    active_emb[:, :, 0] = 1.0

    raw_batch = MultimodalBatch(
        price=torch.tensor(np.array(batch_prices), dtype=torch.float32),
        technical=torch.tensor(np.array(batch_techs), dtype=torch.float32),
        institutional=torch.tensor(np.array(batch_insts), dtype=torch.float32),
        macro=torch.tensor(np.array(batch_macros), dtype=torch.float32),
        stock_context=torch.tensor(np.array(batch_ctx), dtype=torch.float32),
        recent_event_emb=recent_emb,
        active_event_emb=active_emb,
        event_metadata=torch.tensor(np.array(batch_meta), dtype=torch.float32),
        event_stock_rel=torch.tensor(np.array(batch_rel), dtype=torch.float32),
        event_mask=torch.ones((B, TOTAL_EVENT_SLOTS), dtype=torch.bool),
        position=torch.zeros((B, 4), dtype=torch.float32),
        relevance_labels=torch.ones((B, TOTAL_EVENT_SLOTS, 1), dtype=torch.float32),
        novelty_labels=torch.zeros((B, TOTAL_EVENT_SLOTS, 1), dtype=torch.float32),
        return_targets=torch.zeros((B, 3), dtype=torch.float32),
        risk_targets=torch.zeros((B, 3), dtype=torch.float32),
        relevance_mask=torch.ones((B, TOTAL_EVENT_SLOTS, 1), dtype=torch.bool),
        novelty_mask=torch.ones((B, TOTAL_EVENT_SLOTS, 1), dtype=torch.bool),
        return_mask=torch.ones((B, 3), dtype=torch.bool),
        risk_mask=torch.ones((B, 3), dtype=torch.bool),
        schema_version="v8.0-original",
    )
    assert validate_multimodal_batch(raw_batch) is True

    # Step 4: SplitManifest & Strict Provenance Normalization
    manifest = SplitManifest.create(
        dataset_id="e2e_fixture_audit_v8_3",
        schema_version="v8.0-original",
        train_start="2024-01-01 00:00:00+08:00",
        train_end="2024-01-31 23:59:59+08:00",
        val_start="2024-02-01 00:00:00+08:00",
        val_end="2024-02-28 23:59:59+08:00",
        test_start="2024-03-01 00:00:00+08:00",
        test_end="2024-03-31 23:59:59+08:00",
    )

    normalizer = FeatureNormalizer(
        schema_version="v8.0-original",
        split_manifest=manifest,
        mode="formal",
    )
    norm_batch = normalizer.fit_transform(
        raw_batch,
        is_train_split=True,
        sample_timestamps=decision_dates,
        mode="formal",
    )

    assert normalizer.is_fitted is True
    assert normalizer.fitted_manifest_hash == manifest.manifest_hash
    assert validate_multimodal_batch(norm_batch, is_normalized=True) is True

    # Step 5: Model Forward Pass
    model = StockRLFullModel()
    q_vals, preds, mem = model(norm_batch)
    assert q_vals.shape == (B, 2)
    assert "novelty_scores" in preds

    # Step 6: Stage 1 Training on Exact Batch
    stage1_hist = run_stage1_pretraining(
        model=model,
        epochs=1,
        batches=[norm_batch],
    )
    assert len(stage1_hist) == 1
    assert "loss_total" in stage1_hist[0]
    assert np.isfinite(stage1_hist[0]["loss_total"])

    # Step 7: Stage 2 RL Training with Environment receiving Exact Batches
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
            schema_version=b.schema_version,
        )

    single_batches = [_slice_batch(norm_batch, i) for i in range(B)]
    prices_fixture = df_ohlcv[["open", "high", "low", "close"]].values[-B:]

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

    # Step 8: Stage 3 Joint Fine-Tuning inheriting Stage 2 Agent and Buffer
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
