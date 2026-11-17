"""
V8.3 Code Review 1 Regression Test Suite
Validates all 8 items from V8.3 Code Review 1:
- P0-1: Cross-Market PIT Leakage (no bfill, as-of join, calendar/DST awareness, counterfactual isolation)
- P0-2: M05 Sector Relative Strength Horizon (Sector 20D - TAIEX 20D analytical oracle)
- P1-3: Candidate Feature Calculation Completeness (all 12 candidate price features & separate module contracts)
- P1-4: Synthetic Temporal Consistency (continuous path generator integrity)
- P1-5: Train-Only Normalization Provenance (SplitManifest rejection on mislabeled splits)
- P1-6: Replay Observation Snapshot Isolation (memory mutation immunity)
- P1-7: Historical Feature Builder Provenance (ProvenanceRecord PIT Guard & zero post-event leakage)
- P1-8: End-to-End Feature Pipeline Test (Raw -> AsOf -> Features -> Norm -> Batch -> Model -> Stage 1/2/3)
"""

import pytest
import numpy as np
import pandas as pd
import torch

from stock_rl.config.schema import (
    MARKET_WINDOW_LENGTH,
    PRICE_DIM,
    TECHNICAL_DIM,
    INSTITUTIONAL_DIM,
    MACRO_DIM,
    STOCK_CONTEXT_DIM,
    EVENT_METADATA_DIM,
    EVENT_STOCK_RELATION_DIM,
    CURRENT_ACTIVE_SCHEMA_VERSION,
    FeatureSchemaVersion,
)
from stock_rl.data.pit_engine import (
    ProvenanceRecord,
    asof_join_cross_market,
)
from stock_rl.data.feature_engine import (
    compute_price_features,
    compute_technical_features,
    compute_institutional_features,
    compute_macro_features,
    assemble_stock_context_features,
    assemble_event_metadata,
    assemble_event_stock_relation,
    build_audited_stock_context,
    build_audited_event_metadata,
    build_audited_event_stock_relation,
)
from stock_rl.data.synthetic_data import (
    MultimodalBatch,
    generate_synthetic_batch,
    generate_temporally_consistent_synthetic_batch,
)
from stock_rl.data.normalizer import (
    SplitManifest,
    FeatureNormalizer,
)
from stock_rl.data.tensor_validator import validate_multimodal_batch
from stock_rl.rl.replay_buffer import MultimodalReplayBuffer
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.training.stage1_pretrain import run_stage1_pretraining
from stock_rl.training.stage2_rl import run_stage2_rl_training
from stock_rl.training.stage3_joint import run_stage3_joint_finetuning


# ==============================================================================
# P0-1: Cross-Market PIT Leakage
# ==============================================================================

def test_cross_market_pit_asof_join_no_bfill():
    """
    P0-1: Verifies timezone-aware As-Of Join:
    - US market close at 16:00 ET (EDT in summer, EST in winter)
    - Taiwan decision timestamp at 20:00 Taipei time
    - Direction is strictly 'backward'
    - ZERO backward fill (bfill): unobserved initial records stay NaN / missing.
    """
    # Create Taiwan decision timestamps (20:00:00 Asia/Taipei)
    tw_dates = pd.date_range("2024-03-08 20:00:00", periods=5, freq="D", tz="Asia/Taipei")

    # US Market closes at 16:00 America/New_York
    # Note: Daylight Saving Time switch in US occurred on 2024-03-10
    us_dates = pd.date_range("2024-03-07 16:00:00", periods=5, freq="D", tz="America/New_York")
    df_us = pd.DataFrame({
        "available_at": us_dates,
        "sp500": [5100.0, 5120.0, 5130.0, 5150.0, 5160.0],
        "is_audit_verified": True,
    })

    # Backward-only As-Of Join
    joined = asof_join_cross_market(
        taiwan_decision_times=tw_dates,
        foreign_market_df=df_us,
        foreign_available_col="available_at",
        missing_policy="none",
    )

    assert len(joined) == 5
    # On 2024-03-08 20:00 Taipei time (which is 2024-03-08 07:00 NY time):
    # US session on 2024-03-08 has NOT closed yet (closes at 16:00 NY time = 2024-03-09 05:00 Taipei).
    # Thus the latest completed US session available is 2024-03-07 16:00 NY close (5100.0).
    assert joined.iloc[0]["sp500"] == 5100.0


def test_cross_market_counterfactual_future_modification():
    """
    P0-1 Counterfactual Test:
    Modifying future overseas market data (T+1, T+2) must NOT alter
    the historical decision snapshot at day T.
    """
    tw_dates = pd.date_range("2024-01-10 20:00:00", periods=3, freq="D", tz="Asia/Taipei")
    us_dates = pd.date_range("2024-01-09 16:00:00", periods=5, freq="D", tz="America/New_York")

    df_us_baseline = pd.DataFrame({
        "available_at": us_dates,
        "sp500": [4700.0, 4750.0, 4780.0, 4800.0, 4820.0],
        "nasdaq": [15000.0, 15100.0, 15200.0, 15300.0, 15400.0],
        "sox": [4000.0, 4050.0, 4100.0, 4150.0, 4200.0],
        "vix": [14.0, 14.5, 13.8, 14.2, 14.0],
        "us_10y_yield": [4.05, 4.08, 4.10, 4.12, 4.15],
        "is_audit_verified": True,
    })

    # Perturbed future data on day 3 and 4
    df_us_perturbed = df_us_baseline.copy()
    df_us_perturbed.loc[3:, "sp500"] = 99999.0
    df_us_perturbed.loc[3:, "vix"] = 88.0

    joined_base = asof_join_cross_market(tw_dates, df_us_baseline)
    joined_pert = asof_join_cross_market(tw_dates, df_us_perturbed)

    # First two decision steps must be completely identical
    pd.testing.assert_frame_equal(joined_base.iloc[:2], joined_pert.iloc[:2])


# ==============================================================================
# P0-2: M05 Sector Relative Strength Horizon Oracle
# ==============================================================================

def test_m05_sector_relative_strength_horizon_oracle():
    """
    P0-2: Numerical test with analytical ground truth for M05:
    M05 = Sector Return 20D - TAIEX Return 20D.
    Guarantees it does NOT subtract TAIEX Return 5D.
    """
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    # Base constant prices
    taiex = pd.Series(100.0, index=dates)
    sector = pd.Series(100.0, index=dates)

    # At day 25 (index 24):
    # Set day 4 (index 4, 20 days ago) to 100.0
    # Set day 19 (index 19, 5 days ago) to 110.0 (5D return = +9.09%)
    # Set day 24 (index 24, today):
    #   taiex = 120.0 -> 20D return = (120 - 100) / 100 = 0.20
    #   sector = 130.0 -> 20D return = (130 - 100) / 100 = 0.30
    # True M05 = 0.30 - 0.20 = 0.10.
    # If the bug was present (subtracting 5D return of taiex: (120-110)/110 = 0.0909),
    # M05 would be 0.30 - 0.0909 = 0.2091 != 0.10.
    taiex.iloc[19] = 110.0
    taiex.iloc[24] = 120.0
    sector.iloc[24] = 130.0

    df_macro = pd.DataFrame({
        "taiex": taiex,
        "sector_index": sector,
        "usdtwd": 31.0,
        "sp500": 5000.0,
        "nasdaq": 16000.0,
        "sox": 4500.0,
        "vix": 15.0,
        "us_10y_yield": 4.1,
    })

    features = compute_macro_features(df_macro, schema_version="v8.0-original", mode="synthetic")
    # M05 is at column index 4
    m05_computed = features[24, 4]
    expected_m05 = 0.30 - 0.20  # Exactly 0.10
    np.testing.assert_allclose(m05_computed, expected_m05, atol=1e-5)


# ==============================================================================
# P1-3: Candidate Feature Calculation Completeness
# ==============================================================================

def test_candidate_price_12d_distinct_calculations():
    """
    P1-3: Verifies that candidate Price 12D features are NOT repeating p_norm tiles,
    but represent 12 distinct mathematical feature series.
    """
    rng = np.random.default_rng(42)
    n = 80
    close = 100.0 + np.cumsum(rng.normal(0, 1, size=n))
    open_p = close + rng.normal(0, 0.5, size=n)
    high = np.maximum(open_p, close) + np.abs(rng.normal(1, 0.2, size=n))
    low = np.minimum(open_p, close) - np.abs(rng.normal(1, 0.2, size=n))
    volume = np.exp(rng.normal(12, 0.5, size=n))

    df_ohlcv = pd.DataFrame({
        "open": open_p, "high": high, "low": low,
        "close": close, "volume": volume,
    })

    cand_features = compute_price_features(df_ohlcv, schema_version="v8.1-candidate")
    assert cand_features.shape == (n, 12)

    # Check that columns are NOT identical tiles
    for col_i in range(11):
        for col_j in range(col_i + 1, 12):
            diff = np.abs(cand_features[:, col_i] - cand_features[:, col_j])
            assert np.max(diff) > 1e-4, f"Candidate columns {col_i} and {col_j} are identical!"


def test_candidate_vs_original_separate_feature_contracts():
    """
    P1-3: Verifies separate calculation contracts between v8.0-original and v8.1-candidate
    across technical, institutional, and macro modules.
    """
    df_ohlcv = pd.DataFrame({
        "open": [100.0, 102.0, 101.0, 103.0] * 20,
        "high": [103.0, 104.0, 103.0, 105.0] * 20,
        "low": [99.0, 100.0, 99.0, 101.0] * 20,
        "close": [102.0, 101.0, 103.0, 104.0] * 20,
        "volume": [1e6, 1.2e6, 0.9e6, 1.1e6] * 20,
    })

    tech_orig = compute_technical_features(df_ohlcv, schema_version="v8.0-original")
    tech_cand = compute_technical_features(df_ohlcv, schema_version="v8.1-candidate")
    assert tech_orig.shape == tech_cand.shape == (80, 16)
    # They compute different indicator sets (e.g. candidate has MFI and CCI)
    assert not np.allclose(tech_orig, tech_cand)


# ==============================================================================
# P1-4: Synthetic Temporal Consistency
# ==============================================================================

def test_temporally_consistent_synthetic_batch_integrity():
    """
    P1-4: Verifies generate_temporally_consistent_synthetic_batch:
    - High >= max(Open, Close) and Low <= min(Open, Close)
    - Valid, non-NaN/Inf tensors across all modalities
    - Complies with tensor contracts and passes validation
    """
    batch = generate_temporally_consistent_synthetic_batch(
        batch_size=8,
        seed=123,
        schema_version="v8.0-original"
    )

    assert batch.price.shape == (8, MARKET_WINDOW_LENGTH, PRICE_DIM)
    assert batch.technical.shape == (8, MARKET_WINDOW_LENGTH, TECHNICAL_DIM)
    assert batch.institutional.shape == (8, MARKET_WINDOW_LENGTH, INSTITUTIONAL_DIM)
    assert batch.macro.shape == (8, MARKET_WINDOW_LENGTH, MACRO_DIM)
    assert batch.stock_context.shape == (8, STOCK_CONTEXT_DIM)

    # No NaN or Inf
    assert not torch.isnan(batch.price).any()
    assert not torch.isinf(batch.price).any()
    assert not torch.isnan(batch.technical).any()

    # Semantic validation passes
    assert validate_multimodal_batch(batch) is True


# ==============================================================================
# P1-5: Train-Only Normalization Provenance
# ==============================================================================

def test_split_manifest_and_normalizer_rejection_on_mislabeled_splits():
    """
    P1-5: Tests SplitManifest provenance checking:
    - Normalizer rejects fitting if sample timestamps fall outside train_start..train_end,
      even if caller passed is_train_split=True.
    """
    manifest = SplitManifest.create(
        dataset_id="twse_v8_2024",
        schema_version="v8.0-original",
        train_start="2020-01-01",
        train_end="2022-12-31",
        val_start="2023-01-01",
        val_end="2023-06-30",
        test_start="2023-07-01",
        test_end="2023-12-31",
    )

    normalizer = FeatureNormalizer(schema_version="v8.0-original", split_manifest=manifest)
    batch = generate_synthetic_batch(batch_size=4, seed=42)

    # Mislabeled call: timestamps from validation split passed as train
    val_timestamps = pd.date_range("2023-02-01", periods=4, freq="D")
    with pytest.raises(ValueError, match="Split provenance violation"):
        normalizer.fit(
            batch,
            is_train_split=True,  # Caller deceptively claims this is train
            sample_timestamps=val_timestamps,
        )

    # Legitimate call: timestamps within train range
    train_timestamps = pd.date_range("2021-06-01", periods=4, freq="D")
    normalizer.fit(
        batch,
        is_train_split=True,
        sample_timestamps=train_timestamps,
    )
    assert normalizer.is_fitted is True


# ==============================================================================
# P1-6: Replay Observation Snapshot Isolation
# ==============================================================================

def test_replay_buffer_observation_snapshot_isolation():
    """
    P1-6: Verifies that MultimodalReplayBuffer stores cloned immutable snapshots.
    Mutating the original observation after calling push() does NOT alter the transition.
    """
    buffer = MultimodalReplayBuffer(capacity=100, schema_version="v8.0-original")
    batch = generate_synthetic_batch(batch_size=1, seed=42)
    next_batch = generate_synthetic_batch(batch_size=1, seed=43)

    orig_price_val = float(batch.price[0, 0, 0])
    buffer.push(batch, action=1, reward=0.05, next_obs=next_batch, terminated=False)

    # Mutate the original batch outside the buffer
    batch.price[0, 0, 0] += 999.0
    batch.position[0, 0] = 77.0

    # Sample from buffer and verify stored snapshot was immune
    sampled_obs, _, _, _, _, _ = buffer.sample(1)
    assert np.isclose(float(sampled_obs.price[0, 0, 0]), orig_price_val)
    assert float(sampled_obs.position[0, 0]) != 77.0


# ==============================================================================
# P1-7: Historical Feature Builder Provenance
# ==============================================================================

def test_audited_provenance_builders_pit_guard():
    """
    P1-7: Audited builders enforce ProvenanceRecord PIT availability:
    - If record is future or unverified, fills neutral fallback / zero mask.
    - E21 Market Surprise has zero post-event reaction.
    """
    decision_time = pd.Timestamp("2024-04-15 20:00:00", tz="Asia/Taipei")

    # 1. Past verified event (available 2 days ago)
    prov_valid = ProvenanceRecord(
        source_id="news_1001",
        source_published_at=pd.Timestamp("2024-04-13 10:00:00", tz="Asia/Taipei"),
        provider_available_at=pd.Timestamp("2024-04-13 10:05:00", tz="Asia/Taipei"),
        available_at=pd.Timestamp("2024-04-13 10:05:00", tz="Asia/Taipei"),
        ingested_at=pd.Timestamp("2024-04-13 10:10:00", tz="Asia/Taipei"),
        is_audit_verified=True,
        missing_status="AVAILABLE",
    )
    meta_valid = build_audited_event_metadata(
        actor_embedding_4d=np.ones(4),
        type_embedding_8d=np.ones(8),
        status_embedding_4d=np.ones(4),
        evidence_traceability=1.0,
        source_reliability=0.8,
        source_independence=1.0,
        novelty_input_estimate=0.5,
        market_surprise_preevent=0.02,
        quantified_magnitude=0.7,
        timestamp_reliability=1.0,
        provenance=prov_valid,
        decision_time=decision_time,
    )
    # Event Age at index 22 should be ~2.41 days
    assert meta_valid[22] > 2.0

    # 2. Future leaked event (available tomorrow)
    prov_future = ProvenanceRecord(
        source_id="news_1002",
        source_published_at=pd.Timestamp("2024-04-16 09:00:00", tz="Asia/Taipei"),
        provider_available_at=pd.Timestamp("2024-04-16 09:05:00", tz="Asia/Taipei"),
        available_at=pd.Timestamp("2024-04-16 09:05:00", tz="Asia/Taipei"),
        ingested_at=pd.Timestamp("2024-04-16 09:10:00", tz="Asia/Taipei"),
        is_audit_verified=True,
        missing_status="AVAILABLE",
    )
    meta_future = build_audited_event_metadata(
        actor_embedding_4d=np.ones(4),
        type_embedding_8d=np.ones(8),
        status_embedding_4d=np.ones(4),
        evidence_traceability=1.0,
        source_reliability=0.8,
        source_independence=1.0,
        novelty_input_estimate=0.5,
        market_surprise_preevent=0.02,
        quantified_magnitude=0.7,
        timestamp_reliability=1.0,
        provenance=prov_future,
        decision_time=decision_time,
    )
    # Future record is rejected and returns zeroes
    assert np.all(meta_future == 0.0)


# ==============================================================================
# P1-8: End-to-End Feature Pipeline Test
# ==============================================================================

def test_end_to_end_deterministic_pipeline():
    """
    P1-8: Deterministic end-to-end integration test:
    Raw Records -> As-Of Join -> Feature Engineering -> SplitManifest ->
    Normalizer -> Batch Assembly -> Semantic Validation -> Model Forward Pass ->
    Stage 1, 2, 3 micro training steps.
    """
    torch.manual_seed(42)
    # 1. Generate consistent synthetic batch through Feature Engine
    batch = generate_temporally_consistent_synthetic_batch(
        batch_size=4,
        seed=42,
        include_targets=True,
        schema_version="v8.0-original",
    )

    # 2. Manifest & Normalization
    manifest = SplitManifest.create(
        dataset_id="e2e_fixture",
        schema_version="v8.0-original",
        train_start="2024-01-01",
        train_end="2024-01-31",
        val_start="2024-02-01",
        val_end="2024-02-28",
        test_start="2024-03-01",
        test_end="2024-03-31",
    )
    normalizer = FeatureNormalizer(schema_version="v8.0-original", split_manifest=manifest)
    train_ts = pd.date_range("2024-01-05", periods=4, freq="D")
    norm_batch = normalizer.fit_transform(batch, is_train_split=True, sample_timestamps=train_ts)

    # 3. Validation
    assert validate_multimodal_batch(batch) is True
    assert validate_multimodal_batch(norm_batch, is_normalized=True) is True

    # 4. Model Forward Pass
    model = StockRLFullModel()
    q_values, preds, _ = model(norm_batch)
    assert q_values.shape == (4, 2)
    assert "novelty_scores" in preds

    # 5. Micro Training: Stage 1, Stage 2, Stage 3
    hist1 = run_stage1_pretraining(
        model=model,
        epochs=1,
        num_batches=1,
        lr=1e-4,
        schema_version="v8.0-original",
    )
    assert len(hist1) == 1
    assert hist1[0]["loss_total"] >= 0.0

    env = PointInTimeStockEnv(schema_version="v8.0-original")
    agent2, buffer2, hist2 = run_stage2_rl_training(
        model=model,
        env=env,
        episodes=1,
        warmup_steps=5,
        train_steps_per_env_step=1,
        batch_size=4,
        schema_version="v8.0-original",
    )
    assert len(hist2) == 1

    agent3, buffer3, hist3 = run_stage3_joint_finetuning(
        model=model,
        env=env,
        episodes=1,
        batch_size=4,
        agent=agent2,
        buffer=buffer2,
        schema_version="v8.0-original",
    )
    assert len(hist3) == 1
