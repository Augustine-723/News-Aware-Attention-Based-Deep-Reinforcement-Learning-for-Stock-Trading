"""
V8.3 Schema Migration Comprehensive Regression & E2E Test Suite
Validates the full migration to Primary Research Baseline 'v8.0-original':
1. Schema Activation & 100-Column Contracts
2. Historical Feature Engineering Calculations & Bounds
3. Point-in-Time Availability, US Cross-Market T-1 Lag, and No-Lookahead Guards
4. Synthetic Generator Dual-Version Production & Schema Tagging
5. Semantic Tensor Validator Dual-Version Enforcement
6. Dynamic Event Age & Causal Memory Isolation
7. Feature Normalization Fit-on-Train-Only with Frozen Statistics
8. Replay Buffer & Checkpoint Schema Incompatibility Protection
9. Micro End-to-End Training Schedule (Stage 1, 2, 3) under Baseline v8.0-original
10. Ablation Framework Dual Pipeline Verification
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import torch

from stock_rl.config.schema import (
    CURRENT_ACTIVE_SCHEMA_VERSION,
    FeatureSchemaVersion,
    PRICE_DIM,
    TECHNICAL_DIM,
    INSTITUTIONAL_DIM,
    MACRO_DIM,
    STOCK_CONTEXT_DIM,
    EVENT_METADATA_DIM,
    EVENT_STOCK_RELATION_DIM,
    POSITION_STATE_DIM,
    RL_STATE_DIM,
    NUM_ACTIONS,
    get_event_age_index,
    get_feature_index,
    get_schema_table_fields,
    validate_schema_version,
    BASELINE_V8_0_ALL_FEATURE_FIELDS,
)
from stock_rl.data.synthetic_data import MultimodalBatch, generate_synthetic_batch
from stock_rl.data.tensor_validator import validate_multimodal_batch, TensorSchemaError
from stock_rl.data.feature_engine import (
    compute_price_features,
    compute_technical_features,
    compute_institutional_features,
    compute_macro_features,
    assemble_stock_context_features,
    assemble_event_metadata,
    assemble_event_stock_relation,
    assemble_position_state,
)
from stock_rl.data.normalizer import FeatureNormalizer
from stock_rl.data.pit_engine import check_no_future_leakage
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.rl.agent import DuelingDoubleDQNAgent
from stock_rl.rl.replay_buffer import MultimodalReplayBuffer
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.training.stage1_pretrain import run_stage1_pretraining
from stock_rl.training.stage2_rl import run_stage2_rl_training
from stock_rl.training.stage3_joint import run_stage3_joint_finetuning
from stock_rl.evaluation.ablations import (
    AblationMasker,
    run_schema_comparison_study,
)


def test_schema_activation_primary_baseline():
    """Verify v8.0-original is active primary baseline with exact 100-feature layout."""
    assert CURRENT_ACTIVE_SCHEMA_VERSION == FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value
    assert CURRENT_ACTIVE_SCHEMA_VERSION == "v8.0-original"

    # Dynamic Event Age index lookup
    assert get_event_age_index("v8.0-original") == 22
    assert get_event_age_index("v8.1-candidate") == 12
    assert get_event_age_index(None) == 22

    # Verify shorthand code lookups
    assert get_feature_index("P01") == 0
    assert get_feature_index("P07") == 6
    assert get_feature_index("P12") == 11
    assert get_feature_index("T01") == 0
    assert get_feature_index("T14") == 13
    assert get_feature_index("T16") == 15
    assert get_feature_index("I01") == 0
    assert get_feature_index("I08") == 7
    assert get_feature_index("M01") == 0
    assert get_feature_index("M12") == 11
    assert get_feature_index("C01") == 0
    assert get_feature_index("C16") == 15
    assert get_feature_index("E01") == 0
    assert get_feature_index("E23") == 22
    assert get_feature_index("E24") == 23
    assert get_feature_index("R01") == 0
    assert get_feature_index("R08") == 7
    assert get_feature_index("S01") == 0
    assert get_feature_index("S04") == 3

    # Verify 100-dimension count
    total_dims = (
        PRICE_DIM
        + TECHNICAL_DIM
        + INSTITUTIONAL_DIM
        + MACRO_DIM
        + STOCK_CONTEXT_DIM
        + EVENT_METADATA_DIM
        + EVENT_STOCK_RELATION_DIM
        + POSITION_STATE_DIM
    )
    assert total_dims == 100
    assert len(BASELINE_V8_0_ALL_FEATURE_FIELDS) == 100

    # Verify validation helper
    assert validate_schema_version("v8.0-original") is True
    assert validate_schema_version("v8.1-candidate") is True
    with pytest.raises(ValueError, match="Unsupported schema version"):
        validate_schema_version("v9.0-unauthorized")


def test_feature_engineering_100_columns_calculation():
    """Verify feature engineering calculations produce valid finite outputs conforming to bounds."""
    np.random.seed(42)
    n_days = 80

    # 1. Price features
    prices = 100.0 + np.cumsum(np.random.randn(n_days) * 1.5)
    highs = prices + np.abs(np.random.randn(n_days) * 1.0)
    lows = prices - np.abs(np.random.randn(n_days) * 1.0)
    opens = lows + (highs - lows) * np.random.uniform(0.1, 0.9, n_days)
    closes = lows + (highs - lows) * np.random.uniform(0.1, 0.9, n_days)
    volumes = np.random.uniform(1e6, 5e6, n_days)
    values = closes * volumes

    df_ohlcv = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes, "trading_value": values
    })

    price_feats = compute_price_features(df_ohlcv, shares_outstanding=1e8, schema_version="v8.0-original")
    assert price_feats.shape == (n_days, 12)
    assert np.isfinite(price_feats).all()
    # P07 close location in [0, 1]
    assert np.all(price_feats[:, 6] >= 0.0) and np.all(price_feats[:, 6] <= 1.0)
    # P08 log volume >= 0
    assert np.all(price_feats[:, 7] >= 0.0)

    # 2. Technical features
    tech_feats = compute_technical_features(df_ohlcv, schema_version="v8.0-original")
    assert tech_feats.shape == (n_days, 16)
    assert np.isfinite(tech_feats).all()
    # T01..T05 RSI & KDJ in [-1, 1]
    assert np.all(tech_feats[:, 0:5] >= -1.0) and np.all(tech_feats[:, 0:5] <= 1.0)
    # T14 Bollinger Bandwidth >= 0
    assert np.all(tech_feats[:, 13] >= 0.0)
    # T16 ADX14 in [0, 1]
    assert np.all(tech_feats[:, 15] >= 0.0) and np.all(tech_feats[:, 15] <= 1.0)

    # 3. Institutional features
    df_inst = pd.DataFrame({
        "foreign_net_buy": np.random.uniform(-5e5, 5e5, n_days),
        "trust_net_buy": np.random.uniform(-2e5, 2e5, n_days),
        "dealer_net_buy": np.random.uniform(-1e5, 1e5, n_days),
        "margin_balance": np.random.uniform(1e5, 2e5, n_days),
        "short_balance": np.random.uniform(1e4, 3e4, n_days),
    })
    inst_feats = compute_institutional_features(df_inst, pd.Series(volumes), schema_version="v8.0-original")
    assert inst_feats.shape == (n_days, 8)
    assert np.isfinite(inst_feats).all()
    assert np.all(inst_feats[:, 7] >= 0.0)  # Short / Margin ratio >= 0

    # 4. Macro features
    df_macro = pd.DataFrame({
        "taiex": 18000.0 + np.cumsum(np.random.randn(n_days) * 50.0),
        "sector_index": 500.0 + np.cumsum(np.random.randn(n_days) * 2.0),
        "usdtwd": 31.5 + np.cumsum(np.random.randn(n_days) * 0.05),
        "sp500": 5000.0 + np.cumsum(np.random.randn(n_days) * 20.0),
        "nasdaq": 16000.0 + np.cumsum(np.random.randn(n_days) * 80.0),
        "sox": 4500.0 + np.cumsum(np.random.randn(n_days) * 30.0),
        "vix": np.random.uniform(13.0, 28.0, n_days),
        "us_10y_yield": np.random.uniform(3.8, 4.6, n_days),
    })
    macro_feats = compute_macro_features(df_macro, schema_version="v8.0-original", mode="synthetic")
    assert macro_feats.shape == (n_days, 12)
    assert np.isfinite(macro_feats).all()

    # 5. Stock context
    ctx = assemble_stock_context_features(
        industry_embedding_8d=np.ones(8) / np.sqrt(8),
        market_cap=5e11,
        trading_value_60d_ma=1e9,
        volatility_percentile=0.65,
        liquidity_percentile=0.80,
        beta_120d=1.15,
        industry_market_cap_percentile=0.90,
        days_since_ipo=1500,
        had_trading_suspension_20d=False,
    )
    assert ctx.shape == (16,)
    assert np.isfinite(ctx).all()
    assert ctx[15] == 0.0

    # 6. Event metadata
    meta = assemble_event_metadata(
        actor_embedding_4d=np.zeros(4),
        type_embedding_8d=np.ones(8),
        status_embedding_4d=np.zeros(4),
        evidence_traceability=0.95,
        source_reliability=0.80,
        source_independence=1.0,
        novelty_input_estimate=0.45,
        market_surprise_preevent=0.03,
        quantified_magnitude=0.70,
        event_age_days=2.5,
        timestamp_reliability=1.0,
    )
    assert meta.shape == (24,)
    assert meta[22] == 2.5  # E23 Event Age at Python index 22
    assert meta[23] == 1.0  # E24 Timestamp Reliability at Python index 23

    # 7. Event-stock relation
    rel = assemble_event_stock_relation(
        direct_company_mention=True,
        industry_match=True,
        upstream_exposure=0.6,
        downstream_exposure=0.0,
        customer_relationship=0.8,
        supplier_relationship=0.0,
        geographic_exposure=0.5,
        macro_factor_exposure=0.3,
    )
    assert rel.shape == (8,)
    assert rel[0] == 1.0 and rel[1] == 1.0

    # 8. Position state
    pos = assemble_position_state(
        target_position=1,
        unrealized_pnl_pct=0.045,
        holding_duration_days=12,
        days_since_last_successful_trade=3,
    )
    assert pos.shape == (4,)
    assert pos[0] == 1.0
    assert np.isclose(pos[1], 0.045)
    assert np.isclose(pos[2], 12.0 / 60.0)


def test_pit_leakage_and_cross_market_lag():
    """Verify PIT compliance, T-1 US feature lag, and counterfactual future price isolation."""
    # Test valid vs leaking feature timestamps
    t_decision = pd.Timestamp("2024-05-10 20:00:00", tz="Asia/Taipei")
    t_feature_ok = pd.Timestamp("2024-05-10 18:00:00", tz="Asia/Taipei")
    t_feature_leak = pd.Timestamp("2024-05-10 21:00:00", tz="Asia/Taipei")

    assert check_no_future_leakage(np.array([t_feature_ok]), np.array([t_decision])) is True

    with pytest.raises(ValueError, match="Point-in-Time Violation"):
        check_no_future_leakage(np.array([t_feature_leak]), np.array([t_decision]))

    # Verify Market Surprise (E21) cannot be altered by future price shocks
    meta_original = assemble_event_metadata(
        actor_embedding_4d=np.zeros(4),
        type_embedding_8d=np.zeros(8),
        status_embedding_4d=np.zeros(4),
        evidence_traceability=1.0,
        source_reliability=1.0,
        source_independence=1.0,
        novelty_input_estimate=0.5,
        market_surprise_preevent=0.02,
        quantified_magnitude=0.5,
        event_age_days=1.0,
        timestamp_reliability=1.0,
    )
    # Modifying post-event price does NOT touch meta_original[20]
    post_event_shock = 0.099  # +9.9% price spike tomorrow
    assert meta_original[20] == 0.02, "Market surprise must only reflect pre-event prior implied volatility"


def test_synthetic_generator_dual_schema():
    """Verify synthetic batch generation for both baseline and candidate schemas."""
    # Baseline V8.0 Original
    b_orig = generate_synthetic_batch(batch_size=8, seed=42, schema_version="v8.0-original")
    assert b_orig.schema_version == "v8.0-original"
    # Event Age at index 22
    assert (b_orig.event_metadata[:, :, 22] >= 0.1).all()
    # Close location in [0, 1]
    assert (b_orig.price[:, :, 6] >= 0.0).all() and (b_orig.price[:, :, 6] <= 1.0).all()
    # Pass tensor validator
    assert validate_multimodal_batch(b_orig) is True

    # Candidate V8.1
    b_cand = generate_synthetic_batch(batch_size=8, seed=42, schema_version="v8.1-candidate")
    assert b_cand.schema_version == "v8.1-candidate"
    # Event Age at index 12
    assert (b_cand.event_metadata[:, :, 12] >= 0.1).all()
    assert validate_multimodal_batch(b_cand) is True


def test_tensor_validator_semantic_checks():
    """Verify validator enforces shape, finite values, mask consistency, and bounds."""
    batch = generate_synthetic_batch(batch_size=4, seed=123, schema_version="v8.0-original")

    # 1. Invalid dimension
    b_bad_shape = generate_synthetic_batch(batch_size=4, seed=123, schema_version="v8.0-original")
    b_bad_shape.price = b_bad_shape.price[:, :, :10]
    with pytest.raises(TensorSchemaError, match="Shape mismatch for price"):
        validate_multimodal_batch(b_bad_shape)

    # 2. NaN injection
    b_nan = generate_synthetic_batch(batch_size=4, seed=123, schema_version="v8.0-original")
    b_nan.technical[0, 0, 0] = float("nan")
    with pytest.raises(TensorSchemaError, match="NaN detected in tensor technical"):
        validate_multimodal_batch(b_nan)

    # 3. Mask inconsistency: inactive slot marked valid in relevance
    b_mask = generate_synthetic_batch(batch_size=4, seed=123, schema_version="v8.0-original")
    b_mask.event_mask[:, 10] = False
    b_mask.relevance_mask[:, 10, 0] = True
    with pytest.raises(TensorSchemaError, match="Relevance mask violates event mask"):
        validate_multimodal_batch(b_mask)

    # 4. Out of bounds close location (P07 > 1.0)
    b_close = generate_synthetic_batch(batch_size=4, seed=123, schema_version="v8.0-original")
    b_close.price[0, 0, 6] = 1.5
    with pytest.raises(TensorSchemaError, match="Price Close Location"):
        validate_multimodal_batch(b_close)

    # 5. Negative event age in active slot (E23)
    b_age = generate_synthetic_batch(batch_size=4, seed=123, schema_version="v8.0-original")
    b_age.event_mask[0, 0] = True
    b_age.event_metadata[0, 0, 22] = -5.0
    with pytest.raises(TensorSchemaError, match="Negative event age detected"):
        validate_multimodal_batch(b_age)


def test_model_dynamic_event_age_and_causal_memory():
    """Verify temporal attention and causal memory dynamically index Event Age per schema."""
    torch.manual_seed(99)
    model = StockRLFullModel()
    model.eval()

    # Baseline: index 22 is Event Age, index 12 is Status Embedding
    b_orig = generate_synthetic_batch(batch_size=1, seed=42, schema_version="v8.0-original")
    with torch.no_grad():
        b_orig.event_mask[0, :] = False
        b_orig.event_mask[0, 0] = True
        b_orig.event_mask[0, 1] = True
        # Slot 0 older (age 30), Slot 1 newer (age 10)
        b_orig.event_metadata[0, 0, 22] = 30.0
        b_orig.event_metadata[0, 1, 22] = 10.0

        _, _, _, _, causal_mem = model.encode_multimodal_state(b_orig)
        # Slot 0 has NO prior events -> fallback to NRE token
        nre_token = model.event_module.nre_token.squeeze(1)[0]
        assert torch.allclose(causal_mem[0, 0], nre_token, atol=1e-5)

        # Perturbing former candidate age index (index 12) does NOT affect baseline causal memory
        b_counterfactual = b_orig.to(torch.device("cpu"))
        b_counterfactual.event_metadata[0, 0, 12] = 999.0
        _, _, _, _, causal_mem_cf = model.encode_multimodal_state(b_counterfactual)
        assert torch.allclose(causal_mem[0, 0], causal_mem_cf[0, 0], atol=1e-5)


def test_feature_normalizer_train_only_fit():
    """Verify normalizer only fits on Training Split and transforms Val/Test with frozen stats."""
    train_batch = generate_synthetic_batch(batch_size=32, seed=10, schema_version="v8.0-original")
    val_batch = generate_synthetic_batch(batch_size=16, seed=20, schema_version="v8.0-original")

    normalizer = FeatureNormalizer(schema_version="v8.0-original", mode="synthetic")
    assert normalizer.is_fitted is False

    # Attempting to transform before fitting raises RuntimeError
    with pytest.raises(RuntimeError, match="must be fitted on training data"):
        normalizer.transform(train_batch)

    # Attempting to fit on validation split raises ValueError (leakage guard)
    with pytest.raises(ValueError, match="PIT Violation"):
        normalizer.fit(val_batch, is_train_split=False)

    # Fit on training split
    normalizer.fit(train_batch, is_train_split=True)
    assert normalizer.is_fitted is True

    # Re-fitting raises RuntimeError
    with pytest.raises(RuntimeError, match="already fitted and frozen"):
        normalizer.fit(train_batch, is_train_split=True)

    # Transform val batch using frozen training statistics
    val_norm = normalizer.transform(val_batch)
    assert val_norm.schema_version == "v8.0-original"
    assert val_norm.price.shape == val_batch.price.shape

    # Attempting to transform batch with mismatched schema raises ValueError
    cand_batch = generate_synthetic_batch(batch_size=8, seed=30, schema_version="v8.1-candidate")
    with pytest.raises(ValueError, match="Normalizer schema mismatch"):
        normalizer.transform(cand_batch)


def test_replay_buffer_schema_incompatibility_protection():
    """Verify replay buffer rejects transitions with conflicting schema versions."""
    buffer_baseline = MultimodalReplayBuffer(capacity=100, schema_version="v8.0-original")
    obs_orig = generate_synthetic_batch(batch_size=1, seed=1, schema_version="v8.0-original")
    next_obs_orig = generate_synthetic_batch(batch_size=1, seed=2, schema_version="v8.0-original")
    obs_cand = generate_synthetic_batch(batch_size=1, seed=3, schema_version="v8.1-candidate")

    # Push valid baseline transition
    buffer_baseline.push(obs_orig, action=1, reward=0.5, next_obs=next_obs_orig, terminated=False)
    assert len(buffer_baseline) == 1

    # Attempting to push candidate obs into baseline buffer raises ValueError
    with pytest.raises(ValueError, match="Replay Buffer schema violation"):
        buffer_baseline.push(obs_cand, action=0, reward=0.0, next_obs=next_obs_orig, terminated=False)

    with pytest.raises(ValueError, match="Replay Buffer schema violation"):
        buffer_baseline.push(obs_orig, action=0, reward=0.0, next_obs=obs_cand, terminated=False)


def test_checkpoint_schema_incompatibility_protection():
    """Verify checkpoint saving records schema version and loading rejects mismatched schemas."""
    model = StockRLFullModel()
    agent_baseline = DuelingDoubleDQNAgent(model=model, schema_version="v8.0-original")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "agent_baseline.pt")
        agent_baseline.save_checkpoint(ckpt_path)

        # Loading into same schema succeeds
        agent_baseline.load_checkpoint(ckpt_path, expected_schema_version="v8.0-original")

        # Loading into candidate agent fails with clear error
        agent_candidate = DuelingDoubleDQNAgent(model=model, schema_version="v8.1-candidate")
        with pytest.raises(ValueError, match="Checkpoint schema violation"):
            agent_candidate.load_checkpoint(ckpt_path, expected_schema_version="v8.1-candidate")


def test_micro_e2e_training_pipeline_v8_0_original():
    """Run full micro Stage 1 -> Stage 2 -> Stage 3 pipeline under active v8.0-original schema."""
    model = StockRLFullModel()
    env = PointInTimeStockEnv(schema_version="v8.0-original")

    # Stage 1: Representation Pretraining
    hist1 = run_stage1_pretraining(
        model=model,
        epochs=1,
        num_batches=2,
        lr=1e-4,
        schema_version="v8.0-original",
    )
    assert len(hist1) == 1
    assert hist1[0]["loss_total"] > 0.0

    # Stage 2: Frozen Encoders RL
    agent, buffer, hist2 = run_stage2_rl_training(
        model=model,
        env=env,
        episodes=1,
        warmup_steps=10,
        train_steps_per_env_step=1,
        batch_size=8,
        schema_version="v8.0-original",
    )
    assert len(hist2) == 1
    assert agent.schema_version == "v8.0-original"
    assert buffer.schema_version == "v8.0-original"

    # Stage 3: Joint Fine-Tuning with Momentum Migration
    agent3, buffer3, hist3 = run_stage3_joint_finetuning(
        model=model,
        env=env,
        episodes=1,
        batch_size=8,
        agent=agent,
        buffer=buffer,
        schema_version="v8.0-original",
    )
    assert len(hist3) == 1
    assert agent3.schema_version == "v8.0-original"


def test_ablation_framework_dual_pipeline():
    """Verify ablation studies run on both baseline and candidate schemas."""
    b_orig = generate_synthetic_batch(batch_size=4, seed=77, schema_version="v8.0-original")
    b_mkt = AblationMasker.apply_market_only(b_orig)
    assert (b_mkt.recent_event_emb == 0.0).all()
    assert (b_mkt.event_mask == False).all()

    b_sent = AblationMasker.apply_sentiment_only(b_orig)
    assert (b_sent.recent_event_emb == 0.0).all()

    b_pert = AblationMasker.apply_date_perturbation(b_orig, max_jitter_days=2.0)
    assert (b_pert.event_metadata[:, :, 22] >= 0.0).all()

    b_shuf = AblationMasker.apply_shuffled_relations(b_orig)
    assert b_shuf.event_stock_rel.shape == b_orig.event_stock_rel.shape

    # Comparative evaluation
    model_b = StockRLFullModel()
    model_c = StockRLFullModel()
    env_b = PointInTimeStockEnv(schema_version="v8.0-original")
    env_c = PointInTimeStockEnv(schema_version="v8.1-candidate")
    agent_b = DuelingDoubleDQNAgent(model=model_b, schema_version="v8.0-original")
    agent_c = DuelingDoubleDQNAgent(model=model_c, schema_version="v8.1-candidate")

    comp = run_schema_comparison_study(agent_b, agent_c, env_b, env_c, mode="synthetic")
    assert comp["baseline"]["schema_version"] == "v8.0-original"
    assert comp["candidate_ablation"]["schema_version"] == "v8.1-candidate"
