"""
Comprehensive Regression Test Suite for V8.0 Code Audit Corrective Patches
Verifies fixes for P0-1 through P1-8 per user authorization.
"""

import subprocess
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime

from stock_rl.config.schema import (
    PRICE_FEATURE_FIELDS,
    TECHNICAL_FEATURE_FIELDS,
    INSTITUTIONAL_FEATURE_FIELDS,
    MACRO_FEATURE_FIELDS,
    STOCK_CONTEXT_FEATURE_FIELDS,
    EVENT_METADATA_FEATURE_FIELDS,
    EVENT_STOCK_RELATION_FEATURE_FIELDS,
    POSITION_STATE_FEATURE_FIELDS,
    TOTAL_EVENT_SLOTS,
    NUM_HORIZONS,
)
from stock_rl.data.pit_engine import (
    DocumentVersion,
    VersionedEventRecord,
    AvailabilityStatus,
    PointInTimeGuard,
)
from stock_rl.data.synthetic_data import generate_synthetic_batch
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.models.event_module import EventModule
from stock_rl.models.multi_task_heads import MultiTaskHeads, MultiTaskLoss
from stock_rl.rl.replay_buffer import MultimodalReplayBuffer, Transition
from stock_rl.rl.agent import DuelingDoubleDQNAgent
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.training.stage2_rl import run_stage2_rl_training
from stock_rl.training.stage3_joint import run_stage3_joint_finetuning
from stock_rl.evaluation.walk_forward import WalkForwardEngine


# ==============================================================================
# P0-1: .gitignore does not ignore stock_rl/env/
# ==============================================================================
def test_regression_p0_1_env_files_not_ignored():
    """Verify stock_rl/env/ files are not ignored by .gitignore."""
    res1 = subprocess.run(["git", "check-ignore", "stock_rl/env/execution.py"], capture_output=True, text=True)
    res2 = subprocess.run(["git", "check-ignore", "stock_rl/env/stock_trading_env.py"], capture_output=True, text=True)
    assert res1.returncode != 0, f"stock_rl/env/execution.py is still ignored! stdout: {res1.stdout}"
    assert res2.returncode != 0, f"stock_rl/env/stock_trading_env.py is still ignored! stdout: {res2.stdout}"


# ==============================================================================
# P0-2: Versioned Point-in-Time Availability & Modern Backfills
# ==============================================================================
def test_regression_p0_2_versioned_pit_availability():
    t_decision_2022 = pd.Timestamp("2022-05-15 20:00:00", tz="Asia/Taipei")
    modern_ingest_time = pd.Timestamp("2026-11-16 12:00:00", tz="Asia/Taipei")

    # 1. Original Release: 2022 event with modern 2026 backfill
    ev_backfill = VersionedEventRecord(
        event_id="EV_2022_001",
        stock_id="2330",
        occur_time=pd.Timestamp("2022-05-15 10:00:00", tz="Asia/Taipei"),
        system_ingest_time=modern_ingest_time,
    )
    v1 = DocumentVersion(
        version_id=1,
        pub_time=pd.Timestamp("2022-05-15 11:00:00", tz="Asia/Taipei"),
        vendor_available_time=pd.Timestamp("2022-05-15 11:30:00", tz="Asia/Taipei"),
        is_verified=True,
        headline="TSMC 2022 Fab Expansion Confirmed",
    )
    ev_backfill.add_version(v1)

    # In historical backtest, vendor_available_time <= decision -> VERIFIED
    ver, status = ev_backfill.get_version_as_of(t_decision_2022, live_mode=False)
    assert status == AvailabilityStatus.VERIFIED
    assert ver.version_id == 1

    # 2. Historical Revision: Revision v2 published on 2022-05-16
    v2 = DocumentVersion(
        version_id=2,
        pub_time=pd.Timestamp("2022-05-16 09:00:00", tz="Asia/Taipei"),
        vendor_available_time=pd.Timestamp("2022-05-16 09:15:00", tz="Asia/Taipei"),
        is_verified=True,
        headline="TSMC Fab Expansion Budget Amended to $40B",
    )
    ev_backfill.add_version(v2)

    # At 2022-05-15 decision, v2 MUST NOT leak! Only v1 must be returned.
    ver_t1, _ = ev_backfill.get_version_as_of(t_decision_2022, live_mode=False)
    assert ver_t1.version_id == 1
    assert "Amended" not in ver_t1.headline

    # At 2022-05-16 20:00 decision, v2 is now available.
    t_decision_next = pd.Timestamp("2022-05-16 20:00:00", tz="Asia/Taipei")
    ver_t2, _ = ev_backfill.get_version_as_of(t_decision_next, live_mode=False)
    assert ver_t2.version_id == 2
    assert "Amended" in ver_t2.headline

    # 3. Unknown Historical Availability: Unverified timestamp must be marked UNKNOWN
    ev_unverified = VersionedEventRecord(
        event_id="EV_UNVERIFIED",
        stock_id="2330",
        occur_time=pd.Timestamp("2022-05-15 10:00:00", tz="Asia/Taipei"),
    )
    v_unknown = DocumentVersion(
        version_id=1,
        pub_time=pd.Timestamp("2022-05-15 11:00:00", tz="Asia/Taipei"),
        vendor_available_time=None,  # Unverified vendor timestamp
        is_verified=False,
        headline="Rumor from unverified source",
    )
    ev_unverified.add_version(v_unknown)

    ver_u, status_u = ev_unverified.get_version_as_of(t_decision_2022, live_mode=False)
    assert status_u == AvailabilityStatus.UNKNOWN
    assert ver_u is None


# ==============================================================================
# P0-3: Replay Buffer & DDQN Bootstrapping (Terminated vs Truncated)
# ==============================================================================
def test_regression_p0_3_replay_buffer_and_ddqn_bootstrapping():
    buffer = MultimodalReplayBuffer(capacity=100)
    model = StockRLFullModel()
    agent = DuelingDoubleDQNAgent(model=model)

    obs = generate_synthetic_batch(batch_size=1, seed=1)
    next_obs = generate_synthetic_batch(batch_size=1, seed=2)

    # Transition A: True termination (e.g. delisting) -> terminated=True, truncated=False
    buffer.push(obs, action=1, reward=0.05, next_obs=next_obs, terminated=True, truncated=False)

    # Transition B: Time limit cutoff -> terminated=False, truncated=True
    buffer.push(obs, action=1, reward=0.05, next_obs=next_obs, terminated=False, truncated=True)

    assert len(buffer) == 2

    # Sample both transitions
    b_obs, b_act, b_rew, b_next_obs, b_term, b_trunc = buffer.sample(batch_size=2)
    assert b_term.shape == (2,)
    assert b_trunc.shape == (2,)

    # In Double DQN target calculation:
    with torch.no_grad():
        q_online_next, _, _ = agent.online_model(b_next_obs)
        best_a = torch.argmax(q_online_next, dim=1, keepdim=True)
        q_target_next, _, _ = agent.target_model(b_next_obs)
        target_next_q = q_target_next.gather(1, best_a).squeeze(1)

        # Expected Q
        expected_q = b_rew + (1.0 - b_term) * (agent.config.gamma * target_next_q)

    # For sample where terminated=True (index with term=1.0), expected_q should EQUAL reward!
    term_idx = (b_term == 1.0).nonzero(as_tuple=True)[0]
    trunc_idx = (b_term == 0.0).nonzero(as_tuple=True)[0]

    assert torch.isclose(expected_q[term_idx], b_rew[term_idx]), "Terminated transition bootstrap must be 0!"
    # For sample where truncated=True but terminated=False, expected_q should include gamma * Q_target!
    assert not torch.isclose(expected_q[trunc_idx], b_rew[trunc_idx]), "Truncated transition must retain bootstrap!"


# ==============================================================================
# P0-4: Stage 3 Inherits Stage 2 Agent, Target Network, and Replay Buffer
# ==============================================================================
def test_regression_p0_4_stage3_inherits_stage2_state():
    model = StockRLFullModel()
    env = PointInTimeStockEnv()

    # Run Stage 2
    agent_s2, buffer_s2, s2_hist = run_stage2_rl_training(
        model=model, env=env, episodes=1, warmup_steps=4, batch_size=4
    )
    initial_buffer_len = len(buffer_s2)
    assert initial_buffer_len > 0
    epsilon_after_s2 = agent_s2.epsilon

    # Run Stage 3 inheriting Stage 2 agent and buffer
    agent_s3, buffer_s3, s3_hist = run_stage3_joint_finetuning(
        model=model, env=env, episodes=1, batch_size=4, agent=agent_s2, buffer=buffer_s2
    )

    # Buffer must have accumulated new transitions on top of Stage 2 transitions
    assert len(buffer_s3) > initial_buffer_len
    # Epsilon schedule must continue from Stage 2 rather than resetting to 1.0
    assert agent_s3.epsilon <= epsilon_after_s2
    assert agent_s3 is agent_s2


# ==============================================================================
# P1-5: Complete Feature Dictionary Metadata
# ==============================================================================
def test_regression_p1_5_feature_dictionary_metadata_completeness():
    tables = [
        ("Price", PRICE_FEATURE_FIELDS, 12),
        ("Technical", TECHNICAL_FEATURE_FIELDS, 16),
        ("Institutional", INSTITUTIONAL_FEATURE_FIELDS, 8),
        ("Macro", MACRO_FEATURE_FIELDS, 12),
        ("Stock Context", STOCK_CONTEXT_FEATURE_FIELDS, 16),
        ("Event Metadata", EVENT_METADATA_FEATURE_FIELDS, 24),
        ("Event Stock Relation", EVENT_STOCK_RELATION_FEATURE_FIELDS, 8),
        ("Position State", POSITION_STATE_FEATURE_FIELDS, 4),
    ]

    for table_name, fields, expected_dim in tables:
        assert len(fields) == expected_dim, f"{table_name} dimension mismatch: {len(fields)} != {expected_dim}"
        for f in fields:
            assert f.name, f"Empty name in {table_name}"
            assert f.math_definition, f"Empty math definition in {table_name}: {f.name}"
            assert f.unit, f"Empty unit in {table_name}: {f.name}"
            assert f.source, f"Empty source in {table_name}: {f.name}"
            assert f.available_timing, f"Empty available timing in {table_name}: {f.name}"
            assert f.missing_policy, f"Empty missing policy in {table_name}: {f.name}"
            assert f.version in ("v8.0", "V8.0"), f"Incorrect version in {table_name}: {f.name}"


# ==============================================================================
# P1-6: Multi-Task Independent Masks and 3-Level Novelty Targets {0.0, 0.5, 1.0}
# ==============================================================================
def test_regression_p1_6_multitask_independent_masks_and_novelty():
    B = 4
    batch = generate_synthetic_batch(batch_size=B, seed=777)
    model = StockRLFullModel()
    loss_module = MultiTaskLoss(use_uncertainty_weighting=True, novelty_loss_type="huber")

    # Tamper masks: Set return_mask and novelty_mask to partially masked out
    batch.novelty_mask[:, :10] = False
    batch.return_mask[:, 1] = False  # Horizon 3D missing

    q_vals, preds, _ = model(batch)

    # Verify Novelty predictions are in [0, 1]
    assert (preds["novelty_scores"] >= 0.0).all()
    assert (preds["novelty_scores"] <= 1.0).all()

    # Compute loss with independent masks
    loss, loss_dict = loss_module(
        predictions=preds,
        relevance_labels=batch.relevance_labels,
        novelty_labels=batch.novelty_labels,
        return_targets=batch.return_targets,
        risk_targets=batch.risk_targets,
        relevance_mask=batch.relevance_mask,
        novelty_mask=batch.novelty_mask,
        return_mask=batch.return_mask,
        risk_mask=batch.risk_mask,
    )

    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    # Test backward pass
    loss.backward()
    for p in model.multi_task_heads.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()


# ==============================================================================
# P1-7: Temporal Attention Conditioned Query (Stock & Market Conditions)
# ==============================================================================
def test_regression_p1_7_conditioned_temporal_attention():
    event_mod = EventModule()
    B = 2

    event_text = torch.randn(B, TOTAL_EVENT_SLOTS, 128)
    event_meta = torch.rand(B, TOTAL_EVENT_SLOTS, 24)
    event_rel = torch.rand(B, TOTAL_EVENT_SLOTS, 8)
    event_mask = torch.ones(B, TOTAL_EVENT_SLOTS, dtype=torch.bool)

    # 1. Condition 1: Stock A with Market State A
    stock_ctx_a = torch.randn(B, 16)
    market_a = torch.randn(B, 128)
    _, agg_a = event_mod(event_text, event_meta, event_rel, event_mask, stock_ctx_a, market_a)

    # 2. Condition 2: Stock B with Market State B
    stock_ctx_b = torch.randn(B, 16) + 5.0
    market_b = torch.randn(B, 128) - 5.0
    _, agg_b = event_mod(event_text, event_meta, event_rel, event_mask, stock_ctx_b, market_b)

    # Attention summaries must differ based on conditioning queries!
    assert not torch.allclose(agg_a, agg_b, atol=1e-3), "Conditioned queries must produce distinct event tokens!"

    # 3. No-Event Scenario: All event slots masked out -> NRE token must be safely attended
    all_masked = torch.zeros(B, TOTAL_EVENT_SLOTS, dtype=torch.bool)
    _, agg_empty = event_mod(event_text, event_meta, event_rel, all_masked, stock_ctx_a, market_a)
    assert not torch.isnan(agg_empty).any(), "No-event scenario must produce valid output via NRE Token!"


# ==============================================================================
# P1-8: Walk-Forward Engine Primary Split and Dynamic Purging
# ==============================================================================
def test_regression_p1_8_walk_forward_splits_and_purging():
    # Construct 5 years of daily trading timestamps (Nov 2021 to Nov 2026)
    dates = pd.date_range("2021-11-01", "2026-11-01", freq="B")

    # Primary ~3-year split
    primary_split = WalkForwardEngine.get_v8_primary_split(dates, max_label_horizon_days=5)

    assert primary_split.train_start == pd.Timestamp("2021-11-01")
    assert primary_split.train_end == pd.Timestamp("2024-10-31")
    assert primary_split.val_start == pd.Timestamp("2024-11-01")
    assert primary_split.test_start == pd.Timestamp("2025-11-01")
    assert primary_split.test_end == pd.Timestamp("2026-11-01")

    # Dynamic purging: purged_train_end must be strictly before train_end by at least 5 trading days
    train_days_count = len(dates[(dates >= primary_split.purged_train_end) & (dates <= primary_split.train_end)])
    assert train_days_count >= 5, "Dynamic purging must clear at least the max label horizon window!"

    # Rolling splits comparative test
    rolling_splits = WalkForwardEngine.generate_rolling_splits(dates, train_months=12, val_months=3, test_months=3)
    assert len(rolling_splits) > 0
    assert rolling_splits[0].train_start == pd.Timestamp("2021-11-01")
