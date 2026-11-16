"""
Comprehensive Regression Tests for V8.1 Code Review 2
Verifies all 7 requested changes across P0 and P1:
1. P0-1: Stage 3 AdamW optimizer momentum state migration vs fresh reset.
2. P0-2: Label boundary purging off-by-one fix (T+1 Open to T+6 Open) & dataset cutoff.
3. P1-3: Feature dictionary V8 column counts & PIT historical availability validation.
4. P1-4: Novelty causal historical memory, padding exclusion, and publication order invariance.
5. P1-5: DocumentVersion chronological consistency (pub_time <= vendor_time) and evidence retrieval.
6. P1-6: Multi-task loss with zero valid labels, NaN target sanitization, and uncertainty weighting.
7. P1-7: Trading environment order rejection holding duration and entry price consistency.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from stock_rl.config.schema import (
    PRICE_DIM,
    TECHNICAL_DIM,
    INSTITUTIONAL_DIM,
    MACRO_DIM,
    STOCK_CONTEXT_DIM,
    EVENT_RAW_FEATURE_DIM,
    EVENT_METADATA_DIM,
    EVENT_STOCK_RELATION_DIM,
    POSITION_STATE_DIM,
    TOTAL_EVENT_SLOTS,
    ACTION_CASH,
    ACTION_LONG,
    PRICE_FEATURE_FIELDS,
    TECHNICAL_FEATURE_FIELDS,
    INSTITUTIONAL_FEATURE_FIELDS,
    MACRO_FEATURE_FIELDS,
    STOCK_CONTEXT_FEATURE_FIELDS,
    EVENT_METADATA_FEATURE_FIELDS,
    EVENT_STOCK_RELATION_FEATURE_FIELDS,
    POSITION_STATE_FEATURE_FIELDS,
)
from stock_rl.data.pit_engine import (
    DocumentVersion,
    VersionedEventRecord,
    AvailabilityStatus,
    PointInTimeGuard,
    validate_feature_historical_availability,
)
from stock_rl.data.synthetic_data import generate_synthetic_batch
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.models.multi_task_heads import MultiTaskHeads, MultiTaskLoss
from stock_rl.rl.agent import DuelingDoubleDQNAgent
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.env.execution import TWSEExecutionSimulator, ExecutionResult


# ==============================================================================
# 1. P0-1: Stage 3 Optimizer Momentum Migration vs Reset
# ==============================================================================
def test_stage3_optimizer_momentum_migration_and_reset():
    model = StockRLFullModel()
    agent = DuelingDoubleDQNAgent(model=model)

    # Freeze encoders as in Stage 2
    model.freeze_encoders()

    # Step optimizer so DQN parameters have AdamW momentum state (step, exp_avg, exp_avg_sq)
    batch = generate_synthetic_batch(batch_size=2, seed=42)
    next_batch = generate_synthetic_batch(batch_size=2, seed=43)
    actions = torch.tensor([0, 1], dtype=torch.long)
    rewards = torch.tensor([0.01, -0.01], dtype=torch.float32)
    terminated = torch.tensor([False, False], dtype=torch.bool)

    agent.train_step(batch, actions, rewards, next_batch, terminated)

    # Verify online DQN head parameters now have optimizer momentum state
    dqn_params = list(agent.online_model.dqn.parameters())
    assert len(dqn_params) > 0
    dqn_param = dqn_params[0]
    assert dqn_param in agent.optimizer.state
    assert "exp_avg" in agent.optimizer.state[dqn_param]
    old_step = agent.optimizer.state[dqn_param]["step"]
    old_exp_avg = agent.optimizer.state[dqn_param]["exp_avg"].clone()

    # Unfreeze model for Stage 3
    model.unfreeze_for_stage3()

    # Option A: Reconfigure with momentum inheritance (True)
    report_inherit = agent.reconfigure_optimizer_for_stage3(lr=1e-5, inherit_momentum=True)
    assert report_inherit["inherit_momentum"] is True
    assert report_inherit["migrated_parameters"] > 0

    # Verify that dqn_param retained its momentum state exactly
    assert dqn_param in agent.optimizer.state
    assert torch.allclose(agent.optimizer.state[dqn_param]["exp_avg"], old_exp_avg)
    assert agent.optimizer.state[dqn_param]["step"] == old_step

    # Option B: Reconfigure with fresh reset (False)
    report_reset = agent.reconfigure_optimizer_for_stage3(lr=1e-5, inherit_momentum=False)
    assert report_reset["inherit_momentum"] is False
    assert report_reset["migrated_parameters"] == 0
    # In fresh reset mode, optimizer state is freshly initialized and empty until first step
    assert len(agent.optimizer.state) == 0


# ==============================================================================
# 2. P0-2: Label Boundary Purging Off-by-One & Cutoff
# ==============================================================================
def test_label_boundary_purging_off_by_one_and_cutoff():
    # 10 consecutive business days: 2024-01-01 to 2024-01-12
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    # max_horizon = 5 trading days
    # Execution is at T+1 Open. Realization for 5-day horizon is at T+1+5 = T+6 Open.
    # For dates[idx], realization index is idx + 1 + 5 = idx + 6.
    # At idx = 3: realization index is 3 + 6 = 9 (dates[9]). Since 9 < 10 and dates[9] <= dates[9], valid!
    # At idx = 4: realization index is 4 + 6 = 10 >= 10 -> EXCEEDS DATASET CUTOFF -> must be purged!

    split_boundaries = [
        ("train", dates[0], dates[9]),
    ]

    masks = PointInTimeGuard.purge_label_overlap_boundaries(
        decision_dates=dates,
        label_horizons_days=[1, 3, 5],
        split_boundaries=split_boundaries,
    )

    train_mask = masks["train"]
    # Indices 0, 1, 2, 3 must be True (valid)
    assert train_mask[0] == True
    assert train_mask[1] == True
    assert train_mask[2] == True
    assert train_mask[3] == True

    # Indices 4, 5, 6, 7, 8, 9 must be False (purged due to 5-day span reaching >= 10)
    assert train_mask[4] == False
    assert train_mask[5] == False
    assert train_mask[6] == False
    assert train_mask[7] == False
    assert train_mask[8] == False
    assert train_mask[9] == False

    # Also test boundary into next split: split ends at dates[7]
    split_early = [("train", dates[0], dates[7])]
    masks_early = PointInTimeGuard.purge_label_overlap_boundaries(
        decision_dates=dates,
        label_horizons_days=[5],
        split_boundaries=split_early,
    )
    # For end at dates[7]:
    # idx = 1: 1 + 6 = 7 (dates[7] <= dates[7]) -> valid
    # idx = 2: 2 + 6 = 8 (dates[8] > dates[7]) -> crosses into next split -> purged!
    assert masks_early["train"][1] == True
    assert masks_early["train"][2] == False


# ==============================================================================
# 3. P1-3: Feature Dictionary & PIT Historical Availability
# ==============================================================================
def test_feature_dictionary_and_pit_availability_validation():
    # Assert exact V8 authoritative dimensions
    assert len(PRICE_FEATURE_FIELDS) == PRICE_DIM == 12
    assert len(TECHNICAL_FEATURE_FIELDS) == TECHNICAL_DIM == 16
    assert len(INSTITUTIONAL_FEATURE_FIELDS) == INSTITUTIONAL_DIM == 8
    assert len(MACRO_FEATURE_FIELDS) == MACRO_DIM == 12
    assert len(STOCK_CONTEXT_FEATURE_FIELDS) == STOCK_CONTEXT_DIM == 16
    assert len(EVENT_METADATA_FEATURE_FIELDS) == EVENT_METADATA_DIM == 24
    assert len(EVENT_STOCK_RELATION_FEATURE_FIELDS) == EVENT_STOCK_RELATION_DIM == 8
    assert len(POSITION_STATE_FEATURE_FIELDS) == POSITION_STATE_DIM == 4

    # Test historical availability validation
    t_decision = pd.Timestamp("2024-05-10 20:00:00", tz="Asia/Taipei")
    t_source_valid = pd.Timestamp("2024-05-10 18:00:00", tz="Asia/Taipei")
    t_source_future = pd.Timestamp("2024-05-10 21:00:00", tz="Asia/Taipei")

    # Audit verified timestamp before decision is valid
    assert validate_feature_historical_availability(
        "foreign_net_buy_shares_norm", t_source_valid, t_decision, is_audit_verified=True
    ) is True

    # Future timestamp violates PIT
    assert validate_feature_historical_availability(
        "foreign_net_buy_shares_norm", t_source_future, t_decision, is_audit_verified=True
    ) is False

    # Unverified timestamp must not be assumed available
    assert validate_feature_historical_availability(
        "foreign_net_buy_shares_norm", t_source_valid, t_decision, is_audit_verified=False
    ) is False


# ==============================================================================
# 4. P1-4: Novelty Causal Historical Memory & Padding
# ==============================================================================
def test_novelty_causal_memory_and_padding_invariance():
    model = StockRLFullModel()
    batch = generate_synthetic_batch(batch_size=2, seed=10)

    # Scenario:
    # Slot 0: Age 5 days (occurred 5 days ago)
    # Slot 1: Age 2 days (occurred 2 days ago, newer than slot 0)
    # Slot 2: Padded / invalid slot
    # Slots 3..39: Padded / invalid slots
    batch.event_metadata[0, :, 4] = 0.0
    batch.event_metadata[0, 0, 4] = 5.0
    batch.event_metadata[0, 1, 4] = 2.0
    batch.event_metadata[0, 2, 4] = 10.0  # Older, but padded!

    batch.event_mask[0, :] = False
    batch.event_mask[0, 0] = True
    batch.event_mask[0, 1] = True
    # Slot 2 is explicitly False (padded)

    fused_rep, rl_state, projected_events, context_token, causal_prior_memory = model.encode_multimodal_state(batch)

    # 1. Slot 0 has age 5.0. No other valid slot has age > 5.0.
    # Therefore Slot 0 has NO prior events and falls back to default memory.
    # 2. Slot 1 has age 2.0. Slot 0 has age 5.0 > 2.0 and is valid.
    # Slot 2 has age 10.0 > 2.0 but is PADDED (event_mask=False).
    # Therefore Slot 1's prior memory must be EXACTLY Slot 0's projected event!
    assert torch.allclose(causal_prior_memory[0, 1], projected_events[0, 0], atol=1e-5)

    # 3. Test Publication Order Invariance:
    # Swap Slot 0 and Slot 1 (Slot 0 now age 2.0, Slot 1 age 5.0)
    batch_swapped = generate_synthetic_batch(batch_size=2, seed=10)
    batch_swapped.event_metadata[0, :, 4] = 0.0
    batch_swapped.event_metadata[0, 0, 4] = 2.0
    batch_swapped.event_metadata[0, 1, 4] = 5.0
    batch_swapped.event_mask[0, :] = False
    batch_swapped.event_mask[0, 0] = True
    batch_swapped.event_mask[0, 1] = True

    _, _, proj_swapped, _, causal_mem_swapped = model.encode_multimodal_state(batch_swapped)

    # In batch_swapped, Slot 0 has age 2.0, so its prior event is Slot 1 (age 5.0)
    assert torch.allclose(causal_mem_swapped[0, 0], proj_swapped[0, 1], atol=1e-5)

    # 4. Test Empty History / All Padded:
    batch_empty = generate_synthetic_batch(batch_size=2, seed=10)
    batch_empty.event_mask[:] = False
    _, _, _, _, causal_mem_empty = model.encode_multimodal_state(batch_empty)
    assert not torch.isnan(causal_mem_empty).any()


# ==============================================================================
# 5. P1-5: DocumentVersion Chronological Consistency & Text Evidence
# ==============================================================================
def test_document_version_chronological_consistency_and_evidence():
    t_pub = pd.Timestamp("2024-05-15 10:00:00", tz="Asia/Taipei")
    t_vendor_early = pd.Timestamp("2024-05-15 09:00:00", tz="Asia/Taipei")
    t_vendor_valid = pd.Timestamp("2024-05-15 10:15:00", tz="Asia/Taipei")

    # Chronological violation: vendor cannot distribute before publication
    with pytest.raises(ValueError, match="Chronological inconsistency"):
        DocumentVersion(
            version_id=1,
            pub_time=t_pub,
            vendor_available_time=t_vendor_early,
            is_verified=True,
            headline="Invalid timestamp document",
        )

    # Valid version retains evidence snippet
    v1 = DocumentVersion(
        version_id=1,
        pub_time=t_pub,
        vendor_available_time=t_vendor_valid,
        is_verified=True,
        headline="TSMC Reports Record Revenue",
        content="TSMC announced monthly consolidated revenue of NT$230 billion, up 30% YoY.",
    )
    ev = VersionedEventRecord(
        event_id="EV_2330_TEST",
        stock_id="2330",
        occur_time=t_pub,
    )
    ev.add_version(v1)

    # Retrieve before vendor availability -> None
    t_before = pd.Timestamp("2024-05-15 10:05:00", tz="Asia/Taipei")
    h_before, c_before = ev.get_text_at(t_before)
    assert h_before is None
    assert c_before is None

    # Retrieve after vendor availability -> v1 text
    t_after = pd.Timestamp("2024-05-15 20:00:00", tz="Asia/Taipei")
    h_after, c_after = ev.get_text_at(t_after)
    assert h_after == "TSMC Reports Record Revenue"
    assert "NT$230 billion" in c_after


# ==============================================================================
# 6. P1-6: Multi-Task Loss Zero-Label Skipping & NaN Target Sanitization
# ==============================================================================
def test_multi_task_loss_zero_labels_and_nan_targets():
    loss_module = MultiTaskLoss(use_uncertainty_weighting=True)

    B = 2
    predictions = {
        "relevance_logits": torch.randn(B, 40, 1),
        "novelty_scores": torch.rand(B, 40, 1),
        "pred_returns": torch.randn(B, 3),
        "pred_risk": torch.rand(B, 3),
    }

    # Scenario: Return and Risk targets contain NaNs at unmasked locations
    rel_labels = torch.zeros(B, 40, 1)
    nov_labels = torch.zeros(B, 40, 1)
    ret_targets = torch.tensor([[float("nan"), 0.02, float("nan")], [float("nan"), float("nan"), 0.05]])
    risk_targets = torch.tensor([[0.01, float("nan"), 0.03], [float("nan"), float("nan"), float("nan")]])

    # Masks: only valid where targets are numbers
    return_mask = ~torch.isnan(ret_targets)
    risk_mask = ~torch.isnan(risk_targets)

    # Task 1 (Relevance) has ZERO valid labels:
    relevance_mask = torch.zeros(B, 40, 1, dtype=torch.bool)
    novelty_mask = torch.ones(B, 40, 1, dtype=torch.bool)

    total_loss, loss_dict = loss_module(
        predictions=predictions,
        relevance_labels=rel_labels,
        novelty_labels=nov_labels,
        return_targets=ret_targets,
        risk_targets=risk_targets,
        relevance_mask=relevance_mask,
        novelty_mask=novelty_mask,
        return_mask=return_mask,
        risk_mask=risk_mask,
    )

    # Assert no NaNs anywhere in loss
    assert not torch.isnan(total_loss)
    assert not np.isnan(loss_dict["loss_total"])
    assert loss_dict["loss_relevance"] == 0.0  # Skipped because zero valid labels
    assert loss_dict["active_tasks"] == 3      # 3 of 4 tasks active

    # Test completely empty batch (all masks False)
    total_empty, dict_empty = loss_module(
        predictions=predictions,
        relevance_labels=rel_labels,
        novelty_labels=nov_labels,
        return_targets=ret_targets,
        risk_targets=risk_targets,
        relevance_mask=torch.zeros_like(relevance_mask),
        novelty_mask=torch.zeros_like(novelty_mask),
        return_mask=torch.zeros_like(return_mask),
        risk_mask=torch.zeros_like(risk_mask),
    )
    assert not torch.isnan(total_empty)
    assert dict_empty["loss_total"] == 0.0
    assert dict_empty["active_tasks"] == 0


# ==============================================================================
# 7. P1-7: Trading Environment Order Rejection Holding Days & Entry Price
# ==============================================================================
def test_trading_env_order_rejection_holding_and_entry_price():
    env = PointInTimeStockEnv()
    env.reset()

    # Step 1: Execute BUY from Cash -> Long
    # Force open price to be within +-5% of prev close so it executes normally
    env.prices_df[0, 3] = 100.0
    env.prices_df[1, 0] = 100.0  # Open at 100.0
    env.prices_df[1, 3] = 102.0

    obs, reward, terminated, truncated, info = env.step(ACTION_LONG)
    assert env.current_position == ACTION_LONG
    assert env.holding_days == 1
    assert env.entry_price > 0.0
    initial_entry_price = env.entry_price

    # Step 2: Agent attempts to SELL (Cash), but order is REJECTED (e.g. limit-down lock)
    # Set day 2 open to limit-down (-10% from day 1 close)
    env.prices_df[1, 3] = 100.0
    env.prices_df[2, 0] = 89.0   # Locked limit down
    env.prices_df[2, 3] = 89.0

    obs, reward, terminated, truncated, info = env.step(ACTION_CASH)

    # Since sell was rejected at open:
    # 1. Position remains LONG
    assert env.current_position == ACTION_LONG
    # 2. Holding days accumulated to 2
    assert env.holding_days == 2
    # 3. Entry price maintained original buy price
    assert env.entry_price == initial_entry_price

    # Step 3: Now reset env to Cash, and attempt BUY that gets REJECTED (limit-up lock)
    env.reset()
    assert env.current_position == ACTION_CASH
    assert env.holding_days == 0
    assert env.entry_price == 0.0

    env.prices_df[0, 3] = 100.0
    env.prices_df[1, 0] = 111.0  # Locked limit up (+11%)
    env.prices_df[1, 3] = 111.0

    obs, reward, terminated, truncated, info = env.step(ACTION_LONG)

    # Buy rejected:
    # 1. Position remains CASH
    assert env.current_position == ACTION_CASH
    # 2. Holding days remains 0
    assert env.holding_days == 0
    # 3. Entry price remains 0.0
    assert env.entry_price == 0.0
