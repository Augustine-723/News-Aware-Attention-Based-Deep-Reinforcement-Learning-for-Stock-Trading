"""
Unit Tests for Model Forward Pass, Temporal Attention, and Dueling DDQN per V8.0
"""

import pytest
import torch
from stock_rl.data.synthetic_data import generate_synthetic_batch
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.models.multi_task_heads import MultiTaskLoss
from stock_rl.config.schema import (
    MULTIMODAL_FUSION_DIM,
    RL_STATE_DIM,
    NUM_ACTIONS,
    NUM_HORIZONS,
)


def test_full_model_forward():
    B = 8
    batch = generate_synthetic_batch(batch_size=B, seed=123)
    model = StockRLFullModel()

    q_values, preds, rl_state = model(batch)

    # 1. Verify RL State Dimension = 132
    assert rl_state.shape == (B, RL_STATE_DIM)
    assert not torch.isnan(rl_state).any()

    # 2. Verify Q-value output = [B, 2]
    assert q_values.shape == (B, NUM_ACTIONS)
    assert not torch.isnan(q_values).any()

    # 3. Verify Multi-Task Predictions
    assert preds["relevance_logits"].shape == (B, 40, 1)
    assert preds["novelty_logits"].shape == (B, 40, 1)
    assert preds["pred_returns"].shape == (B, NUM_HORIZONS)
    assert preds["pred_risk"].shape == (B, NUM_HORIZONS)

    # Risk must be strictly non-negative (via Softplus)
    assert (preds["pred_risk"] >= 0.0).all()


def test_multi_task_loss_backward():
    B = 4
    batch = generate_synthetic_batch(batch_size=B, seed=456)
    model = StockRLFullModel()
    loss_module = MultiTaskLoss(use_uncertainty_weighting=True)

    q_values, preds, _ = model(batch)
    loss, loss_dict = loss_module(
        predictions=preds,
        relevance_labels=batch.relevance_labels,
        novelty_labels=batch.novelty_labels,
        return_targets=batch.return_targets,
        risk_targets=batch.risk_targets,
        event_mask=batch.event_mask,
    )

    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    # Test backward pass
    loss.backward()

    # Check gradients exist in encoders
    for p in model.market_encoders.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()
