"""
Unit Tests for Evaluation Metrics, Ablation Maskers, and Benchmarks V8.0
"""

import pytest
import numpy as np
import torch

from stock_rl.evaluation.metrics import compute_trading_metrics
from stock_rl.evaluation.ablations import AblationMasker, run_buy_and_hold_benchmark
from stock_rl.evaluation.walk_forward import evaluate_policy_on_env
from stock_rl.data.synthetic_data import generate_synthetic_batch
from stock_rl.config.schema import (
    EVENT_AGE_METADATA_INDEX,
    CURRENT_ACTIVE_SCHEMA_VERSION,
    get_event_age_index,
)
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.models.full_model import StockRLFullModel


def test_trading_metrics_computation():
    # Equity doubles over 250 trading days
    equities = np.linspace(100_000, 200_000, 251)
    actions = [1] * 250

    metrics = compute_trading_metrics(equities, actions)
    assert abs(metrics["total_return"] - 1.0) < 1e-4
    assert metrics["annualized_return"] > 0.0
    assert metrics["sharpe_ratio"] > 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["turnover_trade_count"] == 0


def test_ablation_maskers():
    batch = generate_synthetic_batch(batch_size=4, seed=42)

    # 1. Market Only
    m_batch = AblationMasker.apply_market_only(batch)
    assert (m_batch.recent_event_emb == 0.0).all()
    assert (m_batch.active_event_emb == 0.0).all()
    assert (m_batch.event_mask == False).all()

    # 2. Sentiment Only
    s_batch = AblationMasker.apply_sentiment_only(batch)
    assert (s_batch.recent_event_emb == 0.0).all()
    assert (s_batch.event_stock_rel == 0.0).all()

    # 3. Date Perturbation
    d_batch = AblationMasker.apply_date_perturbation(batch, max_jitter_days=3.0)
    age_idx = get_event_age_index(batch.schema_version)
    assert (d_batch.event_metadata[:, :, age_idx : age_idx + 1] >= 0.0).all()


def test_buy_and_hold_benchmark():
    env = PointInTimeStockEnv()
    metrics = run_buy_and_hold_benchmark(env)
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics


def test_policy_evaluation_walk_forward():
    model = StockRLFullModel()
    env = PointInTimeStockEnv()
    metrics, equity_curve, actions = evaluate_policy_on_env(model, env)

    assert len(equity_curve) > 0
    assert len(actions) == len(equity_curve) - 1
    assert "sharpe_ratio" in metrics
