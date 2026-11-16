"""
Ablation Study Suite V8.0
Implements model ablations: Market-Only, Sentiment-Only, Event-Removal,
Date-Perturbation, and Shuffled-Relation.
"""

from typing import Dict, Any, List
import copy
import torch
import numpy as np

from stock_rl.data.synthetic_data import MultimodalBatch
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.evaluation.walk_forward import evaluate_policy_on_env


class AblationMasker:
    """Modifies multimodal batches to implement controlled ablation experiments."""

    @staticmethod
    def apply_market_only(batch: MultimodalBatch) -> MultimodalBatch:
        """Zeros out all event signals and masks all event slots."""
        b = copy.copy(batch)
        b.recent_event_emb = torch.zeros_like(batch.recent_event_emb)
        b.active_event_emb = torch.zeros_like(batch.active_event_emb)
        b.event_metadata = torch.zeros_like(batch.event_metadata)
        b.event_stock_rel = torch.zeros_like(batch.event_stock_rel)
        b.event_mask = torch.zeros_like(batch.event_mask)  # All masked out
        return b

    @staticmethod
    def apply_sentiment_only(batch: MultimodalBatch) -> MultimodalBatch:
        """Keeps only sentiment features in metadata; zeroes event text & relation graphs."""
        b = copy.copy(batch)
        b.recent_event_emb = torch.zeros_like(batch.recent_event_emb)
        b.active_event_emb = torch.zeros_like(batch.active_event_emb)
        b.event_stock_rel = torch.zeros_like(batch.event_stock_rel)
        # Keep only metadata indices 2:4 (sentiment score & confidence)
        meta_zeroed = torch.zeros_like(batch.event_metadata)
        meta_zeroed[:, :, 2:4] = batch.event_metadata[:, :, 2:4]
        b.event_metadata = meta_zeroed
        return b

    @staticmethod
    def apply_date_perturbation(batch: MultimodalBatch, max_jitter_days: float = 5.0) -> MultimodalBatch:
        """Adds random temporal noise to event ages to test PIT timestamp sensitivity."""
        b = copy.copy(batch)
        meta = batch.event_metadata.clone()
        jitter = torch.FloatTensor(meta[:, :, 4:5].shape).uniform_(-max_jitter_days, max_jitter_days)
        meta[:, :, 4:5] = torch.clamp(meta[:, :, 4:5] + jitter, min=0.0)
        b.event_metadata = meta
        return b

    @staticmethod
    def apply_shuffled_relations(batch: MultimodalBatch) -> MultimodalBatch:
        """Randomly permutes entity supply-chain & competitor roles across event slots."""
        b = copy.copy(batch)
        rel = batch.event_stock_rel.clone()
        # Permute along event slot dimension (dim 1)
        perm = torch.randperm(rel.shape[1])
        b.event_stock_rel = rel[:, perm, :]
        return b


def run_buy_and_hold_benchmark(env: PointInTimeStockEnv) -> Dict[str, float]:
    """Runs a pure Buy-and-Hold benchmark on the specified environment."""
    from stock_rl.config.schema import ACTION_LONG
    from stock_rl.evaluation.metrics import compute_trading_metrics

    obs, _ = env.reset()
    done = False
    actions = []

    while not done:
        action = ACTION_LONG
        actions.append(action)
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return compute_trading_metrics(env.equity_history, actions)
