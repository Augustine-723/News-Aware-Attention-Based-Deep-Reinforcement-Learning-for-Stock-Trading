"""
Ablation Study Suite V8.0
Implements model ablations: Market-Only, Sentiment-Only, Event-Removal,
Date-Perturbation, and Shuffled-Relation.
"""

from typing import Dict, Any, List, Optional
import copy
import torch
import numpy as np

from stock_rl.data.synthetic_data import MultimodalBatch
from stock_rl.config.schema import (
    EVENT_AGE_METADATA_INDEX,
    CURRENT_ACTIVE_SCHEMA_VERSION,
    FeatureSchemaVersion,
    get_event_age_index,
    get_feature_index,
)
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
        """Keeps only sentiment / signal features in metadata; zeroes event text & relation graphs."""
        b = copy.copy(batch)
        b.recent_event_emb = torch.zeros_like(batch.recent_event_emb)
        b.active_event_emb = torch.zeros_like(batch.active_event_emb)
        b.event_stock_rel = torch.zeros_like(batch.event_stock_rel)
        meta_zeroed = torch.zeros_like(batch.event_metadata)

        schema = getattr(batch, "schema_version", CURRENT_ACTIVE_SCHEMA_VERSION)
        if schema in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value, "v8.0-original", "v8.0"):
            # In baseline v8.0-original, keep evidence/reliability/surprise (E17..E22, indices 16..21)
            meta_zeroed[:, :, 16:22] = batch.event_metadata[:, :, 16:22]
        else:
            # Candidate schema: keep sentiment score & confidence (EV_META_10..11, indices 9..10)
            sent_start = get_feature_index("EV_META_10", "event_metadata")
            meta_zeroed[:, :, sent_start : sent_start + 2] = batch.event_metadata[:, :, sent_start : sent_start + 2]

        b.event_metadata = meta_zeroed
        return b

    @staticmethod
    def apply_date_perturbation(batch: MultimodalBatch, max_jitter_days: float = 5.0) -> MultimodalBatch:
        """Adds random temporal noise to event ages to test PIT timestamp sensitivity."""
        b = copy.copy(batch)
        meta = batch.event_metadata.clone()
        schema = getattr(batch, "schema_version", CURRENT_ACTIVE_SCHEMA_VERSION)
        age_idx = get_event_age_index(schema)
        age_slice = slice(age_idx, age_idx + 1)
        jitter = torch.FloatTensor(meta[:, :, age_slice].shape).uniform_(-max_jitter_days, max_jitter_days)
        meta[:, :, age_slice] = torch.clamp(meta[:, :, age_slice] + jitter, min=0.0)
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


def run_schema_comparison_study(
    agent_baseline,
    agent_candidate,
    env_baseline: PointInTimeStockEnv,
    env_candidate: PointInTimeStockEnv,
) -> Dict[str, Any]:
    """
    Executes comparative evaluation across Baseline ('v8.0-original')
    and Candidate ('v8.1-candidate') pipelines.
    Returns:
        comparison_results: Dict containing metrics tagged with schema_version.
    """
    metrics_baseline = evaluate_policy_on_env(agent_baseline.online_model, env_baseline)
    metrics_candidate = evaluate_policy_on_env(agent_candidate.online_model, env_candidate)

    return {
        "baseline": {
            "schema_version": FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value,
            "metrics": metrics_baseline,
        },
        "candidate_ablation": {
            "schema_version": FeatureSchemaVersion.ALTERNATIVE_V8_1_CANDIDATE.value,
            "metrics": metrics_candidate,
        },
    }


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
