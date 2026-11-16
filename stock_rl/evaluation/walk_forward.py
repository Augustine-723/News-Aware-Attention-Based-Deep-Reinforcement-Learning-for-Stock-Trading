"""
Walk-Forward Validation Engine V8.0
Primary Experiment: ~3-Year Initial Training Window (Nov 2021 – Oct 2024),
Validation Window (Nov 2024 – Oct 2025), and Sealed Final Holdout (Nov 2025 – Nov 2026).
Supports Dynamic Label-Span Purging and separates synthetic unit tests from historical backtests.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch

from stock_rl.config.schema import CURRENT_ACTIVE_SCHEMA_VERSION
from stock_rl.evaluation.metrics import compute_trading_metrics
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.rl.agent import DuelingDoubleDQNAgent


@dataclass(frozen=True)
class WalkForwardSplit:
    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purged_train_end: pd.Timestamp
    purged_val_end: pd.Timestamp


class WalkForwardEngine:
    """
    Manages chronological splits and dynamic label-horizon boundary purging.
    """

    @staticmethod
    def get_v8_primary_split(
        decision_dates: pd.DatetimeIndex,
        max_label_horizon_days: int = 5
    ) -> WalkForwardSplit:
        """
        Primary V8.0 Configuration:
        - Train: 2021-11-01 to 2024-10-31 (~3 years)
        - Validation: 2024-11-01 to 2025-10-31 (1 year)
        - Final Sealed Holdout Test: 2025-11-01 to 2026-11-01 (1 year)

        Dynamic Purging Rule:
        A sample at date t predicting horizon H has label span [t+1 Open, t+1+H Open].
        If t + H trading days > split_boundary, it must be purged from training/validation.
        """
        train_start = pd.Timestamp("2021-11-01")
        train_end = pd.Timestamp("2024-10-31")
        val_start = pd.Timestamp("2024-11-01")
        val_end = pd.Timestamp("2025-10-31")
        test_start = pd.Timestamp("2025-11-01")
        test_end = pd.Timestamp("2026-11-01")

        # Dynamically determine the last valid trading day before boundary
        # For horizon H, label realization is at T+1+H Open.
        # To ensure T+1+H <= split_end, sample index <= N - (H + 2).
        train_trading_days = decision_dates[(decision_dates >= train_start) & (decision_dates <= train_end)]
        if len(train_trading_days) >= max_label_horizon_days + 2:
            purged_train_end = train_trading_days[-(max_label_horizon_days + 2)]
        else:
            purged_train_end = train_end

        val_trading_days = decision_dates[(decision_dates >= val_start) & (decision_dates <= val_end)]
        if len(val_trading_days) >= max_label_horizon_days + 2:
            purged_val_end = val_trading_days[-(max_label_horizon_days + 2)]
        else:
            purged_val_end = val_end

        return WalkForwardSplit(
            split_id=1,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
            purged_train_end=purged_train_end,
            purged_val_end=purged_val_end,
        )

    @staticmethod
    def generate_rolling_splits(
        decision_dates: pd.DatetimeIndex,
        train_months: int = 12,
        val_months: int = 3,
        test_months: int = 3,
        step_months: int = 3,
        max_label_horizon_days: int = 5
    ) -> List[WalkForwardSplit]:
        """
        Generates rolling walk-forward slices for comparative stability tests.
        """
        if len(decision_dates) == 0:
            return []

        start_dt = decision_dates[0]
        end_dt = decision_dates[-1]

        splits = []
        cur_train_start = start_dt
        split_id = 1

        while True:
            cur_train_end = cur_train_start + pd.DateOffset(months=train_months)
            cur_val_start = cur_train_end + pd.Timedelta(days=1)
            cur_val_end = cur_val_start + pd.DateOffset(months=val_months)
            cur_test_start = cur_val_end + pd.Timedelta(days=1)
            cur_test_end = cur_test_start + pd.DateOffset(months=test_months)

            if cur_test_end > end_dt:
                break

            train_days = decision_dates[(decision_dates >= cur_train_start) & (decision_dates <= cur_train_end)]
            purged_train_end = train_days[-max_label_horizon_days - 1] if len(train_days) > max_label_horizon_days else cur_train_end

            val_days = decision_dates[(decision_dates >= cur_val_start) & (decision_dates <= cur_val_end)]
            purged_val_end = val_days[-max_label_horizon_days - 1] if len(val_days) > max_label_horizon_days else cur_val_end

            splits.append(WalkForwardSplit(
                split_id=split_id,
                train_start=cur_train_start,
                train_end=cur_train_end,
                val_start=cur_val_start,
                val_end=cur_val_end,
                test_start=cur_test_start,
                test_end=cur_test_end,
                purged_train_end=purged_train_end,
                purged_val_end=purged_val_end,
            ))

            split_id += 1
            cur_train_start = cur_train_start + pd.DateOffset(months=step_months)

        return splits


def evaluate_policy_on_env(
    model: StockRLFullModel,
    env: PointInTimeStockEnv,
    device: torch.device = torch.device("cpu"),
    schema_version: Optional[str] = None,
) -> Tuple[Dict[str, float], List[float], List[int]]:
    """
    Evaluates a frozen policy model deterministically on an environment instance.
    Returns:
        metrics: Dict[str, float] (tagged with schema_version)
        equity_curve: List[float]
        actions_taken: List[int]
    """
    effective_schema = schema_version or getattr(env, "schema_version", CURRENT_ACTIVE_SCHEMA_VERSION)
    agent = DuelingDoubleDQNAgent(model=model, device=device, schema_version=effective_schema)
    obs, _ = env.reset()
    done = False
    actions_taken = []

    while not done:
        action = agent.select_action(obs, evaluate=True)
        actions_taken.append(action)
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = next_obs

    metrics = compute_trading_metrics(env.equity_history, actions_taken)
    metrics["schema_version"] = effective_schema
    return metrics, env.equity_history, actions_taken
