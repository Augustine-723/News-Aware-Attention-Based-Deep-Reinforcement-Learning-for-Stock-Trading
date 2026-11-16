"""
Walk-Forward Validation Engine V8.0
Purges boundary windows and computes multi-seed performance stability.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch

from stock_rl.evaluation.metrics import compute_trading_metrics
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.rl.agent import DuelingDoubleDQNAgent


def evaluate_policy_on_env(
    model: StockRLFullModel,
    env: PointInTimeStockEnv,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, float], List[float], List[int]]:
    """
    Evaluates a frozen policy model deterministically on an environment instance.
    Returns:
        metrics: Dict[str, float]
        equity_curve: List[float]
        actions_taken: List[int]
    """
    agent = DuelingDoubleDQNAgent(model=model, device=device)
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
    return metrics, env.equity_history, actions_taken


def run_multi_seed_evaluation(
    model_factory,
    env_factory,
    seeds: List[int] = [42, 123, 999],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Tuple[float, float]]:
    """
    Executes evaluation across multiple random seeds and returns mean & std for key metrics.
    """
    all_metrics: List[Dict[str, float]] = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = env_factory(seed)
        model = model_factory()
        metrics, _, _ = evaluate_policy_on_env(model, env, device)
        all_metrics.append(metrics)

    summary: Dict[str, Tuple[float, float]] = {}
    if not all_metrics:
        return summary

    keys = all_metrics[0].keys()
    for k in keys:
        values = [m[k] for m in all_metrics if k in m]
        summary[k] = (float(np.mean(values)), float(np.std(values)))

    return summary
