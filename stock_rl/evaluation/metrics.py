"""
Financial Performance & Risk Evaluation Metrics V8.0
Trading-cost-adjusted returns, Sharpe, Sortino, Drawdown, Turnover, and Calmar ratios.
"""

from typing import Dict, List, Union
import numpy as np


def compute_trading_metrics(
    equity_curve: Union[List[float], np.ndarray],
    actions: Union[List[int], np.ndarray],
    trading_days_per_year: int = 250,
    risk_free_rate: float = 0.015,
) -> Dict[str, float]:
    """
    Computes standard quantitative investment metrics from daily portfolio equity values.
    """
    equities = np.array(equity_curve, dtype=np.float64)
    if len(equities) < 2:
        return {}

    # Daily percentage returns
    daily_returns = (equities[1:] - equities[:-1]) / np.maximum(equities[:-1], 1e-4)
    num_days = len(daily_returns)

    # 1. Total Return & Annualized Return
    total_return = float((equities[-1] - equities[0]) / equities[0])
    annualized_return = float((1.0 + total_return) ** (trading_days_per_year / max(num_days, 1)) - 1.0)

    # 2. Volatility & Downside Volatility
    daily_vol = float(np.std(daily_returns))
    ann_vol = daily_vol * np.sqrt(trading_days_per_year)

    negative_returns = daily_returns[daily_returns < 0]
    downside_vol = float(np.std(negative_returns)) if len(negative_returns) > 0 else 1e-6
    ann_downside_vol = downside_vol * np.sqrt(trading_days_per_year)

    # 3. Sharpe & Sortino
    rf_daily = risk_free_rate / trading_days_per_year
    excess_returns = daily_returns - rf_daily
    mean_excess = np.mean(excess_returns)
    sharpe_ratio = float(np.sqrt(trading_days_per_year) * mean_excess / (daily_vol + 1e-8))
    sortino_ratio = float(np.sqrt(trading_days_per_year) * mean_excess / (downside_vol + 1e-8))

    # 4. Maximum Drawdown & Calmar Ratio
    peak = np.maximum.accumulate(equities)
    drawdowns = (equities - peak) / np.maximum(peak, 1e-4)
    max_drawdown = float(np.min(drawdowns))  # negative number
    calmar_ratio = float(annualized_return / (abs(max_drawdown) + 1e-8))

    # 5. Turnover & Trading Activity
    acts = np.array(actions, dtype=np.int32)
    position_changes = np.abs(acts[1:] - acts[:-1]) if len(acts) > 1 else np.array([0])
    turnover_count = int(np.sum(position_changes))
    daily_turnover = float(turnover_count / max(num_days, 1))
    annualized_turnover = daily_turnover * trading_days_per_year

    # 6. Win Rate
    positive_days = np.sum(daily_returns > 0)
    win_rate = float(positive_days / max(num_days, 1))

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "annualized_turnover": annualized_turnover,
        "turnover_trade_count": turnover_count,
        "win_rate": win_rate,
        "evaluation_days": num_days,
    }
