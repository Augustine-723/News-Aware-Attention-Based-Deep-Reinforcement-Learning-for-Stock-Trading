"""Reproduce the Q1 2024 market-only replay on the three cached TWSE stocks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_rl.config.schema import ACTION_CASH, ACTION_LONG
from stock_rl.data.real_data_poc import DATA_DIR, STOCK_METADATA, load_real_market_feeds
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.evaluation.metrics import compute_trading_metrics


def replay(stock_id, data_dir=DATA_DIR):
    prices, inst, macro, foreign, context, _ = load_real_market_feeds(
        stock_id, data_dir=data_dir, mode="estimated"
    )
    calendar = pd.DatetimeIndex(prices.index)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError(f"{stock_id}: invalid trading calendar")
    benchmark_dates = pd.DatetimeIndex(
        pd.read_csv(Path(data_dir) / "taiex.csv", usecols=["date"], parse_dates=["date"])["date"]
    ).tz_localize("Asia/Taipei").sort_values()
    if not calendar.equals(benchmark_dates):
        raise ValueError(f"{stock_id}: stock and TAIEX calendars differ")
    sessions = prices.loc["2024-01-01":"2024-03-31"]
    if len(sessions) < 2:
        raise ValueError(f"{stock_id}: insufficient Q1 sessions")
    builder = HistoricalSnapshotBuilder(
        prices, inst, macro, foreign, stock_context_records=context,
        event_records=[], active_event_records=[],
        shares_outstanding=STOCK_METADATA[stock_id]["shares_outstanding"],
        mode="estimated",
    )
    observations = []
    trend = []
    for session in sessions.index:
        decision = session.normalize() + pd.Timedelta(hours=20)
        snapshot = builder.build_snapshot(decision)
        if snapshot.event_mask.any():
            raise RuntimeError("market-only replay received event slots")
        observations.append(snapshot)
        observed = prices.loc[(prices.index <= session) & (prices.available_at <= decision)]
        if len(observed) < 20 or observed.index[-1] != session:
            raise ValueError(f"{stock_id}: missing decision-session market data")
        close_window = observed.close.tail(20)
        trend.append(ACTION_LONG if close_window.iloc[-1] > close_window.mean() else ACTION_CASH)

    ohlc = sessions[["open", "high", "low", "close"]].to_numpy()
    if not np.isfinite(ohlc).all() or (ohlc <= 0).any():
        raise ValueError(f"{stock_id}: invalid execution prices")
    steps = len(sessions) - 1
    policies = {
        "cash": [ACTION_CASH] * steps,
        "buy_and_hold": [ACTION_LONG] * steps,
        "close_above_sma20": trend[:-1],
    }
    results = {}
    for name, actions in policies.items():
        env = PointInTimeStockEnv(prices_df=ohlc, multimodal_batches=observations)
        env.reset()
        fees = []
        for i, action in enumerate(actions, start=1):
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated != (i == steps) or info["step"] != i:
                raise RuntimeError("unexpected execution session")
            fees.append({key: float(info[key]) for key in ("commission", "tax", "slippage")})
        metrics = compute_trading_metrics(env.equity_history, actions)
        results[name] = {
            "total_return": metrics["total_return"],
            "max_drawdown": metrics["max_drawdown"],
            "costs": {key: sum(fee[key] for fee in fees) for key in fees[0]},
            "daily_equity": [float(value) for value in env.equity_history],
            "target_actions": actions,
        }

    return {
        "cached_sessions": len(calendar),
        "evaluation_sessions": [session.date().isoformat() for session in sessions.index],
        "decision_steps": steps,
        "populated_event_slots": 0,
        "policies": results,
    }


def main():
    stocks = ("2330", "2603", "2881")
    report = {
        "research_status": "ESTIMATED_MARKET_DATA_ONLY",
        "decision_timezone": "Asia/Taipei",
        "decision_time": "20:00",
        "execution": "NEXT_TRADING_SESSION_OPEN",
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(DATA_DIR).glob("*.csv"))
        },
        "stocks": {stock_id: replay(stock_id) for stock_id in stocks},
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
