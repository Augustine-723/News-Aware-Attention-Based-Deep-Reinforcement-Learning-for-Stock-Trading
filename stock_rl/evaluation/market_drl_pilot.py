"""Chronological M0 pilot using the existing multimodal model with empty events."""

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stock_rl.config.schema import ACTION_CASH, ACTION_LONG, CURRENT_ACTIVE_SCHEMA_VERSION
from stock_rl.config.settings import RLConfig, TradingConfig
from stock_rl.data.normalizer import FeatureNormalizer, SplitManifest
from stock_rl.data.real_data_poc import DATA_DIR, STOCK_METADATA, load_real_market_feeds
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.evaluation.metrics import compute_trading_metrics
from stock_rl.models.full_model import StockRLFullModel
from stock_rl.rl.agent import DuelingDoubleDQNAgent
from stock_rl.rl.replay_buffer import MultimodalReplayBuffer

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("train", "validation", "test")


def prepare_inputs(protocol, manifest, timing_events=None, *, include_event_text=False):
    inputs = {phase: {} for phase in PHASES}
    calendar = pd.DatetimeIndex(pd.read_csv(
        Path(DATA_DIR) / "taiex.csv", usecols=["date"], parse_dates=["date"]
    )["date"]).tz_localize(protocol["timezone"]).sort_values()
    for stock in protocol["stocks"]:
        prices, inst, macro, foreign, context, _ = load_real_market_feeds(stock, mode="estimated")
        if prices.index.has_duplicates or not prices.index.equals(calendar):
            raise ValueError(f"{stock}: invalid calendar")
        builder = HistoricalSnapshotBuilder(
            prices, inst, macro, foreign, stock_context_records=context,
            event_records=[] if timing_events is None else timing_events[stock],
            active_event_records=[], shares_outstanding=STOCK_METADATA[stock]["shares_outstanding"],
            mode="estimated",
        )
        for phase in PHASES:
            start, end = protocol["splits"][phase]
            sessions = prices.loc[start:end]
            if len(sessions) < 2:
                raise ValueError(f"{stock}: insufficient {phase} sessions")
            decisions = sessions.index.normalize() + pd.Timedelta(hours=protocol["decision_hour"])
            batches = [builder.build_snapshot(timestamp) for timestamp in decisions]
            if timing_events is not None:
                from stock_rl.evaluation.news_input_pilot import mask_event_inputs
                timing_batches = [mask_event_inputs(batch, "M1") for batch in batches]
                if include_event_text:
                    for timing, original in zip(timing_batches, batches):
                        timing.recent_event_emb.copy_(original.recent_event_emb)
                        timing.active_event_emb.copy_(original.active_event_emb)
                batches = timing_batches
            elif any(batch.event_mask.any() for batch in batches):
                raise RuntimeError("M0 requires empty event slots")
            ohlc = sessions[["open", "high", "low", "close"]].to_numpy()
            if not np.isfinite(ohlc).all() or (ohlc <= 0).any():
                raise ValueError(f"{stock}: invalid {phase} prices")
            if (sessions.available_at > decisions).any():
                raise ValueError(f"{stock}: closing prices unavailable at the decision")
            sma = prices.close.rolling(20).mean().loc[sessions.index]
            inputs[phase][stock] = (decisions, batches, ohlc, sessions.close.gt(sma).tolist())

    train_dates, train_batches = [], []
    for decisions, batches, _, _ in inputs["train"].values():
        train_dates.extend(decisions)
        train_batches.extend(batches)
    normalizer = FeatureNormalizer().fit(
        train_batches, sample_timestamps=train_dates, split_manifest=manifest
    )
    for items in inputs.values():
        for stock, (decisions, batches, ohlc, trend) in items.items():
            items[stock] = (decisions, [normalizer.transform(batch) for batch in batches], ohlc, trend)
    return inputs, normalizer


def evaluate(sample, choose_action):
    dates, batches, ohlc, _ = sample
    env = PointInTimeStockEnv(prices_df=ohlc, multimodal_batches=batches)
    observation, _ = env.reset()
    costs = dict(commission=0.0, tax=0.0, slippage=0.0)
    actions, trades = [], 0
    for index in range(len(dates) - 1):
        previous_position = env.current_position
        action = choose_action(observation, index)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated != (index == len(dates) - 2):
            raise RuntimeError("unexpected evaluation boundary")
        actions.append(action)
        trades += int(previous_position != env.current_position)
        for key in costs:
            costs[key] += float(info[key])
    metrics = compute_trading_metrics(env.equity_history, actions)
    return {
        "net_return": metrics["total_return"],
        "sharpe": metrics["sharpe_ratio"] if metrics["annualized_volatility"] > 0 else None,
        "max_drawdown": metrics["max_drawdown"],
        "executed_position_changes": trades,
        "costs": costs,
        "target_actions": actions,
        "equity": [float(value) for value in env.equity_history],
        "session_dates": [date.date().isoformat() for date in dates],
    }


def fit_seed(seed, protocol, inputs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = replace(RLConfig(), epsilon_decay_steps=protocol["epsilon_decay_steps"])
    agent = DuelingDoubleDQNAgent(StockRLFullModel(), config=config)
    replay = MultimodalReplayBuffer(capacity=protocol["replay_capacity"])
    best_score, best_state, best_epoch = -math.inf, None, None
    trace = []
    for epoch in range(1, protocol["epochs"] + 1):
        losses = []
        for _, batches, ohlc, _ in inputs["train"].values():
            env = PointInTimeStockEnv(prices_df=ohlc, multimodal_batches=batches)
            observation, _ = env.reset()
            while True:
                action = agent.select_action(observation)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                replay.push(observation, action, reward, next_obs, terminated, truncated)
                observation = next_obs
                agent.step_epsilon()
                if len(replay) >= protocol["batch_size"] and agent.total_steps % protocol["update_every"] == 0:
                    obs, actions, rewards, following, terminal, _ = replay.sample(protocol["batch_size"])
                    loss = agent.train_step(obs, actions, rewards, following, terminal)
                    if not math.isfinite(loss):
                        raise RuntimeError("non-finite M0 training loss")
                    losses.append(loss)
                if terminated or truncated:
                    break
        greedy = lambda obs, _: agent.select_action(obs, evaluate=True)
        score = float(np.mean([evaluate(sample, greedy)["net_return"]
                               for sample in inputs["validation"].values()]))
        trace.append(dict(epoch=epoch, validation_mean_net_return=score, updates=len(losses),
                          mean_loss=float(np.mean(losses)) if losses else None))
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(agent.online_model.state_dict())
    if not any(row["updates"] for row in trace):
        raise RuntimeError("M0 pilot produced zero gradient updates")
    agent.online_model.load_state_dict(best_state)
    return agent, best_epoch, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "configs/v8_22_study.json").read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    boundaries = [
        pd.Timestamp(day, tz=protocol["timezone"]) + pd.Timedelta(hours=protocol["decision_hour"])
        for phase in PHASES for day in protocol["splits"][phase]
    ]
    input_digests = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(DATA_DIR).glob("*.csv"))
    }
    input_hash = SplitManifest.compute_content_hash(input_digests)
    manifest = SplitManifest.create(
        "twse-cached-v8.22", CURRENT_ACTIVE_SCHEMA_VERSION, *boundaries,
        raw_dataset_content_hash=input_hash,
    )
    inputs, normalizer = prepare_inputs(protocol, manifest)
    results = []
    baselines = {
        stock: {
            "cash": evaluate(sample, lambda *_: ACTION_CASH),
            "buy_and_hold": evaluate(sample, lambda *_: ACTION_LONG),
            "sma20": evaluate(sample, lambda _, i: ACTION_LONG if sample[3][i] else ACTION_CASH),
        }
        for stock, sample in inputs["test"].items()
    }
    for seed in protocol["seeds"]:
        agent, epoch, trace = fit_seed(seed, protocol, inputs)
        torch.save({
            "model_state_dict": agent.online_model.state_dict(),
            "normalizer": normalizer.state_dict(),
            "schema_version": manifest.schema_version,
            "seed": seed,
            "selected_epoch": epoch,
            "protocol": protocol,
            "raw_dataset_content_hash": input_hash,
        }, args.checkpoints / f"m0-seed-{seed}.pt")
        results.append({
            "seed": seed, "selected_epoch": epoch, "training": trace,
            "test": {stock: evaluate(sample, lambda obs, _: agent.select_action(obs, evaluate=True))
                     for stock, sample in inputs["test"].items()},
        })
    print(json.dumps({
        "status": "M0_ESTIMATED_MARKET_DATA_PILOT",
        "protocol": protocol,
        "schema_version": manifest.schema_version,
        "raw_dataset_content_hash": input_hash,
        "input_sha256": input_digests,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__},
        "split_manifest_hash": manifest.manifest_hash,
        "fitting_data_hash": normalizer.fitted_fitting_data_hash,
        "trading_config": asdict(TradingConfig()),
        "sessions": {phase: {stock: len(sample[0]) for stock, sample in items.items()}
                     for phase, items in inputs.items()},
        "baselines": baselines, "m0_runs": results,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
