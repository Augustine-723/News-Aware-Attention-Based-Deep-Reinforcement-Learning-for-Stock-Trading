"""Evaluation script for Transaction Friction Hysteresis Band across Test and April Holdout."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from stock_rl.data.normalizer import FeatureNormalizer
from stock_rl.data.real_data_poc import STOCK_METADATA, load_real_market_feeds
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.evaluation.april_holdout import prepare_files
from stock_rl.evaluation.market_drl_pilot import DATA_DIR, StockRLFullModel, evaluate
from stock_rl.evaluation.trend_gated_holdout import compute_ma20_series


def run_hysteresis_sweep(cache_dir, tg_dir, output_path):
    torch.set_num_threads(1)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Test Split (2024-02-01 to 2024-03-29, 34 sessions)
    calendar = pd.DatetimeIndex(pd.read_csv(Path(DATA_DIR) / "taiex.csv", usecols=["date"], parse_dates=["date"])["date"])
    dates_test = calendar[(calendar >= "2024-02-01") & (calendar <= "2024-03-29")]
    decisions_test = dates_test.tz_localize("Asia/Taipei") + pd.Timedelta(hours=20)

    test_stocks_data = {}
    for stock, meta in STOCK_METADATA.items():
        prices, inst, macro, foreign, context, _ = load_real_market_feeds(stock, DATA_DIR, mode="estimated")
        builder = HistoricalSnapshotBuilder(
            prices, inst, macro, foreign, context,
            event_records=[], shares_outstanding=meta["shares_outstanding"],
            mode="estimated",
        )
        sma_all = prices.close.rolling(20, min_periods=1).mean()
        is_above = [bool(prices.close.loc[:d.normalize()].iloc[-1] >= sma_all.loc[:d.normalize()].iloc[-1]) for d in decisions_test]
        snapshots = [builder.build_snapshot(d) for d in decisions_test]
        ohlc = prices.loc[dates_test.tz_localize("Asia/Taipei"), ["open", "high", "low", "close"]].to_numpy()
        test_stocks_data[stock] = {
            "decisions": decisions_test,
            "snapshots": snapshots,
            "ohlc": ohlc,
            "is_above": is_above,
        }

    # 2. Prepare April Holdout Split (2024-04-01 to 2024-04-30, 19 return sessions)
    holdout_stocks_data = {}
    with tempfile.TemporaryDirectory() as tmp:
        prepare_files(cache, Path(tmp))
        for stock, meta in STOCK_METADATA.items():
            prices, inst, macro, foreign, context, _ = load_real_market_feeds(stock, tmp, mode="estimated")
            builder = HistoricalSnapshotBuilder(
                prices, inst, macro, foreign, context,
                event_records=[], shares_outstanding=meta["shares_outstanding"],
                mode="estimated",
            )
            april = prices.loc["2024-04"]
            dates = april.index.normalize() + pd.Timedelta(hours=20)
            closes = prices["close"]
            sma_all = closes.rolling(20, min_periods=1).mean()
            is_above = [bool(closes.loc[:d.normalize()].iloc[-1] >= sma_all.loc[:d.normalize()].iloc[-1]) for d in dates]
            snapshots = [builder.build_snapshot(d) for d in dates]
            ohlc = april[["open", "high", "low", "close"]].to_numpy()
            holdout_stocks_data[stock] = {
                "decisions": dates,
                "snapshots": snapshots,
                "ohlc": ohlc,
                "is_above": is_above,
            }

    thetas = [0.0, 0.005, 0.010, 0.015, 0.020]
    seeds = [42, 100, 2024]
    results = {"thetas": thetas, "test_window": {}, "april_holdout": {}}

    for seed in seeds:
        tg_path = Path(tg_dir) / f"trend_gated_flow-seed-{seed}.pt"
        tg_state = torch.load(tg_path, map_location="cpu", weights_only=True)
        tg_norm = FeatureNormalizer(schema_version=tg_state["schema_version"])
        tg_norm.load_state_dict(tg_state["normalizer"])
        tg_model = StockRLFullModel()
        tg_model.load_state_dict(tg_state["model_state_dict"])
        tg_model.eval()

        results["test_window"][str(seed)] = {}
        results["april_holdout"][str(seed)] = {}

        # Pre-transform batches
        test_batches = {}
        for stock, sdata in test_stocks_data.items():
            tb = []
            for i, snap in enumerate(sdata["snapshots"]):
                b = snap.clone()
                if sdata["is_above"][i]:
                    b.institutional[..., 5:8].zero_()
                    b.institutional_missing_mask[..., 5:8].fill_(True)
                else:
                    b.institutional[..., :].zero_()
                    b.institutional_missing_mask[..., :].fill_(True)
                tb.append(tg_norm.transform(b))
            test_batches[stock] = tb

        holdout_batches = {}
        for stock, sdata in holdout_stocks_data.items():
            hb = []
            for i, snap in enumerate(sdata["snapshots"]):
                b = snap.clone()
                if sdata["is_above"][i]:
                    b.institutional[..., 5:8].zero_()
                    b.institutional_missing_mask[..., 5:8].fill_(True)
                else:
                    b.institutional[..., :].zero_()
                    b.institutional_missing_mask[..., :].fill_(True)
                hb.append(tg_norm.transform(b))
            holdout_batches[stock] = hb

        for theta in thetas:
            # Evaluate Test Window
            t_eval = {}
            for stock, sdata in test_stocks_data.items():
                cur_pos = [0]
                def act_test(obs, _):
                    with torch.inference_mode():
                        q = tg_model(obs)[0][0].numpy()
                        if cur_pos[0] == 1:
                            action = 0 if (q[0] - q[1] > theta) else 1
                        else:
                            action = 1 if (q[1] > q[0]) else 0
                        cur_pos[0] = action
                        return action
                res = evaluate((sdata["decisions"], test_batches[stock], sdata["ohlc"], None), act_test)
                t_eval[stock] = {
                    "net_return": float(res["net_return"]),
                    "sharpe": float(res["sharpe"]) if res["sharpe"] is not None else None,
                    "trades": int(res["executed_position_changes"]),
                    "costs": float(sum(res["costs"].values())),
                }
            results["test_window"][str(seed)][f"{theta:.3f}"] = {
                "mean_net_return": float(np.mean([v["net_return"] for v in t_eval.values()])),
                "total_trades": int(sum(v["trades"] for v in t_eval.values())),
                "total_costs": float(sum(v["costs"] for v in t_eval.values())),
                "stocks": t_eval,
            }

            # Evaluate April Holdout
            h_eval = {}
            for stock, sdata in holdout_stocks_data.items():
                cur_pos = [0]
                def act_holdout(obs, _):
                    with torch.inference_mode():
                        q = tg_model(obs)[0][0].numpy()
                        if cur_pos[0] == 1:
                            action = 0 if (q[0] - q[1] > theta) else 1
                        else:
                            action = 1 if (q[1] > q[0]) else 0
                        cur_pos[0] = action
                        return action
                res = evaluate((sdata["decisions"], holdout_batches[stock], sdata["ohlc"], None), act_holdout)
                h_eval[stock] = {
                    "net_return": float(res["net_return"]),
                    "sharpe": float(res["sharpe"]) if res["sharpe"] is not None else None,
                    "trades": int(res["executed_position_changes"]),
                    "costs": float(sum(res["costs"].values())),
                }
            results["april_holdout"][str(seed)][f"{theta:.3f}"] = {
                "mean_net_return": float(np.mean([v["net_return"] for v in h_eval.values()])),
                "total_trades": int(sum(v["trades"] for v in h_eval.values())),
                "total_costs": float(sum(v["costs"] for v in h_eval.values())),
                "stocks": h_eval,
            }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Hysteresis sweep saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "april_source_cache")
    parser.add_argument("--tg-dir", type=Path, default=ROOT / "checkpoints/v8_43_trend_gated")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/v8_44_hysteresis_sweep.json")
    args = parser.parse_args()

    run_hysteresis_sweep(args.cache, args.tg_dir, args.output)
