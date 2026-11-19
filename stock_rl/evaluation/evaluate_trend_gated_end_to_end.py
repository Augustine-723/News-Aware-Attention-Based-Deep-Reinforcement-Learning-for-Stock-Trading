"""End-to-end trained Trend-Gated Flow policy evaluation across test and April holdout."""
import argparse
import hashlib
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
from stock_rl.evaluation.market_drl_pilot import StockRLFullModel, evaluate
from stock_rl.evaluation.paired_performance import compare


def compute_ma20_series(closes):
    series = pd.Series(closes)
    return series.rolling(20, min_periods=1).mean().to_numpy()


def run_evaluation(cache_dir, control_dir, flow_dir, tg_dir, test_results_path):
    torch.set_num_threads(1)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # 1. Prepare April holdout market feeds
    from stock_rl.evaluation.april_holdout import prepare_files
    stocks_data = {}
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
            closes = prices["close"].loc[:"2024-04"].to_numpy()
            ma20 = compute_ma20_series(closes)
            april_indices = [prices.index.get_loc(d.normalize()) for d in dates]
            april_ma20 = ma20[april_indices]
            april_closes = closes[april_indices]
            is_above_ma20 = (april_closes >= april_ma20)

            stocks_data[stock] = {
                "dates": dates,
                "snapshots": [builder.build_snapshot(d) for d in dates],
                "ohlc": april[["open", "high", "low", "close"]].to_numpy(),
                "is_above_ma20": is_above_ma20,
            }

    # Load test results from fitting
    test_json = json.loads(Path(test_results_path).read_text())
    test_runs = {r["seed"]: r for r in test_json["trend_gated_flow_runs"]}

    results = []
    for seed in (42, 100, 2024):
        c_path = Path(control_dir) / f"without_institutional-seed-{seed}.pt"
        f_path = Path(flow_dir) / f"flow_only-seed-{seed}.pt"
        tg_path = Path(tg_dir) / f"trend_gated_flow-seed-{seed}.pt"

        c_state = torch.load(c_path, map_location="cpu", weights_only=True)
        f_state = torch.load(f_path, map_location="cpu", weights_only=True)
        tg_state = torch.load(tg_path, map_location="cpu", weights_only=True)

        c_norm = FeatureNormalizer(schema_version=c_state["schema_version"])
        c_norm.load_state_dict(c_state["normalizer"])
        c_model = StockRLFullModel()
        c_model.load_state_dict(c_state["model_state_dict"])
        c_model.eval()

        f_norm = FeatureNormalizer(schema_version=f_state["schema_version"])
        f_norm.load_state_dict(f_state["normalizer"])
        f_model = StockRLFullModel()
        f_model.load_state_dict(f_state["model_state_dict"])
        f_model.eval()

        tg_norm = FeatureNormalizer(schema_version=tg_state["schema_version"])
        tg_norm.load_state_dict(tg_state["normalizer"])
        tg_model = StockRLFullModel()
        tg_model.load_state_dict(tg_state["model_state_dict"])
        tg_model.eval()

        holdout_arms = {
            "without_institutional": {},
            "flow_only": {},
            "trend_gated_flow": {},
        }

        for stock, sdata in stocks_data.items():
            dates = sdata["dates"]
            ohlc = sdata["ohlc"]
            is_above = sdata["is_above_ma20"]

            # Control batches: zero out institutional features
            c_batches = []
            for snap in sdata["snapshots"]:
                b = snap.clone()
                b.institutional[..., :].zero_()
                b.institutional_missing_mask[..., :].fill_(True)
                c_batches.append(c_norm.transform(b))

            # Flow only batches: zero out margin features (5:8)
            f_batches = []
            for snap in sdata["snapshots"]:
                b = snap.clone()
                b.institutional[..., 5:8].zero_()
                b.institutional_missing_mask[..., 5:8].fill_(True)
                f_batches.append(f_norm.transform(b))

            # Trend-Gated batches: if below MA20, zero all institutional; else zero margin
            tg_batches = []
            for i, snap in enumerate(sdata["snapshots"]):
                b = snap.clone()
                if is_above[i]:
                    b.institutional[..., 5:8].zero_()
                    b.institutional_missing_mask[..., 5:8].fill_(True)
                else:
                    b.institutional[..., :].zero_()
                    b.institutional_missing_mask[..., :].fill_(True)
                tg_batches.append(tg_norm.transform(b))

            def act_c(obs, _):
                with torch.inference_mode():
                    return int(c_model(obs)[0][0].argmax())
            holdout_arms["without_institutional"][stock] = evaluate((dates, c_batches, ohlc, None), act_c)

            def act_f(obs, _):
                with torch.inference_mode():
                    return int(f_model(obs)[0][0].argmax())
            holdout_arms["flow_only"][stock] = evaluate((dates, f_batches, ohlc, None), act_f)

            def act_tg(obs, _):
                with torch.inference_mode():
                    return int(tg_model(obs)[0][0].argmax())
            holdout_arms["trend_gated_flow"][stock] = evaluate((dates, tg_batches, ohlc, None), act_tg)

        tg_hash = hashlib.sha256(tg_path.read_bytes()).hexdigest()
        c_hash = hashlib.sha256(c_path.read_bytes()).hexdigest()
        f_hash = hashlib.sha256(f_path.read_bytes()).hexdigest()

        seed_result = {
            "seed": seed,
            "checkpoints": {
                "trend_gated_flow": {"path": str(tg_path), "sha256": tg_hash},
                "without_institutional": {"path": str(c_path), "sha256": c_hash},
                "flow_only": {"path": str(f_path), "sha256": f_hash},
            },
            "test_window": {
                "selected_epoch": test_runs[seed]["selected_epoch"],
                "evaluation": test_runs[seed]["test"],
                "mean_net_return": float(np.mean([s["net_return"] for s in test_runs[seed]["test"].values()])),
            },
            "april_holdout": {
                "arms": holdout_arms,
                "comparisons": {
                    "trend_gated_vs_control": compare(holdout_arms["without_institutional"], holdout_arms["trend_gated_flow"]),
                    "trend_gated_vs_flow": compare(holdout_arms["flow_only"], holdout_arms["trend_gated_flow"]),
                    "flow_vs_control": compare(holdout_arms["without_institutional"], holdout_arms["flow_only"]),
                },
                "mean_net_returns": {
                    arm: float(np.mean([s["net_return"] for s in holdout_arms[arm].values()]))
                    for arm in holdout_arms
                },
            },
        }
        results.append(seed_result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "april_source_cache")
    parser.add_argument("--control-dir", type=Path, default=ROOT / "checkpoints/v8_38")
    parser.add_argument("--flow-dir", type=Path, default=ROOT / "checkpoints/v8_39_flow")
    parser.add_argument("--tg-dir", type=Path, default=ROOT / "checkpoints/v8_43_trend_gated")
    parser.add_argument("--test-results", type=Path, default=ROOT / "docs/results/v8_43_trend_gated_fitting.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/v8_43_trend_gated_flow.json")
    args = parser.parse_args()

    results = run_evaluation(args.cache, args.control_dir, args.flow_dir, args.tg_dir, args.test_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"End-to-end evaluation results saved to {args.output}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Seed':<6} | {'Test TG':<10} | {'Apr Control':<12} | {'Apr Flow':<10} | {'Apr TG':<10} | {'Delta vs Flow':<14} | {'Delta vs Ctrl':<14}")
    print("-" * 90)
    for r in results:
        seed = r["seed"]
        test_tg = r["test_window"]["mean_net_return"] * 100
        apr_c = r["april_holdout"]["mean_net_returns"]["without_institutional"] * 100
        apr_f = r["april_holdout"]["mean_net_returns"]["flow_only"] * 100
        apr_tg = r["april_holdout"]["mean_net_returns"]["trend_gated_flow"] * 100
        d_flow = apr_tg - apr_f
        d_ctrl = apr_tg - apr_c
        print(f"{seed:<6} | {test_tg:+8.2f}% | {apr_c:+10.2f}% | {apr_f:+8.2f}% | {apr_tg:+8.2f}% | {d_flow:+12.2f} pp | {d_ctrl:+12.2f} pp")
    print("=" * 90)
