"""Trend-Gated Decision Policy evaluation across test and April holdout windows."""
import argparse
import copy
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

from stock_rl.config.schema import ACTION_CASH, ACTION_LONG
from stock_rl.data.normalizer import FeatureNormalizer
from stock_rl.data.real_data_poc import STOCK_METADATA, load_real_market_feeds
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.evaluation.market_drl_pilot import DATA_DIR, ROOT, StockRLFullModel, evaluate
from stock_rl.evaluation.paired_performance import compare


def compute_ma20_series(closes):
    """Return rolling 20-day SMA aligned with closes array."""
    series = pd.Series(closes)
    return series.rolling(20, min_periods=1).mean().to_numpy()


def run_trend_gated_evaluation(cache_dir, control_dir, flow_dir):
    torch.set_num_threads(1)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # 1. Prepare April files
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
            # Slice MA20 for the april decision dates
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

    results = []
    for seed in (42, 100, 2024):
        # Load models
        c_path = Path(control_dir) / f"without_institutional-seed-{seed}.pt"
        f_path = Path(flow_dir) / f"flow_only-seed-{seed}.pt"

        c_state = torch.load(c_path, map_location="cpu", weights_only=True)
        f_state = torch.load(f_path, map_location="cpu", weights_only=True)

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

        arms = {
            "without_institutional": {},
            "flow_only": {},
            "flow_gated_strict": {},
            "flow_gated_entry": {},
            "flow_gated_feature_mask": {},
        }

        for stock, sdata in stocks_data.items():
            dates = sdata["dates"]
            ohlc = sdata["ohlc"]
            is_above = sdata["is_above_ma20"]

            # Batches for control (omitted I01-I08)
            c_batches = []
            for snap in sdata["snapshots"]:
                b = snap.clone()
                b.institutional[..., :].zero_()
                b.institutional_missing_mask[..., :].fill_(True)
                c_batches.append(c_norm.transform(b))

            # Batches for flow (omitted I06-I08)
            f_batches = []
            for snap in sdata["snapshots"]:
                b = snap.clone()
                b.institutional[..., 5:8].zero_()
                b.institutional_missing_mask[..., 5:8].fill_(True)
                f_batches.append(f_norm.transform(b))

            # Batches for dynamic feature mask (when below MA20, zero I01-I05 as well)
            mask_batches = []
            for i, snap in enumerate(sdata["snapshots"]):
                b = snap.clone()
                if not is_above[i]:
                    b.institutional[..., :].zero_()
                    b.institutional_missing_mask[..., :].fill_(True)
                else:
                    b.institutional[..., 5:8].zero_()
                    b.institutional_missing_mask[..., 5:8].fill_(True)
                mask_batches.append(f_norm.transform(b))

            # 1. Control policy
            def act_ctrl(obs, _):
                with torch.inference_mode():
                    return int(c_model(obs)[0][0].argmax())
            arms["without_institutional"][stock] = evaluate((dates, c_batches, ohlc, None), act_ctrl)

            # 2. Flow only policy
            def act_flow(obs, _):
                with torch.inference_mode():
                    return int(f_model(obs)[0][0].argmax())
            arms["flow_only"][stock] = evaluate((dates, f_batches, ohlc, None), act_flow)

            # 3. Flow Gated Strict (Force CASH if below MA20)
            def act_gated_strict(obs, index):
                with torch.inference_mode():
                    base = int(f_model(obs)[0][0].argmax())
                    return base if is_above[index] else ACTION_CASH
            arms["flow_gated_strict"][stock] = evaluate((dates, f_batches, ohlc, None), act_gated_strict)

            # 4. Flow Gated Entry (Prohibit new entry if below MA20, allow existing to trail)
            current_pos = [ACTION_CASH]
            def act_gated_entry(obs, index):
                with torch.inference_mode():
                    base = int(f_model(obs)[0][0].argmax())
                    if current_pos[0] == ACTION_CASH and base == ACTION_LONG:
                        action = ACTION_LONG if is_above[index] else ACTION_CASH
                    else:
                        action = base
                    current_pos[0] = action
                    return action
            arms["flow_gated_entry"][stock] = evaluate((dates, f_batches, ohlc, None), act_gated_entry)

            # 5. Flow Gated Feature Mask
            def act_gated_mask(obs, _):
                with torch.inference_mode():
                    return int(f_model(obs)[0][0].argmax())
            arms["flow_gated_feature_mask"][stock] = evaluate((dates, mask_batches, ohlc, None), act_gated_mask)

        seed_res = {
            "seed": seed,
            "arms": arms,
            "comparisons": {
                "flow_vs_control": compare(arms["without_institutional"], arms["flow_only"]),
                "gated_strict_vs_control": compare(arms["without_institutional"], arms["flow_gated_strict"]),
                "gated_entry_vs_control": compare(arms["without_institutional"], arms["flow_gated_entry"]),
                "gated_mask_vs_control": compare(arms["without_institutional"], arms["flow_gated_feature_mask"]),
                "gated_strict_vs_flow": compare(arms["flow_only"], arms["flow_gated_strict"]),
            },
        }
        results.append(seed_res)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "april_source_cache")
    parser.add_argument("--control-dir", type=Path, default=ROOT / "checkpoints/v8_38")
    parser.add_argument("--flow-dir", type=Path, default=ROOT / "checkpoints/v8_39_flow")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/v8_42_trend_gated_holdout.json")
    args = parser.parse_args()

    res = run_trend_gated_evaluation(args.cache, args.control_dir, args.flow_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"Trend gated evaluation results saved to {args.output}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Seed':<6} | {'Control':<9} | {'Flow Only':<10} | {'Gated Strict':<13} | {'Gated Entry':<12} | {'Gated Mask':<11} | {'Delta Strict':<12}")
    print("-" * 90)
    for r in res:
        seed = r["seed"]
        c_ret = r["comparisons"]["flow_vs_control"]["control_net_return"] * 100
        f_ret = r["comparisons"]["flow_vs_control"]["treatment_net_return"] * 100
        gs_ret = r["comparisons"]["gated_strict_vs_control"]["treatment_net_return"] * 100
        ge_ret = r["comparisons"]["gated_entry_vs_control"]["treatment_net_return"] * 100
        gm_ret = r["comparisons"]["gated_mask_vs_control"]["treatment_net_return"] * 100
        d_gs = gs_ret - c_ret
        print(f"{seed:<6} | {c_ret:+8.2f}% | {f_ret:+9.2f}% | {gs_ret:+12.2f}% | {ge_ret:+11.2f}% | {gm_ret:+10.2f}% | {d_gs:+11.2f} pp")
    print("=" * 90)
