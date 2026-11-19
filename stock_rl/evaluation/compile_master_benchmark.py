"""Compile master benchmark across all policy models, baselines, and evaluation periods."""
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
from stock_rl.evaluation.market_drl_pilot import DATA_DIR, evaluate


def compile_benchmark(output_path):
    # 1. Load intermediate artifacts
    v8_43_fit = json.load(open(ROOT / "docs/results/v8_43_trend_gated_fitting.json"))
    v8_43_flow = json.load(open(ROOT / "docs/results/v8_43_trend_gated_flow.json"))
    v8_44_hys = json.load(open(ROOT / "docs/results/v8_44_hysteresis_sweep.json"))
    v8_38 = json.load(open(ROOT / "docs/results/v8_38_no_institutional.json"))
    v8_39 = json.load(open(ROOT / "docs/results/v8_39_flow.json"))

    cache = ROOT / "april_source_cache"

    # Compute April baselines (Buy & Hold, SMA20)
    with tempfile.TemporaryDirectory() as tmp:
        prepare_files(cache, Path(tmp))
        apr_bah = {}
        apr_sma = {}
        for stock in ["2330", "2603", "2881"]:
            prices, inst, macro, foreign, context, _ = load_real_market_feeds(stock, tmp, mode="estimated")
            builder = HistoricalSnapshotBuilder(
                prices, inst, macro, foreign, context,
                event_records=[], shares_outstanding=STOCK_METADATA[stock]["shares_outstanding"],
                mode="estimated",
            )
            april = prices.loc["2024-04"]
            dates = april.index.normalize() + pd.Timedelta(hours=20)
            ohlc = april[["open", "high", "low", "close"]].to_numpy()
            batches = [builder.build_snapshot(d) for d in dates]

            res_bah = evaluate((dates, batches, ohlc, None), lambda obs, idx: 1)
            apr_bah[stock] = {
                "net_return": float(res_bah["net_return"]),
                "trades": int(res_bah["executed_position_changes"]),
                "costs": float(sum(res_bah["costs"].values())),
            }

            closes = prices["close"]
            sma_all = closes.rolling(20, min_periods=1).mean()
            is_above = [int(closes.loc[:d.normalize()].iloc[-1] >= sma_all.loc[:d.normalize()].iloc[-1]) for d in dates]
            res_sma = evaluate((dates, batches, ohlc, None), lambda obs, idx: is_above[idx])
            apr_sma[stock] = {
                "net_return": float(res_sma["net_return"]),
                "trades": int(res_sma["executed_position_changes"]),
                "costs": float(sum(res_sma["costs"].values())),
            }

    baselines_test = v8_43_fit["baselines"]
    test_bah = {
        s: {
            "net_return": float(baselines_test[s]["buy_and_hold"]["net_return"]),
            "trades": int(baselines_test[s]["buy_and_hold"]["executed_position_changes"]),
            "costs": float(sum(baselines_test[s]["buy_and_hold"]["costs"].values())),
        }
        for s in ["2330", "2603", "2881"]
    }
    test_sma = {
        s: {
            "net_return": float(baselines_test[s]["sma20"]["net_return"]),
            "trades": int(baselines_test[s]["sma20"]["executed_position_changes"]),
            "costs": float(sum(baselines_test[s]["sma20"]["costs"].values())),
        }
        for s in ["2330", "2603", "2881"]
    }

    seeds = [42, 100, 2024]
    benchmark = {
        "metadata": {
            "title": "Master Benchmark: Interaction Threshold & Trend-Gated Architectures",
            "schema_version": "v8.0-original",
            "formal_pit": False,
            "stocks": ["2330", "2603", "2881"],
        },
        "baselines": {
            "buy_and_hold": {
                "test_window": {
                    "mean_net_return": float(np.mean([v["net_return"] for v in test_bah.values()])),
                    "stocks": test_bah,
                },
                "april_holdout": {
                    "mean_net_return": float(np.mean([v["net_return"] for v in apr_bah.values()])),
                    "stocks": apr_bah,
                },
            },
            "pure_sma20_rule": {
                "test_window": {
                    "mean_net_return": float(np.mean([v["net_return"] for v in test_sma.values()])),
                    "stocks": test_sma,
                },
                "april_holdout": {
                    "mean_net_return": float(np.mean([v["net_return"] for v in apr_sma.values()])),
                    "stocks": apr_sma,
                },
            },
        },
        "drl_models": {},
    }

    for seed in seeds:
        # V8.38 Control
        r38 = next(r for r in v8_38["without_institutional_runs"] if r["seed"] == seed)
        t38 = {
            s: {
                "net_return": float(v["net_return"]),
                "trades": int(v["executed_position_changes"]),
                "costs": float(sum(v["costs"].values())),
            }
            for s, v in r38["test"].items()
        }

        # V8.39 Flow Only
        r39 = next(r for r in v8_39["flow_only_runs"] if r["seed"] == seed)
        t39 = {
            s: {
                "net_return": float(v["net_return"]),
                "trades": int(v["executed_position_changes"]),
                "costs": float(sum(v["costs"].values())),
            }
            for s, v in r39["test"].items()
        }

        # April Holdout for Control and Flow
        r43_h = next(r for r in v8_43_flow if r["seed"] == seed)["april_holdout"]
        h38 = {
            s: {
                "net_return": float(v["net_return"]),
                "trades": int(v["executed_position_changes"]),
                "costs": float(sum(v["costs"].values())),
            }
            for s, v in r43_h["arms"]["without_institutional"].items()
        }
        h39 = {
            s: {
                "net_return": float(v["net_return"]),
                "trades": int(v["executed_position_changes"]),
                "costs": float(sum(v["costs"].values())),
            }
            for s, v in r43_h["arms"]["flow_only"].items()
        }

        # V8.43 Trend-Gated E2E (theta=0.000)
        t43 = v8_44_hys["test_window"][str(seed)]["0.000"]
        h43 = v8_44_hys["april_holdout"][str(seed)]["0.000"]

        # V8.44 Trend-Gated + Hysteresis (theta=0.015)
        t44 = v8_44_hys["test_window"][str(seed)]["0.015"]
        h44 = v8_44_hys["april_holdout"][str(seed)]["0.015"]

        benchmark["drl_models"][str(seed)] = {
            "no_institutional_control": {
                "test_window": {"mean_net_return": float(np.mean([v["net_return"] for v in t38.values()])), "stocks": t38},
                "april_holdout": {"mean_net_return": float(np.mean([v["net_return"] for v in h38.values()])), "stocks": h38},
            },
            "flow_only": {
                "test_window": {"mean_net_return": float(np.mean([v["net_return"] for v in t39.values()])), "stocks": t39},
                "april_holdout": {"mean_net_return": float(np.mean([v["net_return"] for v in h39.values()])), "stocks": h39},
            },
            "trend_gated_flow_e2e": {
                "test_window": {"mean_net_return": t43["mean_net_return"], "total_trades": t43["total_trades"], "total_costs": t43["total_costs"], "stocks": t43["stocks"]},
                "april_holdout": {"mean_net_return": h43["mean_net_return"], "total_trades": h43["total_trades"], "total_costs": h43["total_costs"], "stocks": h43["stocks"]},
            },
            "trend_gated_flow_hysteresis_015": {
                "test_window": {"mean_net_return": t44["mean_net_return"], "total_trades": t44["total_trades"], "total_costs": t44["total_costs"], "stocks": t44["stocks"]},
                "april_holdout": {"mean_net_return": h44["mean_net_return"], "total_trades": h44["total_trades"], "total_costs": h44["total_costs"], "stocks": h44["stocks"]},
            },
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(benchmark, indent=2, ensure_ascii=False))
    print(f"Master benchmark saved to {output_path}")


if __name__ == "__main__":
    compile_benchmark(ROOT / "docs/results/v8_45_master_benchmark.json")
