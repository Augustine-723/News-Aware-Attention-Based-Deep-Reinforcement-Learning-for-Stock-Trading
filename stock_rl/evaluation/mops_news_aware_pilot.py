"""News-Aware DRL comparative experiment on 404 authentic MOPS statutory disclosures.

Four experimental groups:
- Group A: Flow Benchmark (No announcement input / M0)
- Group B: Announcement Timing & Recency Only (M1: event_mask and event_age, text masked)
- Group C: Group B + Real Headline BGE-small-zh Embeddings (M3)
- Group D: Group B + Constant Sentence BGE-small-zh Embeddings (Placebo Control / M3)
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from stock_rl.config.schema import CURRENT_ACTIVE_SCHEMA_VERSION
from stock_rl.data.normalizer import FeatureNormalizer, SplitManifest
from stock_rl.data.pit_engine import MissingStatus, ProvenanceRecord
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.env.stock_trading_env import PointInTimeStockEnv
from stock_rl.evaluation.april_holdout import prepare_files
from stock_rl.evaluation.market_drl_pilot import (
    DATA_DIR,
    PHASES,
    ROOT,
    STOCK_METADATA,
    StockRLFullModel,
    compute_trading_metrics,
    evaluate,
    fit_seed,
    load_real_market_feeds,
)
from stock_rl.evaluation.news_input_pilot import mask_event_inputs
from stock_rl.evaluation.provenance import load_trusted_pretrained_embedding_engine

MOPS_JSON = ROOT / "data" / "official_disclosures" / "mops_official_disclosures_2023_2024.json"
RESULTS_JSON = ROOT / "docs" / "results" / "v8_47_mops_news_aware_experiment.json"
CONSTANT_SENTENCE = "本公司公告重大訊息。"


def load_mops_disclosures(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"MOPS disclosures not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["disclosures"]


def build_mops_event_ledgers(
    disclosures: List[Dict[str, Any]],
    embedding_engine,
    constant_sentence: str = CONSTANT_SENTENCE,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """Build event records with real BGE embeddings (events_real) and constant embeddings (events_const)."""
    # 1. Precompute embeddings for all unique headlines
    unique_headlines = sorted(set(d["headline"] for d in disclosures))
    print(f"Embedding {len(unique_headlines)} unique MOPS headlines with BGE-small-zh-v1.5...")
    real_emb_map = {h: embedding_engine.embed_text(h) for h in unique_headlines}
    const_emb = embedding_engine.embed_text(constant_sentence)

    events_real: Dict[str, List[Dict[str, Any]]] = {"2330": [], "2603": [], "2881": []}
    events_const: Dict[str, List[Dict[str, Any]]] = {"2330": [], "2603": [], "2881": []}

    for d in disclosures:
        stock = d["stock_id"]
        if stock not in events_real:
            continue
        f_dt = pd.Timestamp(d["filing_datetime"])
        event_id = f"MOPS_{stock}_{d['roc_date'].replace('/', '')}_{d['seq_no']}"
        headline = d["headline"]

        prov = ProvenanceRecord(
            source_id=event_id,
            source_published_at=f_dt,
            available_at=f_dt,
            is_audit_verified=False,
            missing_status=MissingStatus.AVAILABLE.value,
            mode="estimated",
        )

        rec_real = {
            "event_id": event_id,
            "provenance": prov,
            "headline": headline,
            "text_embedding": real_emb_map[headline],
        }
        rec_const = {
            "event_id": event_id,
            "provenance": prov,
            "headline": headline,
            "text_embedding": const_emb,
        }

        events_real[stock].append(rec_real)
        events_const[stock].append(rec_const)

    return events_real, events_const


def build_study_inputs(
    protocol: Dict[str, Any],
    manifest: SplitManifest,
    events_real: Dict[str, List[Dict[str, Any]]],
    events_const: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Build 4 normalized input bundles (A, B, C, D) for train, validation, and test phases.
    Uses institutional_only='flow'.
    """
    calendar = pd.DatetimeIndex(
        pd.read_csv(Path(DATA_DIR) / "taiex.csv", usecols=["date"], parse_dates=["date"])["date"]
    ).tz_localize(protocol["timezone"]).sort_values()

    raw_snapshots_real = {phase: {} for phase in PHASES}
    raw_snapshots_const = {phase: {} for phase in PHASES}

    print("Building historical snapshots with MOPS disclosures...")
    for stock in protocol["stocks"]:
        prices, inst, macro, foreign, context, _ = load_real_market_feeds(stock, mode="estimated")
        meta = STOCK_METADATA[stock]

        builder_real = HistoricalSnapshotBuilder(
            prices, inst, macro, foreign, stock_context_records=context,
            event_records=events_real[stock],
            active_event_records=[],
            shares_outstanding=meta["shares_outstanding"],
            mode="estimated",
        )
        builder_const = HistoricalSnapshotBuilder(
            prices, inst, macro, foreign, stock_context_records=context,
            event_records=events_const[stock],
            active_event_records=[],
            shares_outstanding=meta["shares_outstanding"],
            mode="estimated",
        )

        for phase in PHASES:
            start, end = protocol["splits"][phase]
            sessions = prices.loc[start:end]
            decisions = sessions.index.normalize() + pd.Timedelta(hours=protocol["decision_hour"])
            batches_real = [builder_real.build_snapshot(ts) for ts in decisions]
            batches_const = [builder_const.build_snapshot(ts) for ts in decisions]

            # Apply institutional flow filter (I06-I08 ablated)
            for b in batches_real:
                b.institutional[..., 5:8].zero_()
                b.institutional_missing_mask[..., 5:8].fill_(True)
            for b in batches_const:
                b.institutional[..., 5:8].zero_()
                b.institutional_missing_mask[..., 5:8].fill_(True)

            ohlc = sessions[["open", "high", "low", "close"]].to_numpy()
            raw_snapshots_real[phase][stock] = (decisions, batches_real, ohlc)
            raw_snapshots_const[phase][stock] = (decisions, batches_const, ohlc)

    # Construct the 4 treatment groups
    groups = {}
    for group_key in ("A", "B", "C", "D"):
        group_inputs = {phase: {} for phase in PHASES}
        for phase in PHASES:
            for stock in protocol["stocks"]:
                decisions, b_real, ohlc = raw_snapshots_real[phase][stock]
                _, b_const, _ = raw_snapshots_const[phase][stock]

                if group_key == "A":
                    # M0: completely empty/masked events
                    batches = [mask_event_inputs(b, "M0") for b in b_real]
                elif group_key == "B":
                    # M1: event timing and recency preserved, text masked
                    batches = [mask_event_inputs(b, "M1") for b in b_real]
                elif group_key == "C":
                    # M3: real BGE headline embeddings
                    batches = [b.clone() for b in b_real]
                elif group_key == "D":
                    # M3: constant sentence BGE embeddings
                    batches = [b.clone() for b in b_const]

                group_inputs[phase][stock] = (decisions, batches, ohlc, None)

        # Fit normalizer on training split of this group
        train_dates, train_batches = [], []
        for decisions, batches, _, _ in group_inputs["train"].values():
            train_dates.extend(decisions)
            train_batches.extend(batches)
        normalizer = FeatureNormalizer().fit(
            train_batches, sample_timestamps=train_dates, split_manifest=manifest
        )

        # Normalize all splits
        for phase in PHASES:
            for stock, (decisions, batches, ohlc, trend) in group_inputs[phase].items():
                group_inputs[phase][stock] = (
                    decisions,
                    [normalizer.transform(b) for b in batches],
                    ohlc,
                    trend,
                )

        groups[group_key] = {"inputs": group_inputs, "normalizer": normalizer}

    return groups


def build_april_holdout_inputs(
    protocol: Dict[str, Any],
    normalizers: Dict[str, FeatureNormalizer],
    events_real: Dict[str, List[Dict[str, Any]]],
    events_const: Dict[str, List[Dict[str, Any]]],
    cache_dir: Path,
) -> Dict[str, Dict[str, Tuple[pd.DatetimeIndex, List[Any], np.ndarray]]]:
    """Build normalized April holdout inputs for each group A, B, C, D."""
    stocks_raw_real = {}
    stocks_raw_const = {}

    with tempfile.TemporaryDirectory() as tmp:
        prepare_files(cache_dir, Path(tmp))
        for stock in protocol["stocks"]:
            meta = STOCK_METADATA[stock]
            prices, inst, macro, foreign, context, _ = load_real_market_feeds(stock, tmp, mode="estimated")

            builder_real = HistoricalSnapshotBuilder(
                prices, inst, macro, foreign, context,
                event_records=events_real[stock],
                shares_outstanding=meta["shares_outstanding"],
                mode="estimated",
            )
            builder_const = HistoricalSnapshotBuilder(
                prices, inst, macro, foreign, context,
                event_records=events_const[stock],
                shares_outstanding=meta["shares_outstanding"],
                mode="estimated",
            )

            april = prices.loc["2024-04"]
            dates = april.index.normalize() + pd.Timedelta(hours=20)
            batches_real = [builder_real.build_snapshot(d) for d in dates]
            batches_const = [builder_const.build_snapshot(d) for d in dates]

            # Flow only ablation
            for b in batches_real:
                b.institutional[..., 5:8].zero_()
                b.institutional_missing_mask[..., 5:8].fill_(True)
            for b in batches_const:
                b.institutional[..., 5:8].zero_()
                b.institutional_missing_mask[..., 5:8].fill_(True)

            ohlc = april[["open", "high", "low", "close"]].to_numpy()
            stocks_raw_real[stock] = (dates, batches_real, ohlc)
            stocks_raw_const[stock] = (dates, batches_const, ohlc)

    april_groups = {}
    for group_key in ("A", "B", "C", "D"):
        norm = normalizers[group_key]
        april_items = {}
        for stock in protocol["stocks"]:
            dates, b_real, ohlc = stocks_raw_real[stock]
            _, b_const, _ = stocks_raw_const[stock]

            if group_key == "A":
                batches = [mask_event_inputs(b, "M0") for b in b_real]
            elif group_key == "B":
                batches = [mask_event_inputs(b, "M1") for b in b_real]
            elif group_key == "C":
                batches = [b.clone() for b in b_real]
            elif group_key == "D":
                batches = [b.clone() for b in b_const]

            norm_batches = [norm.transform(b) for b in batches]
            april_items[stock] = (dates, norm_batches, ohlc, None)
        april_groups[group_key] = april_items

    return april_groups


def run_experiment():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((ROOT / "configs/v8_22_study.json").read_text())

    boundaries = [
        pd.Timestamp(day, tz=protocol["timezone"]) + pd.Timedelta(hours=protocol["decision_hour"])
        for phase in PHASES for day in protocol["splits"][phase]
    ]
    digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(DATA_DIR).glob("*.csv"))}
    raw_hash = SplitManifest.compute_content_hash(digest)
    manifest = SplitManifest.create("mops-news-aware-pilot", CURRENT_ACTIVE_SCHEMA_VERSION, *boundaries, raw_dataset_content_hash=raw_hash)

    # 1. Load MOPS disclosures & BGE engine
    disclosures = load_mops_disclosures(MOPS_JSON)
    print(f"Loaded {len(disclosures)} authentic MOPS disclosures.")
    engine = load_trusted_pretrained_embedding_engine()
    if engine is None:
        raise RuntimeError("Verified BGE embedding engine required but could not be loaded.")

    events_real, events_const = build_mops_event_ledgers(disclosures, engine)

    # 2. Build Study Inputs for Train/Val/Test
    study_groups = build_study_inputs(protocol, manifest, events_real, events_const)
    normalizers = {k: v["normalizer"] for k, v in study_groups.items()}

    # 3. Build April Holdout Inputs
    cache_dir = ROOT / "april_source_cache"
    april_groups = build_april_holdout_inputs(protocol, normalizers, events_real, events_const, cache_dir)

    seeds = protocol["seeds"]
    results = {
        "metadata": {
            "experiment": "V8.47 MOPS News-Aware Controlled Experiment",
            "mops_disclosures_count": len(disclosures),
            "groups": {
                "A": "Flow Benchmark (No announcement input / M0)",
                "B": "Announcement Timing & Recency Only (M1: event_mask + event_age, text masked)",
                "C": "Group B + Real Headline BGE-small-zh Embeddings (M3)",
                "D": "Group B + Constant Sentence BGE-small-zh Embeddings (Placebo / M3)",
            },
            "constant_sentence": CONSTANT_SENTENCE,
            "bge_model_id": "BAAI/bge-small-zh-v1.5",
            "seeds": seeds,
        },
        "retrained_models": {},
        "frozen_benchmark_sensitivity": {},
    }

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Retraining Models Under Each Treatment (A, B, C, D)")
    print("=" * 70)

    for group_key in ("A", "B", "C", "D"):
        print(f"\n>>> Running Group {group_key}: {results['metadata']['groups'][group_key]}")
        inputs = study_groups[group_key]["inputs"]
        april_in = april_groups[group_key]
        group_res = {"test": {}, "holdout": {}}

        for seed in seeds:
            t0 = time.time()
            agent, best_epoch, trace = fit_seed(seed, protocol, inputs)
            fit_time = time.time() - t0

            # Evaluate on Test period (Feb-Mar)
            test_eval = {
                stock: evaluate(sample, lambda obs, _: agent.select_action(obs, evaluate=True))
                for stock, sample in inputs["test"].items()
            }
            # Portfolio test return (equal weighted 3 stocks)
            test_net_ret = float(np.mean([test_eval[s]["net_return"] for s in protocol["stocks"]]))
            test_trades = sum(test_eval[s]["executed_position_changes"] for s in protocol["stocks"])
            test_cost = sum(sum(test_eval[s]["costs"].values()) for s in protocol["stocks"])

            # Evaluate on April Holdout
            holdout_eval = {
                stock: evaluate(sample, lambda obs, _: agent.select_action(obs, evaluate=True))
                for stock, sample in april_in.items()
            }
            holdout_net_ret = float(np.mean([holdout_eval[s]["net_return"] for s in protocol["stocks"]]))
            holdout_trades = sum(holdout_eval[s]["executed_position_changes"] for s in protocol["stocks"])
            holdout_cost = sum(sum(holdout_eval[s]["costs"].values()) for s in protocol["stocks"])

            print(f"  Seed {seed} (Epoch {best_epoch}, {fit_time:.1f}s): "
                  f"Test Net={test_net_ret*100:+.2f}%, Trades={test_trades} | "
                  f"April Net={holdout_net_ret*100:+.2f}%, Trades={holdout_trades}")

            group_res["test"][str(seed)] = {
                "best_epoch": best_epoch,
                "net_return": test_net_ret,
                "trades": test_trades,
                "costs": test_cost,
                "per_stock": test_eval,
            }
            group_res["holdout"][str(seed)] = {
                "net_return": holdout_net_ret,
                "trades": holdout_trades,
                "costs": holdout_cost,
                "per_stock": holdout_eval,
            }

        results["retrained_models"][group_key] = group_res

    # 4. Action Difference Analysis
    print("\n" + "=" * 70)
    print("ACTION COMPARISON & DECISION SHIFT AUDIT (Retrained Models)")
    print("=" * 70)
    comparisons = [("A", "B", "Effect of Knowing Announcement Occurred (Timing Only)"),
                   ("B", "C", "Effect of Real Headline BGE Semantic Embeddings"),
                   ("C", "D", "Specificity of Headline Content (Real vs Constant Placebo)")]

    action_audit = {}
    for g1, g2, desc in comparisons:
        action_audit[f"{g1}_vs_{g2}"] = {"description": desc, "test_diffs": {}, "holdout_diffs": {}}
        print(f"\n--- Comparing Group {g1} vs Group {g2} ({desc}) ---")
        for period, period_key in (("Test (Feb-Mar)", "test"), ("Holdout (April)", "holdout")):
            diff_counts = []
            total_actions = 0
            for seed in seeds:
                s_str = str(seed)
                seed_diff = 0
                for stock in protocol["stocks"]:
                    act1 = results["retrained_models"][g1][period_key][s_str]["per_stock"][stock]["target_actions"]
                    act2 = results["retrained_models"][g2][period_key][s_str]["per_stock"][stock]["target_actions"]
                    total_actions += len(act1)
                    diff = sum(a1 != a2 for a1, a2 in zip(act1, act2))
                    seed_diff += diff
                diff_counts.append(seed_diff)
                print(f"  {period} Seed {seed}: {seed_diff} actions differed.")
            action_audit[f"{g1}_vs_{g2}"][f"{period_key}_diffs"] = {
                "diff_by_seed": dict(zip(seeds, diff_counts)),
                "total_differences": sum(diff_counts),
                "total_evaluated_decisions": total_actions,
                "exact_match_rate": 1.0 - (sum(diff_counts) / total_actions if total_actions > 0 else 0.0),
            }

    results["action_audit"] = action_audit

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSuccessfully compiled news-aware experiment results to: {RESULTS_JSON}")


if __name__ == "__main__":
    run_experiment()
