"""Evaluate newly arrived MOPS disclosures on unexamined periods (validation & train)."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from stock_rl.config.schema import get_event_age_index
from . import mops_news_aware_pilot as pilot
from .mops_fixed_text import prepare_inputs, weights_digest
from .mops_new_arrivals import hide_arrivals, previous_history


@torch.inference_mode()
def evaluate_unexamined_arrivals(model, sample, previous_date, previous_batch, counts_stock):
    """Evaluate factual vs hidden vs restored newly arrived disclosures."""
    dates = sample[0]
    elapsed = [(day - prev).total_seconds() / 86400 for day, prev in zip(dates, [previous_date, *dates[:-1]])]
    prior_batches = [previous_batch, *sample[1][:-1]]
    rows = []

    def score(obs):
        q = model(obs)[0][0]
        if not q.isfinite().all():
            raise ValueError("Non-finite action values")
        return int(q.argmax()), float(q[1] - q[0])

    def paired(obs, i):
        date = dates[i].date().isoformat()
        other, n = hide_arrivals(obs, elapsed[i])
        if counts_stock.get(date, 0) != n:
            raise ValueError(f"Ledger and visible new slots disagree on {date}: ledger={counts_stock.get(date, 0)} vs visible={n}")
        action, gap = score(obs)
        hidden, hidden_gap = score(other)
        restored, restored_gap = score(previous_history(obs, prior_batches[i], elapsed[i])) if n else (action, gap)
        rows.append(dict(
            date=date,
            new_announcements=n,
            real_gap=gap,
            hidden_gap=hidden_gap,
            real_action=action,
            hidden_action=hidden,
            restored_gap=restored_gap,
            restored_action=restored,
        ))
        return action

    real = pilot.evaluate(sample, paired)
    hidden = pilot.evaluate(sample, lambda obs, i: score(hide_arrivals(obs, elapsed[i])[0])[0])

    def restored_action(obs, i):
        if not rows[i]["new_announcements"]:
            return score(obs)[0]
        return score(previous_history(obs, prior_batches[i], elapsed[i]))[0]

    restored = pilot.evaluate(sample, restored_action)
    metrics = ("net_return", "executed_position_changes", "costs")
    return dict(
        decisions=rows,
        strategies={
            label: {k: outcome[k] for k in metrics}
            for label, outcome in (("real", real), ("hidden", hidden), ("restored", restored))
        },
    )


def run(cache, april, checkpoints, phases=("validation", "train")):
    protocol, factual, periods = prepare_inputs(cache, april, "B")
    ledger = pilot.load_mops_disclosures(pilot.MOPS_JSON)
    counts = {
        stock: Counter(row["effective_decision_day"][:10] for row in ledger if row["stock_id"] == stock)
        for stock in protocol["stocks"]
    }
    previous_phase_map = {
        "validation": "train",
        "test": "validation",
        "holdout": "test",
    }

    reference = json.loads((pilot.ROOT / "docs/results/v8_50_new_arrivals.json").read_text())
    expected = {(row["seed"], row["weights_sha256"]) for row in reference}
    results = []
    for seed in protocol["seeds"]:
        model = pilot.StockRLFullModel().eval()
        ckpt_path = checkpoints / f"group-b-seed-{seed}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu"))
        digest = weights_digest(model)
        if (seed, digest) not in expected:
            raise ValueError(f"Group B checkpoint differs from V8.50 for seed {seed}: {digest}")

        for phase in phases:
            for stock in protocol["stocks"]:
                sample = factual[phase][stock]
                if phase == "train":
                    dates = sample[0]
                    # For train, decision session 0 has no preceding train session; use synthetic 1-day step
                    prev_date = dates[0] - (dates[1] - dates[0])
                    prev_batch = sample[1][0]
                else:
                    prev_phase = previous_phase_map[phase]
                    prior_dates, prior_inputs, _, _ = factual[prev_phase][stock]
                    prev_date = prior_dates[-1]
                    prev_batch = prior_inputs[-1]

                out = evaluate_unexamined_arrivals(model, sample, prev_date, prev_batch, counts[stock])
                results.append(dict(
                    seed=seed,
                    phase=phase,
                    stock=stock,
                    weights_sha256=digest,
                    **out,
                ))

    # Calculate summary metrics per phase
    summary = {}
    for phase in phases:
        phase_records = [r for r in results if r["phase"] == phase]
        total_decisions = sum(len(r["decisions"]) for r in phase_records)
        active_pairs = [d for r in phase_records for d in r["decisions"] if d["new_announcements"] > 0]
        flips_hidden = sum(1 for d in active_pairs if d["real_action"] != d["hidden_action"])
        flips_restored = sum(1 for d in active_pairs if d["real_action"] != d["restored_action"])
        q_diffs_hidden = [abs(d["real_gap"] - d["hidden_gap"]) for d in active_pairs]
        q_diffs_restored = [abs(d["real_gap"] - d["restored_gap"]) for d in active_pairs]

        stock_days_with_new = len({(r["stock"], d["date"]) for r in phase_records for d in r["decisions"] if d["new_announcements"] > 0})
        total_new_announcements = sum(d["new_announcements"] for r in phase_records if r["seed"] == protocol["seeds"][0] for d in r["decisions"])

        portfolio_returns = {}
        for seed in protocol["seeds"]:
            portfolio_returns[str(seed)] = {
                strat: float(np.mean([r["strategies"][strat]["net_return"] for r in phase_records if r["seed"] == seed]))
                for strat in ("real", "hidden", "restored")
            }

        summary[phase] = {
            "sessions": len(factual[phase][protocol["stocks"][0]][0]) - 1,
            "stock_days_with_new": stock_days_with_new,
            "total_new_announcements": total_new_announcements,
            "paired_decisions_with_new": len(active_pairs),
            "flips_hidden": flips_hidden,
            "flips_restored": flips_restored,
            "hidden_q_gap_median_change": float(np.median(q_diffs_hidden)) if q_diffs_hidden else 0.0,
            "hidden_q_gap_max_change": float(np.max(q_diffs_hidden)) if q_diffs_hidden else 0.0,
            "restored_q_gap_median_change": float(np.median(q_diffs_restored)) if q_diffs_restored else 0.0,
            "restored_q_gap_max_change": float(np.max(q_diffs_restored)) if q_diffs_restored else 0.0,
            "portfolio_net_returns": portfolio_returns,
        }

    return {"summary": summary, "per_stock_results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bge-cache")
    parser.add_argument("--april-cache", type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--phases", default="validation,train", help="Comma-separated phases to evaluate")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    phase_list = [p.strip() for p in args.phases.split(",") if p.strip()]
    output_data = run(args.bge_cache, args.april_cache, args.checkpoints, phases=phase_list)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2, allow_nan=False) + "\n")
