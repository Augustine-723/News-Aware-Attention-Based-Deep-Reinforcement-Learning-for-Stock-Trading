"""Remove only newly visible MOPS announcements on frozen V8.49 timing models."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from stock_rl.config.schema import get_event_age_index
from . import mops_news_aware_pilot as pilot
from .mops_fixed_text import prepare_inputs, weights_digest


def hide_arrivals(obs, elapsed_days):
    if elapsed_days <= 0:
        raise ValueError("Previous decision must precede the current decision")
    age = get_event_age_index(obs.schema_version)
    new = obs.event_mask & (obs.event_metadata[..., age] < elapsed_days)
    other = obs.clone()
    other.event_mask[new] = False
    other.event_metadata[new] = 0
    if other.event_metadata_missing_mask is not None:
        other.event_metadata_missing_mask[new] = True
    return other, int(new.sum())


def previous_history(obs, previous, elapsed_days):
    # B inputs contain only occupied slots and E23. Retain the current market and position.
    other = obs.clone()
    age = get_event_age_index(obs.schema_version)
    other.event_mask.copy_(previous.event_mask)
    other.event_metadata.copy_(previous.event_metadata)
    other.event_metadata[..., age] += elapsed_days * previous.event_mask
    mask = previous.event_metadata_missing_mask
    other.event_metadata_missing_mask = mask.clone() if mask is not None else None
    return other


@torch.inference_mode()
def evaluate_arrivals(model, sample, previous_date, previous_batch, expected, counts):
    dates = sample[0]
    if len(expected) != len(dates) - 1:
        raise ValueError("Archived evaluation length mismatch")
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
        if counts[date] != n:
            raise ValueError(f"Ledger and visible new slots disagree on {date}")
        action, gap = score(obs)
        if (date, action, gap) != (expected[i]["date"], expected[i]["real_action"], expected[i]["real_gap"]):
            raise ValueError("Factual replay differs from frozen V8.49 output")
        hidden, hidden_gap = score(other)
        restored, restored_gap = score(previous_history(obs, prior_batches[i], elapsed[i])) if n else (action, gap)
        rows.append(dict(date=date, new_announcements=n, real_gap=gap, hidden_gap=hidden_gap,
                         real_action=action, hidden_action=hidden, restored_gap=restored_gap, restored_action=restored))
        return action

    real = pilot.evaluate(sample, paired)
    hidden = pilot.evaluate(sample, lambda obs, i: score(hide_arrivals(obs, elapsed[i])[0])[0])
    def restored_action(obs, i):
        if not rows[i]["new_announcements"]:
            return score(obs)[0]
        return score(previous_history(obs, prior_batches[i], elapsed[i]))[0]

    restored = pilot.evaluate(sample, restored_action)
    metrics = ("net_return", "executed_position_changes", "costs")
    return dict(decisions=rows, strategies={label: {k: outcome[k] for k in metrics}
                for label, outcome in (("real", real), ("hidden", hidden), ("restored", restored))})


def run(cache, april, checkpoints):
    protocol, factual, periods = prepare_inputs(cache, april, "B")
    archived = json.loads((pilot.ROOT / "docs/results/v8_49_fixed_timing.json").read_text())
    previous_phase = {"test": "validation", "holdout": "test"}
    ledger = pilot.load_mops_disclosures(pilot.MOPS_JSON)
    counts = {stock: Counter(row["effective_decision_day"][:10] for row in ledger if row["stock_id"] == stock)
              for stock in protocol["stocks"]}
    result = []
    for seed in protocol["seeds"]:
        model = pilot.StockRLFullModel().eval()
        model.load_state_dict(torch.load(checkpoints / f"group-b-seed-{seed}.pt", weights_only=True, map_location="cpu"))
        digest = weights_digest(model)
        for ref in archived:
            phase, stock = ref["phase"], ref["stock"]
            if ref["seed"] != seed or phase not in periods:
                continue
            if digest != ref["weights_sha256"]:
                raise ValueError("Checkpoint differs from V8.49")
            prior_dates, prior_inputs, _, _ = factual[previous_phase[phase]][stock]
            out = evaluate_arrivals(model, periods[phase][0][stock], prior_dates[-1], prior_inputs[-1],
                                    ref["decisions"], counts[stock])
            result.append(dict(seed=seed, phase=phase, stock=stock, weights_sha256=digest, **out))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bge-cache")
    parser.add_argument("--april-cache", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.bge_cache, args.april_cache, args.checkpoints),
                                      indent=2, allow_nan=False) + "\n")
