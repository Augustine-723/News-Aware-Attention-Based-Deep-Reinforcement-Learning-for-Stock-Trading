"""Probe new headlines using text previously visible for the same issuer."""
import argparse
import json
from pathlib import Path
from statistics import median

import torch

from . import mops_news_aware_pilot as pilot
from .mops_arrival_text import new_slot_mask, replace_new_text
from .mops_fixed_text import prepare_inputs, weights_digest


def text_controls(obs, previous, days, arrivals):
    mask = new_slot_mask(obs, days, arrivals)
    swapped, blank = obs.clone(), obs.clone()
    pool = torch.cat((previous.recent_event_emb, previous.active_event_emb), dim=1)[previous.event_mask]
    donors = []
    for attr, slots in zip(("recent_event_emb", "active_event_emb"),
                           mask.split((obs.recent_event_emb.size(1), obs.active_event_emb.size(1)), dim=1)):
        for batch, slot in slots.nonzero().tolist():
            current = getattr(obs, attr)[batch, slot]
            # Cycle from the target slot to avoid assigning one donor to every arrival.
            for offset in range(len(pool)):
                index = (slot + offset) % len(pool)
                if not torch.equal(pool[index], current):
                    getattr(swapped, attr)[batch, slot] = pool[index]
                    getattr(blank, attr)[batch, slot].zero_()
                    donors.append(index)
                    break
            else:
                raise ValueError("Missing distinct prior-session donor")
    return swapped, blank, donors


@torch.inference_mode()
def run(cache, checkpoints):
    protocol, factual, periods = prepare_inputs(cache, None, "C")
    refs = json.loads((pilot.ROOT / "docs/results/v8_52_arrival_text.json").read_text())["decisions"]
    lookup = {(r["seed"], r["stock"], r["date"]): r for r in refs}
    digests = {r["seed"]: r["weights_sha256"] for r in
               json.loads((pilot.ROOT / "docs/results/v8_48_fixed_text.json").read_text())}
    rows = []
    for seed in protocol["seeds"]:
        model = pilot.StockRLFullModel().eval()
        model.load_state_dict(torch.load(checkpoints / f"group-c-seed-{seed}.pt",
                                         map_location="cpu", weights_only=True))
        if weights_digest(model) != digests[seed]:
            raise ValueError("Weight digest mismatch")
        for stock in protocol["stocks"]:
            sample = factual["test"][stock]
            valid = factual["validation"][stock]

            def choose(obs, i):
                dates, batches = sample[:2]
                prior_dates, prior_batches = (dates, batches) if i else valid[:2]
                prev_date = prior_dates[i-1]
                date = dates[i].date().isoformat()
                ref = lookup[seed, stock, date]
                days = (dates[i] - prev_date).total_seconds() / 86400
                swapped, blank, donors = text_controls(obs, prior_batches[i-1], days, ref["arrivals"])
                fixed = replace_new_text(obs, periods["test"][1][stock][1][i], days, ref["arrivals"])
                row = dict(ref, donor_date=prev_date.date().isoformat(), donor_indices=donors)
                for label, view in (("real", obs), ("prior", swapped), ("blank", blank), ("replaced", fixed)):
                    q = model(view)[0][0]
                    if not q.isfinite().all():
                        raise ValueError("Non-finite scores")
                    gap, action = float(q[1] - q[0]), int(q.argmax())
                    if label in ("real", "replaced"):
                        prefix = "" if label == "real" else "replaced_"
                        if abs(gap-ref[prefix+"gap"]) > 1e-6 or action != ref[prefix+"action"]:
                            raise ValueError(f"Archived reference mismatch: {seed} {stock} {date} {label}")
                    if label != "real" and not donors and gap != row["real_gap"]:
                        raise ValueError("No-arrival negative control mismatch")
                    row[label + "_gap"], row[label + "_action"] = gap, action
                rows.append(row)
                return row["action"]

            pilot.evaluate(sample, choose)
    if len(rows) != len(lookup):
        raise ValueError("Incomplete replay")
    exposed = [r for r in rows if r["arrivals"]]
    summary = {}
    for label in ("replaced", "blank", "prior"):
        changes = [abs(r[label+"_gap"]-r["real_gap"]) for r in exposed]
        summary[label] = dict(changed_gaps=sum(x>0 for x in changes), median_abs_change=median(changes),
                              max_abs_change=max(changes), flips=sum(r[label+"_action"]!=r["action"] for r in exposed))
    return dict(weights_sha256=digests, summary=summary, decisions=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bge-cache", required=True)
    ap.add_argument("--checkpoints", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    args.output.write_text(json.dumps(run(args.bge_cache, args.checkpoints), indent=2, allow_nan=False)+"\n")
