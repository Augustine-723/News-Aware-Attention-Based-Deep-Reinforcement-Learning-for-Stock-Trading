"""Replace only newly arrived headlines on frozen V8.48 factual trajectories."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

import torch

from stock_rl.config.schema import get_event_age_index
from . import mops_news_aware_pilot as pilot
from .mops_fixed_text import prepare_inputs, weights_digest


def new_slot_mask(obs, days, expected_count):
    if days <= 0:
        raise ValueError("Invalid interval")
    new = obs.event_mask & (obs.event_metadata[..., get_event_age_index(obs.schema_version)] < days)
    if int(new.sum()) != expected_count:
        raise ValueError("New slot count differs from the disclosure ledger")
    return new


def replace_new_text(obs, constant, days, expected_count):
    if not torch.equal(obs.event_mask, constant.event_mask):
        raise ValueError("Unaligned event slots")
    new = new_slot_mask(obs, days, expected_count)
    other = obs.clone()
    recent, active = new.split([obs.recent_event_emb.size(1), obs.active_event_emb.size(1)], dim=1)
    other.recent_event_emb[recent] = constant.recent_event_emb[recent]
    other.active_event_emb[active] = constant.active_event_emb[active]
    return other


@torch.inference_mode()
def run(cache, checkpoints):
    protocol, factual, periods = prepare_inputs(cache, None, "C")
    refs = json.loads((pilot.ROOT / "docs/results/v8_48_fixed_text.json").read_text())
    counts = Counter((r["stock_id"], r["effective_decision_day"][:10])
                     for r in pilot.load_mops_disclosures(pilot.MOPS_JSON))
    rows = []
    for seed in protocol["seeds"]:
        model = pilot.StockRLFullModel().eval()
        model.load_state_dict(torch.load(checkpoints / f"group-c-seed-{seed}.pt", weights_only=True, map_location="cpu"))
        digest = weights_digest(model)
        for stock in protocol["stocks"]:
            ref, = [r for r in refs if (r["phase"], r["seed"], r["stock"]) == ("test", seed, stock)]
            actual, constant = (inputs[stock] for inputs in periods["test"])
            dates = actual[0]
            if digest != ref["weights_sha256"] or not dates.equals(constant[0]) or len(ref["decisions"]) != len(dates)-1:
                raise ValueError("Frozen reference or date mismatch")
            previous = factual["validation"][stock][0][-1]

            def choose(obs, i):
                date = dates[i].date().isoformat()
                days = (dates[i] - (dates[i-1] if i else previous)).total_seconds() / 86400
                n = counts[stock, date]
                other = replace_new_text(obs, constant[1][i], days, n)
                real, replaced = (model(x)[0][0] for x in (obs, other))
                if not (real.isfinite().all() and replaced.isfinite().all()):
                    raise ValueError("Non-finite action scores")
                gap, new_gap = (float(q[1]-q[0]) for q in (real, replaced))
                r = ref["decisions"][i]
                if (date, gap, int(real.argmax())) != (r["date"], r["real_gap"], r["real_action"]):
                    raise ValueError("Factual decision differs from V8.48")
                if not n and new_gap != gap:
                    raise ValueError("No-arrival negative control failed")
                rows.append(dict(stock=stock, seed=seed, date=date, arrivals=n, gap=gap, replaced_gap=new_gap,
                                 action=int(real.argmax()), replaced_action=int(replaced.argmax())))
                return int(real.argmax())

            pilot.evaluate(actual, choose)
    exposed = [r for r in rows if r["arrivals"]]
    changes = [abs(r["replaced_gap"]-r["gap"]) for r in exposed]
    return dict(phase="test", summary=dict(decisions=len(rows), exposed_pairs=len(exposed),
                exposed_stock_days=len({(r["stock"], r["date"]) for r in exposed}),
                flips=sum(r["action"]!=r["replaced_action"] for r in exposed),
                changed_q_gaps=sum(x>0 for x in changes), median_abs_change=median(changes),
                max_abs_change=max(changes)), decisions=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bge-cache")
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.bge_cache, args.checkpoints), indent=2, allow_nan=False)+"\n")
