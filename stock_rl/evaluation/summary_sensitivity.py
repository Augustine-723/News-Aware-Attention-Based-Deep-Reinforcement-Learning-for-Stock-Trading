"""Compare the three retrospective-summary treatments to their matched controls."""
import json

from .market_drl_pilot import ROOT
from .paired_performance import compare


def load_runs(filename):
    result = json.loads((ROOT / "docs" / "results" / filename).read_text())
    return {run["seed"]: run for run in result["runs"]}


def build_report():
    timing = load_runs("v8_34_timing_policy_pilot.json")
    notes = load_runs("v8_37_notes.json")
    constant = load_runs("v8_37_constant.json")
    delayed = load_runs("v8_37_delayed.json")
    if not notes.keys() == timing.keys() == constant.keys() == delayed.keys():
        raise ValueError("unmatched training seeds")
    results = []
    for seed, run in notes.items():
        results.append(dict(seed=seed, notes_vs_market=run["comparison"],
                            notes_vs_timing=compare(timing[seed]["test"], run["test"]),
                            notes_vs_constant=compare(constant[seed]["test"], run["test"]),
                            delayed_vs_notes=compare(run["test"], delayed[seed]["test"])))
    return results


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, allow_nan=False))
