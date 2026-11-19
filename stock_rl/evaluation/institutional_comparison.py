"""Compare retained versus omitted institutional features on matching test dates."""
import argparse
import json
from pathlib import Path

from .market_drl_pilot import ROOT
from .paired_performance import compare


def compare_runs(ablated, full):
    if ablated["status"] != "NO_INSTITUTIONAL_ESTIMATED_PILOT":
        raise ValueError("expected a run without I01-I08")
    for key in ("protocol", "input_sha256", "trading_config", "sessions"):
        if ablated[key] != full[key]:
            raise ValueError(f"mismatched {key}")
    controls = {r["seed"]: r for r in ablated["without_institutional_runs"]}
    if set(controls) != {r["seed"] for r in full["m0_runs"]}:
        raise ValueError("mismatched seeds")
    return [dict(seed=r["seed"], **compare(controls[r["seed"]]["test"], r["test"]))
            for r in full["m0_runs"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablated", type=Path)
    args = parser.parse_args()
    baseline = ROOT / "docs/results/v8_22_market_drl_pilot.json"
    print(json.dumps(compare_runs(json.loads(args.ablated.read_text()),
                                  json.loads(baseline.read_text())), indent=2, allow_nan=False))
