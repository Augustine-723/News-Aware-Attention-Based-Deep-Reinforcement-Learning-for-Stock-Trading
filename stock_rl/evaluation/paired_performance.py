"""Paired time-block bootstrap for matched out-of-sample equity sleeves."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def aligned_sleeves(control, treatment):
    """Convert matched, independently funded stock sleeves into portfolio returns."""
    if not control or control.keys() != treatment.keys():
        raise ValueError("both arms require the same nonempty stock set")
    calendar = None
    arms = []
    for arm in (control, treatment):
        normalized = []
        for stock in sorted(arm):
            sleeve = arm[stock]
            dates = pd.DatetimeIndex(pd.to_datetime(sleeve["session_dates"], format="%Y-%m-%d"))
            equity = np.asarray(sleeve["equity"], dtype=np.float64)
            if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
                raise ValueError(f"{stock}: invalid session dates")
            if equity.shape != (len(dates),) or len(equity) < 3:
                raise ValueError(f"{stock}: invalid equity curve length")
            if not np.isfinite(equity).all() or (equity <= 0).any():
                raise ValueError(f"{stock}: equity values must be finite and positive")
            if calendar is None:
                calendar = dates
            if not dates.equals(calendar):
                raise ValueError(f"{stock}: unmatched trading sessions")
            normalized.append(equity / equity[0])
        portfolio = np.mean(normalized, axis=0)
        arms.append(portfolio[1:] / portfolio[:-1] - 1)
    return calendar[1:], np.column_stack(arms)


def compare(control, treatment, *, block_length=5, replicates=5000, seed=42):
    """Estimate treatment minus control on identical resampled date blocks."""
    dates, paired = aligned_sleeves(control, treatment)
    days = len(paired)
    if not 2 <= block_length < days or replicates < 1:
        raise ValueError("require 2 <= block length < returned sessions and positive replicates")

    def sharpe(returns):
        volatility = returns.std(axis=-2)
        excess = returns.mean(axis=-2) - 0.015 / 250
        return np.divide(
            np.sqrt(250) * excess, volatility,
            out=np.full_like(volatility, np.nan), where=volatility > 1e-12,
        )

    net_returns = np.prod(1 + paired, axis=0) - 1
    scores = sharpe(paired)
    result = {
        "first_return_session": dates[0].date().isoformat(),
        "last_return_session": dates[-1].date().isoformat(),
        "matched_return_sessions": days,
        "stocks": sorted(control),
        "control_net_return": float(net_returns[0]),
        "treatment_net_return": float(net_returns[1]),
        "delta_net_return": float(net_returns[1] - net_returns[0]),
        "control_sharpe": float(scores[0]) if np.isfinite(scores[0]) else None,
        "treatment_sharpe": float(scores[1]) if np.isfinite(scores[1]) else None,
        "delta_sharpe": None,
        "delta_sharpe_ci95": None,
        "block_length": block_length,
        "bootstrap_seed": seed,
        "bootstrap_requested": replicates,
        "bootstrap_defined": 0,
    }
    if not np.isfinite(scores).all():
        result["status"] = "undefined_sharpe_zero_volatility"
        return result

    result["delta_sharpe"] = float(scores[1] - scores[0])
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, days, size=(replicates, (days + block_length - 1) // block_length))
    indices = ((starts[..., None] + np.arange(block_length)) % days).reshape(replicates, -1)[:, :days]
    simulated = sharpe(paired[indices])
    deltas = simulated[:, 1] - simulated[:, 0]
    finite = deltas[np.isfinite(deltas)]
    result["bootstrap_defined"] = len(finite)
    if len(finite) != replicates:
        result["status"] = "ci_undefined_in_some_resamples"
        return result
    result["status"] = "descriptive_interval"
    result["delta_sharpe_ci95"] = np.quantile(finite, [0.025, 0.975]).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path, help="JSON stock mapping with session_dates and equity")
    parser.add_argument("treatment", type=Path, help="JSON stock mapping with session_dates and equity")
    parser.add_argument("--block-length", type=int, default=5)
    args = parser.parse_args()
    report = compare(json.loads(args.control.read_text()), json.loads(args.treatment.read_text()),
                     block_length=args.block_length)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
