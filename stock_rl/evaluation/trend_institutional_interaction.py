"""Cross-analyze institutional flow impact conditioned on MA20 trend regimes."""
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
STOCKS = ["2330", "2603", "2881"]


def load_full_price_series(cache_dir=None):
    if cache_dir is None:
        cache_dir = ROOT / "april_source_cache"
    data_dir = ROOT / "data/poc_real_data"
    closes = {}
    dates = {}
    for s in STOCKS:
        with open(data_dir / f"{s}_ohlcv.csv") as f:
            reader = list(csv.DictReader(f))
        hist = [(r["date"], float(r["close"])) for r in reader]
        apr_file = cache_dir / f"{s}_ohlcv.json"
        if apr_file.exists():
            apr = [(r["date"], float(r["close"])) for r in json.loads(apr_file.read_text())["data"]]
        else:
            apr = []
        merged = hist + apr
        dates[s] = [x[0] for x in merged]
        closes[s] = [x[1] for x in merged]
    return dates, closes


def classify_regimes(dates_list, closes_list):
    """Classify each date as ABOVE or BELOW 20-day simple moving average."""
    date_to_idx = {d: i for i, d in enumerate(dates_list)}
    regimes = {}
    for d, idx in date_to_idx.items():
        if idx < 19:
            continue
        c_t = closes_list[idx]
        ma20 = sum(closes_list[idx - 19:idx + 1]) / 20.0
        regimes[d] = "ABOVE" if c_t >= ma20 else "BELOW"
    return regimes


def run_interaction_analysis():
    dates_dict, closes_dict = load_full_price_series()
    regimes_dict = {s: classify_regimes(dates_dict[s], closes_dict[s]) for s in STOCKS}

    with open(ROOT / "docs/results/v8_39_flow.json") as f:
        flow_feb = json.load(f)
    with open(ROOT / "docs/results/v8_38_no_institutional.json") as f:
        ctrl_feb = json.load(f)
    with open(ROOT / "docs/results/v8_40_april_holdout.json") as f:
        april_hold = json.load(f)

    results = {}
    for seed_idx, seed in enumerate([42, 100, 2024]):
        c_feb_runs = ctrl_feb["without_institutional_runs"][seed_idx]["test"]
        t_feb_runs = flow_feb["flow_only_runs"][seed_idx]["test"]
        feb_dates = c_feb_runs["2330"]["session_dates"]

        c_apr_runs = april_hold[seed_idx]["arms"]["without_institutional"]
        t_apr_runs = april_hold[seed_idx]["arms"]["flow_only"]
        apr_dates = c_apr_runs["2330"]["session_dates"]

        seed_data = {"feb_mar": {"ABOVE": 0.0, "BELOW": 0.0},
                     "april": {"ABOVE": 0.0, "BELOW": 0.0},
                     "aggregated_52d": {"ABOVE": 0.0, "BELOW": 0.0, "total": 0.0},
                     "stocks": {s: {"ABOVE": {"ctrl_pnl": 0.0, "treat_pnl": 0.0, "days": 0},
                                    "BELOW": {"ctrl_pnl": 0.0, "treat_pnl": 0.0, "days": 0}}
                                for s in STOCKS}}

        for s in STOCKS:
            # Feb-Mar window
            c_eq = c_feb_runs[s]["equity"]
            t_eq = t_feb_runs[s]["equity"]
            for i in range(len(feb_dates) - 1):
                d = feb_dates[i]
                r = regimes_dict[s].get(d, "ABOVE")
                c_diff = c_eq[i + 1] - c_eq[i]
                t_diff = t_eq[i + 1] - t_eq[i]
                d_delta = (t_diff - c_diff) / 3e4
                seed_data["feb_mar"][r] += d_delta
                seed_data["aggregated_52d"][r] += d_delta
                seed_data["aggregated_52d"]["total"] += d_delta
                seed_data["stocks"][s][r]["ctrl_pnl"] += c_diff
                seed_data["stocks"][s][r]["treat_pnl"] += t_diff
                seed_data["stocks"][s][r]["days"] += 1

            # April window
            c_eq_a = c_apr_runs[s]["equity"]
            t_eq_a = t_apr_runs[s]["equity"]
            for i in range(len(apr_dates) - 1):
                d = apr_dates[i]
                r = regimes_dict[s].get(d, "ABOVE")
                c_diff = c_eq_a[i + 1] - c_eq_a[i]
                t_diff = t_eq_a[i + 1] - t_eq_a[i]
                d_delta = (t_diff - c_diff) / 3e4
                seed_data["april"][r] += d_delta
                seed_data["aggregated_52d"][r] += d_delta
                seed_data["aggregated_52d"]["total"] += d_delta
                seed_data["stocks"][s][r]["ctrl_pnl"] += c_diff
                seed_data["stocks"][s][r]["treat_pnl"] += t_diff
                seed_data["stocks"][s][r]["days"] += 1

        results[str(seed)] = seed_data
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/v8_41_trend_interaction.json")
    args = parser.parse_args()
    res = run_interaction_analysis()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"Results written to {args.output}")
    print(json.dumps(res, indent=2))
