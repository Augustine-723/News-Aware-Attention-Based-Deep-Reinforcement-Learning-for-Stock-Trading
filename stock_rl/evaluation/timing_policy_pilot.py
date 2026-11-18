"""Train an estimated timing-only policy and compare it to the frozen M0 runs."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from stock_rl.config.schema import CURRENT_ACTIVE_SCHEMA_VERSION
from stock_rl.data.normalizer import SplitManifest
from stock_rl.data.pit_engine import MissingStatus, ProvenanceRecord
from .market_drl_pilot import DATA_DIR, PHASES, ROOT, evaluate, fit_seed, prepare_inputs
from .paired_performance import compare


def timing_records(ledger, stocks):
    events = {stock: [] for stock in stocks}
    for row in ledger:
        if row["status"] != "estimated_candidate" or "phase" not in row:
            continue
        published = pd.Timestamp(row["source_published_at"])
        available = pd.Timestamp(row["estimated_available_at"])
        if published.tz is None or available.tz is None or available < published:
            raise ValueError("invalid estimated timing")
        events[row["stock"]].append(dict(event_id=row["url"], provenance=ProvenanceRecord(
            source_id=row["url"], source_published_at=published, available_at=available,
            is_audit_verified=False, missing_status=MissingStatus.AVAILABLE.value, mode="estimated")))
    return events


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    baseline = json.loads((ROOT / "docs/results/v8_22_market_drl_pilot.json").read_bytes())
    protocol = json.loads((ROOT / "configs/v8_22_study.json").read_bytes())
    digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(DATA_DIR).glob("*.csv")}
    if protocol != baseline["protocol"] or digest != baseline["input_sha256"]:
        raise ValueError("market inputs or protocol differ from reference")
    raw = (ROOT / "docs/results/v8_33_news_split_coverage.json").read_bytes()
    ledger_sha = hashlib.sha256(raw).hexdigest()
    events = timing_records(json.loads(raw)["ledger"], protocol["stocks"])
    bounds = [pd.Timestamp(day, tz=protocol["timezone"]) + pd.Timedelta(hours=protocol["decision_hour"])
              for phase in PHASES for day in protocol["splits"][phase]]
    manifest = SplitManifest.create("estimated-timing-pilot", CURRENT_ACTIVE_SCHEMA_VERSION, *bounds,
                                   raw_dataset_content_hash=SplitManifest.compute_content_hash(
                                       dict(market=digest, ledger=ledger_sha)))
    inputs, _ = prepare_inputs(protocol, manifest, timing_events=events)
    counts = {phase: {stock: sum(bool(b.event_mask.any()) for b in sample[1][:-1])
                      for stock, sample in items.items()} for phase, items in inputs.items()}
    runs = []
    for reference in baseline["m0_runs"]:
        seed = reference["seed"]
        agent, epoch, trace = fit_seed(seed, protocol, inputs)
        test = {stock: evaluate(sample, lambda obs, _: agent.select_action(obs, evaluate=True))
                for stock, sample in inputs["test"].items()}
        runs.append(dict(seed=seed, epoch=epoch, training=trace, test=test,
                         comparison=compare(reference["test"], test)))
    return dict(status="ESTIMATED_M1_TIMING_ONLY", ledger_sha256=ledger_sha,
                decisions_with_visible_event=counts, runs=runs)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
