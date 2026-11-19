"""Train an estimated timing-only policy and compare it to the frozen M0 runs."""
import argparse
import hashlib
import json
from dataclasses import replace
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


def run(text_mode=None, delay_session=False, cache_dir=None):
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
    metadata = {}
    if text_mode is not None:
        from .provenance import load_trusted_pretrained_embedding_engine
        engine = load_trusted_pretrained_embedding_engine(cache_dir=cache_dir)
        if engine is None:
            raise ValueError("verified local BGE snapshot required")
        note_bytes = (ROOT / "docs/evidence/retrospective_event_notes.json").read_bytes()
        notes = json.loads(note_bytes)
        if set(notes) != {e["event_id"] for group in events.values() for e in group}:
            raise ValueError("notes must cover the frozen event ledger exactly")
        for group in events.values():
            for event in group:
                text = notes[event["event_id"]] if text_mode == "notes" else "公司相關事件。"
                event["text_embedding"] = engine.embed_text(text)
        metadata = dict(text_mode=text_mode, notes_sha256=hashlib.sha256(note_bytes).hexdigest(),
                        encoder_weights_sha256=engine.weights_sha256)
    if delay_session:
        days = pd.DatetimeIndex(pd.read_csv(Path(DATA_DIR) / "taiex.csv", parse_dates=["date"]).date)
        decisions = days.tz_localize(protocol["timezone"]) + pd.Timedelta(hours=protocol["decision_hour"])
        for group in events.values():
            for event in group:
                record = event["provenance"]
                idx = decisions.searchsorted(record.available_at)
                if idx + 1 >= len(decisions):
                    raise ValueError("delay exceeds observed calendar")
                event["provenance"] = replace(record, available_at=decisions[idx + 1])
        metadata["availability_delay_sessions"] = 1
    bounds = [pd.Timestamp(day, tz=protocol["timezone"]) + pd.Timedelta(hours=protocol["decision_hour"])
              for phase in PHASES for day in protocol["splits"][phase]]
    manifest = SplitManifest.create("estimated-timing-pilot", CURRENT_ACTIVE_SCHEMA_VERSION, *bounds,
                                   raw_dataset_content_hash=SplitManifest.compute_content_hash(
                                       dict(market=digest, ledger=ledger_sha, **metadata)))
    inputs, _ = prepare_inputs(protocol, manifest, timing_events=events, include_event_text=text_mode is not None)
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
    return dict(status="ESTIMATED_M1_TIMING_ONLY" if text_mode is None else "RETROSPECTIVE_TEXT_PILOT",
                **metadata, ledger_sha256=ledger_sha,
                decisions_with_visible_event=counts, runs=runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-mode", choices=("notes", "constant"))
    parser.add_argument("--delay-session", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.text_mode, args.delay_session, args.cache_dir), indent=2, allow_nan=False))
