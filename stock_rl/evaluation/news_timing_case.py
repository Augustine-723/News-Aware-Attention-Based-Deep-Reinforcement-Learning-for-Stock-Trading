"""Replay the two January 18 publisher evidence cases with M1 timing inputs."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from stock_rl.data.pit_engine import MissingStatus, ProvenanceRecord
from stock_rl.data.real_data_poc import STOCK_METADATA, load_real_market_feeds
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.evaluation.news_input_pilot import checkpoint_observer, mask_event_inputs

EVIDENCE = Path(__file__).resolve().parents[2] / "docs/evidence/finmind_stock_day_sample.json"


def candidate(row):
    probe = row["probe"]
    times = probe["og_published_time"]
    if not probe["matched_title_and_article_paragraphs"] or len(times) != 1:
        return None, "missing_article_or_publication_evidence"
    published = pd.Timestamp(times[0])
    if pd.isna(published) or published.tz is None:
        return None, "missing_publication_timezone"
    published = published.tz_convert("Asia/Taipei")
    if published.date().isoformat() != row["date"]:
        return None, "index_and_publisher_dates_conflict"
    event_id = probe["url"]
    provenance = ProvenanceRecord(
        source_id=event_id, source_published_at=published,
        available_at=published + pd.Timedelta(minutes=5), mode="estimated",
        is_audit_verified=False, missing_status=MissingStatus.AVAILABLE.value,
    )
    return {"event_id": event_id, "provenance": provenance}, "estimated_candidate"


def replay(row, observe=None):
    stock = row["stock"]
    event, status = candidate(row)
    market, inst, macro, foreign, context, _ = load_real_market_feeds(stock, mode="estimated")
    builder = HistoricalSnapshotBuilder(
        df_ohlcv=market, df_inst=inst, df_macro=macro, df_foreign=foreign,
        stock_context_records=context, shares_outstanding=STOCK_METADATA[stock]["shares_outstanding"],
        event_records=[event] if event else [], mode="estimated",
    )
    sessions = market.index.normalize()
    i = sessions.get_loc(pd.Timestamp(row["date"], tz=sessions.tz))
    if i == 0 or i + 1 >= len(sessions):
        raise ValueError("missing adjacent market session")
    readings = []
    for j in (i - 1, i):
        decision = sessions[j] + pd.Timedelta(hours=20)
        snapshot = builder.build_snapshot(decision)
        batch = mask_event_inputs(snapshot, "M1")
        visible = [eid for eid, used in zip(snapshot.slot_event_ids[0], batch.event_mask[0])
                   if used]
        readings.append(dict(decision_at=decision.isoformat(), visible_event_ids=visible,
                             next_open=(sessions[j+1] + pd.Timedelta(hours=9)).isoformat(),
                             text_nonzeros=int(batch.recent_event_emb.count_nonzero()
                                               + batch.active_event_emb.count_nonzero())))
        if observe is not None:
            readings[-1]["inference"] = observe(mask_event_inputs(snapshot, "M0"), batch)
    return dict(stock=stock, status=status, publisher_timestamps=row["probe"]["og_published_time"],
                estimated_available_at=event["provenance"].available_at.isoformat() if event else None,
                verified_provider_available_at=None, observations=readings)


def build_report(observe=None):
    rows = json.loads(EVIDENCE.read_text())
    selected = [row for row in rows if row["date"] == "2024-01-18" and row["stock"] in ("2603", "2881")]
    if sorted(row["stock"] for row in selected) != ["2603", "2881"]:
        raise ValueError("expected both January 18 stock cases")
    return [replay(row, observe) for row in selected]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+")
    args = parser.parse_args()
    if args.checkpoint:
        torch.set_num_threads(1)
        results = []
        for path in args.checkpoint:
            observe, metadata = checkpoint_observer(path, ("M0", "M1"))
            results.append(dict(checkpoint=metadata, cases=build_report(observe)))
    else:
        results = build_report()
    print(json.dumps(results, indent=2, allow_nan=False))
