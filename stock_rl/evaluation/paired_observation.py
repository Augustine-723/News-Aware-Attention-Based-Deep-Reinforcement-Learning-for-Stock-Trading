"""Compare market snapshots with and without isolated synthetic events."""

import json

import pandas as pd
import torch

from stock_rl.config.schema import EVENT_TEXT_DIM, get_event_age_index
from stock_rl.data.pit_engine import MissingStatus, ProvenanceRecord
from stock_rl.data.real_data_poc import (
    STOCK_METADATA,
    generate_stable_embedding,
    load_real_market_feeds,
)
from stock_rl.data.snapshot_builder import HistoricalSnapshotBuilder
from stock_rl.data.tensor_validator import validate_multimodal_batch

MARKET_INPUTS = (
    "price", "technical", "institutional", "macro", "stock_context", "position",
    "institutional_missing_mask", "macro_missing_mask", "stock_context_missing_mask",
)


def build_report(observe=None):
    stock_id = "2330"
    ohlcv, institutional, macro, foreign, context, _ = load_real_market_feeds(
        stock_id, mode="estimated"
    )
    calendar = ohlcv.index.normalize()
    fixture_times = {
        "SYNTH-AT-CUTOFF": "2024-02-05 20:00:00",
        "SYNTH-AFTER-CUTOFF": "2024-02-05 20:00:01",
        "SYNTH-NON-SESSION": "2024-02-06 18:00:00",
    }
    events = []
    for event_id, timestamp in fixture_times.items():
        available_at = pd.Timestamp(timestamp, tz=calendar.tz)
        published_at = available_at
        if event_id == "SYNTH-AFTER-CUTOFF":
            published_at = available_at - pd.Timedelta(seconds=2)
        events.append({
            "event_id": event_id,
            "relation": {"direct_company_mention": True},
            "text_embedding": generate_stable_embedding(event_id, EVENT_TEXT_DIM),
            "provenance": ProvenanceRecord(
                source_id=event_id, source_published_at=published_at, available_at=available_at,
                is_audit_verified=False, missing_status=MissingStatus.AVAILABLE.value,
                mode="estimated",
            ),
        })

    inputs = dict(
        df_ohlcv=ohlcv, df_inst=institutional, df_macro=macro, df_foreign=foreign,
        stock_context_records=context, shares_outstanding=STOCK_METADATA[stock_id]["shares_outstanding"],
        mode="estimated",
    )
    control = HistoricalSnapshotBuilder(**inputs, event_records=[])
    treatment = HistoricalSnapshotBuilder(**inputs, event_records=events)

    observed = []
    first_seen = {}
    for date in ("2024-02-02", "2024-02-05", "2024-02-15"):
        session = pd.Timestamp(date, tz=calendar.tz)
        index = calendar.get_loc(session)
        decision = session + pd.Timedelta(hours=20)
        next_open = calendar[index + 1] + pd.Timedelta(hours=9)
        empty = control.build_snapshot(decision)
        paired = treatment.build_snapshot(decision)
        validate_multimodal_batch(empty)
        validate_multimodal_batch(paired)
        equal = all(torch.equal(getattr(empty, key), getattr(paired, key)) for key in MARKET_INPUTS)
        if not equal or empty.event_mask.any():
            raise RuntimeError(f"invalid paired observation at {decision}")
        slots = []
        age_index = get_event_age_index(paired.schema_version)
        for slot, event_id in enumerate(paired.slot_event_ids[0]):
            if bool(paired.event_mask[0, slot]) != (event_id is not None):
                raise RuntimeError(f"event ID and mask disagree at {decision}, slot {slot}")
            if event_id is not None:
                first_seen.setdefault(event_id, decision.isoformat())
                slots.append({
                    "slot": slot, "event_id": event_id,
                    "event_age_days": paired.event_metadata[0, slot, age_index].item(),
                })
        row = {
            "decision_at": decision.isoformat(), "next_session_open": next_open.isoformat(),
            "market_tensors_equal": equal, "control_event_count": int(empty.event_mask.sum()),
            "fixture_event_count": len(slots), "fixture_slots": slots,
        }
        if observe is not None:
            row["frozen_inference"] = observe(empty, paired)
        observed.append(row)
    return {
        "purpose": "SYNTHETIC_EVENT_INTEGRATION_ONLY",
        "stock_id": stock_id,
        "schema_version": control.schema_version,
        "compared_market_fields": list(MARKET_INPUTS),
        "events": [
            {
                "event_id": record.source_id,
                "source_published_at": record.source_published_at.isoformat(),
                "available_at": record.available_at.isoformat(),
                "first_observed_at": first_seen.get(record.source_id),
            }
            for record in (event["provenance"] for event in events)
        ],
        "observations": observed,
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, allow_nan=False))
