import json

import pandas as pd

from stock_rl.evaluation.timing_policy_pilot import ROOT, timing_records


def test_frozen_ledger_excludes_quarantine_and_future_events():
    ledger = json.loads((ROOT / "docs/results/v8_33_news_split_coverage.json").read_text())["ledger"]
    events = timing_records(ledger, ["2330", "2603", "2881"])
    assert sum(map(len, events.values())) == 16
    decision = pd.Timestamp("2023-09-01 20:00", tz="Asia/Taipei")
    visible = [e for records in events.values() for e in records
               if e["provenance"].is_available_as_of(decision, mode="estimated")]
    assert len(visible) == 2
    assert all(not e["provenance"].is_audit_verified for e in visible)


def test_event_notes_cover_only_frozen_candidates():
    notes = json.loads((ROOT / "docs/evidence/retrospective_event_notes.json").read_text())
    ledger = json.loads((ROOT / "docs/results/v8_33_news_split_coverage.json").read_text())["ledger"]
    events = timing_records(ledger, ["2330", "2603", "2881"])
    assert set(notes) == {e["event_id"] for group in events.values() for e in group}
    assert len(set(notes.values())) == 16
    assert all(note.strip() for note in notes.values())
