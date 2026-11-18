import pytest

from stock_rl.evaluation.paired_observation import build_report


def test_synthetic_event_cutoff_and_holiday_on_real_market_snapshots():
    report = build_report()
    before, cutoff, reopened = report["observations"]
    assert all(row["market_tensors_equal"] and row["control_event_count"] == 0
               for row in report["observations"])
    assert before["fixture_slots"] == []
    assert cutoff["fixture_slots"] == [
        {"slot": 0, "event_id": "SYNTH-AT-CUTOFF", "event_age_days": 0.0},
    ]
    assert cutoff["next_session_open"] == "2024-02-15T09:00:00+08:00"
    assert reopened["decision_at"] == "2024-02-15T20:00:00+08:00"
    assert reopened["next_session_open"] == "2024-02-16T09:00:00+08:00"
    assert [slot["event_id"] for slot in reopened["fixture_slots"]] == [
        "SYNTH-NON-SESSION", "SYNTH-AFTER-CUTOFF", "SYNTH-AT-CUTOFF",
    ]
    assert [slot["event_age_days"] for slot in reopened["fixture_slots"]] == pytest.approx(
        [9 + 2 / 24, 10 - 1 / 86400, 10]
    )
