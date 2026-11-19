import json
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime
import pandas as pd
import pytest

from stock_rl.data.acquire_official_mops_disclosures import (
    load_trading_calendar,
    resolve_effective_decision_day,
    resolve_split_from_effective_day,
)

ROOT = Path(__file__).resolve().parent.parent


def test_resolve_effective_decision_day_trading_day_before_cutoff():
    # 2024-01-02 was a trading day
    calendar = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ])
    dt = datetime(2024, 1, 2, 17, 30, 0, tzinfo=ZoneInfo("Asia/Taipei"))
    eff = resolve_effective_decision_day(dt, calendar)
    assert eff == "2024-01-02T20:00:00+08:00"


def test_resolve_effective_decision_day_trading_day_after_cutoff():
    calendar = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ])
    dt = datetime(2024, 1, 2, 20, 15, 0, tzinfo=ZoneInfo("Asia/Taipei"))
    eff = resolve_effective_decision_day(dt, calendar)
    assert eff == "2024-01-03T20:00:00+08:00"


def test_resolve_effective_decision_day_cny_holiday():
    # Chinese New Year break in 2024: Feb 6 to Feb 14 were holidays. First trading day was Feb 15.
    calendar = pd.DatetimeIndex([
        pd.Timestamp("2024-02-05"),
        pd.Timestamp("2024-02-15"),
    ])
    # Feb 6 afternoon disclosure (e.g. TSMC Kumamoto)
    dt = datetime(2024, 2, 6, 18, 55, 24, tzinfo=ZoneInfo("Asia/Taipei"))
    eff = resolve_effective_decision_day(dt, calendar)
    assert eff == "2024-02-15T20:00:00+08:00"


def test_unevaluable_when_no_future_trading_day():
    calendar = pd.DatetimeIndex([
        pd.Timestamp("2024-01-02"),
    ])
    # Filing on 2024-01-02 after 20:00 with no next trading day in calendar
    dt = datetime(2024, 1, 2, 21, 0, 0, tzinfo=ZoneInfo("Asia/Taipei"))
    eff = resolve_effective_decision_day(dt, calendar)
    assert eff is None
    split = resolve_split_from_effective_day(eff)
    assert split == "unevaluable"


def test_acquired_mops_disclosures_integrity():
    json_path = ROOT / "data" / "official_disclosures" / "mops_official_disclosures_2023_2024.json"
    if not json_path.exists():
        pytest.skip("mops_official_disclosures_2023_2024.json not found")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "metadata" in data
    assert "query_receipts" in data
    assert "disclosures" in data

    disclosures = data["disclosures"]
    assert len(disclosures) == 404

    required_keys = {
        "stock_id", "seq_no", "roc_date", "spoke_date", "spoke_time",
        "filing_datetime", "headline", "is_after_20_00", "effective_decision_day",
        "split", "source"
    }
    for d in disclosures:
        assert required_keys.issubset(d.keys())
        assert d["stock_id"] in ["2330", "2603", "2881"]
        assert d["split"] in ["train", "validation", "test", "holdout"]
        assert len(d["headline"]) > 0
        assert d["source"] == "TWSE_MOPS_OFFICIAL"


def test_causality_effective_decision_day_never_earlier_than_filing():
    """
    Regression test for the 5 causality violation cases where decision occurred before filing.
    Verifies that across all 404 disclosures, effective_decision_day is strictly >= filing_datetime.
    """
    json_path = ROOT / "data" / "official_disclosures" / "mops_official_disclosures_2023_2024.json"
    if not json_path.exists():
        pytest.skip("mops_official_disclosures_2023_2024.json not found")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    disclosures = data["disclosures"]

    earlier_count = 0
    earlier_cases = []
    for d in disclosures:
        f_dt = pd.Timestamp(d["filing_datetime"])
        e_dt = pd.Timestamp(d["effective_decision_day"])
        if e_dt < f_dt:
            earlier_count += 1
            earlier_cases.append((d["stock_id"], d["filing_datetime"], d["effective_decision_day"]))

    assert earlier_count == 0, f"Found {earlier_count} disclosures where decision < filing: {earlier_cases}"


def test_april_five_fixed_target_cases():
    """
    Specifically assert that the 5 edge cases in April are now correctly mapped to valid trading days:
    - 2330 on 2024-04-03 23:23:25 -> 2024-04-08T20:00:00+08:00
    - 2330 on 2024-04-04 21:05:31 -> 2024-04-08T20:00:00+08:00
    - 2330 on 2024-04-05 21:23:03 -> 2024-04-08T20:00:00+08:00
    - 2881 on 2024-04-23 20:07:36 -> 2024-04-24T20:00:00+08:00
    - 2881 on 2024-04-23 20:09:23 -> 2024-04-24T20:00:00+08:00
    """
    json_path = ROOT / "data" / "official_disclosures" / "mops_official_disclosures_2023_2024.json"
    if not json_path.exists():
        pytest.skip("mops_official_disclosures_2023_2024.json not found")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    by_stock_and_filing = {
        (d["stock_id"], d["filing_datetime"]): d
        for d in data["disclosures"]
    }

    # 1. 2330 earthquake after 20:00
    d1 = by_stock_and_filing.get(("2330", "2024-04-03T23:23:25+08:00"))
    assert d1 is not None
    assert d1["effective_decision_day"] == "2024-04-08T20:00:00+08:00"
    assert d1["split"] == "holdout"

    # 2. 2330 Qingming holiday April 4
    d2 = by_stock_and_filing.get(("2330", "2024-04-04T21:05:31+08:00"))
    assert d2 is not None
    assert d2["effective_decision_day"] == "2024-04-08T20:00:00+08:00"
    assert d2["split"] == "holdout"

    # 3. 2330 Qingming holiday April 5
    d3 = by_stock_and_filing.get(("2330", "2024-04-05T21:23:03+08:00"))
    assert d3 is not None
    assert d3["effective_decision_day"] == "2024-04-08T20:00:00+08:00"
    assert d3["split"] == "holdout"

    # 4. 2881 April 23 after 20:00 case 1
    d4 = by_stock_and_filing.get(("2881", "2024-04-23T20:07:36+08:00"))
    assert d4 is not None
    assert d4["effective_decision_day"] == "2024-04-24T20:00:00+08:00"
    assert d4["split"] == "holdout"

    # 5. 2881 April 23 after 20:00 case 2
    d5 = by_stock_and_filing.get(("2881", "2024-04-23T20:09:23+08:00"))
    assert d5 is not None
    assert d5["effective_decision_day"] == "2024-04-24T20:00:00+08:00"
    assert d5["split"] == "holdout"


def test_april_holdout_statistics():
    """
    All 87 April disclosures must have valid effective_decision_days within April 2024
    and end in T20:00:00+08:00.
    """
    json_path = ROOT / "data" / "official_disclosures" / "mops_official_disclosures_2023_2024.json"
    if not json_path.exists():
        pytest.skip("mops_official_disclosures_2023_2024.json not found")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    april_disclosures = [d for d in data["disclosures"] if d["spoke_date"].startswith("2024-04")]
    assert len(april_disclosures) == 87

    for d in april_disclosures:
        assert d["split"] == "holdout"
        assert d["effective_decision_day"].endswith("T20:00:00+08:00")
        assert d["effective_decision_day"] >= d["filing_datetime"]
