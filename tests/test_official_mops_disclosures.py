import json
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime
import pandas as pd
import pytest

from stock_rl.data.acquire_official_mops_disclosures import (
    resolve_effective_decision_day,
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

    # Verify each disclosure has valid structure and fields
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
