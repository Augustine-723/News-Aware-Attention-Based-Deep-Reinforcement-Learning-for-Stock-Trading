"""
Unit Tests for Point-in-Time Integrity & Boundary Purging per V8.0
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from stock_rl.data.pit_engine import (
    EventTimestampRecord,
    PointInTimeGuard,
    check_no_future_leakage,
)


def test_pit_event_availability():
    decision_ts = pd.Timestamp("2024-05-15 20:00:00", tz="Asia/Taipei")

    # Available event (vendor received at 18:30)
    ev_valid = EventTimestampRecord(
        event_id="ev_001",
        occur_time=decision_ts - pd.Timedelta(hours=4),
        pub_time=decision_ts - pd.Timedelta(hours=2),
        vendor_time=decision_ts - pd.Timedelta(hours=1, minutes=30),
        ingest_time=decision_ts - pd.Timedelta(hours=1),
        revision_time=decision_ts - pd.Timedelta(hours=1),
        headline="TSMC announces new fab expansion",
        stock_id="2330",
    )
    assert ev_valid.is_available_as_of(decision_ts) is True

    # Future event (vendor received after 20:00)
    ev_future = EventTimestampRecord(
        event_id="ev_002",
        occur_time=decision_ts - pd.Timedelta(hours=1),
        pub_time=decision_ts - pd.Timedelta(minutes=30),
        vendor_time=decision_ts + pd.Timedelta(minutes=15),
        ingest_time=decision_ts + pd.Timedelta(minutes=20),
        revision_time=decision_ts + pd.Timedelta(minutes=20),
        headline="Late-breaking earnings revision",
        stock_id="2330",
    )
    assert ev_future.is_available_as_of(decision_ts) is False


def test_no_future_leakage_assertion():
    t0 = np.array([100, 105, 110])
    t1 = np.array([101, 106, 115])
    # All features available at or before decision -> True
    assert check_no_future_leakage(t0, t1) is True

    # Future feature available after decision -> raises ValueError
    t_bad = np.array([102, 105, 110])
    with pytest.raises(ValueError, match="Point-in-Time Violation"):
        check_no_future_leakage(t_bad, t1)


def test_boundary_purging():
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="B")
    split_dates = [("train", ("2024-01-01", "2024-01-15")), ("val", ("2024-01-16", "2024-01-31"))]

    masks = PointInTimeGuard.purge_train_val_test_boundaries(
        df_index=dates,
        split_dates=split_dates,
        purge_window_days=5
    )

    train_dates = dates[masks["train"]]
    val_dates = dates[masks["val"]]

    # Ensure train dates do not overlap or touch the last 5 days before the split boundary
    train_end_purged = train_dates.max()
    assert train_end_purged < pd.Timestamp("2024-01-15")
