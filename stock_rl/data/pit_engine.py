"""
Point-in-Time (PIT) Data Engine & Timestamp Guard
Enforces strict historical availability rules per V8.0 Section 3.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class EventTimestampRecord:
    event_id: str
    occur_time: datetime      # Actual event occurrence
    pub_time: datetime        # Media publication time
    vendor_time: datetime     # Vendor feed availability time
    ingest_time: datetime     # System internal ingestion time
    revision_time: datetime   # Most recent revision time if amended
    headline: str
    stock_id: str

    def is_available_as_of(self, decision_timestamp: datetime) -> bool:
        """
        PIT Verification: Event is strictly available only if vendor feed
        was received at or before the decision timestamp.
        """
        effective_availability = max(self.vendor_time, self.ingest_time)
        return effective_availability <= decision_timestamp


class PointInTimeGuard:
    """
    Enforces no-lookahead policies, temporal purging, and source availability guards.
    """

    def __init__(self, decision_time_str: str = "20:00:00", timezone: str = "Asia/Taipei"):
        self.decision_time_str = decision_time_str
        self.timezone = timezone

    def create_decision_timestamp(self, date_str: str) -> pd.Timestamp:
        """Constructs an exact timezone-aware decision timestamp for a given date."""
        ts_str = f"{date_str} {self.decision_time_str}"
        return pd.Timestamp(ts_str, tz=self.timezone)

    def filter_events_pit(
        self,
        events: List[EventTimestampRecord],
        decision_timestamp: datetime
    ) -> List[EventTimestampRecord]:
        """Filters events to only those available at or before the decision timestamp."""
        return [ev for ev in events if ev.is_available_as_of(decision_timestamp)]

    @staticmethod
    def purge_train_val_test_boundaries(
        df_index: pd.DatetimeIndex,
        split_dates: List[Tuple[str, str]],
        purge_window_days: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Generates split masks with boundary purging.
        For an N-day forward label, samples within the final N days before a split
        boundary must be purged to prevent label leakage into subsequent splits.
        """
        masks = {}
        for split_name, (start_str, end_str) in split_dates:
            start_dt = pd.Timestamp(start_str)
            end_dt = pd.Timestamp(end_str)

            # Determine purge cutoff: purge last `purge_window_days` from training/validation
            mask = (df_index >= start_dt) & (df_index <= end_dt)

            # If not the final test set, identify trading days in this split and drop the last N
            split_indices = np.where(mask)[0]
            if len(split_indices) > purge_window_days:
                purged_indices = split_indices[:-purge_window_days]
                purged_mask = np.zeros(len(df_index), dtype=bool)
                purged_mask[purged_indices] = True
                masks[split_name] = purged_mask
            else:
                masks[split_name] = mask

        return masks


def check_no_future_leakage(
    feature_timestamps: np.ndarray,
    decision_timestamps: np.ndarray
) -> bool:
    """
    Asserts that every feature timestamp is strictly <= its corresponding decision timestamp.
    Raises ValueError on violation.
    """
    violations = feature_timestamps > decision_timestamps
    if np.any(violations):
        num_violations = np.sum(violations)
        first_violation_idx = np.where(violations)[0][0]
        feat_ts = feature_timestamps[first_violation_idx]
        dec_ts = decision_timestamps[first_violation_idx]
        raise ValueError(
            f"Point-in-Time Violation: {num_violations} features violate PIT constraint! "
            f"Example at index {first_violation_idx}: Feature available at {feat_ts} > Decision at {dec_ts}"
        )
    return True
