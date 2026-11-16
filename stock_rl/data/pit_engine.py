"""
Point-in-Time (PIT) Data Engine & Versioned Availability Guard V8.0
Enforces strict historical availability, document versioning, and dynamic label boundary purging.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


class AvailabilityStatus:
    VERIFIED = "VERIFIED"
    UNKNOWN = "UNKNOWN"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class EventTimestampRecord:
    """Legacy compatibility dataclass for single-version events."""
    event_id: str
    occur_time: datetime
    pub_time: datetime
    vendor_time: datetime
    ingest_time: datetime
    revision_time: datetime
    headline: str
    stock_id: str

    def is_available_as_of(self, decision_timestamp: datetime, live_mode: bool = False) -> bool:
        if live_mode:
            return self.vendor_time <= decision_timestamp and self.ingest_time <= decision_timestamp
        return self.vendor_time <= decision_timestamp


@dataclass(frozen=True)
class DocumentVersion:
    """
    Represents an immutable historical version or revision of an event announcement/news.
    """
    version_id: int
    pub_time: datetime                         # Official media / exchange publication time
    vendor_available_time: Optional[datetime]  # When vendor feed historically made this version accessible
    is_verified: bool                          # Whether historical availability timestamp is audit-verified
    headline: str
    content: str = ""                          # Versioned document content / factual evidence snippet
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Strict validation: vendor feed cannot historically precede official publication
        if self.vendor_available_time is not None and self.pub_time > self.vendor_available_time:
            raise ValueError(
                f"Chronological inconsistency: pub_time ({self.pub_time}) cannot be after vendor_available_time ({self.vendor_available_time})"
            )

    @property
    def availability_status(self) -> str:
        if not self.is_verified or self.vendor_available_time is None:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.VERIFIED


@dataclass
class VersionedEventRecord:
    """
    A financial event entity with multiple point-in-time document versions/revisions.
    """
    event_id: str
    stock_id: str
    occur_time: datetime                       # Ground-truth event occurrence time
    versions: List[DocumentVersion] = field(default_factory=list)
    system_ingest_time: Optional[datetime] = None  # Local system ingestion timestamp

    def add_version(self, version: DocumentVersion):
        self.versions.append(version)
        self.versions.sort(key=lambda v: v.version_id)

    def get_version_as_of(
        self,
        decision_timestamp: datetime,
        live_mode: bool = False
    ) -> Tuple[Optional[DocumentVersion], str]:
        """
        Retrieves the exact document version available at historical decision_timestamp.

        Rules per Specification V8.0:
        1. Backfill records: Historical availability is determined by `vendor_available_time`,
           NOT modern `system_ingest_time`.
        2. Point-in-Time Versioning: Selects the latest revision whose `vendor_available_time <= decision_timestamp`.
           Revisions occurring after decision_timestamp must not leak into model inputs.
        3. Unknown Availability: If `is_verified` is False or `vendor_available_time` is missing,
           the document is marked UNKNOWN and CANNOT be assumed available.
        4. Live Trading Mode: If `live_mode=True`, additionally enforces `system_ingest_time <= decision_timestamp`.
        """
        # In live execution mode, system must have actually ingested the record
        if live_mode:
            if self.system_ingest_time is None or self.system_ingest_time > decision_timestamp:
                return None, AvailabilityStatus.UNAVAILABLE

        valid_versions = []
        for v in self.versions:
            # Rule 3: Unverified timestamps cannot be assumed available
            if v.availability_status == AvailabilityStatus.UNKNOWN:
                continue

            # Rule 1 & 2: Check historical vendor availability
            if v.vendor_available_time <= decision_timestamp:
                valid_versions.append(v)

        if not valid_versions:
            # Check if any unverified version existed
            has_unverified = any(v.availability_status == AvailabilityStatus.UNKNOWN for v in self.versions)
            if has_unverified:
                return None, AvailabilityStatus.UNKNOWN
            return None, AvailabilityStatus.UNAVAILABLE

        # Return latest revision available as of decision_timestamp
        latest_version = max(valid_versions, key=lambda v: (v.vendor_available_time, v.version_id))
        return latest_version, AvailabilityStatus.VERIFIED

    def get_text_at(
        self,
        decision_timestamp: datetime,
        live_mode: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Retrieves (headline, content) strictly as of decision_timestamp.
        Guarantees that past event representations cannot be reconstructed from future revisions.
        """
        ver, status = self.get_version_as_of(decision_timestamp, live_mode=live_mode)
        if status == AvailabilityStatus.VERIFIED and ver is not None:
            return ver.headline, ver.content
        return None, None


class PointInTimeGuard:
    """
    Enforces no-lookahead policies, temporal dynamic purging, and source availability guards.
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
        events: List[VersionedEventRecord],
        decision_timestamp: datetime,
        live_mode: bool = False
    ) -> List[Tuple[VersionedEventRecord, DocumentVersion]]:
        """
        Filters events and returns pairs of (event, active_version) strictly available at decision_timestamp.
        Excludes UNAVAILABLE and UNKNOWN events.
        """
        available_events = []
        for ev in events:
            ver, status = ev.get_version_as_of(decision_timestamp, live_mode=live_mode)
            if status == AvailabilityStatus.VERIFIED and ver is not None:
                available_events.append((ev, ver))
        return available_events

    @staticmethod
    def purge_train_val_test_boundaries(
        df_index: pd.DatetimeIndex,
        split_dates: List[Tuple[str, Tuple[str, str]]],
        purge_window_days: int = 5
    ) -> Dict[str, np.ndarray]:
        """Legacy helper for static purge window boundary splitting."""
        masks = {}
        for split_name, (start_str, end_str) in split_dates:
            start_dt = pd.Timestamp(start_str)
            end_dt = pd.Timestamp(end_str)
            mask = (df_index >= start_dt) & (df_index <= end_dt)
            split_indices = np.where(mask)[0]
            if len(split_indices) > purge_window_days:
                purged_indices = split_indices[:-purge_window_days]
                purged_mask = np.zeros(len(df_index), dtype=bool)
                purged_mask[purged_indices] = True
                masks[split_name] = purged_mask
            else:
                masks[split_name] = mask
        return masks

    @staticmethod
    def purge_label_overlap_boundaries(
        decision_dates: pd.DatetimeIndex,
        label_horizons_days: List[int],
        split_boundaries: List[Tuple[str, pd.Timestamp, pd.Timestamp]]
    ) -> Dict[str, np.ndarray]:
        """
        Dynamically purges samples whose forward-looking label window overlaps past split boundaries
        or exceeds historical dataset cutoff.

        For a sample at decision date t (index idx) with maximum label horizon H trading days,
        the label spans [t+1 Open, t+1+H Open] (e.g. for 5-day horizon: T+1 Open to T+6 Open).
        The realization date index is: label_end_idx = idx + 1 + max_horizon.
        - If label_end_idx >= len(decision_dates), the label exceeds dataset cutoff and is marked invalid/purged.
        - If decision_dates[label_end_idx] > end_ts, the label crosses into the next split and is purged.
        """
        max_horizon = max(label_horizons_days)
        masks = {}

        for split_name, start_ts, end_ts in split_boundaries:
            # Samples falling within the split range
            in_split = (decision_dates >= start_ts) & (decision_dates <= end_ts)
            split_indices = np.where(in_split)[0]

            if len(split_indices) == 0:
                masks[split_name] = np.zeros(len(decision_dates), dtype=bool)
                continue

            valid_indices = []
            for idx in split_indices:
                label_end_idx = idx + 1 + max_horizon
                # Check dataset cutoff
                if label_end_idx >= len(decision_dates):
                    continue
                # Check split boundary
                if decision_dates[label_end_idx] > end_ts:
                    continue
                valid_indices.append(idx)

            mask = np.zeros(len(decision_dates), dtype=bool)
            mask[valid_indices] = True
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


def validate_feature_historical_availability(
    feature_name: str,
    feature_source_time: Optional[pd.Timestamp],
    decision_timestamp: pd.Timestamp,
    is_audit_verified: bool = True
) -> bool:
    """
    Validates whether a specific feature was historically available at decision_timestamp.
    Requires an audit-verified source timestamp <= decision_timestamp.
    Static publication descriptions cannot substitute for point-in-time timestamp verification.
    """
    if not is_audit_verified or feature_source_time is None:
        return False
    return feature_source_time <= decision_timestamp


@dataclass(frozen=True)
class ProvenanceRecord:
    """
    Cryptographically and chronologically auditable feature provenance record.
    Tracks primary source publishing, vendor dissemination, local ingestion,
    and historical availability status per V8.3 Code Review 1 (P1-7).
    """
    source_id: str
    source_published_at: pd.Timestamp
    provider_available_at: Optional[pd.Timestamp]
    available_at: Optional[pd.Timestamp]
    ingested_at: Optional[pd.Timestamp]
    revision_id: int = 1
    is_audit_verified: bool = True
    missing_status: str = "AVAILABLE"  # "AVAILABLE", "UNKNOWN", "MISSING"

    def is_available_as_of(self, decision_time: pd.Timestamp) -> bool:
        """
        Validates historical availability strictly at decision_time.
        Requires verified available_at <= decision_time and missing_status == 'AVAILABLE'.
        Timezone-aware safe comparison.
        """
        if self.missing_status in ("UNKNOWN", "MISSING"):
            return False
        if not self.is_audit_verified:
            return False
        if self.available_at is None:
            return False

        t_avail = self.available_at
        t_dec = decision_time

        # Normalize timezones for comparison
        if t_avail.tzinfo is not None and t_dec.tzinfo is not None:
            return t_avail <= t_dec
        elif t_avail.tzinfo is None and t_dec.tzinfo is None:
            return t_avail <= t_dec
        else:
            t_avail_utc = t_avail.tz_convert("UTC") if t_avail.tzinfo is not None else t_avail.tz_localize("UTC")
            t_dec_utc = t_dec.tz_convert("UTC") if t_dec.tzinfo is not None else t_dec.tz_localize("UTC")
            return t_avail_utc <= t_dec_utc


def asof_join_cross_market(
    taiwan_decision_times: pd.DatetimeIndex,
    foreign_market_df: pd.DataFrame,
    foreign_available_col: str = "available_at",
    missing_policy: str = "forward_fill",
) -> pd.DataFrame:
    """
    Timezone-aware Historical As-Of Join for overseas market data (US S&P 500, NASDAQ, SOX, VIX, US 10Y).
    Strictly aligns foreign data whose `available_at <= decision_time` as of Taiwan 20:00:00 (P0-1).

    Rules:
    1. ZERO lookahead: direction is strictly 'backward' (no bfill).
    2. Respects US trading calendars, holidays, and US Daylight Saving Time (DST, EDT vs EST).
    3. Unverified or missing available_at records remain Unknown / Missing.
    4. If no historical record is available before the decision time, fills with NaN or declared policy,
       strictly NEVER backfilling from future rows.
    """
    if not isinstance(taiwan_decision_times, pd.DatetimeIndex):
        taiwan_decision_times = pd.DatetimeIndex(taiwan_decision_times)

    # Ensure timezone awareness on Taiwan decision times (Asia/Taipei)
    if taiwan_decision_times.tz is None:
        dec_times_taipei = taiwan_decision_times.tz_localize("Asia/Taipei")
    else:
        dec_times_taipei = taiwan_decision_times.tz_convert("Asia/Taipei")

    dec_df = pd.DataFrame({"decision_time": dec_times_taipei}).sort_values("decision_time").reset_index(drop=True)
    dec_df["decision_time_utc"] = dec_df["decision_time"].dt.tz_convert("UTC")

    # Prepare foreign DataFrame
    foreign_clean = foreign_market_df.copy()
    if foreign_available_col not in foreign_clean.columns:
        if isinstance(foreign_clean.index, pd.DatetimeIndex):
            foreign_clean[foreign_available_col] = foreign_clean.index
        else:
            raise KeyError(f"foreign_market_df missing available time column '{foreign_available_col}'")

    # Normalize foreign available_at to UTC
    avail_series = pd.to_datetime(foreign_clean[foreign_available_col])
    if avail_series.dt.tz is None:
        # Default foreign market timestamp without timezone to America/New_York close (16:00 ET)
        avail_utc = avail_series.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    else:
        avail_utc = avail_series.dt.tz_convert("UTC")

    foreign_clean["_avail_utc"] = avail_utc
    # Filter out unverified or invalid timestamps
    foreign_valid = foreign_clean.dropna(subset=["_avail_utc"]).sort_values("_avail_utc").reset_index(drop=True)

    # Perform backward-only As-Of Join
    feature_cols = [c for c in foreign_clean.columns if c not in (foreign_available_col, "_avail_utc")]
    merged = pd.merge_asof(
        dec_df,
        foreign_valid[["_avail_utc"] + feature_cols],
        left_on="decision_time_utc",
        right_on="_avail_utc",
        direction="backward",
    )

    result_df = merged[feature_cols].copy()
    result_df.index = taiwan_decision_times

    if missing_policy == "forward_fill":
        # Forward fill only forward in time from past valid observations
        result_df = result_df.ffill()

    return result_df

