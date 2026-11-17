"""
Point-in-Time (PIT) Data Engine & Versioned Availability Guard V8.0
Enforces strict historical availability, document versioning, and dynamic label boundary purging.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import warnings
import pandas as pd
import numpy as np


class MissingStatus(str, Enum):
    """
    P1-4: Formal Missing & Epistemic Status.
    Explicitly distinguishes between true valid observations, actual zeros,
    expected-but-missing data, unverified unknown data, and future unreleased data.
    """
    AVAILABLE = "AVAILABLE"
    ACTUAL_ZERO = "ACTUAL_ZERO"
    MISSING = "MISSING"
    UNKNOWN = "UNKNOWN"
    NOT_YET_PUBLISHED = "NOT_YET_PUBLISHED"


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
    and historical availability status per V8.3 Code Review 1, 2 & 3 (P0-1, P0-2, P1-7).
    Fail-closed defaults: is_audit_verified=False, missing_status=UNKNOWN.
    """
    source_id: str
    source_published_at: Optional[pd.Timestamp] = None
    provider_available_at: Optional[pd.Timestamp] = None
    available_at: Optional[pd.Timestamp] = None
    ingested_at: Optional[pd.Timestamp] = None
    revision_id: int = 1
    version: str = "1.0"
    is_audit_verified: bool = False  # P0-2: Default False (Fail-Closed)
    missing_status: str = MissingStatus.UNKNOWN.value  # P0-2: Default UNKNOWN (Fail-Closed)
    missing_reason: Optional[str] = None
    mode: str = "formal"

    def is_available_as_of(self, decision_time: pd.Timestamp, mode: Optional[str] = None) -> bool:
        """
        Validates historical availability strictly at decision_time.
        Requires verified available_at <= decision_time and missing_status in (AVAILABLE, ACTUAL_ZERO).
        Timezone-aware safe comparison. Naive timestamps raise ValueError in formal mode.
        """
        eff_mode = mode or self.mode
        if eff_mode == "formal" and not self.is_audit_verified:
            return False
        if self.missing_status not in (
            MissingStatus.AVAILABLE.value,
            MissingStatus.ACTUAL_ZERO.value,
            "AVAILABLE",
            "ACTUAL_ZERO",
        ):
            return False
        if self.available_at is None:
            return False

        t_avail = self.available_at
        t_dec = decision_time

        # P0-2: Strictly enforce timezone awareness. Silent UTC fallback is prohibited!
        if t_avail.tzinfo is None or t_dec.tzinfo is None:
            if eff_mode == "formal":
                raise ValueError(
                    f"Strict PIT Error: Timestamps in ProvenanceRecord and decision_time must be timezone-aware. "
                    f"available_at={t_avail} (tz={t_avail.tzinfo}), decision_time={t_dec} (tz={t_dec.tzinfo}). "
                    "Naive timestamps cannot be presumed UTC without timezone specification in formal mode."
                )
            return False

        return t_avail <= t_dec


class HistoricalVersionSelector:
    """
    P0-2: Point-In-Time Historical Version Selector.
    Selects the authoritative revision of a document or record as of decision_timestamp.
    Guarantees that unreleased or future revisions cannot overwrite historical values.
    """
    @staticmethod
    def select_as_of(
        revisions: List[ProvenanceRecord],
        decision_time: pd.Timestamp,
        mode: str = "formal",
    ) -> Optional[ProvenanceRecord]:
        if not revisions:
            return None
        valid_revs = [
            r for r in revisions
            if r.is_available_as_of(decision_time, mode=mode)
        ]
        if not valid_revs:
            return None
        return max(valid_revs, key=lambda r: (r.available_at, r.revision_id))


def asof_join_cross_market(
    taiwan_decision_times: pd.DatetimeIndex,
    foreign_market_df: pd.DataFrame,
    foreign_available_col: str = "available_at",
    missing_policy: str = "forward_fill",
    allow_estimated_availability: bool = False,
    mode: str = "formal",
) -> pd.DataFrame:
    """
    Timezone-aware Historical As-Of Join for overseas market data (US S&P 500, NASDAQ, SOX, VIX, US 10Y).
    Strictly aligns foreign data whose `available_at <= decision_time` as of Taiwan 20:00:00 (P0-1).

    Formal Historical PIT Rules (mode="formal"):
    1. Requires explicit, verified `available_at` column.
    2. Prohibits treating Trading Date, DataFrame Index, or Source Date automatically as Historical Availability.
    3. Unknown or timezone-naive timestamps are strictly rejected (raises ValueError).
    4. If synthetic/estimated availability is requested, it must be explicitly declared via allow_estimated_availability=True.
    5. Backward-only join: available_at <= decision_timestamp (ZERO future lookahead, ZERO bfill).
    6. Respects US trading calendars, holidays, Daylight Saving Time (EDT UTC-4 vs EST UTC-5), and future revision exclusion.
    """
    if not isinstance(taiwan_decision_times, pd.DatetimeIndex):
        taiwan_decision_times = pd.DatetimeIndex(taiwan_decision_times)

    # 1. Decision times timezone verification
    if taiwan_decision_times.tz is None:
        if mode == "formal" and not allow_estimated_availability:
            raise ValueError(
                "Strict PIT Error: taiwan_decision_times must have a timezone (e.g. Asia/Taipei). "
                "Naive decision timestamps are prohibited in formal mode."
            )
        dec_times_taipei = taiwan_decision_times.tz_localize("Asia/Taipei")
    else:
        dec_times_taipei = taiwan_decision_times.tz_convert("Asia/Taipei")

    dec_df = pd.DataFrame({"decision_time": dec_times_taipei}).sort_values("decision_time").reset_index(drop=True)
    dec_df["decision_time_utc"] = dec_df["decision_time"].dt.tz_convert("UTC")

    # 2. Foreign DataFrame availability column verification
    foreign_clean = foreign_market_df.copy()
    if foreign_available_col not in foreign_clean.columns:
        if mode == "formal" and not allow_estimated_availability:
            raise ValueError(
                f"Strict PIT Error: foreign_market_df missing required explicit '{foreign_available_col}' column. "
                "Automatic fallback to DataFrame index or trade date is strictly prohibited in formal mode."
            )
        if isinstance(foreign_clean.index, pd.DatetimeIndex):
            warnings.warn(
                f"Estimated Availability Fallback: using DataFrame index as '{foreign_available_col}'. "
                "DO NOT use in formal production evaluation.",
                UserWarning,
            )
            foreign_clean[foreign_available_col] = foreign_clean.index
        else:
            raise KeyError(f"foreign_market_df missing available time column '{foreign_available_col}'")

    # 3. Timezone validation on foreign available_at
    avail_series = pd.to_datetime(foreign_clean[foreign_available_col])
    if avail_series.dt.tz is None:
        if mode == "formal" and not allow_estimated_availability:
            raise ValueError(
                f"Strict PIT Error: foreign_market_df['{foreign_available_col}'] column must be timezone-aware. "
                "Naive timestamps cannot be presumed available without timezone specification in formal mode."
            )
        warnings.warn(
            f"Estimated Availability Fallback: localizing naive timestamps to America/New_York (16:00 close). "
            "DO NOT use in formal production evaluation.",
            UserWarning,
        )
        avail_utc = avail_series.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    else:
        avail_utc = avail_series.dt.tz_convert("UTC")

    foreign_clean["_avail_utc"] = avail_utc

    # 4. Mandatory Audit verification filter (P0-3)
    if "is_audit_verified" not in foreign_clean.columns:
        if mode == "formal":
            raise ValueError(
                "Strict PIT Error: foreign_market_df missing required audit verification column "
                "'is_audit_verified' in formal mode. Unverified sources are strictly rejected."
            )
        warnings.warn(
            "Estimated Availability Fallback: 'is_audit_verified' missing in synthetic mode.",
            UserWarning,
        )
    else:
        if mode == "formal":
            foreign_clean = foreign_clean[foreign_clean["is_audit_verified"] == True]

    # Filter out NaT or invalid timestamps
    foreign_valid = foreign_clean.dropna(subset=["_avail_utc"]).sort_values("_avail_utc").reset_index(drop=True)

    # 5. Backward-only As-Of Join
    feature_cols = [c for c in foreign_clean.columns if c not in (foreign_available_col, "_avail_utc", "is_audit_verified")]
    if len(foreign_valid) == 0:
        # No valid foreign records available
        result_df = pd.DataFrame(np.nan, index=taiwan_decision_times, columns=feature_cols)
    else:
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
        # Forward fill only forward in time from past valid observations (ZERO bfill)
        result_df = result_df.ffill()

    # Tag estimated availability status
    result_df.attrs["estimated_availability"] = bool(allow_estimated_availability or (mode == "synthetic"))

    return result_df

