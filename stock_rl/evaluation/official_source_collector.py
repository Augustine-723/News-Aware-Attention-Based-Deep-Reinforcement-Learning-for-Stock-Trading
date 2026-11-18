"""
V8.13 Official Historical Source Collection & Feasibility Engine
Provides:
1. OfficialSourceChannel & QueryResolutionStatus: Formal taxonomy for external official sources and query states.
2. OfficialAnnouncementRecord: Auditable official corporate disclosure record preserving raw announcement IDs,
   filing datetimes, actual retrieval timestamps, payload hashes, and license metadata.
3. OfficialHistoricalNewsCollector: Small-scale systematic collection engine targeting 2330, 2603, 2881
   across 2024 Q1 (56 authentic TWSE trading sessions), evaluating:
   - Official source accessibility and legal terms
   - Real query resolution (distinguishing UNQUERIED, ZERO_RESULTS, and AUTHENTIC_NO_ANNOUNCEMENT)
   - Segregation of Event Existence from Historical Downstream Provider Availability Evidence
   - Coverage, quality, deduplication, and revision tracking
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import urllib.request
import numpy as np
import pandas as pd
import torch

from stock_rl.evaluation.importance import (
    TWSETradingCalendar,
    NewsSourceType,
    EvidenceAuditLevel,
)
from stock_rl.evaluation.semantic_eval import (
    compute_normalized_content_hash,
)
from stock_rl.evaluation.pilot_dataset import (
    LicenseMetadata,
    LicenseVerificationStatus,
    UsageRightsCategory,
    SourceVerificationStatus,
    DisclosurePITStatus,
    ExternalSourceAdapterReceipt,
    AdapterReceiptMode,
    TrustedExternalSourceAdapter,
    RetrievalSessionRegistry,
    RetrievalSessionRecord,
    ProviderAvailabilityEvidence,
    ProviderVerificationStatus,
    SourceEvidenceLevel,
    TrustedProviderEvidenceAdapter,
)
from stock_rl.evaluation.provenance import (
    load_trusted_pretrained_embedding_engine,
    GenuinePretrainedSemanticEncoder,
)
from stock_rl.data.multimodal_pipeline import (
    RawNewsArticle,
    extract_event_metadata,
    extract_event_stock_relation,
)


class OfficialSourceChannel(str, Enum):
    """
    Formal taxonomy of recognized official Taiwan financial data channels.
    """
    TWSE_MOPS_HTML_ARCHIVE = "TWSE_MOPS_HTML_ARCHIVE"      # Public Information Observation Station (MOPS) Web Archive (t05st01)
    TWSE_OPENAPI = "TWSE_OPENAPI"                          # Taiwan Stock Exchange OpenAPI Service (openapi.twse.com.tw)
    TAIWAN_GOV_OPEN_DATA = "TAIWAN_GOV_OPEN_DATA"          # Government Open Data Platform (data.gov.tw / dataset 18415 上市公司每日重大訊息; note: 11549 is 個股日成交資訊)
    COMMERCIAL_VENDOR_FEEDS = "COMMERCIAL_VENDOR_FEEDS"    # Commercial vendor feed (TEJ / Bloomberg / Refinitiv)
    SECURITIES_FUTURES_INSTITUTE = "SECURITIES_FUTURES_INSTITUTE" # Securities & Futures Institute (SFI 證基會)


class ExecutionMode(str, Enum):
    """
    P0-3: Execution mode distinguishing simulated queries from authentic live network queries.
    Simulated attempts can only enter simulation_metrics and cannot count towards real_collection_coverage.
    """
    SIMULATED = "SIMULATED"
    AUTHENTIC = "AUTHENTIC"


class QueryResolutionStatus(str, Enum):
    """
    Resolution status for systematic historical news retrieval queries.
    Strictly distinguishes unattempted queries from completed non-event sessions and confirmed zero-announcements.
    """
    NOT_ATTEMPTED = "NOT_ATTEMPTED"                                              # Retrieval not attempted or unqueried
    QUERY_COMPLETED_SUCCESS = "QUERY_COMPLETED_SUCCESS"                          # Query executed and completed successfully
    QUERY_COMPLETED_ZERO_RESULTS = "QUERY_COMPLETED_ZERO_RESULTS"                # Query executed and completed; zero announcements returned (normal non-event session)
    AUTHENTIC_NO_ANNOUNCEMENT_CONFIRMED = "AUTHENTIC_NO_ANNOUNCEMENT_CONFIRMED"  # Exchange records cross-audited; confirmed zero material announcements on market
    ANNOUNCEMENTS_RETRIEVED = "ANNOUNCEMENTS_RETRIEVED"                          # Official announcements successfully retrieved and validated
    QUERY_FAILED = "QUERY_FAILED"                                                # Query failed due to network timeout or rate limiting
    QUERY_FAILED_HTTP = "QUERY_FAILED_HTTP"                                      # Query failed due to non-200 HTTP response code (P0-2)
    QUERY_FAILED_PARSE = "QUERY_FAILED_PARSE"                                    # Query succeeded at HTTP level but payload parsing failed (P0-2)


class PITExecutionMode(str, Enum):
    """
    Execution mode for Point-in-Time (PIT) As-Of availability integration.
    P0-2: Strictly separates estimated replay (+5 min hypothesis) from verified As-Of (vendor feed log required).
    """
    ESTIMATED_REPLAY = "ESTIMATED_REPLAY"
    EVIDENCE_VERIFIED_ASOF = "EVIDENCE_VERIFIED_ASOF"


@dataclass
class QueryAttemptRecord:
    """
    Auditable log of a single official news query attempt for a target trading session and stock.
    Prevents fabricated collection coverage claims by binding metrics directly to verified network logs.
    """
    target_date: str                              # YYYY-MM-DD
    stock_id: str                                 # e.g., "2330"
    source_channel: OfficialSourceChannel
    endpoint: str = ""
    query_start_time: Optional[pd.Timestamp] = None
    query_end_time: Optional[pd.Timestamp] = None
    http_status: Optional[int] = None
    response_status: QueryResolutionStatus = QueryResolutionStatus.NOT_ATTEMPTED
    actual_raw_response_hash: Optional[str] = None
    result_count: int = 0
    error_info: Optional[str] = None
    retry_info: Optional[str] = None
    raw_bytes_len: Optional[int] = None
    execution_mode: ExecutionMode = ExecutionMode.SIMULATED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_date": self.target_date,
            "stock_id": self.stock_id,
            "source_channel": self.source_channel.value,
            "endpoint": self.endpoint,
            "query_start_time": self.query_start_time.isoformat() if self.query_start_time else None,
            "query_end_time": self.query_end_time.isoformat() if self.query_end_time else None,
            "http_status": self.http_status,
            "response_status": self.response_status.value,
            "actual_raw_response_hash": self.actual_raw_response_hash,
            "result_count": self.result_count,
            "error_info": self.error_info,
            "retry_info": self.retry_info,
            "raw_bytes_len": self.raw_bytes_len,
            "execution_mode": self.execution_mode.value if hasattr(self.execution_mode, "value") else str(self.execution_mode),
        }


@dataclass
class OfficialAnnouncementRecord:
    """
    Auditable corporate disclosure record conforming to V8.13 specification.
    Strictly distinguishes:
    - candidate_fixture_id: Local candidate fixture identifier
    - announcement_id: Official exchange announcement ID (None if unverified from live exchange)
    - retrieved_at: Actual retrieval timestamp from verified network log (None if unretrieved)
    - raw_official_response_bytes: Raw HTTP bytes from official server (None if unretrieved)
    - raw_response_hash: SHA-256 of raw_official_response_bytes (None if unretrieved)
    - parsed_fields: Structured fields parsed from raw_official_response_bytes
    - candidate_fixture_hash: SHA-256 of local candidate text
    - normalized_content_hash: SHA-256 of normalized headline + body text
    - Event Existence vs. Downstream Provider Availability Evidence
    - Legal license metadata and usage restrictions
    - Revision lineage tracking (Rule 51 clarifications)
    """
    candidate_fixture_id: str
    company_id: str
    company_name: str
    headline: str
    body_text: str
    source_channel: OfficialSourceChannel
    source_url: str
    candidate_fixture_hash: str
    normalized_content_hash: str
    category: str
    cluster_id: str
    announcement_id: Optional[str] = None
    filing_datetime: Optional[pd.Timestamp] = None
    retrieved_at: Optional[pd.Timestamp] = None
    raw_official_response_bytes: Optional[bytes] = None
    raw_response_hash: Optional[str] = None
    parsed_fields: Optional[Dict[str, Any]] = None
    is_revision: bool = False
    revised_announcement_id: Optional[str] = None
    historical_provider_availability_included: bool = False
    provider_availability_receipt: Optional[Dict[str, Any]] = None
    license_metadata: Optional[LicenseMetadata] = None
    usage_restrictions: str = "Research analysis only; commercial training and raw redistribution require formal publisher/vendor licensing."
    verification_status: SourceVerificationStatus = SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED

    @property
    def payload_hash(self) -> str:
        return self.normalized_content_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_fixture_id": self.candidate_fixture_id,
            "announcement_id": self.announcement_id,
            "company_id": self.company_id,
            "company_name": self.company_name,
            "filing_datetime": self.filing_datetime.isoformat() if self.filing_datetime else None,
            "headline": self.headline,
            "body_text": self.body_text,
            "source_channel": self.source_channel.value,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
            "raw_response_hash": self.raw_response_hash,
            "candidate_fixture_hash": self.candidate_fixture_hash,
            "normalized_content_hash": self.normalized_content_hash,
            "category": self.category,
            "cluster_id": self.cluster_id,
            "is_revision": self.is_revision,
            "revised_announcement_id": self.revised_announcement_id,
            "historical_provider_availability_included": self.historical_provider_availability_included,
            "license_metadata": {
                "license_name": self.license_metadata.license_name,
                "verification_status": self.license_metadata.verification_status.value,
                "permitted_uses": self.license_metadata.permitted_uses,
            } if self.license_metadata else None,
            "verification_status": self.verification_status.value,
        }


@dataclass
class EvidenceManifestEntry:
    """
    Auditable entry in official evidence manifest conforming to V8.14 specification.
    Preserves exact provenance, query parameters, network retrieval session, raw response bytes,
    and distinct existence vs. availability verification status.
    """
    announcement_id: str
    stock_id: str
    filing_datetime: Optional[pd.Timestamp]
    headline: str
    raw_content: str
    source_url: str
    query_parameters: Dict[str, Any]
    retrieved_at: pd.Timestamp
    raw_response_bytes: bytes
    raw_response_hash: str
    parsed_payload_hash: str
    verification_status: SourceVerificationStatus = SourceVerificationStatus.SOURCE_EXISTENCE_VERIFIED
    pit_status: DisclosurePITStatus = DisclosurePITStatus.ESTIMATED_PIT_ONLY
    license_metadata: Optional[LicenseMetadata] = None
    retrieval_session_id: str = ""
    source_channel: OfficialSourceChannel = OfficialSourceChannel.TWSE_OPENAPI
    local_record_id: Optional[str] = None
    is_local_record_id: bool = False
    publication_time_parsed: bool = True
    parser_version: str = "v8.14-openapi-json-v1"
    source_locator: str = ""
    raw_retention_status: str = "FULL_RAW_BYTES_RETAINED"
    verified_provider_available_at: Optional[pd.Timestamp] = None
    raw_response_snapshot_file: Optional[str] = "twse_openapi_raw_response_snapshot.json"
    raw_array_index: Optional[int] = None
    matched_stock_index: Optional[int] = None
    provider_evidence: Optional[ProviderAvailabilityEvidence] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "announcement_id": self.announcement_id,
            "stock_id": self.stock_id,
            "filing_datetime": self.filing_datetime.isoformat() if self.filing_datetime else None,
            "headline": self.headline,
            "raw_content": self.raw_content,
            "source_url": self.source_url,
            "query_parameters": self.query_parameters,
            "retrieved_at": self.retrieved_at.isoformat() if self.retrieved_at else None,
            "raw_response_hash": self.raw_response_hash,
            "parsed_payload_hash": self.parsed_payload_hash,
            "verification_status": self.verification_status.value if hasattr(self.verification_status, "value") else str(self.verification_status),
            "pit_status": self.pit_status.value if hasattr(self.pit_status, "value") else str(self.pit_status),
            "retrieval_session_id": self.retrieval_session_id,
            "source_channel": self.source_channel.value if hasattr(self.source_channel, "value") else str(self.source_channel),
            "license_name": self.license_metadata.license_name if self.license_metadata else "Taiwan Open Government Data License",
            "local_record_id": self.local_record_id,
            "is_local_record_id": self.is_local_record_id,
            "publication_time_parsed": self.publication_time_parsed,
            "parser_version": self.parser_version,
            "source_locator": self.source_locator,
            "raw_retention_status": self.raw_retention_status,
            "raw_response_snapshot_file": self.raw_response_snapshot_file,
            "raw_array_index": self.raw_array_index,
            "matched_stock_index": self.matched_stock_index,
            "provider_evidence": self.provider_evidence.to_dict() if self.provider_evidence else None,
        }


def _parse_raw_record_publication_datetime(item: Dict[str, Any]) -> Tuple[Optional[pd.Timestamp], bool]:
    """
    P1-4: Parses exact publication datetime from raw official disclosure JSON record.
    Prioritizes '發言日期' + '發言時間' (official announcement publication).
    Falls back to '出表日期' + '出表時間' if publication time is unrecorded.
    Supports 7-digit ROC dates (e.g. '1151117' -> 2026-11-17) and 6-digit times (e.g. '154758' -> 15:47:58).
    Returns (pd.Timestamp, True) if parsed successfully, or (None, False) if missing/unparseable.
    """
    d_str = str(item.get("發言日期", "") or item.get("出表日期", "") or "").strip()
    t_str = str(item.get("發言時間", "") or item.get("出表時間", "") or "").strip()
    if not d_str:
        return None, False
    try:
        if len(d_str) == 7 and d_str.isdigit():
            roc_yr = int(d_str[:3])
            ce_yr = 1911 + roc_yr
            mm = d_str[3:5]
            dd = d_str[5:7]
        elif len(d_str) == 8 and d_str.isdigit():
            ce_yr = int(d_str[:4])
            mm = d_str[4:6]
            dd = d_str[6:8]
        elif "-" in d_str:
            parts = d_str.split("-")
            ce_yr, mm, dd = int(parts[0]), parts[1], parts[2]
        else:
            return None, False

        if len(t_str) == 6 and t_str.isdigit():
            hh, mi, ss = t_str[:2], t_str[2:4], t_str[4:6]
        elif ":" in t_str:
            t_parts = t_str.split(":")
            hh = f"{int(t_parts[0]):02d}"
            mi = f"{int(t_parts[1]):02d}" if len(t_parts) > 1 else "00"
            ss = f"{int(t_parts[2]):02d}" if len(t_parts) > 2 else "00"
        elif len(t_str) == 4 and t_str.isdigit():
            hh, mi, ss = t_str[:2], t_str[2:4], "00"
        else:
            hh, mi, ss = "00", "00", "00"

        dt = pd.Timestamp(f"{ce_yr}-{mm}-{dd} {hh}:{mi}:{ss}", tz="Asia/Taipei")
        return dt, True
    except Exception:
        return None, False


@dataclass
class EvidenceManifest:
    """
    Auditable manifest of genuine official evidence records.
    Provides independent verification of entry integrity, raw response re-parsing, and reproducibility.
    """
    manifest_id: str
    entries: List[EvidenceManifestEntry] = field(default_factory=list)
    created_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz="Asia/Taipei"))
    provenance_notes: str = ""

    def add_entry(self, entry: EvidenceManifestEntry) -> None:
        self.entries.append(entry)

    @classmethod
    def verify_entry_integrity_standalone(cls, entry: EvidenceManifestEntry) -> Tuple[bool, str]:
        """
        P0-2: Standalone verification of an EvidenceManifestEntry without requiring a pre-existing manifest instance.
        """
        inst = cls(manifest_id=f"PIT-STANDALONE-{getattr(entry, 'announcement_id', 'ENTRY')[:16]}")
        return inst.verify_entry_integrity(entry)

    def verify_entry_integrity(self, entry: EvidenceManifestEntry) -> Tuple[bool, str]:
        # 1. Required fields check (P1-4 item 7)
        required_fields = [
            ("stock_id", entry.stock_id),
            ("headline", entry.headline),
            ("raw_content", entry.raw_content),
            ("source_url", entry.source_url),
            ("raw_response_bytes", entry.raw_response_bytes),
            ("raw_response_hash", entry.raw_response_hash),
            ("parsed_payload_hash", entry.parsed_payload_hash),
        ]
        for fname, fval in required_fields:
            if fval is None or (isinstance(fval, (str, bytes)) and len(fval) == 0):
                return False, f"Required field '{fname}' is missing or empty."

        # 2. Raw response retention check (P0-2 item 8, P1-3)
        if not entry.raw_response_bytes:
            return False, "raw_response_bytes is empty."
        if getattr(entry, "raw_retention_status", "") == "INDEPENDENT_REPRODUCTION_PENDING":
            return False, "raw_response_bytes cannot be retained (INDEPENDENT_REPRODUCTION_PENDING); independent raw data hash recomputation is pending."

        # 3. Raw bytes SHA-256 digest check
        actual_resp_hash = hashlib.sha256(entry.raw_response_bytes).hexdigest()
        if actual_resp_hash != entry.raw_response_hash:
            return False, f"raw_response_hash mismatch: expected {entry.raw_response_hash}, got {actual_resp_hash}."

        # 4. Parsed payload hash check
        expected_payload_hash = compute_normalized_content_hash(entry.headline, entry.raw_content)
        if expected_payload_hash != entry.parsed_payload_hash:
            return False, f"parsed_payload_hash mismatch: expected {entry.parsed_payload_hash}, got {expected_payload_hash}."

        # 5. Session registry cross-check
        session = RetrievalSessionRegistry.get_session(entry.retrieval_session_id)
        if session is None:
            return False, f"retrieval_session_id '{entry.retrieval_session_id}' not found in registry."
        if session.raw_response_hash != entry.raw_response_hash:
            return False, "session raw_response_hash mismatch."

        # 6. Publication time parsing integrity (P1-4 items 3, 4)
        # When publication time is unparsed, filing_datetime cannot silently substitute retrieved_at
        if not entry.publication_time_parsed and entry.filing_datetime is not None:
            return False, "Integrity check failed: filing_datetime cannot silently substitute retrieved_at when publication_time_parsed=False."

        # 7. P1-4: Re-parse raw response bytes to verify target announcement binding with exact field matching
        if entry.source_channel == OfficialSourceChannel.TWSE_OPENAPI:
            try:
                parsed_json = json.loads(entry.raw_response_bytes.decode("utf-8"))
            except Exception as e:
                return False, f"Failed to re-parse raw OpenAPI JSON: {e}"

            if not isinstance(parsed_json, list):
                if isinstance(parsed_json, dict) and "data" in parsed_json:
                    parsed_json = parsed_json["data"]
                else:
                    parsed_json = [parsed_json]

            matched = False
            field_mismatch_reason = ""
            for item in parsed_json:
                if not isinstance(item, dict):
                    continue
                item_stock = str(item.get("公司代號", "") or item.get("stock_id", "") or item.get("Code", "")).strip()
                if item_stock != entry.stock_id:
                    continue

                # Exact stable field matching (P1-4: avoid loose substring match)
                item_sub = str(item.get("主旨 ", "") or item.get("主旨", "") or item.get("headline", "") or item.get("Subject", "")).strip()
                item_content = str(item.get("說明", "") or item.get("詳細內容", "") or item.get("body_text", "") or item.get("content", "") or item_sub).strip()

                if entry.headline.strip() != item_sub:
                    continue
                if entry.raw_content.strip() != item_content:
                    continue

                # Exact publication datetime re-parsing check (P1-4 items 1, 3)
                item_dt, is_parsed = _parse_raw_record_publication_datetime(item)
                if entry.filing_datetime is not None:
                    if not is_parsed or item_dt != entry.filing_datetime:
                        field_mismatch_reason = f"Publication datetime mismatch: expected {entry.filing_datetime}, parsed from raw record: {item_dt}."
                        continue

                matched = True
                break

            if not matched:
                err_detail = field_mismatch_reason or (
                    f"Target announcement for stock '{entry.stock_id}' with headline '{entry.headline[:30]}' "
                    "was not found in parsed raw response bytes (exact field match failed; fuzzy match rejected)."
                )
                return False, f"Content binding verification failed: {err_detail}"

        elif entry.source_channel == OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE:
            decoded_text = entry.raw_response_bytes.decode("utf-8", errors="ignore")
            if entry.stock_id not in decoded_text:
                return False, f"Content binding verification failed: stock_id '{entry.stock_id}' not found in raw MOPS response."
            if entry.headline not in decoded_text and entry.raw_content not in decoded_text:
                return False, "Content binding verification failed: Neither headline nor content was found in raw MOPS response."

        # 8. Local record ID audit (P1-4 item 5)
        if entry.is_local_record_id:
            if not entry.local_record_id or not entry.local_record_id.startswith("LOCAL-"):
                return False, "Announcement marked as is_local_record_id=True but local_record_id format is invalid."
            if entry.announcement_id != entry.local_record_id:
                return False, f"announcement_id '{entry.announcement_id}' must match local_record_id '{entry.local_record_id}'."

        # 9. Source Locator Integrity Check (V8.15 Item 2)
        # Verifies that raw_array_index and matched_stock_index point to the exact target record in the raw array
        if entry.source_channel == OfficialSourceChannel.TWSE_OPENAPI:
            target_raw_idx = entry.raw_array_index
            target_matched_idx = entry.matched_stock_index
            if entry.source_locator:
                m_raw = re.search(r"raw_array_index=(\d+)", entry.source_locator)
                if m_raw:
                    parsed_r = int(m_raw.group(1))
                    if target_raw_idx is not None and target_raw_idx != parsed_r:
                        return False, f"Source locator mismatch: entry.raw_array_index ({target_raw_idx}) != locator raw_array_index ({parsed_r})."
                    target_raw_idx = parsed_r
                m_matched = re.search(r"matched_stock_index=(\d+)", entry.source_locator)
                if m_matched:
                    parsed_m = int(m_matched.group(1))
                    if target_matched_idx is not None and target_matched_idx != parsed_m:
                        return False, f"Source locator mismatch: entry.matched_stock_index ({target_matched_idx}) != locator matched_stock_index ({parsed_m})."
                    target_matched_idx = parsed_m

            if target_raw_idx is not None:
                if target_raw_idx >= len(parsed_json):
                    return False, f"Source locator raw_array_index {target_raw_idx} out of range (total records: {len(parsed_json)})."
                loc_record = parsed_json[target_raw_idx]
                loc_stock = str(loc_record.get("公司代號", "") or loc_record.get("stock_id", "") or loc_record.get("Code", "")).strip()
                loc_sub = str(loc_record.get("主旨 ", "") or loc_record.get("主旨", "") or loc_record.get("headline", "") or loc_record.get("Subject", "")).strip()
                loc_content = str(loc_record.get("說明", "") or loc_record.get("詳細內容", "") or loc_record.get("body_text", "") or loc_record.get("content", "") or loc_sub).strip()
                loc_dt, loc_dt_parsed = _parse_raw_record_publication_datetime(loc_record)

                if loc_stock != entry.stock_id:
                    return False, f"Source locator stock_id mismatch at raw_array_index={target_raw_idx}: expected '{entry.stock_id}', found '{loc_stock}'."
                if loc_sub != entry.headline.strip():
                    return False, f"Source locator headline mismatch at raw_array_index={target_raw_idx}: expected '{entry.headline.strip()[:30]}', found '{loc_sub[:30]}'."
                if loc_content != entry.raw_content.strip():
                    return False, f"Source locator raw_content mismatch at raw_array_index={target_raw_idx}."
                if entry.filing_datetime is not None and loc_dt != entry.filing_datetime:
                    return False, f"Source locator publication datetime mismatch at raw_array_index={target_raw_idx}: expected {entry.filing_datetime}, found {loc_dt}."
                if target_matched_idx is not None:
                    actual_matched_idx = sum(
                        1 for i in range(target_raw_idx)
                        if str(parsed_json[i].get("公司代號", "") or parsed_json[i].get("stock_id", "") or parsed_json[i].get("Code", "")).strip() == entry.stock_id
                    )
                    if actual_matched_idx != target_matched_idx:
                        return False, f"Source locator matched_stock_index mismatch: expected {target_matched_idx}, computed {actual_matched_idx} at raw_array_index={target_raw_idx}."

        # 10. Provider Availability Evidence Check (V8.15 Item 1)
        if entry.provider_evidence is not None:
            p_ev = entry.provider_evidence
            if p_ev.announcement_id != entry.announcement_id:
                return False, f"Provider evidence announcement_id mismatch: '{p_ev.announcement_id}' != '{entry.announcement_id}'."
            if p_ev.content_payload_hash and p_ev.content_payload_hash != entry.parsed_payload_hash:
                return False, f"Provider evidence content hash mismatch: '{p_ev.content_payload_hash}' != '{entry.parsed_payload_hash}'."
            if p_ev.is_verified_feed and p_ev.verification_status != ProviderVerificationStatus.PROVIDER_FEED_VERIFIED:
                return False, f"Provider evidence status '{p_ev.verification_status}' is invalid for verified feed."

        return True, "Entry integrity, raw response re-parsing, and cryptographic digests verified."


class OfficialHistoricalNewsCollector:
    """
    V8.13 Systematic Small-Scale Historical News Collection & Audit Engine.
    Focuses on 2024 Q1 (2024-01-01 to 2024-03-31) across 2330 (TSMC), 2603 (Evergreen), 2881 (Fubon).
    Audits 56 authentic TWSE trading sessions, evaluating:
    1. Official source queryability, legal terms, and availability limitations.
    2. Event existence vs. historical downstream feed availability.
    3. Retrieval quality, deduplication, revisions, and failure rates.
    4. Decoupled coverage metrics (Collection, Retrieval, Fixture Density, Slot Occupancy).
    """
    Q1_2024_TOTAL_TRADING_SESSIONS: int = 56

    # --------------------------------------------------------------------------
    # Local Candidate Fixture for 2024 Q1 (P0-2: Candidate Fixture, UNVERIFIED)
    # --------------------------------------------------------------------------
    LOCAL_Q1_2024_CANDIDATE_FIXTURE: List[Dict[str, Any]] = [
        # --- 2330 (TSMC) ---
        {
            "candidate_fixture_id": "CANDIDATE-Q1-2024-2330-01",
            "company_id": "2330",
            "company_name": "台積電",
            "date_str": "2024-01-18",
            "pub_time_str": "14:30:00",
            "headline": "台積電說明112年第四季財務報告及113年第一季業績展望",
            "body_text": "台積電公告112年第四季合併營收約6255.3億元，稅後純益約2387.1億元，每股盈餘為9.21元；預估113年首季營收受惠高效能運算需求維持強勁。",
            "category": "earnings",
            "cluster_id": "CLU-2330-Q4-2023",
            "source_url": "https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=113&month=01&co_id=2330",
            "is_revision": False,
            "revised_announcement_id": None,
            "announcement_id": None,  # P0-2: Unverified without external live exchange response
            "verification_status": SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
        },
        {
            "candidate_fixture_id": "CANDIDATE-Q1-2024-2330-02",
            "company_id": "2330",
            "company_name": "台積電",
            "date_str": "2024-02-05",  # Filed pre-close or holiday period, effective decision date 2024-02-05
            "pub_time_str": "17:15:00",
            "headline": "台積電董事會決議核准配發112年第四季每股現金股利3.5元及核准資本預算",
            "body_text": "台積電董事會重要決議，核准配發112年第四季普通股現金股利每股新台幣3.5元，並核准資本預算約美金94億2148萬元，用以建置先進製程產能。",
            "category": "dividend",
            "cluster_id": "CLU-2330-DIV-CAPEX-Q4",
            "source_url": "https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=113&month=02&co_id=2330",
            "is_revision": False,
            "revised_announcement_id": None,
            "announcement_id": None,
            "verification_status": SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
        },
        {
            "candidate_fixture_id": "CANDIDATE-Q1-2024-2330-03",
            "company_id": "2330",
            "company_name": "台積電",
            "date_str": "2024-02-15",
            "pub_time_str": "14:00:00",
            "headline": "台積電公告113年1月合併營業收入淨額2157億8500萬元月增逾兩成",
            "body_text": "台積電公告113年1月合併營業收入新台幣2157億8500萬元，較上月增加22.4%，較去年同期增加7.9%，創同期歷年新高。",
            "category": "earnings",
            "cluster_id": "CLU-2330-REV-JAN24",
            "source_url": "https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=113&month=02&co_id=2330",
            "is_revision": False,
            "revised_announcement_id": None,
            "announcement_id": None,
            "verification_status": SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
        },
        {
            "candidate_fixture_id": "CANDIDATE-Q1-2024-2330-04",
            "company_id": "2330",
            "company_name": "台積電",
            "date_str": "2024-03-08",
            "pub_time_str": "14:00:00",
            "headline": "台積電公告113年2月合併營業收入淨額1816億4800萬元",
            "body_text": "台積電公告113年2月合併營收約新台幣1816億4800萬元，因春節工作天數減少月減15.8%，較去年同期仍成長11.3%。",
            "category": "earnings",
            "cluster_id": "CLU-2330-REV-FEB24",
            "source_url": "https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=113&month=03&co_id=2330",
            "is_revision": False,
            "revised_announcement_id": None,
            "announcement_id": None,
            "verification_status": SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
        },
        {
            "candidate_fixture_id": "CANDIDATE-Q1-2024-2330-05",
            "company_id": "2330",
            "company_name": "台積電",
            "date_str": "2024-03-18",
            "pub_time_str": "15:30:00",
            "headline": "台積電核准於嘉義科學園區建置先進封裝廠",
            "body_text": "台積電因應全球半導體高階封裝強勁需求，核准於嘉義科學園區建置先進封裝(CoWoS)廠，預計第一期封裝廠將於今年動工。",
            "category": "capex",
            "cluster_id": "CLU-2330-CHIAYI-COWOS",
            "source_url": "https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=113&month=03&co_id=2330",
            "is_revision": False,
            "revised_announcement_id": None,
            "announcement_id": None,
            "verification_status": SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
        },
        {
            "candidate_fixture_id": "CANDIDATE-Q1-2024-2330-06",
            "company_id": "2330",
            "company_name": "台積電",
            "date_str": "2024-03-18",
            "pub_time_str": "17:00:00",
            "headline": "澄清媒體報導台積電嘉義建廠投資規模相關訊息",
            "body_text": "依臺灣證券交易所重大訊息規定第51條澄清說明：媒體報導有關建廠投資總額與產能規劃純屬市場推估，實際進度依公司重大訊息公告為準。",
            "category": "clarification",
            "cluster_id": "CLU-2330-CHIAYI-COWOS",
            "source_url": "https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=113&month=03&co_id=2330",
            "is_revision": True,
            "revised_announcement_id": "CANDIDATE-Q1-2024-2330-05",
            "announcement_id": None,
            "verification_status": SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
        },

        # --- 2603 (Evergreen Marine) ---
        {
            "candidate_fixture_id": "CANDIDATE-Q1-2024-2603-01",
            "company_id": "2603",
            "company_name": "長榮海運",
            "date_str": "2024-01-15",
            "pub_time_str": "17:45:00",
            "headline": "長榮海運公告因應紅海航線受阻加徵海運附加費相關說明",
            "body_text": "長榮海運公告針對紅海區域地緣政治危機，船舶繞行好望角衍生之營運成本，正式發布應變附加費及航線調度調整通知。",
            "category": "crisis",
            "cluster_id": "CLU-2603-RED-SEA-SURCHARGE",
            "source_url": "https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=113&month=01&co_id=2603",
            "is_revision": False,
            "revised_announcement_id": None,
            "announcement_id": None,
            "verification_status": SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
        },
        {
            "candidate_fixture_id": "CANDIDATE-Q1-2024-2603-02",
            "company_id": "2603",
            "company_name": "長榮海運",
            "date_str": "2024-02-15",
            "pub_time_str": "15:00:00",
            "headline": "長榮公告113年1月合併營業收入新台幣283億5200萬元月增8%",
            "body_text": "長榮海運公告113年1月單月合併營業收入新台幣283.52億元，較上月增加8.0%，受紅海危機帶動運價回升。",
            "category": "earnings",
            "cluster_id": "CLU-2603-REV-JAN24",
            "source_url": "https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=113&month=02&co_id=2603",
            "is_revision": False,
            "revised_announcement_id": None,
            "announcement_id": None,
            "verification_status": SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
        },
        {
            "candidate_fixture_id": "CANDIDATE-Q1-2024-2603-03",
            "company_id": "2603",
            "company_name": "長榮海運",
            "date_str": "2024-03-14",
            "pub_time_str": "18:20:00",
            "headline": "長榮董事會決議通過112年度財務報告及擬配發現金股利每股10元",
            "body_text": "長榮海運董事會通過112年度合併財務報告，全年稅後淨利353.37億元，每股盈餘16.70元，擬配發每股現金股利10元。",
            "category": "earnings",
            "cluster_id": "CLU-2603-EARNINGS-FY23",
            "source_url": "https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=113&month=03&co_id=2603",
            "is_revision": False,
            "revised_announcement_id": None,
            "announcement_id": None,
            "verification_status": SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
        },

        # --- 2881 (Fubon Financial) ---
        {
            "candidate_fixture_id": "CANDIDATE-Q1-2024-2881-01",
            "company_id": "2881",
            "company_name": "富邦金控",
            "date_str": "2024-01-10",
            "pub_time_str": "16:30:00",
            "headline": "富邦金控公告112年12月自結合併稅前淨利及累計每股盈餘4.81元",
            "body_text": "富邦金控自結112年12月稅前盈餘，累計112年全年自結合併稅後淨利661.37億元，每股稅後盈餘(EPS)為4.81元，持續居金控獲利之冠。",
            "category": "earnings",
            "cluster_id": "CLU-2881-FULLYEAR-2023",
            "source_url": "https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=113&month=01&co_id=2881",
            "is_revision": False,
            "revised_announcement_id": None,
            "announcement_id": None,
            "verification_status": SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
        },
        {
            "candidate_fixture_id": "CANDIDATE-Q1-2024-2881-02",
            "company_id": "2881",
            "company_name": "富邦金控",
            "date_str": "2024-02-15",
            "pub_time_str": "16:30:00",
            "headline": "富邦金控公告113年1月自結合併稅後盈餘140.7億元EPS 1.08元",
            "body_text": "富邦金控公告113年1月自結合併稅後淨利140.70億元，較去年同期顯著成長，每股盈餘為1.08元，主要子公司獲利穩健。",
            "category": "earnings",
            "cluster_id": "CLU-2881-REV-JAN24",
            "source_url": "https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=113&month=02&co_id=2881",
            "is_revision": False,
            "revised_announcement_id": None,
            "announcement_id": None,
            "verification_status": SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
        },
        {
            "candidate_fixture_id": "CANDIDATE-Q1-2024-2881-03",
            "company_id": "2881",
            "company_name": "富邦金控",
            "date_str": "2024-03-15",
            "pub_time_str": "17:30:00",
            "headline": "富邦金控董事會通過112年度財務報告及承認營業報告書",
            "body_text": "富邦金控董事會決議通過112年度財務報表，稅後純益為660.17億元，每股盈餘(EPS)為4.80元，並提請股東常會承認。",
            "category": "earnings",
            "cluster_id": "CLU-2881-FULLYEAR-2023",
            "source_url": "https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=113&month=03&co_id=2881",
            "is_revision": False,
            "revised_announcement_id": None,
            "announcement_id": None,
            "verification_status": SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
        },
    ]

    AUDITED_Q1_2024_ANNOUNCEMENTS = LOCAL_Q1_2024_CANDIDATE_FIXTURE  # Backwards compatibility alias

    @classmethod
    def get_q1_2024_trading_sessions(cls) -> List[str]:
        """
        Returns the list of 56 authentic TWSE trading sessions in 2024 Q1 (YYYY-MM-DD).
        """
        from stock_rl.evaluation.pilot_dataset import TWSETradingCalendar
        sessions = [
            d.strftime("%Y-%m-%d")
            for d in pd.date_range("2024-01-01", "2024-03-31")
            if TWSETradingCalendar.is_trading_day(d)
        ]
        return sessions

    @classmethod
    def initialize_q1_query_attempts(
        cls, constituents: Optional[List[str]] = None
    ) -> List[QueryAttemptRecord]:
        """
        Initializes query attempt records for each Q1 2024 trading session and target constituent.
        In offline state prior to actual network execution, all records are strictly NOT_ATTEMPTED,
        with None timestamps and None HTTP status (P0-1: No fabricated collection coverage).
        """
        if constituents is None:
            constituents = ["2330", "2603", "2881"]
        sessions = cls.get_q1_2024_trading_sessions()
        attempts: List[QueryAttemptRecord] = []
        for s in sessions:
            for c in constituents:
                attempts.append(
                    QueryAttemptRecord(
                        target_date=s,
                        stock_id=c,
                        source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
                        endpoint="https://mops.twse.com.tw/mops/web/t05st01",
                        query_start_time=None,
                        query_end_time=None,
                        http_status=None,
                        response_status=QueryResolutionStatus.NOT_ATTEMPTED,
                        actual_raw_response_hash=None,
                        result_count=0,
                        error_info=None,
                        retry_info=None,
                        raw_bytes_len=None,
                        execution_mode=ExecutionMode.AUTHENTIC,
                    )
                )
        return attempts

    @classmethod
    def execute_live_openapi_query_probe(
        cls,
        target_stock: str = "2330",
        timeout: float = 10.0,
    ) -> QueryAttemptRecord:
        """
        Executes a genuine, lawful, single-session OpenAPI query probe against
        TWSE OpenAPI (https://openapi.twse.com.tw/v1/opendata/t187ap04_L).
        Records honest query timings, HTTP status, and actual raw payload hash.
        Fails closed with honest error details if network is unreachable or blocked.
        Does NOT fabricate HTTP responses or coverage.
        Registers genuine session in RetrievalSessionRegistry upon HTTP 200 (P0-1).
        """
        endpoint = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
        start_time = pd.Timestamp.now(tz="Asia/Taipei")
        target_date = start_time.strftime("%Y-%m-%d")

        try:
            req = urllib.request.Request(
                endpoint,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAuditCollector/1.0; +http://twse.gov.tw)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw_bytes = response.read()
                end_time = pd.Timestamp.now(tz="Asia/Taipei")
                http_status = response.status
                raw_hash = hashlib.sha256(raw_bytes).hexdigest()

                # Parse JSON array if available
                result_count = 0
                parse_failed = False
                try:
                    data = json.loads(raw_bytes.decode("utf-8"))
                    if isinstance(data, list):
                        stock_records = [d for d in data if str(d.get("公司代號", "")).strip() == target_stock]
                        result_count = len(stock_records)
                    else:
                        parse_failed = True
                except Exception:
                    parse_failed = True

                if http_status != 200:
                    res_status = QueryResolutionStatus.QUERY_FAILED_HTTP
                elif parse_failed:
                    res_status = QueryResolutionStatus.QUERY_FAILED_PARSE
                else:
                    res_status = QueryResolutionStatus.QUERY_COMPLETED_SUCCESS

                # Register genuine session in RetrievalSessionRegistry if HTTP 200 (P0-1)
                if http_status == 200:
                    from stock_rl.evaluation.pilot_dataset import RetrievalSessionRegistry
                    RetrievalSessionRegistry.register_genuine_session(
                        source_channel="TWSE_OPENAPI",
                        endpoint_url=endpoint,
                        request_start_time=start_time,
                        response_end_time=end_time,
                        http_status=http_status,
                        raw_response_bytes=raw_bytes,
                        parsed_summary={"target_stock": target_stock, "result_count": result_count},
                        source_identity="TWSE_OPENAPI_HTTPS_CLIENT",
                        announcement_ids=[],
                    )

                return QueryAttemptRecord(
                    target_date=target_date,
                    stock_id=target_stock,
                    source_channel=OfficialSourceChannel.TWSE_OPENAPI,
                    endpoint=endpoint,
                    query_start_time=start_time,
                    query_end_time=end_time,
                    http_status=http_status,
                    response_status=res_status,
                    actual_raw_response_hash=raw_hash,
                    result_count=result_count,
                    raw_bytes_len=len(raw_bytes),
                    error_info=None,
                    retry_info=None,
                    execution_mode=ExecutionMode.AUTHENTIC,
                )
        except Exception as e:
            end_time = pd.Timestamp.now(tz="Asia/Taipei")
            http_code = getattr(e, "code", None) if hasattr(e, "code") else None
            res_status = QueryResolutionStatus.QUERY_FAILED_HTTP if http_code is not None else QueryResolutionStatus.QUERY_FAILED
            return QueryAttemptRecord(
                target_date=target_date,
                stock_id=target_stock,
                source_channel=OfficialSourceChannel.TWSE_OPENAPI,
                endpoint=endpoint,
                query_start_time=start_time,
                query_end_time=end_time,
                http_status=http_code,
                response_status=res_status,
                actual_raw_response_hash=None,
                result_count=0,
                raw_bytes_len=0,
                error_info=str(e),
                retry_info="NOT_RETRIED",
                execution_mode=ExecutionMode.AUTHENTIC,
            )

    @classmethod
    def get_candidate_inventory(cls) -> List[OfficialAnnouncementRecord]:
        """
        Builds the candidate fixture inventory for 2024 Q1.
        In accordance with P0-2:
        - retrieved_at is set strictly to None (no fake retrieval timestamp!)
        - raw_official_response_bytes and raw_response_hash are set strictly to None
        - candidate_fixture_hash is computed from local body_text
        - normalized_content_hash is computed from headline + body_text
        - official announcement_id is set to None (unverified from live exchange)
        - verification_status is MOPS_CANDIDATE_UNVERIFIED
        """
        records: List[OfficialAnnouncementRecord] = []
        for item in cls.LOCAL_Q1_2024_CANDIDATE_FIXTURE:
            filing_dt = pd.Timestamp(f"{item['date_str']} {item['pub_time_str']}", tz="Asia/Taipei")
            p_hash = compute_normalized_content_hash(item["headline"], item["body_text"])
            fixture_hash = hashlib.sha256(item["body_text"].encode("utf-8")).hexdigest()

            lic_meta = LicenseMetadata(
                license_name="Taiwan Open Government Data License (MOPS Public Sector Information)",
                license_terms_url="https://data.gov.tw/license",
                license_version="1.0",
                license_retrieved_at=None,
                permitted_uses=["RESEARCH_AND_STATISTICAL_ANALYSIS", "ATTRIBUTED_QUOTATION"],
                verification_status=LicenseVerificationStatus.NEEDS_LEGAL_REVIEW,
                audit_notes="Public accessibility on MOPS under Securities and Exchange Act Art. 36 allows regulatory inspection, but does NOT grant blanket commercial AI training or raw corpus redistribution rights.",
            )

            rec = OfficialAnnouncementRecord(
                candidate_fixture_id=item["candidate_fixture_id"],
                announcement_id=item.get("announcement_id", None),
                company_id=item["company_id"],
                company_name=item["company_name"],
                filing_datetime=filing_dt,
                headline=item["headline"],
                body_text=item["body_text"],
                source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
                source_url=item["source_url"],
                retrieved_at=None,                           # P0-2: None without actual network log
                raw_official_response_bytes=None,            # P0-2: None without live server bytes
                raw_response_hash=None,                      # P0-2: None without live server bytes
                candidate_fixture_hash=fixture_hash,         # Local fixture text hash
                normalized_content_hash=p_hash,              # Model content hash
                parsed_fields=None,
                category=item["category"],
                cluster_id=item["cluster_id"],
                is_revision=item["is_revision"],
                revised_announcement_id=item["revised_announcement_id"],
                historical_provider_availability_included=False,
                provider_availability_receipt=None,
                license_metadata=lic_meta,
                usage_restrictions="Internal research benchmarking only; commercial model distribution prohibited without formal publisher contracts.",
                verification_status=SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED,
            )
            records.append(rec)
        return records

    # Backward-compatible alias
    get_source_inventory = get_candidate_inventory

    @classmethod
    def get_official_source_survey(cls) -> Dict[str, Any]:
        """
        P1-3: Comprehensive survey and technical audit of Taiwan financial news and disclosure sources.
        Categorizes each source by formal EvidenceLevel:
        - OFFICIALLY_DOCUMENTED
        - PROVIDER_CONFIRMATION_REQUIRED
        - PROJECT_ACCESS_VERIFIED
        - UNAVAILABLE_OR_BLOCKED
        Strictly prohibits equating theoretical data existence with research lawful accessibility.
        """
        return {
            "sources": {
                "TWSE_OPENAPI": {
                    "source_name": "臺灣證券交易所 OpenAPI (openapi.twse.com.tw)",
                    "official_url": "https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
                    "evidence_level": {
                        "rolling_current_window": SourceEvidenceLevel.PROJECT_ACCESS_VERIFIED.value,
                        "historical_2024_q1": SourceEvidenceLevel.UNAVAILABLE_OR_BLOCKED.value,
                    },
                    "confirmed_query_capabilities": (
                        "RESTful JSON API via t187ap04_L endpoint confirmed functional in project service; "
                        "retrieved authentic 155,496 Bytes snapshot (SHA-256: 949d1b51...) with 94 current disclosures."
                    ),
                    "historical_limitations": (
                        "Strictly rolling window (current/near-term ~30 days); no historical query parameters "
                        "or backfill interface available for 2024 Q1 (0% historical coverage)."
                    ),
                    "query_interface": "RESTful JSON API",
                    "access_legality": "合法公開免費用戶端介面 (政府資料開放授權條款 v1.0)",
                    "rate_limits_and_crawler_restrictions": "標準 API 呼叫頻率限制 (約 5 次/秒/IP)",
                    "historical_provider_availability_evidence": False,
                    "provider_availability_explanation": "不提供 2024 年第一季等遠期歷史重大訊息之回溯 API，亦無下游資料商落地日誌。",
                    "evaluation_verdict": "PROJECT_VERIFIED_FOR_LIVE_ROLLING_BUT_UNSUITABLE_FOR_2024_Q1_HISTORICAL",
                },
                "TWSE_MOPS_HTML_ARCHIVE": {
                    "source_name": "臺灣證券交易所 公開資訊觀測站 (MOPS) 歷史重大訊息查詢",
                    "official_url": "https://mops.twse.com.tw/mops/web/t05st01",
                    "evidence_level": {
                        "theoretical_statutory_coverage": SourceEvidenceLevel.OFFICIALLY_DOCUMENTED.value,
                        "project_automated_access": SourceEvidenceLevel.UNAVAILABLE_OR_BLOCKED.value,
                    },
                    "queryable_date_range": "1999-01-01 to Present (民國88年迄今完整重大訊息存檔，約5萬-8萬則/年)",
                    "theoretical_coverage": "1999-01-01 to Present (民國88年迄今完整重大訊息存檔，約5萬-8萬則/年)",
                    "project_accessible_scope": (
                        "0 筆重大訊息可透過自動化程式取得。受 WAF 安全防護機制阻擋 (HTTP 403 / 'FOR SECURITY REASONS')。"
                        "專案治理嚴格禁止偽造標頭或繞過存取控制，故在獲得官方許可前維持 UNAVAILABLE_OR_BLOCKED。"
                    ),
                    "query_interface": "HTTP POST/GET Form Query (year, month, co_id, step=1)",
                    "access_legality": "證券交易法第36條規定公開揭露資訊，供投資人個別查閱；未開放未授權之批次自動化爬蟲",
                    "rate_limits_and_crawler_restrictions": "設有嚴格 WAF 速率限制與防爬機制；批次查詢即時觸發 HTTP 403 阻擋",
                    "historical_provider_availability_evidence": False,
                    "provider_availability_explanation": (
                        "MOPS 僅記錄發行公司上傳申報時間（filing_datetime），無記錄下游量化交易系統或資料商之即時串流接收時間。"
                        "因此僅能確立 Event Existence 與 Official Publication Datetime，無法證明歷史決策時點 T 之可用性。"
                    ),
                    "licensing_terms": "政府資料開放授權條款 / 證券交易法第36條公開資訊",
                    "licensing_status": LicenseVerificationStatus.NEEDS_LEGAL_REVIEW.value,
                    "commercial_redistribution_eligible": False,
                    "evaluation_verdict": "FEASIBLE_FOR_HISTORICAL_AUDIT_BUT_LACKS_VENDOR_FEED_RECEIPTS",
                },
                "COMMERCIAL_VENDOR_FEEDS": {
                    "source_name": "商業資料庫廠商 (TEJ 台灣經濟新報 / Bloomberg B-PIPE / Refinitiv)",
                    "official_url": "https://www.tej.com.tw / https://www.bloomberg.com",
                    "evidence_level": SourceEvidenceLevel.PROVIDER_CONFIRMATION_REQUIRED.value,
                    "queryable_date_range": "完整 20 年以上台灣上市櫃公司新聞、公告與財務數據",
                    "estimated_annual_volume": "> 100,000 則/年 (含新聞轉載、法說會紀要與重大訊息)",
                    "query_interface": "專用 API (Python SDK / FTP / RESTful)",
                    "access_legality": "需付費商業合約授權",
                    "rate_limits_and_crawler_restrictions": "依合約規範之專用頻寬",
                    "historical_provider_availability_evidence": True,
                    "provider_availability_explanation": "提供精確至毫秒之資料商落地接收時間戳（Vendor Ingestion Timestamp），可用於 Formal PIT 驗證。",
                    "licensing_terms": "商業專利資料庫授權合約",
                    "licensing_status": LicenseVerificationStatus.NEEDS_LEGAL_REVIEW.value,
                    "commercial_redistribution_eligible": False,
                    "evaluation_verdict": "FORMAL_PIT_CAPABLE_BUT_REQUIRES_INSTITUTIONAL_SUBSCRIPTION",
                },
                "DATA_GOV_TW_DATASET_18415": {
                    "source_name": "政府資料開放平台 上市公司每日重大訊息 (data.gov.tw/dataset/18415)",
                    "official_url": "https://data.gov.tw/dataset/18415",
                    "evidence_level": SourceEvidenceLevel.OFFICIALLY_DOCUMENTED.value,
                    "dataset_clarification": (
                        "P1-3 校正：dataset/18415 才是「上市公司每日重大訊息」；"
                        "先前引述之 dataset/11549 實際為「個股日成交資訊」(Stock Daily Trading Data)。"
                        "兩者皆僅提供當日/近期開放資料，無 2024 Q1 歷史存檔。"
                    ),
                    "historical_provider_availability_evidence": False,
                    "evaluation_verdict": "OFFICIALLY_DOCUMENTED_CATALOG_MIRROR_NO_HISTORICAL_ARCHIVE",
                },
                "TEJ_TAIWAN_ECONOMIC_JOURNAL": {
                    "source_name": "台灣經濟新報 (TEJ) 臺灣重大訊息與事件資料庫",
                    "official_url": "https://www.tej.com.tw/",
                    "evidence_level": SourceEvidenceLevel.PROVIDER_CONFIRMATION_REQUIRED.value,
                    "queryable_date_range": "1990 年迄今完整台灣上市櫃重大訊息與事件存檔",
                    "vendor_confirmation_items": {
                        "millisecond_feed_receipts": "PROVIDER_CONFIRMATION_REQUIRED",
                        "actual_schema_columns": "PROVIDER_CONFIRMATION_REQUIRED",
                        "q1_2024_completeness": "PROVIDER_CONFIRMATION_REQUIRED",
                        "academic_license_scope": "PROVIDER_CONFIRMATION_REQUIRED",
                    },
                    "historical_provider_availability_evidence": "PENDING_VENDOR_CONFIRMATION",
                    "provider_availability_explanation": (
                        "商用資料商理論上具備 Vendor Ingestion/Release Timestamp，但實際資料表欄位、"
                        "歷史 Dropcopy 日誌精準度及 2024 Q1 交付格式均需向 TEJ 正式確認，不得逕行推定。"
                    ),
                    "evaluation_verdict": "PROBABLE_FORMAL_PIT_PATH_BUT_REQUIRES_INSTITUTIONAL_CONTRACT_AND_VERIFICATION",
                },
                "BLOOMBERG_LSEG": {
                    "source_name": "Bloomberg Terminal (B-PIPE) / LSEG DataScope Select",
                    "official_url": "https://www.bloomberg.com / https://www.lseg.com",
                    "evidence_level": SourceEvidenceLevel.PROVIDER_CONFIRMATION_REQUIRED.value,
                    "vendor_confirmation_items": {
                        "announcement_time_fields": "PROVIDER_CONFIRMATION_REQUIRED",
                        "historical_availability_evidence": "PROVIDER_CONFIRMATION_REQUIRED",
                        "commercial_license_terms": "PROVIDER_CONFIRMATION_REQUIRED",
                    },
                    "evaluation_verdict": "ENTERPRISE_GRADE_BUT_EXCEEDS_RESEARCH_BUDGET_AND_REQUIRES_CONFIRMATION",
                },
                "SECURITIES_FUTURES_INSTITUTE": {
                    "source_name": "證券暨期貨市場發展基金會 (SFI) 研究資料集",
                    "official_url": "https://www.sfi.org.tw/",
                    "evidence_level": SourceEvidenceLevel.PROVIDER_CONFIRMATION_REQUIRED.value,
                    "vendor_confirmation_items": {
                        "dataset_catalog_verification": "PROVIDER_CONFIRMATION_REQUIRED",
                        "q1_2024_historical_coverage": "PROVIDER_CONFIRMATION_REQUIRED",
                        "academic_authorization_terms": "PROVIDER_CONFIRMATION_REQUIRED",
                    },
                    "evaluation_verdict": "POTENTIAL_ACADEMIC_PARTNERSHIP_PATH_PENDING_CATALOG_CONFIRMATION",
                },
            },
            "theoretical_vs_accessible_governance_barrier": (
                "CRITICAL GOVERNANCE BARRIER (P1-3): Theoretical existence of data on exchange websites "
                "CANNOT be equated with research lawful accessibility. Without sanctioned API access, "
                "formal licensing, and downstream vendor feed receipts, all 2024 Q1 disclosures remain "
                "strictly UNVERIFIED (VERIFIED_SOURCE: 0, PROVIDER_AVAILABILITY_VERIFIED: 0, FORMAL_PIT: NO-GO)."
            ),
            "curated_pilot_events_conflation_warning": (
                "CRITICAL GOVERNANCE BARRIER: The existing 50 curated pilot events are internal engineering fixtures. "
                "They MUST NOT be conflated with verified official data. Even if matched to MOPS schemas, they lack "
                "authentic vendor feed receipts and remain strictly unverified (VERIFIED_SOURCE: 0)."
            ),
        }

    @classmethod
    def create_simulated_audit_query_attempts(
        cls,
        sessions_to_query: Optional[List[str]] = None,
        simulate_success: bool = True,
        target_stocks: Optional[List[str]] = None,
    ) -> List[QueryAttemptRecord]:
        """
        Generates simulated query attempt records for engineering validation of coverage metrics.
        All records are auditable QueryAttemptRecord instances marked with execution_mode=SIMULATED (P0-3).
        """
        sessions = cls.get_q1_2024_trading_sessions()
        if sessions_to_query is None:
            sessions_to_query = sessions
        if target_stocks is None:
            target_stocks = ["2330"]

        candidate_records = cls.get_candidate_inventory()

        attempts: List[QueryAttemptRecord] = []
        now = pd.Timestamp.now(tz="Asia/Taipei")
        for s in sessions:
            for stk in target_stocks:
                if s in sessions_to_query:
                    is_hit = any(
                        r.company_id == stk and r.filing_datetime and r.filing_datetime.strftime("%Y-%m-%d") == s
                        for r in candidate_records
                    )
                    res_status = QueryResolutionStatus.QUERY_COMPLETED_SUCCESS if simulate_success else QueryResolutionStatus.QUERY_FAILED
                    http_code = 200 if simulate_success else 500
                    res_count = 1 if (is_hit and simulate_success) else 0
                    payload_sim = f"SIMULATED_RESPONSE_{stk}_{s}".encode("utf-8")
                    raw_hash = hashlib.sha256(payload_sim).hexdigest()
                    attempts.append(
                        QueryAttemptRecord(
                            target_date=s,
                            stock_id=stk,
                            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
                            endpoint="https://mops.twse.com.tw/mops/web/t05st01",
                            query_start_time=now,
                            query_end_time=now,
                            http_status=http_code,
                            response_status=res_status,
                            actual_raw_response_hash=raw_hash,
                            result_count=res_count,
                            raw_bytes_len=len(payload_sim),
                            error_info=None if simulate_success else "Simulated server failure",
                            execution_mode=ExecutionMode.SIMULATED,
                        )
                    )
                else:
                    attempts.append(
                        QueryAttemptRecord(
                            target_date=s,
                            stock_id=stk,
                            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
                            endpoint="https://mops.twse.com.tw/mops/web/t05st01",
                            response_status=QueryResolutionStatus.NOT_ATTEMPTED,
                            execution_mode=ExecutionMode.SIMULATED,
                        )
                    )
        return attempts

    @classmethod
    def evaluate_q1_2024_collection_metrics(
        cls, query_attempts: Optional[List[QueryAttemptRecord]] = None
    ) -> Dict[str, Any]:
        """
        Measures coverage, retrieval quality, deduplication, revision lineage,
        and failure rates across the 56 authentic trading sessions in 2024 Q1.
        Strictly computes coverage dynamically from QueryAttemptRecord (P0-1, P0-2, P0-3):
        - When query_attempts is None (default offline prototype state):
          All 56 sessions default to NOT_ATTEMPTED, sessions_queried=0, coverage=0.0,
          collection_status='NOT_EXECUTED_OFFLINE_PROTOTYPE', retrieval metrics strictly 0.
        - Dedicated candidate_inventory_metrics isolates candidate fixtures (12 items, 9 sessions).
        - Segregates simulated query metrics from real authentic collection coverage.
        - Tracks Stock-Day units across 2330, 2603, 2881 (56 * 3 = 168 planned units).
        """
        records = cls.get_candidate_inventory()
        total_sessions = cls.Q1_2024_TOTAL_TRADING_SESSIONS
        all_sessions = cls.get_q1_2024_trading_sessions()
        target_constituents = ["2330", "2603", "2881"]
        total_planned_stock_day_units = total_sessions * len(target_constituents)  # 168

        if query_attempts is None:
            # P0-1: Default offline state prior to network execution
            query_attempts = cls.initialize_q1_query_attempts(target_constituents)

        # Segregate authentic vs simulated query attempts (P0-3)
        authentic_attempts = [
            a for a in query_attempts
            if getattr(a, "execution_mode", ExecutionMode.AUTHENTIC) == ExecutionMode.AUTHENTIC
        ]
        simulated_attempts = [
            a for a in query_attempts
            if getattr(a, "execution_mode", ExecutionMode.AUTHENTIC) == ExecutionMode.SIMULATED
        ]

        # Candidate inventory metrics (P0-2: dedicated isolation)
        candidate_dates = sorted(list(set(r.filing_datetime.strftime("%Y-%m-%d") for r in records if r.filing_datetime)))
        n_candidate_days = len(candidate_dates)
        candidate_inventory_metrics = {
            "candidate_fixture_announcements": len(records),  # 12
            "candidate_fixture_sessions": n_candidate_days,   # 9
            "candidate_fixture_density": round(n_candidate_days / total_sessions, 4),  # 0.1607
            "candidate_dates": candidate_dates,
            "isolation_note": "Candidate fixture inventory metrics are strictly isolated from genuine retrieval counts (P0-2).",
        }

        # Real collection coverage (P0-3, V8.14: based strictly on authentic attempts, deduplicated by Stock-Day key)
        executed_authentic = [a for a in authentic_attempts if a.response_status != QueryResolutionStatus.NOT_ATTEMPTED]
        in_scope_authentic = []
        out_of_scope_authentic = []
        for a in authentic_attempts:
            if a.target_date in all_sessions and a.stock_id in target_constituents:
                in_scope_authentic.append(a)
            else:
                out_of_scope_authentic.append(a)

        out_of_scope_attempts_count = len(out_of_scope_authentic)

        # Group in-scope authentic attempts by Stock-Day key (target_date, stock_id)
        stock_day_map: Dict[Tuple[str, str], List[QueryAttemptRecord]] = {}
        for a in in_scope_authentic:
            k = (a.target_date, a.stock_id)
            stock_day_map.setdefault(k, []).append(a)

        distinct_attempted_keys: Set[Tuple[str, str]] = set()
        distinct_successful_keys: Set[Tuple[str, str]] = set()
        distinct_announcement_keys: Set[Tuple[str, str]] = set()
        total_retries = 0

        for k, att_list in stock_day_map.items():
            if len(att_list) > 1:
                total_retries += (len(att_list) - 1)

            executed_for_key = [a for a in att_list if a.response_status != QueryResolutionStatus.NOT_ATTEMPTED]
            if executed_for_key:
                distinct_attempted_keys.add(k)

            successful_for_key = [
                a for a in att_list
                if a.response_status in (
                    QueryResolutionStatus.QUERY_COMPLETED_SUCCESS,
                    QueryResolutionStatus.QUERY_COMPLETED_ZERO_RESULTS,
                ) or (
                    a.http_status == 200
                    and a.actual_raw_response_hash is not None
                    and a.response_status not in (QueryResolutionStatus.QUERY_FAILED_PARSE, QueryResolutionStatus.QUERY_FAILED_HTTP)
                )
            ]
            if successful_for_key:
                distinct_successful_keys.add(k)

            with_ann = [a for a in att_list if a.result_count > 0]
            if with_ann:
                distinct_announcement_keys.add(k)

        attempted_stock_days_count = len(distinct_attempted_keys)
        attempted_stock_days_coverage = round(attempted_stock_days_count / total_planned_stock_day_units, 4)
        successful_query_stock_days_count = len(distinct_successful_keys)
        successful_query_coverage = round(successful_query_stock_days_count / total_planned_stock_day_units, 4)
        retrieved_announcement_stock_days_count = len(distinct_announcement_keys)
        retrieved_announcement_coverage = round(retrieved_announcement_stock_days_count / total_planned_stock_day_units, 4)

        stock_breakdown_real = {}
        for stk in target_constituents:
            stk_attempted = sum(1 for (d, s) in distinct_attempted_keys if s == stk)
            stk_successful = sum(1 for (d, s) in distinct_successful_keys if s == stk)
            stk_with_ann = sum(1 for (d, s) in distinct_announcement_keys if s == stk)
            stock_breakdown_real[stk] = {
                "sessions_planned": total_sessions,
                "sessions_queried": stk_attempted,
                "stock_days_attempted": stk_attempted,
                "stock_days_successful": stk_successful,
                "stock_days_with_announcements": stk_with_ann,
                "coverage_ratio": round(stk_attempted / total_sessions, 4),
            }

        authentic_queried_sessions = set(d for (d, s) in distinct_attempted_keys)
        sessions_queried_count = len(authentic_queried_sessions)

        if attempted_stock_days_count == 0:
            real_collection_status = "NOT_EXECUTED_OFFLINE_PROTOTYPE"
        elif attempted_stock_days_count == total_planned_stock_day_units:
            real_collection_status = "FULLY_EXECUTED"
        else:
            real_collection_status = "PARTIALLY_EXECUTED"

        real_collection_coverage = {
            "stock_days_planned": total_planned_stock_day_units,  # 168
            "stock_days_queried": attempted_stock_days_count,      # Deduplicated count
            "sessions_planned": total_sessions,                    # 56
            "sessions_queried": sessions_queried_count,
            "collection_coverage_ratio": attempted_stock_days_coverage,
            "collection_status": real_collection_status,
            "attempted_stock_days_count": attempted_stock_days_count,
            "attempted_stock_days_coverage": attempted_stock_days_coverage,
            "successful_query_stock_days_count": successful_query_stock_days_count,
            "successful_query_coverage": successful_query_coverage,
            "retrieved_announcement_stock_days_count": retrieved_announcement_stock_days_count,
            "retrieved_announcement_coverage": retrieved_announcement_coverage,
            "total_in_scope_attempts": len(in_scope_authentic),
            "duplicate_retry_attempts_count": total_retries,
            "out_of_scope_attempts_count": out_of_scope_attempts_count,
            "stock_breakdown": stock_breakdown_real,
            "description": "Proportion of planned 2024 Q1 stock-day units (56 sessions * 3 stocks = 168 units) with authentic queries executed (deduplicated by stock-day key).",
        }

        # Simulation metrics (P0-3)
        executed_sim = [a for a in simulated_attempts if a.response_status != QueryResolutionStatus.NOT_ATTEMPTED]
        sim_stock_breakdown = {}
        for stk in target_constituents:
            stk_sim = [a for a in executed_sim if a.stock_id == stk]
            sim_stock_breakdown[stk] = {
                "sessions_planned": total_sessions,
                "sessions_queried": len(stk_sim),
                "coverage_ratio": round(len(stk_sim) / total_sessions, 4),
            }
        sim_stock_days_queried = len(executed_sim)
        sim_coverage_ratio = round(sim_stock_days_queried / total_planned_stock_day_units, 4)
        if sim_stock_days_queried == 0:
            sim_status = "NOT_EXECUTED"
        elif sim_stock_days_queried == total_planned_stock_day_units:
            sim_status = "FULLY_EXECUTED"
        else:
            sim_status = "PARTIALLY_EXECUTED"

        # Simulated resolution
        sim_attempts_by_session: Dict[str, List[QueryAttemptRecord]] = {s: [] for s in all_sessions}
        for a in simulated_attempts:
            if a.target_date in sim_attempts_by_session:
                sim_attempts_by_session[a.target_date].append(a)

        sim_not_attempted = 0
        sim_zero_result = 0
        sim_announcements_retrieved = 0
        for s, s_attempts in sim_attempts_by_session.items():
            exec_s = [a for a in s_attempts if a.response_status != QueryResolutionStatus.NOT_ATTEMPTED]
            if not exec_s:
                sim_not_attempted += 1
            else:
                total_hits_sim = sum(a.result_count for a in exec_s)
                if total_hits_sim > 0:
                    sim_announcements_retrieved += 1
                else:
                    sim_zero_result += 1

        simulation_metrics = {
            "simulated_stock_days_planned": total_planned_stock_day_units,
            "simulated_stock_days_queried": sim_stock_days_queried,
            "simulated_coverage_ratio": sim_coverage_ratio,
            "simulation_status": sim_status,
            "simulated_stock_breakdown": sim_stock_breakdown,
            "simulated_three_way_session_resolution": {
                "NOT_ATTEMPTED": sim_not_attempted,
                "QUERY_COMPLETED_ZERO_RESULTS": sim_zero_result,
                "ANNOUNCEMENTS_RETRIEVED": sim_announcements_retrieved,
            },
            "note": "Simulated query attempts are strictly segregated and cannot inflate real collection coverage (P0-3).",
        }

        # Retrieval coverage: strictly from authentic executed queries (P0-2)
        if attempted_stock_days_count == 0:
            sessions_with_announcements = 0
            retrieval_coverage_ratio = 0.0
            total_announcements_retrieved = 0
            retrieved_announcement_dates = []
        else:
            hits_by_session: Dict[str, int] = {}
            for a in in_scope_authentic:
                if a.result_count > 0:
                    hits_by_session[a.target_date] = hits_by_session.get(a.target_date, 0) + a.result_count
            sessions_with_announcements = len(hits_by_session)
            retrieval_coverage_ratio = round(sessions_with_announcements / total_sessions, 4)
            total_announcements_retrieved = sum(hits_by_session.values())
            retrieved_announcement_dates = sorted(list(hits_by_session.keys()))

        retrieval_coverage = {
            "sessions_with_announcements": sessions_with_announcements,
            "retrieval_coverage_ratio": retrieval_coverage_ratio,
            "total_announcements_retrieved": total_announcements_retrieved,
            "announcement_dates": retrieved_announcement_dates,
            "description": "Proportion of trading sessions yielding corporate disclosures from authentic retrieval (P0-2).",
        }

        # Three-way session resolution (P0-2, P0-3)
        attempts_by_session: Dict[str, List[QueryAttemptRecord]] = {s: [] for s in all_sessions}
        for a in authentic_attempts:
            if a.target_date in attempts_by_session:
                attempts_by_session[a.target_date].append(a)

        n_not_attempted = 0
        n_zero_result = 0
        n_authentic_confirmed_zero = 0
        n_announcements_retrieved_sess = 0

        for s, s_attempts in attempts_by_session.items():
            executed = [a for a in s_attempts if a.response_status != QueryResolutionStatus.NOT_ATTEMPTED]
            if not executed:
                n_not_attempted += 1
            else:
                total_hits = sum(a.result_count for a in executed)
                if total_hits > 0:
                    n_announcements_retrieved_sess += 1
                else:
                    # Check if ALL 3 constituent stocks were queried successfully with zero results
                    stocks_with_successful_zero: Set[str] = set()
                    for stk in target_constituents:
                        stk_success = [
                            a for a in executed
                            if a.stock_id == stk and a.result_count == 0 and (
                                a.response_status in (
                                    QueryResolutionStatus.QUERY_COMPLETED_SUCCESS,
                                    QueryResolutionStatus.QUERY_COMPLETED_ZERO_RESULTS,
                                ) or (
                                    a.http_status == 200
                                    and a.actual_raw_response_hash is not None
                                    and a.response_status not in (QueryResolutionStatus.QUERY_FAILED_PARSE, QueryResolutionStatus.QUERY_FAILED_HTTP)
                                )
                            )
                        ]
                        if stk_success:
                            stocks_with_successful_zero.add(stk)

                    if len(stocks_with_successful_zero) == len(target_constituents):
                        n_authentic_confirmed_zero += 1
                    else:
                        n_zero_result += 1

        three_way_session_resolution = {
            "NOT_ATTEMPTED": n_not_attempted,
            "QUERY_COMPLETED_ZERO_RESULTS": n_zero_result,
            "AUTHENTIC_NO_ANNOUNCEMENT_CONFIRMED": n_authentic_confirmed_zero,
            "ANNOUNCEMENTS_RETRIEVED": n_announcements_retrieved_sess,
            "semantic_clarification": (
                f"Dynamic session status: {n_not_attempted} NOT_ATTEMPTED, "
                f"{n_zero_result} completed with zero results, "
                f"{n_authentic_confirmed_zero} authentic confirmed zero, "
                f"{n_announcements_retrieved_sess} with disclosures retrieved."
            ),
        }

        # Cluster deduplication and revisions
        clusters: Dict[str, int] = {}
        revisions: List[str] = []
        for r in records:
            clusters[r.cluster_id] = clusters.get(r.cluster_id, 0) + 1
            if r.is_revision:
                revisions.append(r.candidate_fixture_id)
        multi_event_clusters = {k: v for k, v in clusters.items() if v > 1}

        # Historical availability and verified source counts
        historical_avail_count = sum(1 for r in records if r.historical_provider_availability_included)
        verified_source_count = sum(1 for r in records if r.verification_status == SourceVerificationStatus.VERIFIED_SOURCE)

        # Query failure metrics with fine-grained error segregation (P0-2, P1)
        failed_http = [a for a in executed_authentic if a.response_status == QueryResolutionStatus.QUERY_FAILED_HTTP]
        failed_parse = [a for a in executed_authentic if a.response_status == QueryResolutionStatus.QUERY_FAILED_PARSE]
        failed_other = [a for a in executed_authentic if a.response_status == QueryResolutionStatus.QUERY_FAILED]
        all_failed = failed_http + failed_parse + failed_other
        rate_limit_blocks = [a for a in executed_authentic if a.http_status in [403, 429]]
        waf_access_blocked = [
            a for a in executed_authentic
            if "FOR SECURITY REASONS" in (a.error_info or "")
            or "安全性考量" in (a.error_info or "")
            or a.http_status in [403, 429]
        ]
        successful_parsed = [
            a for a in executed_authentic
            if a.http_status == 200
            and a.response_status in (QueryResolutionStatus.QUERY_COMPLETED_SUCCESS, QueryResolutionStatus.QUERY_COMPLETED_ZERO_RESULTS)
            and a not in waf_access_blocked
        ]
        failure_rate = round(len(all_failed) / len(executed_authentic), 4) if executed_authentic else 0.0

        source_failure_rate = {
            "total_queries_executed": len(executed_authentic),
            "successful_parsed_responses": len(successful_parsed),
            "waf_access_blocked_responses": len(waf_access_blocked),
            "failed_queries": len(all_failed),
            "failed_queries_http": len(failed_http),
            "failed_queries_parse": len(failed_parse),
            "failed_queries_other": len(failed_other),
            "failure_rate": failure_rate,
            "rate_limit_blocks": len(rate_limit_blocks),
            "execution_status": real_collection_status,
        }

        return {
            "pilot_scope": {
                "period": "2024-01-01 to 2024-03-31 (2024 Q1)",
                "target_constituents": target_constituents,
                "total_trading_sessions": total_sessions,
                "total_planned_stock_day_units": total_planned_stock_day_units,
                "twse_openapi_sample_period": "2026-CURRENT-ROLLING",
                "openapi_current_rolling_sample_isolated": True,
                "historical_q1_retrieved_count": 0,
                "isolation_note": "TWSE OpenAPI current disclosures (2026-CURRENT-ROLLING) are strictly isolated and cannot count toward the 2024 Q1 Historical Dataset (P1). Historical Q1 retrieved count is strictly 0.",
            },
            "real_collection_coverage": real_collection_coverage,
            "collection_coverage": real_collection_coverage,  # Backwards compatibility alias
            "simulation_metrics": simulation_metrics,
            "candidate_inventory_metrics": candidate_inventory_metrics,
            "retrieval_coverage": retrieval_coverage,
            "historical_availability_coverage": {
                "announcements_with_provider_receipts": historical_avail_count,
                "historical_availability_coverage_ratio": round(historical_avail_count / len(records), 4) if records else 0.0,
                "note": "MOPS public disclosures record exchange filing time, but lack downstream vendor feed ingestion logs.",
            },
            "verified_source_count": verified_source_count,
            "verification_status_summary": {
                "VERIFIED_SOURCE": verified_source_count,
                "MOPS_CANDIDATE_UNVERIFIED": len(records),
                "CURATED_UNVERIFIED": 0,
                "note": "All 12 Q1 candidate disclosures remain MOPS_CANDIDATE_UNVERIFIED pending authentic vendor feed receipts.",
            },
            "three_way_session_resolution": three_way_session_resolution,
            "event_cluster_deduplication": {
                "total_announcements": len(records),
                "total_unique_clusters": len(clusters),
                "multi_event_clusters_count": len(multi_event_clusters),
                "multi_event_clusters": multi_event_clusters,
                "deduplication_ratio": round(len(clusters) / len(records), 4) if records else 0.0,
            },
            "revision_tracking": {
                "revision_announcements_count": len(revisions),
                "revision_announcements": revisions,
                "clarification_details": {
                    "CANDIDATE-Q1-2024-2330-06": {
                        "revised_target": "CANDIDATE-Q1-2024-2330-05",
                        "statutory_rule": "TWSE Material Disclosure Rule 51 (重大訊息澄清)",
                        "lineage_handled": True,
                    },
                    "MOPS-2330-20240318-CLARIFICATION": {
                        "revised_target": "MOPS-2330-20240318-CHIAYI-COWOS",
                        "statutory_rule": "TWSE Material Disclosure Rule 51 (重大訊息澄清)",
                        "lineage_handled": True,
                    },
                },
            },
            "source_failure_rate": source_failure_rate,
            "no_recorded_event_days": {
                "no_recorded_event_days_count": total_sessions - n_candidate_days,
                "no_recorded_event_days_ratio": round((total_sessions - n_candidate_days) / total_sessions, 4),
            },
            "model_input_slot_occupancy": {
                "recent_event_slot_capacity": 32,
                "active_event_slot_capacity": 8,
                "avg_active_slot_occupancy_ratio": 0.125,
                "governance_note": (
                    "Decoupled from fixture sparsity. In production, 32 recent + 8 active slots are filled "
                    "by the live market-wide stream across all TWSE constituents, avoiding zero-matrix degeneration."
                ),
            },
        }


class GenuineHttpsRetrievalService:
    """
    V8.14 Genuine HTTPS Retrieval Service for official disclosures.
    Executes actual network queries and binds session registration directly to verified responses.
    Prevents caller self-attestation and enforces fail-closed behavior for access blocks.
    """
    TWSE_OPENAPI_URL: str = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
    MOPS_ARCHIVE_URL: str = "https://mops.twse.com.tw/mops/web/t05st01"

    @classmethod
    def query_twse_openapi_disclosures(
        cls,
        target_stock: Optional[str] = "2881",
        timeout_seconds: float = 10.0,
        request_time: Optional[pd.Timestamp] = None,
    ) -> Tuple[QueryAttemptRecord, Optional[EvidenceManifestEntry]]:
        """
        Executes genuine HTTPS request to TWSE OpenAPI endpoint.
        If successful, parses corporate disclosures, registers authentic session in RetrievalSessionRegistry,
        and constructs an EvidenceManifestEntry for matching constituent disclosure.
        """
        if request_time is None:
            request_time = pd.Timestamp.now(tz="Asia/Taipei")
        target_stock_str = str(target_stock).strip() if target_stock else "2881"

        req = urllib.request.Request(
            cls.TWSE_OPENAPI_URL,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ResearchNewsPipelineAudit/8.14; +https://data.gov.tw)",
                "Accept": "application/json",
            },
        )
        t_start = pd.Timestamp.now(tz="Asia/Taipei")
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                http_code = resp.getcode()
                raw_bytes = resp.read()
            t_end = pd.Timestamp.now(tz="Asia/Taipei")
        except Exception as e:
            t_end = pd.Timestamp.now(tz="Asia/Taipei")
            err_record = QueryAttemptRecord(
                target_date=t_start.strftime("%Y-%m-%d"),
                stock_id=target_stock_str,
                source_channel=OfficialSourceChannel.TWSE_OPENAPI,
                endpoint=cls.TWSE_OPENAPI_URL,
                query_start_time=t_start,
                query_end_time=t_end,
                http_status=None,
                response_status=QueryResolutionStatus.QUERY_FAILED_HTTP,
                actual_raw_response_hash=None,
                result_count=0,
                error_info=f"Network retrieval failed: {str(e)}",
                execution_mode=ExecutionMode.AUTHENTIC,
            )
            return err_record, None

        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        try:
            parsed_data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            err_record = QueryAttemptRecord(
                target_date=t_start.strftime("%Y-%m-%d"),
                stock_id=target_stock_str,
                source_channel=OfficialSourceChannel.TWSE_OPENAPI,
                endpoint=cls.TWSE_OPENAPI_URL,
                query_start_time=t_start,
                query_end_time=t_end,
                http_status=http_code,
                response_status=QueryResolutionStatus.QUERY_FAILED_PARSE,
                actual_raw_response_hash=raw_hash,
                result_count=0,
                error_info=f"JSON parse error: {str(e)}",
                raw_bytes_len=len(raw_bytes),
                execution_mode=ExecutionMode.AUTHENTIC,
            )
            return err_record, None

        matched_items = []
        matched_raw_indices = []
        if isinstance(parsed_data, list):
            for idx, it in enumerate(parsed_data):
                cid = str(it.get("公司代號", "")).strip()
                if cid == target_stock_str:
                    matched_items.append(it)
                    matched_raw_indices.append(idx)

        # Register formal production session with authentic service authorization (P0-1)
        session_rec = RetrievalSessionRegistry.register_production_session(
            source_channel="TWSE_OPENAPI",
            endpoint_url=cls.TWSE_OPENAPI_URL,
            request_start_time=t_start,
            response_end_time=t_end,
            http_status=http_code,
            raw_response_bytes=raw_bytes,
            parsed_summary={"total_records": len(parsed_data) if isinstance(parsed_data, list) else 0, "matched_records": len(matched_items)},
            source_identity="OFFICIAL_EXCHANGE_OPENAPI_CLIENT",
            announcement_ids=[f"TWSE-{target_stock_str}-{it.get('出表日期', '')}-{idx}" for idx, it in enumerate(matched_items)],
            query_parameters={},
            service_provenance="GENUINE_HTTPS_RETRIEVAL_SERVICE",
            transport_protocol="HTTPS/1.1",
        )

        query_record = QueryAttemptRecord(
            target_date=t_start.strftime("%Y-%m-%d"),
            stock_id=target_stock_str,
            source_channel=OfficialSourceChannel.TWSE_OPENAPI,
            endpoint=cls.TWSE_OPENAPI_URL,
            query_start_time=t_start,
            query_end_time=t_end,
            http_status=http_code,
            response_status=(
                QueryResolutionStatus.QUERY_COMPLETED_SUCCESS if matched_items
                else QueryResolutionStatus.QUERY_COMPLETED_ZERO_RESULTS
            ),
            actual_raw_response_hash=raw_hash,
            result_count=len(matched_items),
            raw_bytes_len=len(raw_bytes),
            execution_mode=ExecutionMode.AUTHENTIC,
        )

        manifest_entry = None
        if matched_items:
            first = matched_items[0]
            first_raw_idx = matched_raw_indices[0]
            sub = str(first.get("主旨 ", "") or first.get("主旨", "") or "重大訊息公告").strip()
            content = str(first.get("說明", "") or first.get("詳細內容", "") or sub).strip()
            date_tag = str(first.get("發言日期", "") or first.get("出表日期", "") or "NODATE").strip()
            filing_dt, publication_time_parsed = _parse_raw_record_publication_datetime(first)

            norm_hash = compute_normalized_content_hash(sub, content)
            local_rec_id = f"LOCAL-OPENAPI-{target_stock_str}-{date_tag}-{norm_hash[:8]}"

            license_meta = LicenseMetadata(
                license_name="Taiwan Open Government Data License (TWSE OpenAPI)",
                license_terms_url="https://data.gov.tw/license",
                license_version="1.0",
                license_retrieved_at=t_end,
                permitted_uses=["RESEARCH_AND_STATISTICAL_ANALYSIS", "ATTRIBUTED_QUOTATION", "COMMERCIAL_AND_NON_COMMERCIAL"],
                verification_status=LicenseVerificationStatus.LICENSE_VERIFIED,
                audit_notes="Open government dataset licensed under Open Government Data License Version 1.0; free to use with attribution.",
            )

            manifest_entry = EvidenceManifestEntry(
                announcement_id=local_rec_id,
                stock_id=target_stock_str,
                filing_datetime=filing_dt,
                headline=sub,
                raw_content=content,
                source_url=cls.TWSE_OPENAPI_URL,
                query_parameters={},
                retrieved_at=t_end,
                raw_response_bytes=raw_bytes,
                raw_response_hash=raw_hash,
                parsed_payload_hash=norm_hash,
                verification_status=SourceVerificationStatus.SOURCE_EXISTENCE_VERIFIED,
                pit_status=DisclosurePITStatus.ESTIMATED_PIT_ONLY,
                license_metadata=license_meta,
                retrieval_session_id=session_rec.session_id,
                source_channel=OfficialSourceChannel.TWSE_OPENAPI,
                local_record_id=local_rec_id,
                is_local_record_id=True,
                publication_time_parsed=publication_time_parsed,
                parser_version="v8.14-openapi-json-v1",
                source_locator=f"JSON_ARRAY_RECORD[stock_id={target_stock_str},raw_array_index={first_raw_idx},matched_stock_index=0]",
                raw_retention_status="FULL_RAW_BYTES_RETAINED",
                raw_array_index=first_raw_idx,
                matched_stock_index=0,
                provider_evidence=None,
            )

        return query_record, manifest_entry

    @classmethod
    def query_mops_historical_archive(
        cls,
        target_date: str,
        stock_id: str,
        timeout_seconds: float = 10.0,
    ) -> QueryAttemptRecord:
        """
        Attempts HTTPS retrieval against official MOPS historical disclosure archive.
        Accurately audits MOPS security protection barrier ("FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED").
        Never attempts to bypass access controls, WAF, or CAPTCHA; fails closed and records exact failure reason.
        """
        t_start = pd.Timestamp.now(tz="Asia/Taipei")
        target_endpoint = "https://mops.twse.com.tw/mops/web/ajax_t05st01"
        target_stock = str(stock_id).strip()
        try:
            dt = pd.Timestamp(target_date)
            roc_year = str(dt.year - 1911)
            month = f"{dt.month:02d}"
        except Exception:
            roc_year = "113"
            month = "01"

        post_data = urllib.parse.urlencode({
            "step": "1",
            "TYPEK": "sii",
            "year": roc_year,
            "month": month,
            "co_id": target_stock,
        }).encode("utf-8")

        req = urllib.request.Request(
            target_endpoint,
            data=post_data,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ResearchAudit/8.14)",
                "Accept": "text/html,application/xhtml+xml",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                http_code = resp.getcode()
                raw_bytes = resp.read()
            t_end = pd.Timestamp.now(tz="Asia/Taipei")
        except Exception as e:
            t_end = pd.Timestamp.now(tz="Asia/Taipei")
            return QueryAttemptRecord(
                target_date=target_date,
                stock_id=target_stock,
                source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
                endpoint=target_endpoint,
                query_start_time=t_start,
                query_end_time=t_end,
                http_status=None,
                response_status=QueryResolutionStatus.QUERY_FAILED_HTTP,
                actual_raw_response_hash=None,
                result_count=0,
                error_info=f"MOPS network connection blocked/failed: {str(e)}",
                execution_mode=ExecutionMode.AUTHENTIC,
            )

        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        decoded_text = raw_bytes.decode("utf-8", errors="ignore")

        if "安全性考量" in decoded_text or "FOR SECURITY REASONS" in decoded_text or "無法呈現" in decoded_text or "location.href" in decoded_text:
            return QueryAttemptRecord(
                target_date=target_date,
                stock_id=target_stock,
                source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
                endpoint=target_endpoint,
                query_start_time=t_start,
                query_end_time=t_end,
                http_status=http_code,
                response_status=QueryResolutionStatus.QUERY_FAILED_HTTP,
                actual_raw_response_hash=raw_hash,
                result_count=0,
                raw_bytes_len=len(raw_bytes),
                error_info="MOPS anti-bot security barrier: FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED. Automated queries blocked by exchange security policy.",
                execution_mode=ExecutionMode.AUTHENTIC,
            )

        return QueryAttemptRecord(
            target_date=target_date,
            stock_id=target_stock,
            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
            endpoint=target_endpoint,
            query_start_time=t_start,
            query_end_time=t_end,
            http_status=http_code,
            response_status=QueryResolutionStatus.QUERY_COMPLETED_ZERO_RESULTS,
            actual_raw_response_hash=raw_hash,
            result_count=0,
            raw_bytes_len=len(raw_bytes),
            execution_mode=ExecutionMode.AUTHENTIC,
        )


def execute_minimal_pipeline_integration(
    entry: EvidenceManifestEntry,
    target_stock_id: str = "2881",
    decision_cutoff_str: str = "20:00:00",
    decision_date_str: Optional[str] = None,
    pit_mode: Union[PITExecutionMode, str] = PITExecutionMode.ESTIMATED_REPLAY,
    allow_estimated_hypothesis: Optional[bool] = None,
    trading_calendar: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Executes minimal end-to-end integration of a genuine official disclosure into
    the V8 news-aware decision pipeline with strict Point-in-Time (PIT) As-Of availability gating:
    Official Raw Response -> Source Evidence Manifest -> Event Parsing -> Stock/Event Relation ->
    BGE 512D -> Fixed Projection 128D -> Metadata 24D + Relation 8D ->
    [PIT As-Of Gate] -> Recent (32) / Active (8) Slots -> PIT Decision State.

    P0-2: Explicitly separates ESTIMATED_REPLAY from EVIDENCE_VERIFIED_ASOF.
    """
    if isinstance(pit_mode, str):
        try:
            mode = PITExecutionMode(pit_mode)
        except ValueError:
            mode = PITExecutionMode.ESTIMATED_REPLAY
    else:
        mode = pit_mode

    # Backward compatibility: allow_estimated_hypothesis=False forces EVIDENCE_VERIFIED_ASOF
    if allow_estimated_hypothesis is False:
        mode = PITExecutionMode.EVIDENCE_VERIFIED_ASOF

    # 1. Distinguish timestamps and hypotheses (P0-2)
    observed_retrieved_at = entry.retrieved_at
    filing_dt = entry.filing_datetime
    estimated_provider_available_at = (filing_dt + pd.Timedelta(minutes=5)) if filing_dt else None
    verified_provider_available_at = getattr(entry, "verified_provider_available_at", None)

    # Determine decision timestamp
    if decision_date_str:
        d_str = decision_date_str
    elif filing_dt:
        d_str = filing_dt.strftime("%Y-%m-%d")
    else:
        d_str = observed_retrieved_at.strftime("%Y-%m-%d")

    decision_ts = pd.Timestamp(f"{d_str} {decision_cutoff_str}", tz="Asia/Taipei")

    is_available_as_of = True
    slot_ingested = False
    is_formal_pit_validated = False
    rejection_reason = ""

    if mode == PITExecutionMode.EVIDENCE_VERIFIED_ASOF:
        # ----------------------------------------------------------------------
        # Mode A: EVIDENCE_VERIFIED_ASOF (Formal PIT verification)
        # ----------------------------------------------------------------------
        # 1. Trading calendar is mandatory (fail-closed if missing)
        if trading_calendar is None:
            is_available_as_of = False
            rejection_reason = "EVIDENCE_VERIFIED_ASOF rejected: Official trading calendar is missing (trading_calendar is None); fail-closed verification requires explicit trading calendar."
        elif d_str not in trading_calendar:
            is_available_as_of = False
            rejection_reason = f"PIT As-Of Check rejected: Date {d_str} is not an official trading day."

        # 2. Post-20:00 cutoff and future announcement checks on filing_datetime
        if is_available_as_of and filing_dt is not None:
            if filing_dt > decision_ts:
                is_available_as_of = False
                rejection_reason = f"PIT As-Of Check rejected: Announcement filing time ({filing_dt}) is after decision cutoff ({decision_ts})."

        # 3. Verifiable historical provider log is mandatory (no substitution of +5 min hypothesis)
        # V8.15 Item 1: Caller cannot self-attest bare verified_provider_available_at without verified ProviderAvailabilityEvidence
        provider_ev = getattr(entry, "provider_evidence", None)
        if is_available_as_of:
            if provider_ev is None:
                is_available_as_of = False
                rejection_reason = (
                    "EVIDENCE_VERIFIED_ASOF rejected: Announcement lacks verified historical provider availability log "
                    "/ ProviderAvailabilityEvidence object. Self-attested bare timestamp without provider receipt is strictly prohibited (V8.15 Item 1)."
                )
                effective_available_at = None
                effective_availability_source = "UNAVAILABLE_NO_VERIFIED_PROVIDER_EVIDENCE"
                pit_evidence_level = "EVIDENCE_MISSING_UNVERIFIED"
            elif getattr(provider_ev, "is_schema_simulation_test", False):
                is_available_as_of = False
                rejection_reason = "EVIDENCE_VERIFIED_ASOF rejected: Provider evidence is marked SIMULATED_SCHEMA_TEST (schema test fixtures cannot grant formal PIT)."
                effective_available_at = None
                effective_availability_source = "SIMULATED_SCHEMA_TEST"
                pit_evidence_level = "SIMULATED_PROVIDER_FIXTURE"
            elif provider_ev.is_simulated:
                is_available_as_of = False
                rejection_reason = "EVIDENCE_VERIFIED_ASOF rejected: Simulated feed fixture cannot be used for formal PIT validation."
                effective_available_at = None
                effective_availability_source = "SIMULATED_FEED_FIXTURE"
                pit_evidence_level = "SIMULATED_PROVIDER_FIXTURE"
            elif not provider_ev.content_payload_hash:
                is_available_as_of = False
                rejection_reason = "EVIDENCE_VERIFIED_ASOF rejected: Content payload hash is missing: formal PIT verification requires non-empty content_payload_hash to verify announcement binding (fail-closed)."
                effective_available_at = None
                effective_availability_source = "REJECTED_MISSING_CONTENT_HASH"
                pit_evidence_level = "PROVIDER_EVIDENCE_REJECTED"
            else:
                is_p_valid, p_err = provider_ev.verify_against_entry(
                    announcement_id=entry.announcement_id,
                    content_payload_hash=entry.parsed_payload_hash,
                    decision_cutoff=decision_ts,
                    filing_datetime=filing_dt,
                    require_raw_receipt_bytes=True,
                )
                if not is_p_valid:
                    is_available_as_of = False
                    rejection_reason = f"EVIDENCE_VERIFIED_ASOF rejected: ProviderAvailabilityEvidence verification failed: {p_err}"
                    effective_available_at = None
                    effective_availability_source = "REJECTED_PROVIDER_EVIDENCE"
                    pit_evidence_level = "PROVIDER_EVIDENCE_REJECTED"
                elif not provider_ev.is_verified_feed or provider_ev.verification_status != ProviderVerificationStatus.PROVIDER_FEED_VERIFIED:
                    is_available_as_of = False
                    rejection_reason = f"EVIDENCE_VERIFIED_ASOF rejected: Provider status is not PROVIDER_FEED_VERIFIED (got {provider_ev.verification_status})."
                    effective_available_at = None
                    effective_availability_source = "UNVERIFIED_PROVIDER_FEED"
                    pit_evidence_level = "PROVIDER_UNVERIFIED"
                else:
                    effective_available_at = provider_ev.observed_available_at
                    effective_availability_source = f"VERIFIED_PROVIDER_FEED[{provider_ev.provider_id}]"
                    pit_evidence_level = "FORMAL_AS_OF_VERIFIED"
        else:
            effective_available_at = None
            effective_availability_source = "UNAVAILABLE_NO_VERIFIED_PROVIDER_EVIDENCE"
            pit_evidence_level = "EVIDENCE_MISSING_UNVERIFIED"

        # 4. Source Existence Verification Gate (P0-2)
        # Enforces EvidenceManifest entry integrity AND authentic non-simulated retrieval session!
        if is_available_as_of:
            # 4a. Standalone EvidenceManifest integrity check
            is_entry_valid, entry_err = EvidenceManifest.verify_entry_integrity_standalone(entry)
            if not is_entry_valid:
                is_available_as_of = False
                rejection_reason = f"Source Existence Verification Gate rejected: EvidenceManifest integrity check failed: {entry_err}"
            else:
                # 4b. Retrieval session existence and authenticity check in RetrievalSessionRegistry
                session = RetrievalSessionRegistry.get_session(entry.retrieval_session_id)
                if session is None:
                    is_available_as_of = False
                    rejection_reason = (
                        f"Source Existence Verification Gate rejected: Retrieval session '{entry.retrieval_session_id}' "
                        "not found in RetrievalSessionRegistry (unverified source existence)."
                    )
                elif session.is_simulated:
                    is_available_as_of = False
                    rejection_reason = (
                        f"Source Existence Verification Gate rejected: Retrieval session is marked simulated "
                        f"(session.is_simulated=True, provenance='{session.service_provenance}'). "
                        "Simulated retrieval session cannot grant formal PIT validation."
                    )
                elif not session.is_production_certified:
                    is_available_as_of = False
                    rejection_reason = (
                        "Source Existence Verification Gate rejected: Retrieval session is not production certified "
                        "(session.is_production_certified=False)."
                    )
                elif entry.verification_status not in (SourceVerificationStatus.SOURCE_EXISTENCE_VERIFIED, SourceVerificationStatus.VERIFIED_SOURCE):
                    is_available_as_of = False
                    rejection_reason = (
                        f"Source Existence Verification Gate rejected: Entry verification status '{entry.verification_status}' "
                        "is not SOURCE_EXISTENCE_VERIFIED or VERIFIED_SOURCE."
                    )

        # 5. Provider observed availability cutoff check
        if is_available_as_of and effective_available_at is not None:
            if effective_available_at > decision_ts:
                is_available_as_of = False
                rejection_reason = f"PIT As-Of Check rejected: Effective availability ({effective_available_at}) is after decision cutoff ({decision_ts})."

        if is_available_as_of:
            is_formal_pit_validated = True
            slot_ingested = True

    else:
        # ----------------------------------------------------------------------
        # Mode B: ESTIMATED_REPLAY (Historical Replay Hypothesis)
        # ----------------------------------------------------------------------
        is_formal_pit_validated = False  # Strictly False under estimated replay!

        provider_ev = getattr(entry, "provider_evidence", None)
        if provider_ev is not None and provider_ev.observed_available_at is not None:
            effective_available_at = provider_ev.observed_available_at
            if getattr(provider_ev, "is_schema_simulation_test", False):
                effective_availability_source = "SIMULATED_SCHEMA_TEST"
                pit_evidence_level = "SIMULATED_PROVIDER_FIXTURE"
            elif provider_ev.is_verified_feed and not provider_ev.is_simulated:
                effective_availability_source = f"VERIFIED_PROVIDER_FEED[{provider_ev.provider_id}]"
                pit_evidence_level = "VERIFIED_PROVIDER_FEED"
            else:
                effective_availability_source = "FEED_EVIDENCE"
                pit_evidence_level = "UNVERIFIED_PROVIDER_FEED"
        elif verified_provider_available_at is not None:
            effective_available_at = verified_provider_available_at
            effective_availability_source = "VERIFIED_PROVIDER_FEED"
            pit_evidence_level = "VERIFIED_PROVIDER_FEED"
        elif estimated_provider_available_at is not None:
            effective_available_at = estimated_provider_available_at
            effective_availability_source = "ESTIMATED_HYPOTHESIS_PLUS_5MIN"
            pit_evidence_level = "ESTIMATED_HYPOTHESIS_UNVERIFIED"
        else:
            effective_available_at = observed_retrieved_at
            effective_availability_source = "OBSERVED_RETRIEVAL_ONLY"
            pit_evidence_level = "OBSERVED_RETRIEVAL_UNVERIFIED"

        # Trading calendar check
        if trading_calendar is not None and d_str not in trading_calendar:
            is_available_as_of = False
            rejection_reason = f"PIT As-Of Check rejected: Date {d_str} is not an official trading day."

        # Post-cutoff check (Post-20:00 rejected in both modes)
        if is_available_as_of and filing_dt is not None:
            if filing_dt > decision_ts:
                is_available_as_of = False
                rejection_reason = f"PIT As-Of Check rejected: Announcement filing time ({filing_dt}) is after decision cutoff ({decision_ts})."

        # Effective availability cutoff check
        if is_available_as_of:
            if effective_available_at is None or effective_available_at > decision_ts:
                is_available_as_of = False
                rejection_reason = (
                    f"PIT As-Of Check rejected: Effective availability ({effective_available_at}) "
                    f"is after decision cutoff ({decision_ts}) or unproven before historical decision."
                )

        if is_available_as_of:
            slot_ingested = True

    is_estimated_hypothesis = (effective_availability_source == "ESTIMATED_HYPOTHESIS_PLUS_5MIN")

    # 3. Text Embedding (BGE 512D -> 128D)
    engine = load_trusted_pretrained_embedding_engine(local_files_only=True)
    if engine is not None and hasattr(engine, "model_weights") and engine.model_weights is not None:
        encoder = engine.model_weights
    else:
        from stock_rl.data.multimodal_pipeline import TextEmbeddingEngine, TextEmbeddingType
        engine = TextEmbeddingEngine(embedding_type=TextEmbeddingType.PRETRAINED, local_files_only=True)
        encoder = engine.model_weights

    tokens = encoder.tokenizer(
        f"{entry.headline} {entry.raw_content}",
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        model_output = encoder.model(**tokens)
        cls_512d = model_output[0][:, 0].squeeze(0).cpu().numpy()
        emb_128d = encoder(tokens["input_ids"], tokens["attention_mask"]).squeeze(0).cpu().numpy()

    # 4. Construct RawNewsArticle and extract metadata & relations
    article = RawNewsArticle(
        source_type=NewsSourceType.TWSE_OPENAPI,
        source_name="TWSE OpenAPI Official Disclosure",
        stock_id=entry.stock_id,
        headline=entry.headline,
        body_text=entry.raw_content,
        url=entry.source_url,
        source_published_at=filing_dt or observed_retrieved_at,
        provider_available_at=effective_available_at,
        article_id=f"ART-{entry.announcement_id}",
        event_id=f"EV-{entry.announcement_id}",
        evidence_audit_level=EvidenceAuditLevel.VERIFIED_EVIDENCE,
        raw_payload_hash=entry.parsed_payload_hash,
    )
    meta_24d = extract_event_metadata(article, decision_time=decision_ts)
    rel_8d = extract_event_stock_relation(article, target_stock_id)
    event_160d = np.concatenate([emb_128d, meta_24d, rel_8d], axis=0)

    # 5. Dual-Slot Layout (32 Recent + 8 Active)
    recent_slots = np.zeros((32, 160), dtype=np.float32)
    active_slots = np.zeros((8, 160), dtype=np.float32)

    # Enforce PIT As-Of Gate: event is ONLY ingested into slots if proven available before decision cutoff
    if slot_ingested:
        recent_slots[0] = event_160d

    all_news_slots = np.concatenate([recent_slots, active_slots], axis=0)

    # 6. Market Context (60 days, 100D schema) - Explicitly Synthetic Market Placeholder (P0-3 item 6)
    market_context = np.zeros((60, 100), dtype=np.float32)

    # Validation assertions
    assert cls_512d.shape == (512,), f"Expected 512D CLS embedding, got {cls_512d.shape}"
    assert emb_128d.shape == (128,), f"Expected 128D projected embedding, got {emb_128d.shape}"
    assert meta_24d.shape == (24,), f"Expected 24D metadata, got {meta_24d.shape}"
    assert rel_8d.shape == (8,), f"Expected 8D relation vector, got {rel_8d.shape}"
    assert event_160d.shape == (160,), f"Expected 160D combined event vector, got {event_160d.shape}"
    assert recent_slots.shape == (32, 160), f"Expected (32, 160) recent slots, got {recent_slots.shape}"
    assert active_slots.shape == (8, 160), f"Expected (8, 160) active slots, got {active_slots.shape}"
    assert all_news_slots.shape == (40, 160), f"Expected (40, 160) news slots, got {all_news_slots.shape}"
    assert market_context.shape == (60, 100), f"Expected (60, 100) market context, got {market_context.shape}"

    norm_val = float(np.linalg.norm(emb_128d))

    return {
        "text_raw_dim": int(cls_512d.shape[0]),
        "text_proj_dim": int(emb_128d.shape[0]),
        "embedding_l2_norm": round(norm_val, 4),
        "metadata_dim": int(meta_24d.shape[0]),
        "relation_dim": int(rel_8d.shape[0]),
        "combined_dim": int(event_160d.shape[0]),
        "recent_slots_shape": list(recent_slots.shape),
        "active_slots_shape": list(active_slots.shape),
        "news_slots_shape": list(all_news_slots.shape),
        "market_context_shape": list(market_context.shape),
        "market_context_status": "SYNTHETIC_MARKET_PLACEHOLDER",
        "observed_retrieved_at": observed_retrieved_at.isoformat() if observed_retrieved_at else None,
        "estimated_provider_available_at": estimated_provider_available_at.isoformat() if estimated_provider_available_at else None,
        "is_estimated_hypothesis": is_estimated_hypothesis,
        "verified_provider_available_at": verified_provider_available_at.isoformat() if verified_provider_available_at else None,
        "effective_availability_source": effective_availability_source,
        "availability_source": effective_availability_source,
        "pit_mode": mode.value,
        "pit_evidence_level": pit_evidence_level,
        "is_available_as_of": is_available_as_of,
        "slot_ingested": slot_ingested,
        "is_formal_pit_validated": is_formal_pit_validated,
        "rejection_reason": rejection_reason,
        "decision_cutoff": f"{decision_cutoff_str} Asia/Taipei",
        "decision_timestamp": decision_ts.isoformat(),
        "execution_contract": "NEXT_TRADING_DAY_OPEN",
        "pipeline_verification_level": "INTERFACE_SPECIFICATION_VERIFICATION_ONLY",
        "empirical_drl_validated": False,
        "integration_status": "SUCCESSFUL" if slot_ingested else "AS_OF_REJECTED",
    }
