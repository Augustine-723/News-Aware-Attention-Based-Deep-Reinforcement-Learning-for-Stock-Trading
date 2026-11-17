"""
V8.14 Comprehensive Evidence-Backed Historical Dataset & Retrieval Boundary Tests.
Validates:
1. Genuine Retrieval Trust Boundary (GenuineHttpsRetrievalService, RetrievalSessionRegistry)
2. Counterexample Rejection (RAW_PILOT_EVENTS fixture bytes cannot register authentic session)
3. Lawful Official Access & MOPS Security Block Audit (fail-closed honest reporting)
4. Stock-Day Coverage Accounting & Deduplication (168 planned units, retry deduplication, out-of-scope quarantine)
5. Three-Way Session Resolution (AUTHENTIC_NO_ANNOUNCEMENT_CONFIRMED requires all 3 stocks)
6. Tripartite PIT Status Segregation (SOURCE_EXISTENCE_VERIFIED vs ESTIMATED_PIT_ONLY vs FORMAL_PIT_ELIGIBLE)
7. Evidence Manifest Reproducibility & Cryptographic Integrity Verification
8. Minimal Pipeline Integration Forward Pass (512D -> 128D -> 24D -> 8D -> (40, 160) -> (60, 100))
"""

import hashlib
import json
import numpy as np
import pandas as pd
import pytest
import torch

from stock_rl.evaluation.importance import (
    NewsSourceType,
    EvidenceAuditLevel,
)
from stock_rl.evaluation.semantic_eval import (
    compute_normalized_content_hash,
)
from stock_rl.evaluation.pilot_dataset import (
    RAW_PILOT_EVENTS,
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
    TrustedSourceCertifier,
    MOPSSourceEvidence,
)
from stock_rl.evaluation.official_source_collector import (
    OfficialSourceChannel,
    ExecutionMode,
    QueryResolutionStatus,
    QueryAttemptRecord,
    OfficialAnnouncementRecord,
    OfficialHistoricalNewsCollector,
    EvidenceManifestEntry,
    EvidenceManifest,
    GenuineHttpsRetrievalService,
    execute_minimal_pipeline_integration,
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


@pytest.fixture(autouse=True)
def clean_session_registry():
    RetrievalSessionRegistry.clear()
    yield
    RetrievalSessionRegistry.clear()


# ==============================================================================
# 1. Genuine Retrieval Trust Boundary Tests
# ==============================================================================

def test_genuine_retrieval_trust_boundary_metadata():
    """
    Test 1: Genuine retrieval session registry requires authentic network metadata,
    stores query parameters, service provenance, session token, and transport protocol.
    """
    raw_bytes = b'{"status": "ok", "announcements": []}'
    t_start = pd.Timestamp("2024-01-18 14:00:00", tz="Asia/Taipei")
    t_end = pd.Timestamp("2024-01-18 14:00:02", tz="Asia/Taipei")

    session = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_OPENAPI",
        endpoint_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        request_start_time=t_start,
        response_end_time=t_end,
        http_status=200,
        raw_response_bytes=raw_bytes,
        parsed_summary={"announcements": 0},
        source_identity="OFFICIAL_EXCHANGE_OPENAPI_CLIENT",
        announcement_ids=[],
        query_parameters={"type": "daily_material"},
        service_provenance="GENUINE_HTTPS_RETRIEVAL_SERVICE",
        transport_protocol="HTTPS/1.1",
    )

    assert session.session_id.startswith("REG-SESS-")
    assert session.http_status == 200
    assert session.service_provenance == "GENUINE_HTTPS_RETRIEVAL_SERVICE"
    assert session.transport_protocol == "HTTPS/1.1"
    assert session.query_parameters == {"type": "daily_material"}
    assert session.raw_response_hash == hashlib.sha256(raw_bytes).hexdigest()
    assert session.raw_bytes_len == len(raw_bytes)

    # RetrievalSessionRegistry query
    fetched = RetrievalSessionRegistry.get_session(session.session_id)
    assert fetched is not None
    assert fetched.session_id == session.session_id


def test_reject_untrusted_caller_provenance():
    """
    Test 1b: If session service_provenance is untrusted or self-attested,
    TrustedSourceCertifier MUST reject verification fail-closed.
    """
    ev_target = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-01"][0]
    ev_dt = pd.Timestamp(f"{ev_target['date_str']} {ev_target['pub_time_str']}", tz="Asia/Taipei")
    raw_content = ev_target.get("body_text", "")
    raw_bytes = raw_content.encode("utf-8")
    ev_hash = compute_normalized_content_hash(ev_target["headline"], raw_content)

    session = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_MOPS_HTML_ARCHIVE",
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=112&month=06&co_id=2330",
        request_start_time=pd.Timestamp.now(tz="Asia/Taipei"),
        response_end_time=pd.Timestamp.now(tz="Asia/Taipei"),
        http_status=200,
        raw_response_bytes=raw_bytes,
        source_identity="UNVERIFIED_CALLER",
        service_provenance="UNTRUSTED_CALLER_INJECTION",
        announcement_ids=["MOPS-2330-20230606-AGM"],
        allow_legacy_fixture=True,
    )

    evidence = TrustedExternalSourceAdapter.certify_with_live_retrieval_evidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=raw_content,
        raw_http_response_bytes=raw_bytes,
        retrieval_session_id=session.session_id,
        headline=ev_target["headline"],
        payload_hash=ev_hash,
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=112&month=06&co_id=2330",
        is_existence_verified=True,
    )

    status, evidence_dict, notes = TrustedSourceCertifier.verify_evidence(ev_target, evidence)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert evidence_dict is None
    assert "service provenance 'UNTRUSTED_CALLER_INJECTION' is untrusted" in notes


# ==============================================================================
# 2. Counterexample Rejection Tests
# ==============================================================================

def test_counterexample_raw_pilot_events_bytes_rejection():
    """
    Test 2 (P0-1 item 7 Counterexample Test):
    Passing RAW_PILOT_EVENTS local bytes directly into register_genuine_session()
    MUST raise ValueError and CANNOT achieve VERIFIED_SOURCE or SOURCE_EXISTENCE_VERIFIED.
    """
    ev_target = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-01"][0]
    raw_fixture_bytes = ev_target.get("body_text", "").encode("utf-8")

    # Direct registration with RAW_PILOT_EVENTS bytes without allow_legacy_fixture MUST fail
    with pytest.raises(ValueError, match="Counterexample rejected"):
        RetrievalSessionRegistry.register_genuine_session(
            source_channel="TWSE_MOPS_HTML_ARCHIVE",
            endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
            request_start_time=pd.Timestamp.now(tz="Asia/Taipei"),
            response_end_time=pd.Timestamp.now(tz="Asia/Taipei"),
            http_status=200,
            raw_response_bytes=raw_fixture_bytes,
            service_provenance="GENUINE_HTTPS_RETRIEVAL_SERVICE",
        )


def test_counterexample_headline_bytes_rejection():
    """
    Test 2b: Passing RAW_PILOT_EVENTS headline text into register_genuine_session()
    is also detected as a known fixture and rejected.
    """
    ev_target = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2603-01"][0]
    headline_bytes = ev_target["headline"].encode("utf-8")

    with pytest.raises(ValueError, match="Counterexample rejected"):
        RetrievalSessionRegistry.register_genuine_session(
            source_channel="TWSE_MOPS_HTML_ARCHIVE",
            endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
            request_start_time=pd.Timestamp.now(tz="Asia/Taipei"),
            response_end_time=pd.Timestamp.now(tz="Asia/Taipei"),
            http_status=200,
            raw_response_bytes=headline_bytes,
        )


# ==============================================================================
# 3. MOPS Security Block Audit & Fail-Closed Reporting Tests
# ==============================================================================

def test_mops_security_block_audit_and_fail_closed():
    """
    Test 3: Attempts HTTPS retrieval against MOPS historical archive.
    MOPS server responds with security barrier ('FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED').
    System must fail closed, record QUERY_FAILED_HTTP, preserve barrier response hash,
    and never attempt to bypass WAF or fabricate disclosure text.
    """
    record = GenuineHttpsRetrievalService.query_mops_historical_archive(
        target_date="2024-03-18",
        stock_id="2330",
        timeout_seconds=5.0,
    )

    assert record.stock_id == "2330"
    assert record.target_date == "2024-03-18"
    assert record.source_channel == OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE
    assert record.execution_mode == ExecutionMode.AUTHENTIC
    assert record.result_count == 0
    # Must be marked failed due to security barrier or network block
    assert record.response_status in [
        QueryResolutionStatus.QUERY_FAILED_HTTP,
        QueryResolutionStatus.QUERY_FAILED,
    ]
    if record.actual_raw_response_hash is not None:
        assert "FOR SECURITY REASONS" in record.error_info or "安全性考量" in record.error_info


def test_mops_failure_does_not_yield_verified_sources():
    """
    Test 3b: When official interface fails to provide historical Q1 announcements,
    verified source count remains strictly 0 (P0-1 item 8).
    """
    mops_record = GenuineHttpsRetrievalService.query_mops_historical_archive("2024-01-18", "2330")
    metrics = OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics([mops_record])

    assert metrics["verified_source_count"] == 0
    assert metrics["real_collection_coverage"]["retrieved_announcement_stock_days_count"] == 0
    assert metrics["real_collection_coverage"]["collection_status"] == "PARTIALLY_EXECUTED"


# ==============================================================================
# 4. Stock-Day Coverage Accounting & Deduplication Tests
# ==============================================================================

def test_stock_day_accounting_planned_population():
    """
    Test 4a: Target universe is 56 sessions * 3 constituents = 168 planned Stock-Days.
    Default offline state has 0 queried, coverage = 0.0.
    """
    metrics = OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics()
    cov = metrics["real_collection_coverage"]
    assert cov["stock_days_planned"] == 168
    assert cov["sessions_planned"] == 56
    assert cov["stock_days_queried"] == 0
    assert cov["collection_coverage_ratio"] == 0.0
    assert cov["attempted_stock_days_coverage"] == 0.0
    assert cov["successful_query_coverage"] == 0.0
    assert cov["retrieved_announcement_coverage"] == 0.0
    assert cov["collection_status"] == "NOT_EXECUTED_OFFLINE_PROTOTYPE"


def test_stock_day_retry_deduplication():
    """
    Test 4b: Retrying the same (target_date, stock_id) multiple times must NOT
    inflate coverage. Duplicate retries are tracked under duplicate_retry_attempts_count.
    """
    t_date = "2024-01-18"
    stk = "2330"
    attempts = [
        # Attempt 1: Failed
        QueryAttemptRecord(
            target_date=t_date,
            stock_id=stk,
            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
            endpoint="https://mops.twse.com.tw/mops/web/t05st01",
            response_status=QueryResolutionStatus.QUERY_FAILED_HTTP,
            execution_mode=ExecutionMode.AUTHENTIC,
            http_status=403,
            error_info="WAF Block",
        ),
        # Attempt 2: Retry, failed
        QueryAttemptRecord(
            target_date=t_date,
            stock_id=stk,
            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
            endpoint="https://mops.twse.com.tw/mops/web/t05st01",
            response_status=QueryResolutionStatus.QUERY_FAILED_HTTP,
            execution_mode=ExecutionMode.AUTHENTIC,
            http_status=403,
            error_info="WAF Block (Retry 1)",
            retry_info="Retry 1/2",
        ),
        # Attempt 3: Retry, completed zero results
        QueryAttemptRecord(
            target_date=t_date,
            stock_id=stk,
            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
            endpoint="https://mops.twse.com.tw/mops/web/t05st01",
            response_status=QueryResolutionStatus.QUERY_COMPLETED_ZERO_RESULTS,
            execution_mode=ExecutionMode.AUTHENTIC,
            http_status=200,
            actual_raw_response_hash="abc123hash",
            result_count=0,
            retry_info="Retry 2/2",
        ),
    ]

    metrics = OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics(attempts)
    cov = metrics["real_collection_coverage"]

    # 3 attempts for 1 stock-day unit
    assert cov["total_in_scope_attempts"] == 3
    assert cov["duplicate_retry_attempts_count"] == 2
    # Deduplicated stock-days queried must be exactly 1, NOT 3!
    assert cov["stock_days_queried"] == 1
    assert cov["attempted_stock_days_count"] == 1
    assert cov["attempted_stock_days_coverage"] == round(1 / 168, 4)
    # Successful query count is 1
    assert cov["successful_query_stock_days_count"] == 1
    assert cov["successful_query_coverage"] == round(1 / 168, 4)
    # Retrieved announcement count is 0
    assert cov["retrieved_announcement_stock_days_count"] == 0
    assert cov["retrieved_announcement_coverage"] == 0.0


def test_out_of_scope_query_isolation():
    """
    Test 4c: Queries for dates outside 2024 Q1 or stocks outside {2330, 2603, 2881}
    must be quarantined under out_of_scope_attempts_count and NOT inflate in-scope coverage.
    """
    attempts = [
        # In-scope: 2024-01-18 for 2330
        QueryAttemptRecord(
            target_date="2024-01-18",
            stock_id="2330",
            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
            response_status=QueryResolutionStatus.QUERY_COMPLETED_ZERO_RESULTS,
            execution_mode=ExecutionMode.AUTHENTIC,
            http_status=200,
            actual_raw_response_hash="hash1",
        ),
        # Out-of-scope date: 2024-04-02 (2024 Q2)
        QueryAttemptRecord(
            target_date="2024-04-02",
            stock_id="2330",
            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
            response_status=QueryResolutionStatus.QUERY_COMPLETED_SUCCESS,
            execution_mode=ExecutionMode.AUTHENTIC,
            http_status=200,
            actual_raw_response_hash="hash2",
            result_count=1,
        ),
        # Out-of-scope stock: 2317 (Hon Hai, not in target universe)
        QueryAttemptRecord(
            target_date="2024-01-18",
            stock_id="2317",
            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
            response_status=QueryResolutionStatus.QUERY_COMPLETED_SUCCESS,
            execution_mode=ExecutionMode.AUTHENTIC,
            http_status=200,
            actual_raw_response_hash="hash3",
            result_count=1,
        ),
    ]

    metrics = OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics(attempts)
    cov = metrics["real_collection_coverage"]

    assert cov["out_of_scope_attempts_count"] == 2
    assert cov["total_in_scope_attempts"] == 1
    assert cov["stock_days_queried"] == 1
    assert cov["attempted_stock_days_count"] == 1
    assert cov["successful_query_stock_days_count"] == 1
    # The out-of-scope announcement for 2317 or Q2 must NOT count towards in-scope retrieved announcements
    assert cov["retrieved_announcement_stock_days_count"] == 0


def test_single_stock_partial_coverage():
    """
    Test 4d: If only 2330 is queried across all 56 sessions,
    in-scope stock-days queried = 56 / 168 = 33.33%, status = PARTIALLY_EXECUTED.
    """
    all_sessions = OfficialHistoricalNewsCollector.get_q1_2024_trading_sessions()
    attempts = [
        QueryAttemptRecord(
            target_date=d,
            stock_id="2330",
            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
            response_status=QueryResolutionStatus.QUERY_COMPLETED_ZERO_RESULTS,
            execution_mode=ExecutionMode.AUTHENTIC,
            http_status=200,
            actual_raw_response_hash=f"hash_{d}",
        )
        for d in all_sessions
    ]

    metrics = OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics(attempts)
    cov = metrics["real_collection_coverage"]

    assert cov["stock_days_queried"] == 56
    assert cov["attempted_stock_days_count"] == 56
    assert cov["attempted_stock_days_coverage"] == 0.3333
    assert cov["collection_status"] == "PARTIALLY_EXECUTED"
    assert cov["stock_breakdown"]["2330"]["coverage_ratio"] == 1.0
    assert cov["stock_breakdown"]["2603"]["coverage_ratio"] == 0.0
    assert cov["stock_breakdown"]["2881"]["coverage_ratio"] == 0.0


# ==============================================================================
# 5. Three-Way Session Resolution Tests
# ==============================================================================

def test_confirmed_zero_requires_all_three_stocks():
    """
    Test 5: AUTHENTIC_NO_ANNOUNCEMENT_CONFIRMED requires that ALL 3 target stocks
    were queried successfully with 0 results on that session date.
    If only 1 or 2 stocks were queried with 0 results, it remains QUERY_COMPLETED_ZERO_RESULTS.
    """
    # Case A: Only 2330 and 2603 queried with 0 results on 2024-01-02 (2881 missing)
    partial_session_attempts = [
        QueryAttemptRecord(
            target_date="2024-01-02",
            stock_id="2330",
            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
            response_status=QueryResolutionStatus.QUERY_COMPLETED_ZERO_RESULTS,
            execution_mode=ExecutionMode.AUTHENTIC,
            http_status=200,
            actual_raw_response_hash="hash_2330",
            result_count=0,
        ),
        QueryAttemptRecord(
            target_date="2024-01-02",
            stock_id="2603",
            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
            response_status=QueryResolutionStatus.QUERY_COMPLETED_ZERO_RESULTS,
            execution_mode=ExecutionMode.AUTHENTIC,
            http_status=200,
            actual_raw_response_hash="hash_2603",
            result_count=0,
        ),
    ]

    metrics_a = OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics(partial_session_attempts)
    three_way_a = metrics_a["three_way_session_resolution"]
    assert three_way_a["AUTHENTIC_NO_ANNOUNCEMENT_CONFIRMED"] == 0
    assert three_way_a["QUERY_COMPLETED_ZERO_RESULTS"] == 1
    assert three_way_a["NOT_ATTEMPTED"] == 55

    # Case B: All 3 stocks (2330, 2603, 2881) queried with 0 results on 2024-01-02
    full_session_attempts = partial_session_attempts + [
        QueryAttemptRecord(
            target_date="2024-01-02",
            stock_id="2881",
            source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
            response_status=QueryResolutionStatus.QUERY_COMPLETED_ZERO_RESULTS,
            execution_mode=ExecutionMode.AUTHENTIC,
            http_status=200,
            actual_raw_response_hash="hash_2881",
            result_count=0,
        )
    ]

    metrics_b = OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics(full_session_attempts)
    three_way_b = metrics_b["three_way_session_resolution"]
    assert three_way_b["AUTHENTIC_NO_ANNOUNCEMENT_CONFIRMED"] == 1
    assert three_way_b["QUERY_COMPLETED_ZERO_RESULTS"] == 0
    assert three_way_b["NOT_ATTEMPTED"] == 55


# ==============================================================================
# 6. Tripartite PIT Status Segregation Tests
# ==============================================================================

def test_tripartite_pit_status_segregation():
    """
    Test 6: Formally segregates:
    - SOURCE_EXISTENCE_VERIFIED: Content exists on official exchange.
    - ESTIMATED_PIT_ONLY: Content exists, but arrival time is estimated without vendor feed logs.
    - FORMAL_PIT_ELIGIBLE: Requires verified historical provider availability timestamp.
    """
    ev_target = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-01"][0]
    ev_dt = pd.Timestamp(f"{ev_target['date_str']} {ev_target['pub_time_str']}", tz="Asia/Taipei")
    raw_content = ev_target.get("body_text", "")
    raw_bytes = raw_content.encode("utf-8")
    ev_hash = compute_normalized_content_hash(ev_target["headline"], raw_content)

    session = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_MOPS_HTML_ARCHIVE",
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=112&month=06&co_id=2330",
        request_start_time=pd.Timestamp.now(tz="Asia/Taipei"),
        response_end_time=pd.Timestamp.now(tz="Asia/Taipei"),
        http_status=200,
        raw_response_bytes=raw_bytes,
        service_provenance="GENUINE_HTTPS_RETRIEVAL_SERVICE",
        announcement_ids=["MOPS-2330-20230606-AGM"],
        allow_legacy_fixture=True,
    )

    # Case A: is_formal_pit_eligible=True but is_availability_verified=False -> MUST BE REJECTED
    evidence_invalid_pit = TrustedExternalSourceAdapter.certify_with_live_retrieval_evidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=raw_content,
        raw_http_response_bytes=raw_bytes,
        retrieval_session_id=session.session_id,
        headline=ev_target["headline"],
        payload_hash=ev_hash,
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=112&month=06&co_id=2330",
        is_existence_verified=True,
        is_availability_verified=False,
    )
    evidence_invalid_pit.is_formal_pit_eligible = True

    status_a, dict_a, notes_a = TrustedSourceCertifier.verify_evidence(ev_target, evidence_invalid_pit)
    assert status_a == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert dict_a is None
    assert "ESTIMATED_PIT_ONLY cannot elevate to FORMAL_PIT_ELIGIBLE" in notes_a

    # Case B: is_existence_verified=True, is_availability_verified=False -> VERIFIED with ESTIMATED_PIT_ONLY
    evidence_valid_estimated = TrustedExternalSourceAdapter.certify_with_live_retrieval_evidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=raw_content,
        raw_http_response_bytes=raw_bytes,
        retrieval_session_id=session.session_id,
        headline=ev_target["headline"],
        payload_hash=ev_hash,
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=112&month=06&co_id=2330",
        is_existence_verified=True,
        is_availability_verified=False,
    )
    status_b, dict_b, notes_b = TrustedSourceCertifier.verify_evidence(ev_target, evidence_valid_estimated)
    assert status_b == SourceVerificationStatus.VERIFIED_SOURCE
    assert dict_b is not None
    assert dict_b["pit_status"] == DisclosurePITStatus.ESTIMATED_PIT_ONLY.value
    assert dict_b["is_formal_pit_eligible"] is False


# ==============================================================================
# 7. Evidence Manifest Reproducibility & Integrity Tests
# ==============================================================================

def test_evidence_manifest_entry_verification():
    """
    Test 7: Validates that EvidenceManifest enforces SHA-256 payload digests,
    raw response hashes, and session registry cross-checks.
    """
    raw_bytes = b'[{"\xe5\x85\xac\xe5\x8f\xb8\xe4\xbb\xa3\xe8\x99\x9f": "2881", "\xe4\xb8\xbb\xe6\x97\xa8": "\xe5\x85\xac\xe5\x91\x8a\xe6\x9c\x88\xe7\x87\x9f\xe6\x94\xb6"}]'
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    t_retrieved = pd.Timestamp("2024-03-15 15:30:00", tz="Asia/Taipei")
    headline = "公告月營收"
    content = "富邦金公告2月份自結合併損益"
    parsed_hash = compute_normalized_content_hash(headline, content)

    # Register genuine session
    session = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_OPENAPI",
        endpoint_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        request_start_time=t_retrieved - pd.Timedelta(seconds=1),
        response_end_time=t_retrieved,
        http_status=200,
        raw_response_bytes=raw_bytes,
        service_provenance="GENUINE_HTTPS_RETRIEVAL_SERVICE",
    )

    entry = EvidenceManifestEntry(
        announcement_id="OPENAPI-2881-20240315-001",
        stock_id="2881",
        filing_datetime=t_retrieved,
        headline=headline,
        raw_content=content,
        source_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        query_parameters={},
        retrieved_at=t_retrieved,
        raw_response_bytes=raw_bytes,
        raw_response_hash=raw_hash,
        parsed_payload_hash=parsed_hash,
        verification_status=SourceVerificationStatus.SOURCE_EXISTENCE_VERIFIED,
        pit_status=DisclosurePITStatus.ESTIMATED_PIT_ONLY,
        retrieval_session_id=session.session_id,
        source_channel=OfficialSourceChannel.TWSE_OPENAPI,
    )

    manifest = EvidenceManifest(manifest_id="MANIFEST-2024Q1-AUDIT-001")
    manifest.add_entry(entry)
    assert len(manifest.entries) == 1

    is_valid, msg = manifest.verify_entry_integrity(entry)
    assert is_valid is True
    assert "Entry integrity and cryptographic digests verified" in msg

    # Tampered payload hash test
    tampered_entry = EvidenceManifestEntry(
        announcement_id="OPENAPI-2881-20240315-001",
        stock_id="2881",
        filing_datetime=t_retrieved,
        headline=headline,
        raw_content=content,
        source_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        query_parameters={},
        retrieved_at=t_retrieved,
        raw_response_bytes=raw_bytes,
        raw_response_hash=raw_hash,
        parsed_payload_hash="forged_payload_hash_12345",
        verification_status=SourceVerificationStatus.SOURCE_EXISTENCE_VERIFIED,
        pit_status=DisclosurePITStatus.ESTIMATED_PIT_ONLY,
        retrieval_session_id=session.session_id,
        source_channel=OfficialSourceChannel.TWSE_OPENAPI,
    )
    is_valid_t, msg_t = manifest.verify_entry_integrity(tampered_entry)
    assert is_valid_t is False
    assert "parsed_payload_hash mismatch" in msg_t


# ==============================================================================
# 8. Minimal Pipeline Integration Forward Pass Tests
# ==============================================================================

def test_minimal_pipeline_integration_forward_pass():
    """
    Test 8: End-to-end integration of a genuine official disclosure into the V8 news-aware decision pipeline:
    Official Raw Response -> Source Evidence Manifest -> Event Parsing -> Stock/Event Relation ->
    BGE 512D -> Fixed Projection 128D -> Metadata 24D + Relation 8D ->
    Recent (32) / Active (8) Slots -> PIT Decision State.
    Validates tensor dimensions: (128,), (24,), (8,), (160,), (32, 160), (8, 160), (40, 160), (60, 100).
    """
    filing_dt = pd.Timestamp("2024-03-15 16:00:00", tz="Asia/Taipei")
    headline = "公告本公司董事會決議發放現金股利"
    content = "富邦金公告董事會決議112年度盈餘分派，每股配發現金股利2.5元及股票股利0.5元。"
    raw_bytes = content.encode("utf-8")
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    parsed_hash = compute_normalized_content_hash(headline, content)

    session = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_OPENAPI",
        endpoint_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        request_start_time=filing_dt,
        response_end_time=filing_dt + pd.Timedelta(seconds=1),
        http_status=200,
        raw_response_bytes=raw_bytes,
        service_provenance="GENUINE_HTTPS_RETRIEVAL_SERVICE",
        allow_legacy_fixture=True,
    )

    entry = EvidenceManifestEntry(
        announcement_id="OPENAPI-2881-20240315-DIVIDEND",
        stock_id="2881",
        filing_datetime=filing_dt,
        headline=headline,
        raw_content=content,
        source_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        query_parameters={},
        retrieved_at=filing_dt + pd.Timedelta(seconds=1),
        raw_response_bytes=raw_bytes,
        raw_response_hash=raw_hash,
        parsed_payload_hash=parsed_hash,
        verification_status=SourceVerificationStatus.SOURCE_EXISTENCE_VERIFIED,
        pit_status=DisclosurePITStatus.ESTIMATED_PIT_ONLY,
        retrieval_session_id=session.session_id,
        source_channel=OfficialSourceChannel.TWSE_OPENAPI,
    )

    result = execute_minimal_pipeline_integration(entry, target_stock_id="2881")

    assert result["integration_status"] == "SUCCESSFUL"
    assert result["text_raw_dim"] == 512
    assert result["text_proj_dim"] == 128
    assert result["metadata_dim"] == 24
    assert result["relation_dim"] == 8
    assert result["combined_dim"] == 160
    assert result["recent_slots_shape"] == [32, 160]
    assert result["active_slots_shape"] == [8, 160]
    assert result["news_slots_shape"] == [40, 160]
    assert result["market_context_shape"] == [60, 100]
    assert result["decision_cutoff"] == "20:00:00 Asia/Taipei"
    assert result["execution_contract"] == "NEXT_TRADING_DAY_OPEN"
    # L2 norm of projected 128D embedding should be normalized to ~1.0
    assert 0.99 <= result["embedding_l2_norm"] <= 1.01
