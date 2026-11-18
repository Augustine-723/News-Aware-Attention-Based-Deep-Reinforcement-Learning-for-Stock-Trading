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


def test_unauthorized_production_registration_rejection():
    """
    Test 2c (P0-1 item 1 & 6):
    Unauthorized caller calling register_production_session() without valid
    GenuineHttpsRetrievalService auth key MUST raise PermissionError.
    """
    fake_raw_bytes = b'{"status": "ok", "announcements": []}'
    t_now = pd.Timestamp.now(tz="Asia/Taipei")

    with pytest.raises(PermissionError, match="Unauthorized caller"):
        RetrievalSessionRegistry.register_production_session(
            service_auth_key="FORGED_UNAUTHORIZED_KEY_12345",
            source_channel="TWSE_OPENAPI",
            endpoint_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
            request_start_time=t_now,
            response_end_time=t_now,
            http_status=200,
            raw_response_bytes=fake_raw_bytes,
        )


def test_counterexample_forged_provenance_rejected_in_production_boundary():
    """
    Test 2d (P0-1 item 3 & 6):
    Self-attesting service_provenance='GENUINE_HTTPS_RETRIEVAL_SERVICE' or session_token
    without authorized service key sets is_production_certified=False and is_simulated=True.
    When audited against production boundary, it MUST fail closed to MOPS_CANDIDATE_UNVERIFIED.
    """
    ev_target = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-01"][0]
    ev_dt = pd.Timestamp(f"{ev_target['date_str']} {ev_target['pub_time_str']}", tz="Asia/Taipei")
    raw_content = ev_target.get("body_text", "")
    raw_bytes = raw_content.encode("utf-8")
    ev_hash = compute_normalized_content_hash(ev_target["headline"], raw_content)

    # Ordinary caller calls register_genuine_session without authorized key
    session = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_MOPS_HTML_ARCHIVE",
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=112&month=06&co_id=2330",
        request_start_time=pd.Timestamp.now(tz="Asia/Taipei"),
        response_end_time=pd.Timestamp.now(tz="Asia/Taipei"),
        http_status=200,
        raw_response_bytes=raw_bytes,
        source_identity="ORDINARY_UNAUTHORIZED_CALLER",
        service_provenance="GENUINE_HTTPS_RETRIEVAL_SERVICE",
        announcement_ids=["MOPS-2330-20230606-AGM"],
        allow_legacy_fixture=True,
    )
    assert session.is_production_certified is False
    assert session.is_simulated is True

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

    status, dict_res, notes = TrustedSourceCertifier.verify_evidence(
        ev_target, evidence, enforce_production_boundary=True
    )
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert dict_res is None
    assert "lacks genuine production certification" in notes


def test_counterexample_legacy_fixture_bypass_rejected():
    """
    Test 2e (P0-1 item 3 & 4):
    Passing allow_legacy_fixture=True allows unit test inspection but produces a
    session marked as SIMULATED; it is strictly rejected from formal production certification.
    """
    sim_session = RetrievalSessionRegistry.register_simulated_session(
        source_channel="TWSE_MOPS_HTML_ARCHIVE",
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
        request_start_time=pd.Timestamp.now(tz="Asia/Taipei"),
        response_end_time=pd.Timestamp.now(tz="Asia/Taipei"),
        http_status=200,
        raw_response_bytes=b"simulation_bytes",
    )
    assert sim_session.is_simulated is True
    assert sim_session.execution_mode == "SIMULATED"
    assert sim_session.is_production_certified is False


def test_openapi_session_cannot_certify_mops_historical_disclosure():
    """
    Test 2f (P0-1 item 7):
    A live TWSE_OPENAPI session CANNOT certify historical MOPS disclosure existence.
    """
    ev_target = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-01"][0]
    ev_dt = pd.Timestamp(f"{ev_target['date_str']} {ev_target['pub_time_str']}", tz="Asia/Taipei")
    raw_content = ev_target.get("body_text", "")
    raw_bytes = raw_content.encode("utf-8")
    ev_hash = compute_normalized_content_hash(ev_target["headline"], raw_content)

    session = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_OPENAPI",  # OpenAPI channel!
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01?step=1&TYPEK=sii&year=112&month=06&co_id=2330",
        request_start_time=pd.Timestamp.now(tz="Asia/Taipei"),
        response_end_time=pd.Timestamp.now(tz="Asia/Taipei"),
        http_status=200,
        raw_response_bytes=raw_bytes,
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

    status, dict_res, notes = TrustedSourceCertifier.verify_evidence(ev_target, evidence)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert "Live TWSE_OPENAPI probe session cannot certify historical MOPS disclosure" in notes


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
# 7. Evidence Manifest Reproducibility & Integrity Tests (P0-2)
# ==============================================================================

def test_evidence_manifest_entry_verification():
    """
    Test 7 (P0-2): Validates that EvidenceManifest re-parses official raw response bytes,
    cross-audits stock_id, headline, content, local_record_id, SHA-256 payload digests,
    raw response hashes, and session registry records.
    """
    raw_json_str = json.dumps([
        {
            "公司代號": "2881",
            "主旨": "公告月營收",
            "詳細內容": "富邦金公告2月份自結合併損益",
            "出表日期": "1130315",
            "出表時間": "153000",
        }
    ], ensure_ascii=False)
    raw_bytes = raw_json_str.encode("utf-8")
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

    local_rec_id = f"LOCAL-OPENAPI-2881-1130315-{parsed_hash[:8]}"
    entry = EvidenceManifestEntry(
        announcement_id=local_rec_id,
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
        local_record_id=local_rec_id,
        is_local_record_id=True,
        publication_time_parsed=True,
        parser_version="v8.14-openapi-json-v1",
        source_locator="JSON_ARRAY_RECORD[stock_id=2881,index=0]",
        raw_retention_status="FULL_RAW_BYTES_RETAINED",
    )

    manifest = EvidenceManifest(manifest_id="MANIFEST-2024Q1-AUDIT-001")
    manifest.add_entry(entry)
    assert len(manifest.entries) == 1

    is_valid, msg = manifest.verify_entry_integrity(entry)
    assert is_valid is True
    assert "Entry integrity, raw response re-parsing, and cryptographic digests verified" in msg


def test_counterexample_manifest_tampered_content_rejection():
    """
    Test 7b (P0-2 item 7 Counterexample Test):
    Raw response bytes is authentic official OpenAPI JSON, but the caller puts a
    fabricated headline or tampered body text into EvidenceManifestEntry.
    Manifest re-parsing MUST detect that the announcement content is missing from the raw response
    and fail closed with 'Content binding verification failed'.
    """
    raw_json_str = json.dumps([
        {
            "公司代號": "2881",
            "主旨": "富邦金公告2月營收",
            "詳細內容": "富邦金公告113年2月自結合併稅前淨利及稅後淨利。",
            "出表日期": "1130315",
            "出表時間": "153000",
        }
    ], ensure_ascii=False)
    raw_bytes = raw_json_str.encode("utf-8")
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    t_retrieved = pd.Timestamp("2024-03-15 15:30:00", tz="Asia/Taipei")

    session = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_OPENAPI",
        endpoint_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        request_start_time=t_retrieved - pd.Timedelta(seconds=1),
        response_end_time=t_retrieved,
        http_status=200,
        raw_response_bytes=raw_bytes,
        service_provenance="GENUINE_HTTPS_RETRIEVAL_SERVICE",
    )

    # Caller tampered with the headline and body text (e.g. fabricated M&A rumor)
    fabricated_headline = "富邦金宣布重大海外併購案"
    fabricated_content = "富邦金控董事會決議以新台幣1000億元併購海外大型商業銀行。"
    tampered_hash = compute_normalized_content_hash(fabricated_headline, fabricated_content)

    tampered_entry = EvidenceManifestEntry(
        announcement_id="LOCAL-OPENAPI-2881-1130315-FABRICATED",
        stock_id="2881",
        filing_datetime=t_retrieved,
        headline=fabricated_headline,
        raw_content=fabricated_content,
        source_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        query_parameters={},
        retrieved_at=t_retrieved,
        raw_response_bytes=raw_bytes,  # Authentic raw response bytes!
        raw_response_hash=raw_hash,    # Authentic raw response hash!
        parsed_payload_hash=tampered_hash,
        verification_status=SourceVerificationStatus.SOURCE_EXISTENCE_VERIFIED,
        pit_status=DisclosurePITStatus.ESTIMATED_PIT_ONLY,
        retrieval_session_id=session.session_id,
        source_channel=OfficialSourceChannel.TWSE_OPENAPI,
        local_record_id="LOCAL-OPENAPI-2881-1130315-FABRICATED",
        is_local_record_id=True,
    )

    manifest = EvidenceManifest(manifest_id="MANIFEST-2024Q1-AUDIT-TAMPER-TEST")
    manifest.add_entry(tampered_entry)

    is_valid, msg = manifest.verify_entry_integrity(tampered_entry)
    assert is_valid is False
    assert "Content binding verification failed" in msg


def test_counterexample_manifest_unparsed_time_rejection():
    """
    Test 7c (P0-2 item 5):
    If publication time parsing failed (publication_time_parsed=False),
    filing_datetime cannot silently substitute retrieved_at.
    """
    raw_bytes = json.dumps([{"公司代號": "2881", "主旨": "公告", "詳細內容": "內容"}]).encode("utf-8")
    t_retrieved = pd.Timestamp("2024-03-15 15:30:00", tz="Asia/Taipei")
    parsed_hash = compute_normalized_content_hash("公告", "內容")

    session = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_OPENAPI",
        endpoint_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        request_start_time=t_retrieved,
        response_end_time=t_retrieved,
        http_status=200,
        raw_response_bytes=raw_bytes,
        service_provenance="GENUINE_HTTPS_RETRIEVAL_SERVICE",
    )

    bad_entry = EvidenceManifestEntry(
        announcement_id="LOCAL-OPENAPI-2881-NODATE-001",
        stock_id="2881",
        filing_datetime=t_retrieved,  # Silently substituted retrieved_at!
        headline="公告",
        raw_content="內容",
        source_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        query_parameters={},
        retrieved_at=t_retrieved,
        raw_response_bytes=raw_bytes,
        raw_response_hash=hashlib.sha256(raw_bytes).hexdigest(),
        parsed_payload_hash=parsed_hash,
        retrieval_session_id=session.session_id,
        publication_time_parsed=False,  # Unparsed publication time!
        local_record_id="LOCAL-OPENAPI-2881-NODATE-001",
        is_local_record_id=True,
    )

    manifest = EvidenceManifest(manifest_id="MANIFEST-TEST")
    is_valid, msg = manifest.verify_entry_integrity(bad_entry)
    assert is_valid is False
    assert "cannot silently substitute retrieved_at" in msg


# ==============================================================================
# 8. Tensor Shape Smoke Test (P0-3 item 7)
# ==============================================================================

def test_tensor_shape_smoke_test():
    """
    Test 8a (P0-3 item 7):
    Shape Smoke Test strictly isolated from Historical PIT Integration.
    Verifies tensor dimensions across the pipeline architecture:
    (512,) -> (128,) -> (24,) -> (8,) -> (160,) -> (32, 160) -> (8, 160) -> (40, 160) -> (60, 100).
    Explicitly declared as interface verification only, NOT empirical proof of DRL trading value.
    """
    filing_dt = pd.Timestamp("2024-03-15 16:00:00", tz="Asia/Taipei")
    headline = "台積電公告先進製程資本支出"
    content = "台積電董事會決議核准資本預算，用以建置2奈米先進製程產能。"
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
        announcement_id="OPENAPI-2330-20240315-CAPEX",
        stock_id="2330",
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

    result = execute_minimal_pipeline_integration(entry, target_stock_id="2330")

    assert result["text_raw_dim"] == 512
    assert result["text_proj_dim"] == 128
    assert result["metadata_dim"] == 24
    assert result["relation_dim"] == 8
    assert result["combined_dim"] == 160
    assert result["recent_slots_shape"] == [32, 160]
    assert result["active_slots_shape"] == [8, 160]
    assert result["news_slots_shape"] == [40, 160]
    assert result["market_context_shape"] == [60, 100]
    assert result["market_context_status"] == "SYNTHETIC_MARKET_PLACEHOLDER"
    assert result["pipeline_verification_level"] == "INTERFACE_SPECIFICATION_VERIFICATION_ONLY"
    assert result["empirical_drl_validated"] is False


# ==============================================================================
# 9. Historical PIT Integration Gate Tests (P0-3)
# ==============================================================================

def test_historical_pit_integration_gate():
    """
    Test 9 (P0-3 items 3, 4, 5):
    Rigorous Point-in-Time (PIT) As-Of Availability Gate before slot ingestion.
    Verifies that an event CANNOT be ingested into Recent / Active slots unless
    proven available prior to decision_timestamp (20:00:00 cutoff):
    - Case A: On-time announcement (16:00) with estimated hypothesis -> Accepted into Slot 0.
    - Case B: Post-cutoff announcement (20:30) -> Rejected from Slot 0 (slots remain 0).
    - Case C: Future announcement (T+1) -> Rejected from Slot 0 (slots remain 0).
    - Case D: Non-trading day announcement -> Rejected from Slot 0 (slots remain 0).
    - Case E: Retrieved in 2026 without historical vendor feed log -> Rejected from historical decision.
    """
    trading_calendar = ["2024-03-15", "2024-03-18", "2024-03-19"]
    content = "富邦金控公告董事會決議發放現金股利及股票股利。"
    headline = "富邦金董事會決議股利分派"
    raw_bytes = content.encode("utf-8")
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    parsed_hash = compute_normalized_content_hash(headline, content)

    session = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_OPENAPI",
        endpoint_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        request_start_time=pd.Timestamp.now(tz="Asia/Taipei"),
        response_end_time=pd.Timestamp.now(tz="Asia/Taipei"),
        http_status=200,
        raw_response_bytes=raw_bytes,
        allow_legacy_fixture=True,
    )

    # --------------------------------------------------------------------------
    # Case A: On-time announcement (2024-03-15 16:00:00)
    # Available at 16:05 (hypothesis) <= 20:00 cutoff -> MUST BE INGESTED
    # --------------------------------------------------------------------------
    entry_ontime = EvidenceManifestEntry(
        announcement_id="OPENAPI-2881-20240315-ONTIME",
        stock_id="2881",
        filing_datetime=pd.Timestamp("2024-03-15 16:00:00", tz="Asia/Taipei"),
        headline=headline,
        raw_content=content,
        source_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        query_parameters={},
        retrieved_at=pd.Timestamp("2024-03-15 16:01:00", tz="Asia/Taipei"),
        raw_response_bytes=raw_bytes,
        raw_response_hash=raw_hash,
        parsed_payload_hash=parsed_hash,
        retrieval_session_id=session.session_id,
    )
    res_a = execute_minimal_pipeline_integration(
        entry_ontime,
        target_stock_id="2881",
        decision_cutoff_str="20:00:00",
        decision_date_str="2024-03-15",
        allow_estimated_hypothesis=True,
        trading_calendar=trading_calendar,
    )
    assert res_a["is_available_as_of"] is True
    assert res_a["slot_ingested"] is True
    assert res_a["integration_status"] == "SUCCESSFUL"
    assert res_a["market_context_status"] == "SYNTHETIC_MARKET_PLACEHOLDER"

    # --------------------------------------------------------------------------
    # Case B: Post-cutoff announcement (2024-03-15 20:30:00)
    # Filed at 20:30 > 20:00 cutoff -> MUST BE REJECTED (Lookahead bias prevention)
    # --------------------------------------------------------------------------
    entry_late = EvidenceManifestEntry(
        announcement_id="OPENAPI-2881-20240315-LATE",
        stock_id="2881",
        filing_datetime=pd.Timestamp("2024-03-15 20:30:00", tz="Asia/Taipei"),
        headline=headline,
        raw_content=content,
        source_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        query_parameters={},
        retrieved_at=pd.Timestamp("2024-03-15 20:31:00", tz="Asia/Taipei"),
        raw_response_bytes=raw_bytes,
        raw_response_hash=raw_hash,
        parsed_payload_hash=parsed_hash,
        retrieval_session_id=session.session_id,
    )
    res_b = execute_minimal_pipeline_integration(
        entry_late,
        target_stock_id="2881",
        decision_cutoff_str="20:00:00",
        decision_date_str="2024-03-15",
        allow_estimated_hypothesis=True,
        trading_calendar=trading_calendar,
    )
    assert res_b["is_available_as_of"] is False
    assert res_b["slot_ingested"] is False
    assert res_b["integration_status"] == "AS_OF_REJECTED"
    assert "after decision cutoff" in res_b["rejection_reason"]

    # --------------------------------------------------------------------------
    # Case C: Future announcement (T+1: 2024-03-16 10:00:00) for 2024-03-15 decision
    # Future filing > 2024-03-15 20:00:00 -> MUST BE REJECTED
    # --------------------------------------------------------------------------
    entry_future = EvidenceManifestEntry(
        announcement_id="OPENAPI-2881-20240316-FUTURE",
        stock_id="2881",
        filing_datetime=pd.Timestamp("2024-03-16 10:00:00", tz="Asia/Taipei"),
        headline=headline,
        raw_content=content,
        source_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        query_parameters={},
        retrieved_at=pd.Timestamp("2024-03-16 10:01:00", tz="Asia/Taipei"),
        raw_response_bytes=raw_bytes,
        raw_response_hash=raw_hash,
        parsed_payload_hash=parsed_hash,
        retrieval_session_id=session.session_id,
    )
    res_c = execute_minimal_pipeline_integration(
        entry_future,
        target_stock_id="2881",
        decision_cutoff_str="20:00:00",
        decision_date_str="2024-03-15",
        allow_estimated_hypothesis=True,
        trading_calendar=trading_calendar,
    )
    assert res_c["is_available_as_of"] is False
    assert res_c["slot_ingested"] is False
    assert res_c["integration_status"] == "AS_OF_REJECTED"

    # --------------------------------------------------------------------------
    # Case D: Non-trading day announcement (Sunday 2024-03-17)
    # Sunday is not in trading_calendar -> MUST BE REJECTED
    # --------------------------------------------------------------------------
    entry_weekend = EvidenceManifestEntry(
        announcement_id="OPENAPI-2881-20240317-WEEKEND",
        stock_id="2881",
        filing_datetime=pd.Timestamp("2024-03-17 14:00:00", tz="Asia/Taipei"),
        headline=headline,
        raw_content=content,
        source_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        query_parameters={},
        retrieved_at=pd.Timestamp("2024-03-17 14:01:00", tz="Asia/Taipei"),
        raw_response_bytes=raw_bytes,
        raw_response_hash=raw_hash,
        parsed_payload_hash=parsed_hash,
        retrieval_session_id=session.session_id,
    )
    res_d = execute_minimal_pipeline_integration(
        entry_weekend,
        target_stock_id="2881",
        decision_cutoff_str="20:00:00",
        decision_date_str="2024-03-17",
        allow_estimated_hypothesis=True,
        trading_calendar=trading_calendar,
    )
    assert res_d["is_available_as_of"] is False
    assert res_d["slot_ingested"] is False
    assert "not an official trading day" in res_d["rejection_reason"]

    # --------------------------------------------------------------------------
    # Case E: Observed in 2026 without vendor feed log or estimated hypothesis
    # Earliest proven availability is 2026 > 2024-03-15 -> MUST BE REJECTED
    # --------------------------------------------------------------------------
    entry_2026 = EvidenceManifestEntry(
        announcement_id="OPENAPI-2881-20260401-OBSERVED",
        stock_id="2881",
        filing_datetime=None,  # No parsed publication time
        headline=headline,
        raw_content=content,
        source_url="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        query_parameters={},
        retrieved_at=pd.Timestamp("2026-04-01 10:00:00", tz="Asia/Taipei"),
        raw_response_bytes=raw_bytes,
        raw_response_hash=raw_hash,
        parsed_payload_hash=parsed_hash,
        retrieval_session_id=session.session_id,
        publication_time_parsed=False,
    )
    res_e = execute_minimal_pipeline_integration(
        entry_2026,
        target_stock_id="2881",
        decision_cutoff_str="20:00:00",
        decision_date_str="2024-03-15",
        allow_estimated_hypothesis=False,  # No hypothesis!
        trading_calendar=trading_calendar,
    )
    assert res_e["is_available_as_of"] is False
    assert res_e["slot_ingested"] is False
    assert res_e["integration_status"] == "AS_OF_REJECTED"


# ==============================================================================
# 10. Evidence Reproducibility & Governance Tests (P1)
# ==============================================================================

def test_metrics_segregate_waf_blocks_and_parsed_responses():
    """
    Test 10 (P1):
    evaluate_q1_2024_collection_metrics distinguishes successful_parsed_responses
    from waf_access_blocked_responses.
    """
    t_now = pd.Timestamp.now(tz="Asia/Taipei")
    rec_blocked = QueryAttemptRecord(
        target_date="2024-01-18",
        stock_id="2330",
        source_channel=OfficialSourceChannel.TWSE_MOPS_HTML_ARCHIVE,
        endpoint="https://mops.twse.com.tw/mops/web/ajax_t05st01",
        query_start_time=t_now,
        query_end_time=t_now,
        http_status=200,
        response_status=QueryResolutionStatus.QUERY_FAILED_HTTP,
        actual_raw_response_hash="hash_blocked",
        result_count=0,
        error_info="MOPS anti-bot security barrier: FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED.",
        execution_mode=ExecutionMode.AUTHENTIC,
    )
    rec_success = QueryAttemptRecord(
        target_date="2024-01-18",
        stock_id="2881",
        source_channel=OfficialSourceChannel.TWSE_OPENAPI,
        endpoint="https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        query_start_time=t_now,
        query_end_time=t_now,
        http_status=200,
        response_status=QueryResolutionStatus.QUERY_COMPLETED_SUCCESS,
        actual_raw_response_hash="hash_success",
        result_count=1,
        execution_mode=ExecutionMode.AUTHENTIC,
    )

    metrics = OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics([rec_blocked, rec_success])
    failure_dict = metrics["source_failure_rate"]
    assert failure_dict["waf_access_blocked_responses"] == 1
    assert failure_dict["successful_parsed_responses"] == 1


def test_openapi_live_sample_isolated_from_2024_q1():
    """
    Test 10b (P1):
    TWSE OpenAPI live sample (2026-CURRENT-ROLLING) is strictly isolated and does
    not count toward 2024 Q1 Historical Dataset (historical Q1 retrieved count = 0).
    """
    metrics = OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics()
    pilot_scope = metrics["pilot_scope"]
    assert pilot_scope["openapi_current_rolling_sample_isolated"] is True
    assert pilot_scope["twse_openapi_sample_period"] == "2026-CURRENT-ROLLING"
    assert pilot_scope["historical_q1_retrieved_count"] == 0

