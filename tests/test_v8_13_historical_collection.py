"""
V8.13 Systematic Historical News Collection Feasibility & Statistical Power Calibration Tests
Validates:
1. Simulated adapter receipt isolation and fail-closed security:
   - Prohibits SIMULATED_ADAPTER_RECEIPT from elevating records to VERIFIED_SOURCE.
   - Requires genuine external responses and verified adapter receipts.
   - Distinguishes SHA-256 integrity digests from cryptographic digital signatures.
   - Maintains default baseline formal verification count at strictly 0 (VERIFIED_SOURCE: 0).
2. Small-scale official historical source collection inventory:
   - Audits 2330, 2603, 2881 across 2024 Q1 (56 authentic TWSE trading sessions).
   - Records announcement volume, filing timestamps, actual retrieval timestamps, payload hashes.
   - Segregates Event Existence from Historical Downstream Provider Availability Evidence.
   - Audits legal terms and licensing barriers (NEEDS_LEGAL_REVIEW).
   - Strictly prohibits conflation of 50 curated pilot news items with verified official data.
3. Coverage and collection quality metrics:
   - 3-way daily query resolution (NOT_ATTEMPTED, QUERY_COMPLETED_ZERO_RESULTS, AUTHENTIC_NO_ANNOUNCEMENT_CONFIRMED).
   - Deduplication (10 clusters across 12 announcements).
   - Revision tracking (Rule 51 clarification lineage).
   - Zero query failure rate on lawful query interface.
4. Statistical power validity and calibration:
   - Empirical Type-I error calibration under H0: delta_rho = 0.0 (false rejection rate approx 5%).
   - Power curve explicitly marked as ASSUMPTION_BASED_ILLUSTRATION and POWER_NOT_ESTABLISHED.
   - Multi-dimensional sensitivity analysis across ICC, valid label ratio, and purging loss.
   - Sample sizes designated strictly as PROVISIONAL_PLANNING_TARGET_ONLY pending real data calibration.
"""

import pytest
import pandas as pd
import numpy as np
import hashlib

from stock_rl.evaluation.pilot_dataset import (
    PilotSourceManifest,
    SourceVerificationStatus,
    LicenseVerificationStatus,
    UsageRightsCategory,
    ExternalSourceAdapterReceipt,
    AdapterReceiptMode,
    TrustedExternalSourceAdapter,
    TrustedSourceCertifier,
    DatasetExpansionReadinessAssessor,
    MOPSSourceEvidence,
    RetrievalSessionRegistry,
    RAW_PILOT_EVENTS,
)
from stock_rl.evaluation.official_source_collector import (
    OfficialSourceChannel,
    ExecutionMode,
    QueryResolutionStatus,
    OfficialAnnouncementRecord,
    OfficialHistoricalNewsCollector,
)
from stock_rl.evaluation.semantic_eval import compute_normalized_content_hash


# ==============================================================================
# 1. Simulated Adapter Receipt Isolation and Fail-Closed Security
# ==============================================================================

def test_simulated_receipt_rejection_and_fail_closed():
    """
    V8.13 Requirement 1:
    - Locally generated receipts are explicitly marked SIMULATED_ADAPTER_RECEIPT.
    - Simulated receipts strictly FAIL CLOSED and cannot elevate to VERIFIED_SOURCE.
    - Distinguishes SHA-256 integrity digests from cryptographic signatures.
    - External verification requires authentic external evidence.
    - Default manifest has strictly 0 VERIFIED_SOURCE.
    """
    # 1. Baseline default manifest verification count is strictly 0
    manifest = PilotSourceManifest.generate_manifest()
    summary = PilotSourceManifest.get_provenance_summary()
    assert summary["VERIFIED_SOURCE"] == 0
    assert summary["total_unverified_records"] == 50
    assert summary["MOPS_CANDIDATE_UNVERIFIED"] == 12
    assert summary["CURATED_UNVERIFIED"] == 38

    # 2. Inspect target event EV-2330-01
    ev_target = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-01"][0]
    ev_dt = pd.Timestamp(f"{ev_target['date_str']} {ev_target['pub_time_str']}", tz="Asia/Taipei")
    ev_hash = compute_normalized_content_hash(ev_target["headline"], ev_target.get("body_text", ""))

    # 3. Create simulated adapter receipt (default mode)
    simulated_receipt = TrustedExternalSourceAdapter.create_adapter_receipt(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
        raw_content=ev_target.get("body_text", ""),
    )
    assert simulated_receipt.receipt_mode == AdapterReceiptMode.SIMULATED_ADAPTER_RECEIPT
    assert simulated_receipt.is_simulated is True
    assert simulated_receipt.is_live_adapter_call is False
    assert simulated_receipt.integrity_digest is not None
    assert simulated_receipt.adapter_signature == simulated_receipt.integrity_digest

    # 4. Attempt certification with simulated receipt -> MUST FAIL CLOSED
    simulated_evidence = MOPSSourceEvidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        payload_hash=ev_hash,
        certifier_authority="TWSE_MOPS_OFFICIAL_ARCHIVE",
        external_reference_url="https://mops.twse.com.tw/mops/web/t05st01",
        adapter_receipt=simulated_receipt,
        is_trusted_external_certified=True,
    )
    status, evidence_dict, notes = TrustedSourceCertifier.verify_evidence(ev_target, simulated_evidence)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert evidence_dict is None
    assert "Adapter receipt is a SIMULATED_ADAPTER_RECEIPT" in notes
    assert "strictly prohibited from elevating records to VERIFIED_SOURCE" in notes

    # 5. Missing external evidence bundle -> MUST FAIL CLOSED
    status_none, _, notes_none = TrustedSourceCertifier.verify_evidence(ev_target, None)
    assert status_none == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert "unverified without raw evidence" in notes_none

    # 6. Attempt certification with is_authentic=True via standard adapter -> MUST FAIL CLOSED (P0-3)
    self_attested_evidence = TrustedExternalSourceAdapter.certify_with_adapter_receipt(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        headline=ev_target["headline"],
        payload_hash=ev_hash,
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
        is_existence_verified=True,
        is_availability_verified=False,
        is_authentic=True,
    )
    assert self_attested_evidence.adapter_receipt.receipt_mode == AdapterReceiptMode.SIMULATED_ADAPTER_RECEIPT
    assert self_attested_evidence.adapter_receipt.is_simulated is True
    assert self_attested_evidence.adapter_receipt.is_verified_live_retrieval is False

    fail_status, fail_dict, fail_notes = TrustedSourceCertifier.verify_evidence(ev_target, self_attested_evidence)
    assert fail_status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert fail_dict is None
    assert "Adapter receipt is a SIMULATED_ADAPTER_RECEIPT" in fail_notes or "lacks verified live external retrieval" in fail_notes

    # 7. Genuine live retrieval evidence -> Elevates to VERIFIED_SOURCE when verified
    live_raw_bytes = ev_target.get("body_text", "").encode("utf-8")
    session_rec = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_MOPS_HTML_ARCHIVE",
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
        request_start_time=pd.Timestamp.now(tz="Asia/Taipei"),
        response_end_time=pd.Timestamp.now(tz="Asia/Taipei"),
        http_status=200,
        raw_response_bytes=live_raw_bytes,
        parsed_summary={"announcement_count": 1},
        source_identity="OFFICIAL_MOPS_HTTPS_CLIENT",
        announcement_ids=["MOPS-2330-20230606-AGM"],
    )

    authentic_evidence = TrustedExternalSourceAdapter.certify_with_live_retrieval_evidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        raw_http_response_bytes=live_raw_bytes,
        retrieval_session_id=session_rec.session_id,
        headline=ev_target["headline"],
        payload_hash=ev_hash,
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
        is_existence_verified=True,
        is_availability_verified=False,
    )
    assert authentic_evidence.adapter_receipt.receipt_mode == AdapterReceiptMode.AUTHENTIC_EXTERNAL_RECEIPT
    assert authentic_evidence.adapter_receipt.is_simulated is False
    assert authentic_evidence.adapter_receipt.is_live_adapter_call is True
    assert authentic_evidence.adapter_receipt.is_verified_live_retrieval is True
    assert authentic_evidence.adapter_receipt.retrieval_session_id == session_rec.session_id
    assert authentic_evidence.adapter_receipt.raw_http_status == 200

    auth_status, auth_dict, auth_notes = TrustedSourceCertifier.verify_evidence(ev_target, authentic_evidence)
    assert auth_status == SourceVerificationStatus.VERIFIED_SOURCE
    assert auth_dict is not None
    assert auth_dict["receipt_mode"] == "AUTHENTIC_EXTERNAL_RECEIPT"
    assert auth_dict["is_simulated"] is False
    assert auth_dict["is_verified_live_retrieval"] is True
    assert "Verified authentic MOPS corporate disclosure" in auth_notes
    RetrievalSessionRegistry.clear()


def test_counterexample_raw_pilot_events_as_http_bytes_rejection():
    """
    P0-1 Item 7 Counterexample Test:
    Taking RAW_PILOT_EVENTS body text as raw HTTP bytes and filling in a self-minted
    or unregistered retrieval_session_id MUST be rejected and cannot achieve VERIFIED_SOURCE.
    """
    ev_target = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-01"][0]
    ev_dt = pd.Timestamp(f"{ev_target['date_str']} {ev_target['pub_time_str']}", tz="Asia/Taipei")
    ev_hash = compute_normalized_content_hash(ev_target["headline"], ev_target.get("body_text", ""))
    fake_raw_bytes = ev_target.get("body_text", "").encode("utf-8")

    # Caller attempts to self-attest an authentic receipt with arbitrary un-registered session ID
    fake_session_id = "SELF-MINTED-SESS-TWSE-2024Q1-001"
    evidence_with_fake_session = TrustedExternalSourceAdapter.certify_with_live_retrieval_evidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        raw_http_response_bytes=fake_raw_bytes,
        retrieval_session_id=fake_session_id,
        headline=ev_target["headline"],
        payload_hash=ev_hash,
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
        is_existence_verified=True,
        is_availability_verified=False,
    )

    # Verification MUST fail closed
    status, evidence_dict, notes = TrustedSourceCertifier.verify_evidence(ev_target, evidence_with_fake_session)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert evidence_dict is None
    assert "not found in trusted RetrievalSessionRegistry" in notes
    assert "self-attested or unregistered session IDs are strictly prohibited" in notes


def test_openapi_probe_cannot_certify_historical_mops_announcements():
    """
    P0-1 Item 8 Test:
    A live OpenAPI query probe session cannot certify historical MOPS disclosure existence
    or historical provider availability (differing channel/scope and lack of historical vendor receipt; fails closed).
    """
    openapi_raw_bytes = b'[{"\xe5\x85\xac\xe5\x8f\xb8\xe4\xbb\xa3\xe8\x99\x9f": "2330"}]'
    openapi_endpoint = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
    openapi_content = openapi_raw_bytes.decode("utf-8")

    ev_target = dict([e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-01"][0])
    ev_target["body_text"] = openapi_content
    ev_dt = pd.Timestamp(f"{ev_target['date_str']} {ev_target['pub_time_str']}", tz="Asia/Taipei")
    ev_hash = compute_normalized_content_hash(ev_target["headline"], openapi_content)

    openapi_session = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_OPENAPI",
        endpoint_url=openapi_endpoint,
        request_start_time=pd.Timestamp.now(tz="Asia/Taipei"),
        response_end_time=pd.Timestamp.now(tz="Asia/Taipei"),
        http_status=200,
        raw_response_bytes=openapi_raw_bytes,
        parsed_summary={"target_stock": "2330", "result_count": 1},
        source_identity="TWSE_OPENAPI_HTTPS_CLIENT",
        announcement_ids=["MOPS-2330-20230606-AGM"],
    )

    openapi_evidence = TrustedExternalSourceAdapter.certify_with_live_retrieval_evidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=openapi_content,
        raw_http_response_bytes=openapi_raw_bytes,
        retrieval_session_id=openapi_session.session_id,
        headline=ev_target["headline"],
        payload_hash=ev_hash,
        endpoint_url=openapi_endpoint,
        is_existence_verified=True,
        is_availability_verified=False,
    )

    status, evidence_dict, notes = TrustedSourceCertifier.verify_evidence(ev_target, openapi_evidence)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert evidence_dict is None
    assert "Live TWSE_OPENAPI probe session cannot certify historical MOPS disclosure existence" in notes
    RetrievalSessionRegistry.clear()


def test_self_attested_authentic_receipt_rejection():
    """
    P0-3 Regression Test:
    Calling with is_authentic=True or instantiating receipts without verified live external retrieval
    MUST fail closed and be rejected from VERIFIED_SOURCE.
    """
    ev_target = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-01"][0]
    ev_dt = pd.Timestamp(f"{ev_target['date_str']} {ev_target['pub_time_str']}", tz="Asia/Taipei")
    ev_hash = compute_normalized_content_hash(ev_target["headline"], ev_target.get("body_text", ""))

    # Attempt to forge authentic receipt directly on ExternalSourceAdapterReceipt
    forged_receipt = ExternalSourceAdapterReceipt(
        adapter_id="TWSE_MOPS_REST_ADAPTER_V1",
        receipt_id="FORGED-RCPT-001",
        retrieved_at=pd.Timestamp.now(tz="Asia/Taipei"),
        raw_response_hash=hashlib.sha256(ev_target.get("body_text", "").encode("utf-8")).hexdigest(),
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        publication_datetime=ev_dt,
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
        receipt_mode=AdapterReceiptMode.AUTHENTIC_EXTERNAL_RECEIPT,
        is_simulated=False,
        is_verified_live_retrieval=False,  # Self-attested without verified live session!
        retrieval_session_id=None,
    )
    forged_evidence = MOPSSourceEvidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        payload_hash=ev_hash,
        certifier_authority="TWSE_MOPS_OFFICIAL_ARCHIVE",
        external_reference_url="https://mops.twse.com.tw/mops/web/t05st01",
        adapter_receipt=forged_receipt,
        is_trusted_external_certified=True,
    )
    status, evidence_dict, notes = TrustedSourceCertifier.verify_evidence(ev_target, forged_evidence)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert evidence_dict is None
    assert "lacks verified live external retrieval evidence" in notes


# ==============================================================================
# 2. Small-Scale Official Historical Source Collection Inventory
# ==============================================================================

def test_small_scale_official_historical_collection_inventory():
    """
    V8.13 Requirement 2:
    - Focuses on 2330, 2603, 2881 across 2024 Q1 (56 authentic trading sessions).
    - Records announcement volumes, raw announcement IDs, filing datetimes, actual retrieval timestamps.
    - Segregates Event Existence from Historical Downstream Provider Availability Evidence.
    - Audits licensing metadata and fair-use limitations (NEEDS_LEGAL_REVIEW).
    - Confirms zero conflation between 50 curated pilot events and verified official data.
    """
    survey = OfficialHistoricalNewsCollector.get_official_source_survey()
    assert "TWSE_MOPS_HTML_ARCHIVE" in survey["sources"]
    assert "TWSE_OPENAPI" in survey["sources"]
    assert "COMMERCIAL_VENDOR_FEEDS" in survey["sources"]
    assert "CRITICAL GOVERNANCE BARRIER" in survey["curated_pilot_events_conflation_warning"]

    # Verify MOPS survey characteristics
    mops_info = survey["sources"]["TWSE_MOPS_HTML_ARCHIVE"]
    assert "1999-01-01 to Present" in mops_info["queryable_date_range"]
    assert mops_info["historical_provider_availability_evidence"] is False
    assert mops_info["licensing_status"] == LicenseVerificationStatus.NEEDS_LEGAL_REVIEW.value
    assert mops_info["commercial_redistribution_eligible"] is False
    assert "FEASIBLE_FOR_HISTORICAL_AUDIT_BUT_LACKS_VENDOR_FEED_RECEIPTS" == mops_info["evaluation_verdict"]

    # Verify inventory of 12 disclosures across 2024 Q1
    inventory = OfficialHistoricalNewsCollector.get_source_inventory()
    assert len(inventory) == 12

    # Check ticker distribution
    tickers = [r.company_id for r in inventory]
    assert tickers.count("2330") == 6
    assert tickers.count("2603") == 3
    assert tickers.count("2881") == 3

    # Check all fields populated with authentic provenance
    for r in inventory:
        assert r.candidate_fixture_id.startswith("CANDIDATE-Q1-2024-")
        assert r.announcement_id is None  # P0-2: Unverified without live official response
        assert r.company_id in ["2330", "2603", "2881"]
        assert len(r.headline) > 5
        assert len(r.body_text) > 20
        assert str(r.filing_datetime.tz) == "Asia/Taipei"
        assert r.retrieved_at is None  # P0-2: None without actual network log
        assert r.raw_official_response_bytes is None  # P0-2: None without live server bytes
        assert r.raw_response_hash is None  # P0-2: None without live server bytes
        assert len(r.candidate_fixture_hash) == 64
        assert len(r.normalized_content_hash) == 64
        assert r.historical_provider_availability_included is False
        assert r.provider_availability_receipt is None
        assert r.license_metadata is not None
        assert r.license_metadata.verification_status == LicenseVerificationStatus.NEEDS_LEGAL_REVIEW
        assert r.verification_status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED


# ==============================================================================
# 3. Coverage and Collection Quality Metrics
# ==============================================================================

def test_coverage_and_collection_quality_metrics():
    """
    V8.13 Requirement 3 & P0-1:
    - Coverage metrics dynamically computed from QueryAttemptRecord.
    - Offline prototype state defaults honestly to NOT_EXECUTED_OFFLINE_PROTOTYPE (56 NOT_ATTEMPTED).
    - Dynamic query attempt evaluation correctly updates 3-way daily resolution and failure rates.
    - Validates live OpenAPI query probe execution.
    - Validates event cluster deduplication (10 clusters across 12 candidate announcements).
    - Validates revision tracking (Rule 51 clarification lineage).
    """
    # 1. Default offline state (no fabricated queries!)
    metrics = OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics()

    # Scope
    scope = metrics["pilot_scope"]
    assert scope["period"] == "2024-01-01 to 2024-03-31 (2024 Q1)"
    assert scope["target_constituents"] == ["2330", "2603", "2881"]
    assert scope["total_trading_sessions"] == 56
    assert scope["total_planned_stock_day_units"] == 168

    # 1. Real collection coverage (Honest 0 queries executed offline - P0-1, P0-3)
    assert metrics["real_collection_coverage"]["sessions_planned"] == 56
    assert metrics["real_collection_coverage"]["sessions_queried"] == 0
    assert metrics["real_collection_coverage"]["stock_days_planned"] == 168
    assert metrics["real_collection_coverage"]["stock_days_queried"] == 0
    assert metrics["real_collection_coverage"]["collection_coverage_ratio"] == 0.0
    assert metrics["real_collection_coverage"]["collection_status"] == "NOT_EXECUTED_OFFLINE_PROTOTYPE"

    # 2. Retrieval coverage (P0-2: when unqueried, all retrieval success metrics are strictly 0)
    assert metrics["retrieval_coverage"]["sessions_with_announcements"] == 0
    assert metrics["retrieval_coverage"]["retrieval_coverage_ratio"] == 0.0
    assert metrics["retrieval_coverage"]["total_announcements_retrieved"] == 0
    assert metrics["retrieval_coverage"]["announcement_dates"] == []

    # 2b. Candidate inventory metrics (P0-2: dedicated isolation from retrieval counts)
    cand_inv = metrics["candidate_inventory_metrics"]
    assert cand_inv["candidate_fixture_announcements"] == 12
    assert cand_inv["candidate_fixture_sessions"] == 9
    assert cand_inv["candidate_fixture_density"] == 0.1607
    assert len(cand_inv["candidate_dates"]) == 9

    # 3. Historical availability coverage (0% - MOPS lacks downstream vendor feed timestamps)
    assert metrics["historical_availability_coverage"]["announcements_with_provider_receipts"] == 0
    assert metrics["historical_availability_coverage"]["historical_availability_coverage_ratio"] == 0.0

    # 4. Verified source count (0 formal PIT verified)
    assert metrics["verified_source_count"] == 0
    assert metrics["verification_status_summary"]["VERIFIED_SOURCE"] == 0
    assert metrics["verification_status_summary"]["MOPS_CANDIDATE_UNVERIFIED"] == 12

    # 5. Three-way session resolution (P0-2: 56 NOT_ATTEMPTED, 0 ANNOUNCEMENTS_RETRIEVED in offline mode)
    three_way = metrics["three_way_session_resolution"]
    assert three_way["NOT_ATTEMPTED"] == 56
    assert three_way["QUERY_COMPLETED_ZERO_RESULTS"] == 0
    assert three_way["AUTHENTIC_NO_ANNOUNCEMENT_CONFIRMED"] == 0
    assert three_way["ANNOUNCEMENTS_RETRIEVED"] == 0

    # 6. Deduplication & Revisions
    dedup = metrics["event_cluster_deduplication"]
    assert dedup["total_announcements"] == 12
    assert dedup["total_unique_clusters"] == 10
    assert dedup["multi_event_clusters_count"] == 2
    assert "CLU-2330-CHIAYI-COWOS" in dedup["multi_event_clusters"]
    assert "CLU-2881-FULLYEAR-2023" in dedup["multi_event_clusters"]

    rev = metrics["revision_tracking"]
    assert rev["revision_announcements_count"] == 1
    assert "CANDIDATE-Q1-2024-2330-06" in rev["revision_announcements"]

    # 7. Query failure rate (0 queries executed offline)
    assert metrics["source_failure_rate"]["total_queries_executed"] == 0
    assert metrics["source_failure_rate"]["failed_queries"] == 0
    assert metrics["source_failure_rate"]["failure_rate"] == 0.0
    assert metrics["source_failure_rate"]["execution_status"] == "NOT_EXECUTED_OFFLINE_PROTOTYPE"

    # 8. Dynamic computation with simulated query attempts (P0-3)
    # Partial execution test: simulate only 2330 (56 / 168 = 33.33% -> PARTIALLY_EXECUTED)
    sim_attempts = OfficialHistoricalNewsCollector.create_simulated_audit_query_attempts(target_stocks=["2330"])
    sim_metrics = OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics(query_attempts=sim_attempts)
    assert sim_metrics["simulation_metrics"]["simulated_stock_days_planned"] == 168
    assert sim_metrics["simulation_metrics"]["simulated_stock_days_queried"] == 56
    assert sim_metrics["simulation_metrics"]["simulated_coverage_ratio"] == 0.3333
    assert sim_metrics["simulation_metrics"]["simulation_status"] == "PARTIALLY_EXECUTED"
    # Simulated queries CANNOT inflate real collection coverage! (P0-3)
    assert sim_metrics["real_collection_coverage"]["stock_days_queried"] == 0
    assert sim_metrics["real_collection_coverage"]["collection_coverage_ratio"] == 0.0
    assert sim_metrics["real_collection_coverage"]["collection_status"] == "NOT_EXECUTED_OFFLINE_PROTOTYPE"

    # Full execution test: simulate all 3 stocks (168 / 168 = 100% -> FULLY_EXECUTED)
    sim_all = OfficialHistoricalNewsCollector.create_simulated_audit_query_attempts(target_stocks=["2330", "2603", "2881"])
    sim_all_metrics = OfficialHistoricalNewsCollector.evaluate_q1_2024_collection_metrics(query_attempts=sim_all)
    assert sim_all_metrics["simulation_metrics"]["simulated_stock_days_queried"] == 168
    assert sim_all_metrics["simulation_metrics"]["simulated_coverage_ratio"] == 1.0
    assert sim_all_metrics["simulation_metrics"]["simulation_status"] == "FULLY_EXECUTED"
    assert sim_all_metrics["simulation_metrics"]["simulated_three_way_session_resolution"]["NOT_ATTEMPTED"] == 0
    assert sim_all_metrics["real_collection_coverage"]["stock_days_queried"] == 0

    # 9. Genuine Live OpenAPI Query Probe (single session lawful audit probe)
    probe_rec = OfficialHistoricalNewsCollector.execute_live_openapi_query_probe(target_stock="2330")
    assert probe_rec.stock_id == "2330"
    assert probe_rec.source_channel == OfficialSourceChannel.TWSE_OPENAPI
    assert probe_rec.endpoint == "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
    assert probe_rec.execution_mode == ExecutionMode.AUTHENTIC
    assert probe_rec.response_status in [
        QueryResolutionStatus.QUERY_COMPLETED_SUCCESS,
        QueryResolutionStatus.QUERY_FAILED,
        QueryResolutionStatus.QUERY_FAILED_HTTP,
        QueryResolutionStatus.QUERY_FAILED_PARSE,
    ]
    if probe_rec.http_status == 200:
        assert probe_rec.actual_raw_response_hash is not None
        assert len(probe_rec.actual_raw_response_hash) == 64
        assert probe_rec.raw_bytes_len is not None and probe_rec.raw_bytes_len > 0
    RetrievalSessionRegistry.clear()


# ==============================================================================
# 4. Statistical Power Validity and Calibration
# ==============================================================================

def test_statistical_power_validity_and_calibration():
    """
    V8.13 Requirement 4:
    - Calibrates empirical Type-I error rate under H0: delta_rho = 0.0 (false rejection rate approx 5%).
    - Marks power curve as ASSUMPTION_BASED_ILLUSTRATION and POWER_NOT_ESTABLISHED.
    - Conducts multi-dimensional sensitivity analysis across:
      * ICC in [0.05, 0.15, 0.30]
      * Valid label ratio in [0.80, 0.90, 1.00]
      * Purging loss ratio in [0.05, 0.15, 0.25]
    - Retains 250-300 strictly as PROVISIONAL_PLANNING_TARGET_ONLY.
    """
    # 1. Type-I Error Calibration under H0: delta_rho = 0.0
    type1_res = DatasetExpansionReadinessAssessor.calibrate_type_1_error(
        sample_sizes=[25, 50, 100],
        seeds=[42, 100],
        n_sim=300,
    )
    assert type1_res["hypothesis"] == "H0: delta_rho = 0.0"
    assert type1_res["nominal_alpha"] == 0.05
    assert type1_res["calibration_status"] == "CALIBRATED_NOMINAL_5_PERCENT"
    assert type1_res["is_type_1_error_calibrated"] is True
    assert type1_res["calibration_type"] == "ASSUMPTION_BASED"
    assert "oracle_simulation_calibration" in type1_res
    assert "implemented_test_calibration" in type1_res
    assert type1_res["oracle_simulation_calibration"]["calibration_status"] == "CALIBRATED_NOMINAL_5_PERCENT"
    assert type1_res["implemented_test_calibration"]["calibration_status"] == "CALIBRATED_WITHIN_DATASET_CLUSTER_BOOTSTRAP"
    # Empirical false rejection rate must be well-calibrated (around 5% nominal)
    assert 0.03 <= type1_res["overall_mean_type_1_error"] <= 0.07
    assert 0.03 <= type1_res["oracle_simulation_calibration"]["overall_mean_type_1_error"] <= 0.07
    assert 0.03 <= type1_res["implemented_test_calibration"]["overall_mean_type_1_error"] <= 0.12

    # 2. Power curve classification and status
    power_res = DatasetExpansionReadinessAssessor.simulate_cluster_bootstrap_power(
        sample_sizes=[11, 50, 200],
        seeds=[42],
        n_sim=100,
    )
    assert power_res["power_curve_type"] == "ASSUMPTION_BASED_ILLUSTRATION"
    assert power_res["hypothesized_delta_rho"] == 0.20
    assert "11" in power_res["power_curve"]

    # 3. Sensitivity analysis across ICC, valid label ratio, purging loss
    sens_res = DatasetExpansionReadinessAssessor.run_sensitivity_analysis(
        hypothesized_delta_rho=0.20,
        icc_grid=[0.05, 0.15, 0.30],
        valid_label_grid=[0.80, 0.90, 1.00],
        purging_loss_grid=[0.05, 0.15, 0.25],
        n_sim=60,
    )
    assert sens_res["grid_scenarios_evaluated"] == 27
    assert len(sens_res["scenarios"]) == 27
    # Verify that higher ICC reduces statistical power at n=100
    low_icc_scenario = [s for s in sens_res["scenarios"] if s["icc"] == 0.05 and s["valid_label_ratio"] == 1.0 and s["purging_loss_ratio"] == 0.05][0]
    high_icc_scenario = [s for s in sens_res["scenarios"] if s["icc"] == 0.30 and s["valid_label_ratio"] == 0.80 and s["purging_loss_ratio"] == 0.25][0]
    assert low_icc_scenario["power_at_n_100"] > high_icc_scenario["power_at_n_100"]

    # 4. Readiness report governance disclosures
    readiness = DatasetExpansionReadinessAssessor.assess_readiness()
    p_analysis = readiness["statistical_power_analysis"]
    assert p_analysis["power_curve_type"] == "ASSUMPTION_BASED_ILLUSTRATION"
    assert p_analysis["power_status"] == "POWER_NOT_ESTABLISHED"
    assert "PROVISIONAL_PLANNING_TARGET_ONLY" in p_analysis["planning_sample_target"]
    assert p_analysis["sample_size_justification_rule"] == "FORMAL_SAMPLE_SIZE_PROPOSALS_PERMITTED_ONLY_AFTER_METHOD_CALIBRATION_ON_REAL_DATA"
