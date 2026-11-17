"""
Tests for V8.12 Evidence-Backed Historical Dataset Expansion & Source Feasibility

Verifies:
1. Trusted Source Certification Preflight:
   - VERIFIED_SOURCE can ONLY be issued by a recognized external source verification authority.
   - Rejection of self-asserted or unauthorized certifier authority (fail closed).
   - Rejection of uncertified evidence bundles.
   - Rejection of forged announcement IDs (not in official exchange disclosure catalog).
   - Rejection of matching content with non-existent or untrusted external archive URLs.
   - Rejection of general RAW_PILOT_EVENTS attempting self-elevation via verification_status.
   - Zero verified sources in default baseline (100% unverified honest reporting).
2. Distinction Between Event Existence & Historical Availability Verification:
   - Official filing existence verification is strictly isolated from historical provider feed availability.
3. Dataset Expansion Readiness Assessment:
   - Comprehensive assessment of 50 pilot events across stocks, categories, months, missingness,
     clusters, revisions, walk-forward effective sizes, and statistical power requirements.
4. Legal Usage Rights and Licensing Audit:
   - Categorization into TAIWAN_GOVERNMENT_OPEN_DATA vs COMMERCIAL_FAIR_USE_RESEARCH_ONLY.
   - Zero commercial redistribution eligibility without publisher licensing agreements.
5. Baseline Evaluation Retention & Inconclusive Verdict Verification:
   - Genuine BGE 128D, 32D Metadata, Method B, B-matched, Method C.
   - Paired bootstrap 95% CIs cross zero -> formal status strictly INCONCLUSIVE.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from stock_rl.evaluation.importance import (
    TWSETradingCalendar,
    NewsSourceType,
    EvidenceAuditLevel,
)
from stock_rl.evaluation.semantic_eval import (
    compute_normalized_content_hash,
)
from stock_rl.evaluation.provenance import (
    load_trusted_pretrained_embedding_engine,
)
from stock_rl.evaluation.pilot_dataset import (
    PilotEventSample,
    PilotDatasetBuilder,
    PilotSourceManifest,
    SourceVerificationStatus,
    SupervisionTargetName,
    EventSourceRecord,
    AnnotationProvenance,
    MOPSSourceEvidence,
    ExternalSourceAdapterReceipt,
    TrustedExternalSourceAdapter,
    LicenseVerificationStatus,
    LicenseMetadata,
    validate_official_url,
    TrustedSourceCertifier,
    TrustedCertifierAuthority,
    RetrievalSessionRegistry,
    UsageRightsCategory,
    DatasetExpansionReadinessAssessor,
    RAW_PILOT_EVENTS,
)
from stock_rl.evaluation.pilot_experiment import (
    PilotExperimentRunner,
    PurgedWalkForwardEvaluator,
    compute_paired_differences_and_bootstrap,
)


@pytest.fixture(scope="module")
def pretrained_encoder():
    """Loads trusted pretrained semantic encoder locally."""
    return load_trusted_pretrained_embedding_engine(local_files_only=True)


# ==============================================================================
# 1. Trusted Source Certification Preflight & Rejection Tests
# ==============================================================================

def test_trusted_source_certification_preflight():
    """
    V8.12 Requirement 1 & Code Review P0:
    - Verifies that VERIFIED_SOURCE requires trusted external authority, authentic adapter receipt,
      strict urlparse validation, and local candidate catalog registration.
    - Tests fail-closed rejections for:
      1. Self-asserted or untrusted certifier authority
      2. Caller self-filling True booleans without external adapter receipt (caller self-attestation prohibited)
      3. Forged announcement ID (not found in local candidate catalog)
      4. Forged official URLs (non-https, invalid hostname, subdomain spoofing, invalid path)
      5. Cross-announcement evidence reuse (attaching evidence of one disclosure to another event)
      6. Company ID mismatch
      7. RAW_PILOT_EVENTS attempting self-elevation via verification_status
    - Verifies default manifest maintains strictly 0 VERIFIED_SOURCE (100% unverified honest reporting).
    """
    manifest = PilotSourceManifest.generate_manifest()
    assert len(manifest) == 50

    # Baseline: Default manifest has strictly 0 VERIFIED_SOURCE
    summary = PilotSourceManifest.get_provenance_summary()
    assert summary["VERIFIED_SOURCE"] == 0
    assert summary["MOPS_CANDIDATE_UNVERIFIED"] == 12
    assert summary["CURATED_UNVERIFIED"] == 38
    assert summary["total_unverified_records"] == 50

    ev_target = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-01"][0]
    ev_dt = pd.Timestamp(f"{ev_target['date_str']} {ev_target['pub_time_str']}", tz="Asia/Taipei")
    ev_hash = compute_normalized_content_hash(ev_target["headline"], ev_target.get("body_text", ""))

    # --- Rejection Case 1: Caller self-fills untrusted certifier authority ---
    fake_auth_receipt = TrustedExternalSourceAdapter.create_adapter_receipt(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
        raw_content=ev_target.get("body_text", ""),
    )
    fake_auth_evidence = MOPSSourceEvidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        payload_hash=ev_hash,
        certifier_authority="UNTRUSTED_SELF_ASSERTED",  # Unauthorized certifier!
        external_reference_url="https://mops.twse.com.tw/mops/web/t05st01",
        adapter_receipt=fake_auth_receipt,
        is_trusted_external_certified=True,
    )
    status, evidence_dict, notes = TrustedSourceCertifier.verify_evidence(ev_target, fake_auth_evidence)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert "is not a recognized trusted external source certifier" in notes

    # --- Rejection Case 2: Caller self-fills True booleans without external adapter receipt ---
    self_attested_evidence = MOPSSourceEvidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        payload_hash=ev_hash,
        certifier_authority="TWSE_MOPS_OFFICIAL_ARCHIVE",
        external_reference_url="https://mops.twse.com.tw/mops/web/t05st01",
        adapter_receipt=None,  # Prohibited self-attestation!
        is_trusted_external_certified=True,
        is_existence_verified=True,
    )
    status, evidence_dict, notes = TrustedSourceCertifier.verify_evidence(ev_target, self_attested_evidence)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert "caller self-attestation prohibited" in notes

    # --- Rejection Case 3: Forged announcement ID (not in local candidate catalog) ---
    forged_receipt = TrustedExternalSourceAdapter.create_adapter_receipt(
        announcement_id="MOPS-2330-20230606-FORGED-ID",
        company_id="2330",
        announcement_datetime=ev_dt,
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
        raw_content=ev_target.get("body_text", ""),
    )
    forged_id_evidence = MOPSSourceEvidence(
        announcement_id="MOPS-2330-20230606-FORGED-ID",  # Forged ID!
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        payload_hash=ev_hash,
        certifier_authority="TWSE_MOPS_OFFICIAL_ARCHIVE",
        external_reference_url="https://mops.twse.com.tw/mops/web/t05st01",
        adapter_receipt=forged_receipt,
        is_trusted_external_certified=True,
    )
    status, evidence_dict, notes = TrustedSourceCertifier.verify_evidence(ev_target, forged_id_evidence)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert "not found in local candidate catalog" in notes

    # --- Rejection Case 4: Forged official URLs tested via strict urlparse ---
    # 4a. Scheme must be https (http rejected)
    is_valid, err = validate_official_url("http://mops.twse.com.tw/mops/web/t05st01")
    assert not is_valid
    assert "URL scheme must be 'https'" in err

    # 4b. Invalid host rejected
    is_valid, err = validate_official_url("https://fake-mops.internal/mops/web/t05st01")
    assert not is_valid
    assert "is not in allowed official hosts" in err

    # 4c. Subdomain spoofing rejected (mops.twse.com.tw.attacker.com)
    is_valid, err = validate_official_url("https://mops.twse.com.tw.attacker.com/mops/web/t05st01")
    assert not is_valid
    assert "is not in allowed official hosts" in err

    # 4d. MOPS path must begin with /mops/
    is_valid, err = validate_official_url("https://mops.twse.com.tw/malicious/feed")
    assert not is_valid
    assert "must start with '/mops/'" in err

    untrusted_url_receipt = TrustedExternalSourceAdapter.create_adapter_receipt(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        endpoint_url="http://fake-mops.internal/archive/20230606",
        raw_content=ev_target.get("body_text", ""),
    )
    untrusted_url_evidence = MOPSSourceEvidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        payload_hash=ev_hash,
        certifier_authority="TWSE_MOPS_OFFICIAL_ARCHIVE",
        external_reference_url="http://fake-mops.internal/archive/20230606",  # Non-https & untrusted host!
        adapter_receipt=untrusted_url_receipt,
        is_trusted_external_certified=True,
    )
    status, evidence_dict, notes = TrustedSourceCertifier.verify_evidence(ev_target, untrusted_url_evidence)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert "URL validation failed" in notes

    # --- Rejection Case 5: Reusing evidence from another announcement ---
    ev_other = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-03"][0]
    valid_evidence_agm = TrustedExternalSourceAdapter.certify_with_live_retrieval_evidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        raw_http_response_bytes=ev_target.get("body_text", "").encode("utf-8"),
        retrieval_session_id="SESS-LIVE-20230606-001",
        headline=ev_target["headline"],
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
    )
    status, evidence_dict, notes = TrustedSourceCertifier.verify_evidence(ev_other, valid_evidence_agm)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert "Cross-announcement reuse detected" in notes

    # --- Rejection Case 6: Company ID mismatch ---
    mismatch_co_receipt = TrustedExternalSourceAdapter.create_authentic_receipt(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2881",
        announcement_datetime=ev_dt,
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
        raw_content=ev_target.get("body_text", ""),
    )
    mismatch_co_evidence = MOPSSourceEvidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2881",  # Mismatch company!
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        payload_hash=ev_hash,
        certifier_authority="TWSE_MOPS_OFFICIAL_ARCHIVE",
        external_reference_url="https://mops.twse.com.tw/mops/web/t05st01",
        adapter_receipt=mismatch_co_receipt,
        is_trusted_external_certified=True,
    )
    status, evidence_dict, notes = TrustedSourceCertifier.verify_evidence(ev_target, mismatch_co_evidence)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert "Company ID mismatch" in notes

    # --- Rejection Case 7: RAW_PILOT_EVENTS attempting self-elevation ---
    fake_raw_event = dict(ev_target)
    fake_raw_event["event_id"] = "EV-MEDIA-FAKE-ELEVATE-01"
    fake_raw_event["verification_status"] = "VERIFIED_SOURCE"  # Unauthorized self-elevation!
    
    test_manifest = PilotSourceManifest.generate_manifest(raw_events=[fake_raw_event])
    assert test_manifest["EV-MEDIA-FAKE-ELEVATE-01"].verification_status == SourceVerificationStatus.CURATED_UNVERIFIED
    assert "Self-elevation rejected" in test_manifest["EV-MEDIA-FAKE-ELEVATE-01"].notes

    # --- Rejection Case 8: Tampered raw response hash in adapter receipt ---
    tampered_receipt = TrustedExternalSourceAdapter.create_authentic_receipt(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        endpoint_url="https://mops.twse.com.tw/mops/web/t05st01",
        raw_content="Tampered raw response that differs from actual announcement body",
    )
    tampered_evidence = MOPSSourceEvidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        payload_hash=ev_hash,
        certifier_authority="TWSE_MOPS_OFFICIAL_ARCHIVE",
        external_reference_url="https://mops.twse.com.tw/mops/web/t05st01",
        adapter_receipt=tampered_receipt,
        is_trusted_external_certified=True,
    )
    status, evidence_dict, notes = TrustedSourceCertifier.verify_evidence(ev_target, tampered_evidence)
    assert status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
    assert "Adapter raw response hash mismatch" in notes


# ==============================================================================
# 2. Event Existence vs Historical Provider Availability Distinction
# ==============================================================================

def test_event_existence_vs_historical_availability_distinction():
    """
    V8.12 Requirement 1 & 2:
    - Confirms that official exchange filing existence verification is distinctly
      tracked from historical data provider availability.
    - Without authentic historical vendor feed logs, availability cannot be assumed.
    """
    ev_target = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-01"][0]
    ev_dt = pd.Timestamp(f"{ev_target['date_str']} {ev_target['pub_time_str']}", tz="Asia/Taipei")
    ev_hash = compute_normalized_content_hash(ev_target["headline"], ev_target.get("body_text", ""))

    target_endpoint = "https://mops.twse.com.tw/mops/web/t05st01"
    raw_bytes = ev_target.get("body_text", "").encode("utf-8")
    session_rec = RetrievalSessionRegistry.register_genuine_session(
        source_channel="TWSE_MOPS_HTML_ARCHIVE",
        endpoint_url=target_endpoint,
        request_start_time=pd.Timestamp.now(tz="Asia/Taipei"),
        response_end_time=pd.Timestamp.now(tz="Asia/Taipei"),
        http_status=200,
        raw_response_bytes=raw_bytes,
        parsed_summary={"announcement_count": 1},
        source_identity="OFFICIAL_MOPS_HTTPS_CLIENT",
        announcement_ids=["MOPS-2330-20230606-AGM"],
        allow_legacy_fixture=True,
    )

    valid_evidence = TrustedExternalSourceAdapter.certify_with_live_retrieval_evidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id="2330",
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        raw_http_response_bytes=raw_bytes,
        retrieval_session_id=session_rec.session_id,
        headline=ev_target["headline"],
        payload_hash=ev_hash,
        endpoint_url=target_endpoint,
        is_existence_verified=True,
        is_availability_verified=False,  # Segregated! No historical vendor feed receipt
        audit_notes="Authentic exchange filing verified against raw disclosure archive.",
    )

    manifest = PilotSourceManifest.generate_manifest(evidence_registry={"EV-2330-01": valid_evidence})
    rec = manifest["EV-2330-01"]
    assert rec.verification_status == SourceVerificationStatus.VERIFIED_SOURCE
    assert rec.verification_evidence["is_existence_verified"] is True
    assert rec.verification_evidence["is_availability_verified"] is False
    assert rec.verification_evidence["provider_available_at"] is None
    assert "Existence verified: True; Provider availability verified: False" in rec.notes
    RetrievalSessionRegistry.clear()


# ==============================================================================
# 3. Dataset Expansion Readiness Assessment
# ==============================================================================

def test_dataset_expansion_readiness_assessment():
    """
    V8.12 Requirement 3:
    - Verifies DatasetExpansionReadinessAssessor across all required audit dimensions:
      1. Sample counts by stock (2330, 2603, 2881) and category
      2. Monthly temporal distribution (2023-06 to 2024-03)
      3. Missing news ratios and temporal sparsity (~78.7% zero-event trading days)
      4. Verified vs unverified source proportions (100% unverified in default baseline)
      5. Valid supervision target counts (Human=50, Market=45, Hybrid=45)
      6. Cluster deduplication and revision tracking
      7. Purged Walk-Forward effective sample sizes
      8. Statistical power analysis for dataset expansion
    """
    readiness = DatasetExpansionReadinessAssessor.assess_readiness()

    # 1. Sample totals & stock breakdown
    assert readiness["total_samples"] == 50
    assert readiness["stock_breakdown"] == {"2330": 20, "2603": 16, "2881": 14}
    assert sum(readiness["category_breakdown"].values()) == 50
    assert readiness["category_breakdown"]["earnings"] == 21
    assert readiness["category_breakdown"]["routine"] == 9
    assert readiness["category_breakdown"]["capex"] == 8

    # 2. Monthly distribution (10 months)
    assert len(readiness["monthly_distribution"]) == 10
    assert list(readiness["monthly_distribution"].keys())[0] == "2023-06"
    assert list(readiness["monthly_distribution"].keys())[-1] == "2024-03"

    # 3. Temporal sparsity & missing crawl logs
    missing = readiness["missing_news_ratios"]
    assert missing["total_trading_days"] == 202
    assert missing["distinct_event_days"] == 43
    assert missing["distinct_event_days_in_fixture"] == 43
    assert missing["no_recorded_event_days_in_curated_fixture"] == 159
    assert missing["no_recorded_event_days_ratio"] == 0.7871
    assert missing["zero_event_trading_days"] == 159
    assert missing["zero_event_trading_days_ratio"] == 0.7871
    assert missing["missing_crawl_log_count"] == 50
    assert missing["missing_crawl_log_ratio"] == 1.0
    assert missing["unverified_provider_available_count"] == 50
    assert missing["collection_coverage"]["status"] == "NO_LIVE_CRAWLER_DEPLOYED"
    assert missing["retrieval_coverage_by_source"]["MOPS"]["candidate_event_count"] == 12
    assert missing["retrieval_coverage_by_source"]["MOPS"]["verified_retrieval_count"] == 0
    assert missing["retrieval_coverage_by_source"]["COMMERCIAL_MEDIA"]["curated_event_count"] == 38
    assert missing["fixture_event_density"] == 0.2129
    assert missing["model_input_slot_occupancy"]["recent_event_slot_capacity"] == 32
    assert missing["model_input_slot_occupancy"]["active_event_slot_capacity"] == 8
    assert "Absence of news records in curated pilot fixture does NOT indicate" in missing["interpretation_warning"]

    # 4. Source verification summary (Default: 0 verified)
    v_summary = readiness["source_verification_summary"]
    assert v_summary["VERIFIED_SOURCE"] == 0
    assert v_summary["MOPS_CANDIDATE_UNVERIFIED"] == 12
    assert v_summary["CURATED_UNVERIFIED"] == 38
    assert v_summary["total_unverified_records"] == 50
    assert v_summary["unverified_ratio"] == 1.0

    # 5. Supervision targets
    tgt = readiness["supervision_target_counts"]
    assert tgt["HUMAN_IMPORTANCE"]["valid"] == 50
    assert tgt["HUMAN_IMPORTANCE"]["missing"] == 0
    assert tgt["EX_POST_MARKET_IMPACT"]["valid"] == 45
    assert tgt["EX_POST_MARKET_IMPACT"]["missing"] == 5

    # 6. Cluster deduplication and revision tracking
    clu = readiness["cluster_deduplication_and_revision_tracking"]
    assert clu["total_unique_clusters"] == 45
    assert clu["clusters_with_multiple_events"] == 4
    assert clu["revision_events_count"] == 1
    assert "EV-2330-10-REV2" in clu["revision_event_ids"]
    assert clu["exact_hash_duplicates"] == 0

    # 7. Walk-Forward effective sizes & overlap disclosure
    wf = readiness["purged_walk_forward_effective_sizes"]
    assert wf["num_windows"] == 2
    assert wf["windows"][0]["n_test"] == 11
    assert wf["windows"][1]["n_test"] == 11
    assert wf["temporal_overlap_disclosed"] is True

    # 8. Statistical power analysis (Simulation-based cluster bootstrap)
    power = readiness["statistical_power_analysis"]
    assert power["current_test_sample_size"] == 11
    assert power["is_statistically_underpowered"] is True
    assert power["power_status"] == "POWER_NOT_ESTABLISHED"
    assert "250-300" in power["planning_sample_target"]
    assert "PLANNING_TARGET_ONLY" in power["planning_sample_target"]
    assert power["statistically_justified_minimum"] == "NOT_ESTABLISHED_DUE_TO_PILOT_UNDERPOWERING"
    assert power["simulation_method"] == "CLUSTER_BOOTSTRAP_CORRELATED_SPEARMAN_SIMULATION"
    assert "power_curve" in power
    assert power["power_curve"]["11"]["simulated_power"] < 0.20
    assert power["power_curve"]["200"]["simulated_power"] > 0.40


# ==============================================================================
# 4. Legal Usage Rights and Licensing Audit
# ==============================================================================

def test_usage_rights_and_licensing_audit():
    """
    V8.12 Requirement 2 & 5 & Code Review P1:
    - Verifies that all 50 events in manifest are assigned formal evidence-based LicenseMetadata.
    - Decouples data source identity from actual license grant.
    - MOPS public availability does not grant commercial model training or raw redistribution.
    - Commercial news articles lack formal publisher licenses (LICENSE_UNVERIFIED).
    - Fair use research eligibility is strictly NEEDS_LEGAL_REVIEW.
    - Strictly 0 events are eligible for commercial redistribution.
    """
    manifest = PilotSourceManifest.generate_manifest()
    assert len(manifest) == 50

    mops_count = 0
    media_count = 0
    for eid, rec in manifest.items():
        assert isinstance(rec.usage_rights, UsageRightsCategory)
        assert rec.license_metadata is not None
        assert rec.commercial_redistribution_eligible is False
        if rec.source_name == "MOPS":
            assert rec.usage_rights == UsageRightsCategory.TAIWAN_GOVERNMENT_OPEN_DATA
            assert rec.license_metadata.verification_status == LicenseVerificationStatus.NEEDS_LEGAL_REVIEW
            assert "Securities and Exchange Act Art. 36" in rec.license_metadata.audit_notes
            mops_count += 1
        else:
            assert rec.usage_rights == UsageRightsCategory.COMMERCIAL_FAIR_USE_RESEARCH_ONLY
            assert rec.license_metadata.verification_status == LicenseVerificationStatus.LICENSE_UNVERIFIED
            media_count += 1

    assert mops_count == 12
    assert media_count == 38

    summary = PilotSourceManifest.get_provenance_summary()
    assert summary["commercial_redistribution_eligible"] is False
    assert summary["commercial_redistribution_eligible_count"] == 0
    assert summary["usage_rights_counts"]["TAIWAN_GOVERNMENT_OPEN_DATA"] == 12
    assert summary["usage_rights_counts"]["COMMERCIAL_FAIR_USE_RESEARCH_ONLY"] == 38
    assert summary["license_verification_status_counts"]["NEEDS_LEGAL_REVIEW"] == 12
    assert summary["license_verification_status_counts"]["LICENSE_UNVERIFIED"] == 38
    assert summary["license_verification_status_counts"]["LICENSE_VERIFIED"] == 0


# ==============================================================================
# 5. Baseline Evaluation Retention & Inconclusive Verdict
# ==============================================================================

def test_baseline_evaluation_retention_and_inconclusive_verdict(pretrained_encoder):
    """
    V8.12 Requirement 4:
    - Retains Genuine BGE 128D, 32D Metadata, Method B, B-matched, Method C.
    - Retains 20:00 cutoff and purged chronological splitting.
    - Verifies that paired bootstrap 95% CIs cross zero across all targets,
      and formal status remains strictly INCONCLUSIVE with exploratory positive trend noted.
    """
    builder = PilotDatasetBuilder(encoder=pretrained_encoder)
    samples = builder.build_dataset()

    res = PilotExperimentRunner.evaluate_all_supervision_targets(samples=samples)
    summary_table = res["multi_target_comparison_summary"]
    assert len(summary_table) == 3

    human_row = [r for r in summary_table if r["target"] == "HUMAN_IMPORTANCE"][0]
    market_row = [r for r in summary_table if r["target"] == "EX_POST_MARKET_IMPACT"][0]
    hybrid_row = [r for r in summary_table if r["target"] == "HYBRID_IMPORTANCE"][0]

    # All three targets remain formally INCONCLUSIVE due to CI crossing zero on N=11
    assert human_row["formal_incremental_status"] == "INCONCLUSIVE"
    assert market_row["formal_incremental_status"] == "INCONCLUSIVE"
    assert hybrid_row["formal_incremental_status"] == "INCONCLUSIVE"

    # Human importance retains exploratory positive point trend
    assert human_row["exploratory_descriptor"] == "EXPLORATORY_POSITIVE_TREND"
    assert human_row["delta_rho_c_vs_bm"] > 0.20
    assert human_row["delta_ndcg3_c_vs_bm"] > 0.05
    assert 0.70 <= human_row["bootstrap_sample_positive_proportion"] <= 0.95

    # Market impact retains inconclusive no trend
    assert market_row["exploratory_descriptor"] == "INCONCLUSIVE_NO_TREND"
    assert market_row["delta_rho_c_vs_bm"] < 0.0

    # Ensure samples track usage rights
    for s in samples:
        assert isinstance(s.usage_rights, UsageRightsCategory)
