"""
Tests for V8.11 Historical Evidence Expansion & Supervised Learning Validity

Verifies:
1. Historical Provenance Corrections:
   - Temporal separation: source_published_at, provider_available_at, retrieved_at (None for unrecorded crawls), annotation_created_at.
   - AnnotationProvenance tracking: rubric description, annotation method, time status, and hindsight isolation barrier.
   - Fails closed if annotation time is unknown or accessed as historical decision feature.
2. Historical News Source Verification:
   - Categorical breakdown: 0 VERIFIED_SOURCE, 12 MOPS_CANDIDATE_UNVERIFIED, 38 CURATED_UNVERIFIED, 0 SYNTHETIC_FIXTURE.
   - Mere presence in internal MOPS candidate dictionary does NOT pass formal verification.
   - Evidence-based verification requires authentic MOPSSourceEvidence bundle (announcement ID, company, datetime, raw content, hash match).
   - Distinction between event existence verification and historical provider availability verification.
3. Multi-Target Supervision Quality Contrast:
   - Method B, Method B-matched, Method C across HUMAN_IMPORTANCE, EX_POST_MARKET_IMPACT, HYBRID_IMPORTANCE.
   - Formal status is strictly INCONCLUSIVE when 95% bootstrap CI spans zero on N=11 test events.
   - Exploratory positive trend noted on Human Importance, while Market Impact is dominated by noise.
   - Bootstrap sample positive proportion is documented as empirical resample proportion, not Bayesian posterior.
4. Purged Walk-Forward Evaluation:
   - Multi-window chronological evaluation (Window 1 and Window 2).
   - Zero cluster leakage, strict label embargo, and >= 3 trading session purging gaps.
   - Explicit disclosure of temporal overlap between Window 1 Test and Window 2 Validation.
5. Paired Differences and Bootstrap Uncertainty:
   - Sample-level paired difference metrics (MAE reduction, prediction difference).
   - 1,000-resample bootstrap 95% confidence intervals on Delta rho and Delta NDCG@3.
   - Evidence-based determination under small-sample uncertainty.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from stock_rl.evaluation.importance import (
    TWSETradingCalendar,
    RawNewsArticle,
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
    AnnotationMethod,
    AnnotationTimeStatus,
    MOPSSourceEvidence,
    RAW_PILOT_EVENTS,
)
from stock_rl.evaluation.pilot_experiment import (
    ChronologicalEventGroupedSplitter,
    SupervisedPilotTrainer,
    PilotExperimentRunner,
    PurgedWalkForwardEvaluator,
    compute_paired_differences_and_bootstrap,
    compute_split_metrics,
)


@pytest.fixture(scope="module")
def pretrained_encoder():
    """Loads trusted pretrained semantic encoder locally."""
    return load_trusted_pretrained_embedding_engine(local_files_only=True)


# ==============================================================================
# 1. Historical Provenance Corrections & Hindsight Isolation
# ==============================================================================

def test_provenance_separation_and_hindsight_isolation(pretrained_encoder):
    """
    V8.11 Requirement 1 & P0-2:
    - Verifies distinct temporal timestamps: published_at, available_at, retrieved_at, annotation_created_at.
    - Confirms unrecorded historical crawls set retrieved_at = None (no hardcoded 2024-04-01).
    - Verifies AnnotationProvenance tracks rubric, annotation method, and enforces hindsight isolation.
    - Confirms annotation_created_at is None and time_status is ANNOTATION_TIME_UNKNOWN for unrecorded logs.
    - Tests hindsight isolation barrier raises RuntimeError for:
      a) unrecorded annotation timestamp (ANNOTATION_TIME_UNKNOWN)
      b) future/ex-post timestamp relative to decision time
      c) post-facto annotations even if artificially backdated.
    """
    builder = PilotDatasetBuilder(encoder=pretrained_encoder)
    samples = builder.build_dataset()
    assert len(samples) == 50

    for s in samples:
        # Check source timestamps
        assert isinstance(s.source_published_at, pd.Timestamp)
        assert isinstance(s.provider_available_at, pd.Timestamp)
        assert s.provider_available_at >= s.source_published_at

        # Confirms no hardcoded retrieved_at; unrecorded historical crawls are strictly None
        assert s.retrieved_at is None

        # Check annotation provenance
        ann = s.annotation_provenance
        assert isinstance(ann, AnnotationProvenance)
        assert ann.is_ex_post_annotation is True
        assert ann.hindsight_isolated is True
        assert len(ann.rubric_description) > 0
        assert ann.annotation_created_at is None
        assert ann.time_status == AnnotationTimeStatus.ANNOTATION_TIME_UNKNOWN
        assert ann.annotation_method == AnnotationMethod.HEURISTIC_ASSIGNED

        # Verify hindsight isolation barrier raises exception if queried at decision time
        with pytest.raises(RuntimeError, match="Hindsight isolation barrier: Annotation creation time is unknown"):
            ann.verify_hindsight_isolation(s.decision_timestamp)

    # Test case b: future timestamp relative to decision time
    future_ann = AnnotationProvenance(
        annotation_created_at=pd.Timestamp("2025-01-01 10:00:00", tz="Asia/Taipei"),
        time_status=AnnotationTimeStatus.VERIFIED_TIMESTAMP,
        annotation_method=AnnotationMethod.MANUAL_EXPERT_ANNOTATED,
        is_ex_post_annotation=True,
    )
    decision_t = pd.Timestamp("2024-01-01 13:30:00", tz="Asia/Taipei")
    with pytest.raises(RuntimeError, match="cannot be accessed as a historical feature"):
        future_ann.verify_hindsight_isolation(decision_t)

    # Test case c: backdated timestamp but post-facto / heuristic
    backdated_ann = AnnotationProvenance(
        annotation_created_at=pd.Timestamp("2023-01-01 10:00:00", tz="Asia/Taipei"),
        time_status=AnnotationTimeStatus.VERIFIED_TIMESTAMP,
        annotation_method=AnnotationMethod.HEURISTIC_ASSIGNED,
        is_ex_post_annotation=True,
    )
    with pytest.raises(RuntimeError, match="cannot enter historical decision State"):
        backdated_ann.verify_hindsight_isolation(decision_t)


# ==============================================================================
# 2. Historical News Source Verification Coverage
# ==============================================================================

def test_historical_source_verification_coverage():
    """
    V8.11 Requirement 2 & P0-1:
    - Default manifest: VERIFIED_SOURCE is 0, MOPS_CANDIDATE_UNVERIFIED is 12, CURATED_UNVERIFIED is 38.
    - Confirms zero fabricated URLs, announcement numbers, or synthetic logs.
    - Confirms exact SHA-256 payload hash integrity across all 50 events.
    - Asserts that mere presence in MOPS_CANDIDATE_EVENTS dictionary does NOT pass verification.
    - Tests audit_and_verify_source_evidence: valid evidence bundle promotes candidate to VERIFIED_SOURCE,
      while corrupted or mismatched evidence fails closed to MOPS_CANDIDATE_UNVERIFIED.
    - Confirms distinction between event existence verification and historical provider availability verification.
    """
    manifest = PilotSourceManifest.generate_manifest()
    assert len(manifest) == 50

    summary = PilotSourceManifest.get_provenance_summary()
    assert summary["total_records"] == 50
    assert summary["VERIFIED_SOURCE"] == 0
    assert summary["MOPS_CANDIDATE_UNVERIFIED"] == 12
    assert summary["CURATED_UNVERIFIED"] == 38
    assert summary["total_unverified_records"] == 50
    assert summary["SYNTHETIC_FIXTURE"] == 0
    assert summary["missing_records"] == 0
    assert summary["records_with_external_url"] == 0
    assert summary["records_with_valid_hash"] == 50
    assert summary["records_with_retrieval_log"] == 0

    candidate_ids = [
        "EV-2330-01", "EV-2881-01", "EV-2603-01", "EV-2330-03", "EV-2330-ADD-03",
        "EV-2330-07", "EV-2881-06", "EV-2603-ADD-03", "EV-2603-08", "EV-2603-09",
        "EV-2330-10", "EV-2330-10-REV2"
    ]
    # Check default status of candidates without evidence: MOPS_CANDIDATE_UNVERIFIED (NOT VERIFIED_SOURCE)
    for cid in candidate_ids:
        rec = manifest[cid]
        assert rec.verification_status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED
        assert "unverified without raw evidence" in rec.notes

    # Verify unverified curated events are strictly CURATED_UNVERIFIED
    for eid, rec in manifest.items():
        if eid not in candidate_ids:
            assert rec.verification_status == SourceVerificationStatus.CURATED_UNVERIFIED
        assert rec.url is None or rec.url == ""
        assert len(rec.payload_hash) == 64
        ev = [e for e in RAW_PILOT_EVENTS if e["event_id"] == eid][0]
        recomputed = compute_normalized_content_hash(ev["headline"], ev.get("body_text", ""))
        assert rec.payload_hash == recomputed

    # Test Evidence-Based Verification: construct valid MOPSSourceEvidence for EV-2330-01
    ev_target = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-01"][0]
    ev_dt = pd.Timestamp(f"{ev_target['date_str']} {ev_target['pub_time_str']}", tz="Asia/Taipei")
    ev_hash = compute_normalized_content_hash(ev_target["headline"], ev_target.get("body_text", ""))

    valid_evidence = MOPSSourceEvidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id=ev_target["stock_id"],
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        payload_hash=ev_hash,
        is_existence_verified=True,
        is_availability_verified=False,  # Distinguishes existence vs historical provider availability!
        audit_notes="Authentic exchange filing verified against raw disclosure archive.",
    )

    # Manifest with verified evidence bundle
    custom_manifest = PilotSourceManifest.generate_manifest(evidence_registry={"EV-2330-01": valid_evidence})
    assert custom_manifest["EV-2330-01"].verification_status == SourceVerificationStatus.VERIFIED_SOURCE
    assert "Authentic exchange filing" in custom_manifest["EV-2330-01"].notes
    assert custom_manifest["EV-2330-01"].verification_evidence["existence_verified"] is True
    assert custom_manifest["EV-2330-01"].verification_evidence["availability_verified"] is False

    # Summary with evidence registry
    summary_with_evidence = PilotSourceManifest.get_provenance_summary(evidence_registry={"EV-2330-01": valid_evidence})
    assert summary_with_evidence["VERIFIED_SOURCE"] == 1
    assert summary_with_evidence["MOPS_CANDIDATE_UNVERIFIED"] == 11
    assert summary_with_evidence["total_unverified_records"] == 49

    # Fail closed on corrupted evidence (wrong payload hash)
    corrupted_evidence = MOPSSourceEvidence(
        announcement_id="MOPS-2330-20230606-AGM",
        company_id=ev_target["stock_id"],
        announcement_datetime=ev_dt,
        raw_content=ev_target.get("body_text", ""),
        payload_hash="corrupted_hash_value_that_does_not_match_payload_hash_at_all!!",
        is_existence_verified=True,
    )
    fail_manifest = PilotSourceManifest.generate_manifest(evidence_registry={"EV-2330-01": corrupted_evidence})
    assert fail_manifest["EV-2330-01"].verification_status == SourceVerificationStatus.MOPS_CANDIDATE_UNVERIFIED


# ==============================================================================
# 3. Multi-Target Supervision Quality Contrast
# ==============================================================================

def test_multi_target_supervision_quality_contrast(pretrained_encoder):
    """
    V8.11 Requirement 3 & P1-3:
    - Compares Method B, Method B-matched, Method C across all 3 targets.
    - Rigorous evidence-based research verdicts:
      When 95% bootstrap CI spans zero on N=11 test events, formal status is strictly INCONCLUSIVE.
    - Contrasts exploratory positive trend on HUMAN_IMPORTANCE vs noise on EX_POST_MARKET_IMPACT.
    - Strictly avoids claiming market prediction from hybrid scores alone.
    - Bootstrap sample positive proportion is documented as empirical resample proportion,
      never as Bayesian posterior probability.
    """
    builder = PilotDatasetBuilder(encoder=pretrained_encoder)
    samples = builder.build_dataset()

    res = PilotExperimentRunner.evaluate_all_supervision_targets(samples=samples)
    summary_table = res["multi_target_comparison_summary"]
    assert len(summary_table) == 3

    human_row = [r for r in summary_table if r["target"] == "HUMAN_IMPORTANCE"][0]
    market_row = [r for r in summary_table if r["target"] == "EX_POST_MARKET_IMPACT"][0]
    hybrid_row = [r for r in summary_table if r["target"] == "HYBRID_IMPORTANCE"][0]

    # 1. Human Importance: 95% CI spans zero on N=11 -> formal status is strictly INCONCLUSIVE
    assert human_row["formal_incremental_status"] == "INCONCLUSIVE"
    assert human_row["exploratory_descriptor"] == "EXPLORATORY_POSITIVE_TREND"
    assert human_row["delta_rho_c_vs_bm"] > 0.20
    assert human_row["delta_ndcg3_c_vs_bm"] > 0.05
    assert "INCONCLUSIVE" in human_row["empirical_claim"]
    assert "exploratory positive point trend" in human_row["empirical_claim"]

    # 2. Market Impact on N=11 is strictly inconclusive due to realized return noise
    assert market_row["formal_incremental_status"] == "INCONCLUSIVE"
    assert market_row["exploratory_descriptor"] == "INCONCLUSIVE_NO_TREND"
    assert "INCONCLUSIVE" in market_row["empirical_claim"]

    # 3. Hybrid Importance is also formally inconclusive
    assert hybrid_row["formal_incremental_status"] == "INCONCLUSIVE"

    # 4. Bootstrap interpretation note
    for row in summary_table:
        assert "NOT a Bayesian posterior probability" in row["bootstrap_interpretation_note"]
        assert 0.0 <= row["bootstrap_sample_positive_proportion"] <= 1.0

    # 5. Overall conclusion reflects caution
    assert "strictly inconclusive" in res["overall_conclusion"].lower() or "inconclusive" in res["overall_conclusion"].lower()


# ==============================================================================
# 4. Purged Walk-Forward Evaluation
# ==============================================================================

def test_purged_walk_forward_evaluation(pretrained_encoder):
    """
    V8.11 Requirement 4 & P1-4:
    - Multi-window chronological walk-forward validation (Window 1 & Window 2).
    - Enforces cluster grouping, label embargo, and >= 3 trading session purging gaps.
    - Explicit disclosure of temporal overlap between Window 1 Test and Window 2 Validation.
    - Confirms W1 and W2 are rolling retraining steps, NOT independent sample draws.
    """
    builder = PilotDatasetBuilder(encoder=pretrained_encoder)
    samples = builder.build_dataset(target_name="HUMAN_IMPORTANCE")

    evaluator = PurgedWalkForwardEvaluator(samples=samples, target_name="HUMAN_IMPORTANCE")
    wf_res = evaluator.evaluate_all_windows()

    assert wf_res["num_windows"] == 2
    w1, w2 = wf_res["windows"]

    # Both windows evaluate 11 out-of-sample events
    assert w1["n_test"] == 11
    assert w2["n_test"] == 11

    # Positive semantic incremental value across windows
    assert w1["delta_spearman_rho"] > 0.0
    assert w2["delta_spearman_rho"] > 0.20
    assert wf_res["average_delta_spearman_rho"] > 0.15
    assert wf_res["walk_forward_verdict"] == "EXPLORATORY_ROLLING_POSITIVE"

    # Temporal overlap disclosure (P1-4)
    disc = wf_res["temporal_overlap_disclosure"]
    assert disc["is_independent_sampling_experiment"] is False
    assert disc["evaluation_paradigm"] == "ROLLING_RETRAINING_CHRONOLOGICAL_SIMULATION"
    assert "Window 1 Test" in disc["overlap_description"]
    assert "Window 2 Validation" in disc["overlap_description"]
    assert "NOT independent sample draws" in disc["overlap_description"]
    assert len(disc["windows_specification"]) == 2


# ==============================================================================
# 5. Paired Differences and Bootstrap Uncertainty
# ==============================================================================

def test_paired_differences_and_bootstrap_uncertainty():
    """
    V8.11 Requirement 5:
    - Tests sample-level paired differences (MAE reductions, prediction differences).
    - Executes 1,000 bootstrap resamples for 95% confidence intervals on Delta rho and Delta NDCG@3.
    - Verifies empirical determination and small-sample uncertainty boundaries.
    - Verifies bootstrap_sample_positive_proportion interpretation.
    """
    rng = np.random.default_rng(42)
    n = 11
    y_true = np.array([0.9, 0.85, 0.8, 0.65, 0.6, 0.5, 0.45, 0.3, 0.25, 0.2, 0.1], dtype=np.float32)
    pred_c = y_true + rng.normal(0, 0.05, size=n)
    pred_base = y_true + rng.normal(0, 0.25, size=n)

    boot_res = compute_paired_differences_and_bootstrap(
        y_true=y_true,
        y_pred_c=pred_c,
        y_pred_baseline=pred_base,
        n_bootstrap=1000,
        seed=42,
    )

    assert boot_res["sample_size"] == 11
    assert boot_res["n_bootstrap"] == 1000
    assert len(boot_res["sample_paired_mae_reduction"]) == 11
    assert len(boot_res["sample_paired_pred_diff"]) == 11

    # Check 95% CIs
    ci_rho = boot_res["ci_delta_rho_95"]
    ci_ndcg = boot_res["ci_delta_ndcg3_95"]
    assert len(ci_rho) == 2 and ci_rho[0] <= ci_rho[1]
    assert len(ci_ndcg) == 2 and ci_ndcg[0] <= ci_ndcg[1]

    # Probabilities & resample proportions
    assert 0.0 <= boot_res["p_delta_rho_positive"] <= 1.0
    assert 0.0 <= boot_res["p_delta_ndcg3_positive"] <= 1.0
    assert boot_res["incremental_value_verdict"] in ["SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"]
    assert boot_res["formal_status"] in ["SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"]
    assert boot_res["exploratory_descriptor"] in [
        "STATISTICALLY_SIGNIFICANT",
        "STATISTICALLY_SIGNIFICANT_NEGATIVE",
        "EXPLORATORY_POSITIVE_TREND",
        "INCONCLUSIVE_NO_TREND",
    ]
    assert "NOT a Bayesian posterior probability" in boot_res["bootstrap_interpretation_note"]
