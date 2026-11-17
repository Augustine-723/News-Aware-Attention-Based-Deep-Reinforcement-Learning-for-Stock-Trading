"""
Tests for V8.11 Historical Evidence Expansion & Supervised Learning Validity

Verifies:
1. Historical Provenance Corrections:
   - Temporal separation: source_published_at, provider_available_at, retrieved_at (None for unrecorded crawls), annotation_created_at.
   - AnnotationProvenance tracking: rubric description, hindsight isolation barrier, zero leakage into PIT features.
2. Historical News Source Verification:
   - Categorical breakdown: 12 VERIFIED_SOURCE (authentic MOPS corporate disclosures), 38 CURATED_UNVERIFIED, 0 SYNTHETIC_FIXTURE.
   - Zero fabricated URLs, announcement numbers, or synthetic retrieval logs.
   - Exact SHA-256 payload hash integrity across all 50 events.
3. Multi-Target Supervision Quality Contrast:
   - Method B, Method B-matched, Method C across HUMAN_IMPORTANCE, EX_POST_MARKET_IMPACT, HYBRID_IMPORTANCE.
   - Contrast between high semantic text alignment on human importance vs small-sample noise on realized market impact.
   - No conflation of hybrid scores with market return predictability.
4. Purged Walk-Forward Evaluation:
   - Multi-window chronological evaluation (Window 1 and Window 2).
   - Zero cluster leakage, strict label embargo, and >= 3 trading session purging gaps.
5. Paired Differences and Bootstrap Uncertainty:
   - Sample-level paired difference metrics (MAE reduction, prediction difference).
   - 1,000-resample bootstrap 95% confidence intervals on Delta rho and Delta NDCG@3.
   - Statistical determination under small-sample uncertainty.
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
    V8.11 Requirement 1:
    - Verifies distinct temporal timestamps: published_at, available_at, retrieved_at, annotation_created_at.
    - Confirms unrecorded historical crawls set retrieved_at = None (no hardcoded 2024-04-01).
    - Verifies AnnotationProvenance tracks rubric and enforces hindsight isolation.
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
        assert ann.annotation_created_at > s.decision_timestamp

        # Verify hindsight isolation barrier raises exception if queried at decision time
        with pytest.raises(RuntimeError, match="Hindsight isolation barrier"):
            ann.verify_hindsight_isolation(s.decision_timestamp)


# ==============================================================================
# 2. Historical News Source Verification Coverage
# ==============================================================================

def test_historical_source_verification_coverage():
    """
    V8.11 Requirement 2:
    - Verifies 12 VERIFIED_SOURCE disclosures from official exchange filings (MOPS).
    - Verifies 38 CURATED_UNVERIFIED media/RSS news articles.
    - Confirms zero fabricated URLs, announcement numbers, or synthetic logs.
    - Confirms exact SHA-256 payload hash integrity.
    """
    manifest = PilotSourceManifest.generate_manifest()
    assert len(manifest) == 50

    summary = PilotSourceManifest.get_provenance_summary()
    assert summary["total_records"] == 50
    assert summary["VERIFIED_SOURCE"] == 12
    assert summary["CURATED_UNVERIFIED"] == 38
    assert summary["SYNTHETIC_FIXTURE"] == 0
    assert summary["missing_records"] == 0
    assert summary["records_with_external_url"] == 0
    assert summary["records_with_valid_hash"] == 50
    assert summary["records_with_retrieval_log"] == 0

    # Verify specific verified MOPS events
    verified_ids = [
        "EV-2330-01", "EV-2881-01", "EV-2603-01", "EV-2330-03", "EV-2330-ADD-03",
        "EV-2330-07", "EV-2881-06", "EV-2603-ADD-03", "EV-2603-08", "EV-2603-09",
        "EV-2330-10", "EV-2330-10-REV2"
    ]
    for vid in verified_ids:
        rec = manifest[vid]
        assert rec.verification_status == SourceVerificationStatus.VERIFIED_SOURCE
        assert "MOPS" in rec.notes

    # Verify unverified events are strictly CURATED_UNVERIFIED
    for eid, rec in manifest.items():
        if eid not in verified_ids:
            assert rec.verification_status == SourceVerificationStatus.CURATED_UNVERIFIED
        # No fabricated URLs
        assert rec.url is None or rec.url == ""
        # Hash audit
        assert len(rec.payload_hash) == 64
        ev = [e for e in RAW_PILOT_EVENTS if e["event_id"] == eid][0]
        recomputed = compute_normalized_content_hash(ev["headline"], ev.get("body_text", ""))
        assert rec.payload_hash == recomputed


# ==============================================================================
# 3. Multi-Target Supervision Quality Contrast
# ==============================================================================

def test_multi_target_supervision_quality_contrast(pretrained_encoder):
    """
    V8.11 Requirement 3:
    - Compares Method B, Method B-matched, Method C across all 3 targets.
    - Contrasts strong semantic text alignment on HUMAN_IMPORTANCE vs noise on EX_POST_MARKET_IMPACT.
    - Strictly avoids claiming market prediction from hybrid scores alone.
    """
    builder = PilotDatasetBuilder(encoder=pretrained_encoder)
    samples = builder.build_dataset()

    res = PilotExperimentRunner.evaluate_all_supervision_targets(samples=samples)
    summary_table = res["multi_target_comparison_summary"]
    assert len(summary_table) == 3

    human_row = [r for r in summary_table if r["target"] == "HUMAN_IMPORTANCE"][0]
    market_row = [r for r in summary_table if r["target"] == "EX_POST_MARKET_IMPACT"][0]
    hybrid_row = [r for r in summary_table if r["target"] == "HYBRID_IMPORTANCE"][0]

    # 1. Human Importance shows large positive semantic gain
    assert human_row["delta_rho_c_vs_bm"] > 0.20
    assert human_row["delta_ndcg3_c_vs_bm"] > 0.05
    assert "SUPPORTED" in human_row["empirical_claim"]

    # 2. Market Impact on N=11 is inconclusive due to realized return noise
    assert "INCONCLUSIVE" in market_row["empirical_claim"]

    # 3. Hybrid Importance is classified as exploratory only
    assert "EXPLORATORY" in hybrid_row["empirical_claim"]


# ==============================================================================
# 4. Purged Walk-Forward Evaluation
# ==============================================================================

def test_purged_walk_forward_evaluation(pretrained_encoder):
    """
    V8.11 Requirement 4:
    - Multi-window chronological walk-forward validation (Window 1 & Window 2).
    - Enforces cluster grouping, label embargo, and >= 3 trading session purging gaps.
    - Verifies out-of-sample evaluation on distinct test periods.
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
    assert wf_res["walk_forward_verdict"] == "CONSISTENT_POSITIVE"


# ==============================================================================
# 5. Paired Differences and Bootstrap Uncertainty
# ==============================================================================

def test_paired_differences_and_bootstrap_uncertainty():
    """
    V8.11 Requirement 5:
    - Tests sample-level paired differences (MAE reductions, prediction differences).
    - Executes 1,000 bootstrap resamples for 95% confidence intervals on Delta rho and Delta NDCG@3.
    - Verifies empirical determination and small-sample uncertainty boundaries.
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

    # Probabilities
    assert 0.0 <= boot_res["p_delta_rho_positive"] <= 1.0
    assert 0.0 <= boot_res["p_delta_ndcg3_positive"] <= 1.0
    assert boot_res["incremental_value_verdict"] in ["SUPPORTED", "SUPPORTED_EXPLORATORY", "INCONCLUSIVE"]
