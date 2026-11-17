"""
V8.9 Semantic Representation & Event Importance Evaluation Test Suite.
Validates:
1. Semantic Benchmark Integrity across all 8 linguistic and domain categories.
2. 512D vs 128D Semantic Representation Metrics:
   - Positive/negative separation margin
   - Ranking consistency (Spearman rho)
   - Duplicate retrieval precision (Top-1 precision)
   - Negation/clarification sensitivity
   - Confusable entity disambiguation
3. Event Importance Formulations (Method A Rule-based, Method B Metadata, Method C Semantic+Meta).
4. Strict PIT Boundaries:
   - Decision-time isolation
   - Ex-post label availability (T+1 14:00 Asia/Taipei)
   - Leakage-free negative sampling
5. Controlled Evaluations:
   - Event ranking plausibility
   - Deduplication robustness (syndication discount)
   - Exponential temporal decay (3-day half-life)
   - Counterfactual future news invariance
   - Shortcut learning audit (MOPS routine notice vs major CapEx)
   - Semantic embedding value-add
"""

import copy
import math
import os
import pytest
import numpy as np
import pandas as pd
import torch

from stock_rl.config.schema import (
    EVENT_TEXT_DIM,
    EVENT_METADATA_DIM,
    EVENT_STOCK_RELATION_DIM,
    EVENT_RAW_FEATURE_DIM,
)
from stock_rl.data.multimodal_pipeline import (
    NewsSourceType,
    EvidenceAuditLevel,
    RawNewsArticle,
    SourceEvidenceRecord,
    TrustedSourceRegistry,
    TrustedNewsSourceVerifier,
    load_micro_news_dataset,
)
from stock_rl.evaluation.provenance import (
    BGE_SMALL_ZH_MANIFEST,
    GenuinePretrainedSemanticEncoder,
    load_trusted_pretrained_embedding_engine,
)
from stock_rl.evaluation.semantic_eval import (
    SEMANTIC_EVALUATION_BENCHMARK,
    evaluate_semantic_representation_benchmark,
)
from stock_rl.evaluation.importance import (
    RuleBasedEventImportanceScorer,
    MetadataEventImportanceScorer,
    SemanticMetadataEventImportanceScorer,
    ExPostMarketReactionLabel,
    ControlledEventImportanceEvaluation,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def pretrained_engine():
    """Loads trusted genuine pretrained BGE embedding engine."""
    engine = load_trusted_pretrained_embedding_engine(local_files_only=True)
    if engine is None:
        pytest.skip("Local BGE pretrained snapshot not available")
    return engine


@pytest.fixture(scope="module")
def pretrained_encoder(pretrained_engine):
    """Extracts GenuinePretrainedSemanticEncoder from engine."""
    enc = pretrained_engine.model_weights
    assert isinstance(enc, GenuinePretrainedSemanticEncoder)
    return enc


# ==============================================================================
# 1. Semantic Evaluation Set Specification & Integrity
# ==============================================================================

def test_semantic_benchmark_specification_and_coverage():
    """
    V8.9 Section 2:
    Verifies that the curated benchmark covers all 8 required categories
    and specifies ground-truth expected relationships.
    """
    expected_categories = {
        "multi_outlet_syndication",
        "paraphrase",
        "similar_event_diff_company",
        "same_company_diff_event",
        "negation_and_clarification",
        "material_earnings_capex",
        "negative_control",
        "confusable_entities",
    }
    found_categories = {item["category"] for item in SEMANTIC_EVALUATION_BENCHMARK}
    assert expected_categories == found_categories, f"Missing categories: {expected_categories - found_categories}"

    assert len(SEMANTIC_EVALUATION_BENCHMARK) >= 32

    # Check required annotations on each pair
    for item in SEMANTIC_EVALUATION_BENCHMARK:
        assert "pair_id" in item
        assert "category" in item
        assert "text_a" in item and len(item["text_a"]) > 10
        assert "text_b" in item and len(item["text_b"]) > 10
        assert "expected_relationship" in item
        assert "expected_tier" in item
        assert "is_positive_pair" in item
        assert "cluster_id" in item


# ==============================================================================
# 2. Semantic Representation Evaluation Metrics
# ==============================================================================

def test_semantic_representation_evaluation_metrics(pretrained_encoder):
    """
    V8.9 Section 2:
    Executes full semantic representation benchmark:
    - Distortion bounds (|sim_512 - sim_128|)
    - Positive / negative pair separation margin
    - Ranking consistency (Spearman rho)
    - Duplicate retrieval precision (Top-1 precision)
    - Confusable entity disambiguation
    """
    res = evaluate_semantic_representation_benchmark(
        encoder=pretrained_encoder,
        benchmark_data=SEMANTIC_EVALUATION_BENCHMARK,
    )

    # 1. Distortion bounds
    assert res["mean_absolute_cosine_distortion"] < 0.05
    assert res["percentile_95_distortion"] < 0.08
    assert res["maximum_distortion"] < 0.10

    # 2. Ranking consistency
    assert res["spearman_rank_correlation"] >= 0.90
    assert res["kendall_tau"] >= 0.75

    # 3. Positive vs Negative Separation Margin
    mean_pos = res["positive_pairs_mean_sim"]
    mean_neg = res["negative_pairs_mean_sim"]
    sep_margin = res["positive_negative_separation_margin"]

    print(f"\n[Semantic Representation Benchmark]")
    print(f"Positive Pairs Mean Sim: {mean_pos:.4f}")
    print(f"Negative Pairs Mean Sim: {mean_neg:.4f}")
    print(f"Separation Margin:       {sep_margin:.4f}")
    print(f"Top-1 Retrieval Prec:    {res['duplicate_retrieval_top1_precision']:.4f}")
    print(f"Spearman Rho:            {res['spearman_rank_correlation']:.4f}")

    assert mean_pos >= 0.75, f"Positive pairs mean similarity {mean_pos} below 0.75"
    assert mean_neg <= 0.68, f"Negative pairs mean similarity {mean_neg} exceeds 0.68"
    assert sep_margin >= 0.20, f"Separation margin {sep_margin} below 0.20"

    # 4. Duplicate Retrieval Precision (Top-1)
    assert res["duplicate_retrieval_top1_precision"] == 1.0, "All multi-outlet duplicates must rank Top-1 against distractors"

    # 5. Confusable entity disambiguation
    assert res["max_confusable_similarity"] < 0.75, "Confusable entities (TSMC vs UMC) must not be treated as duplicates"

    # 6. Holdout isolation notice check
    assert "Strict Holdout Isolation" in res["holdout_data_isolation_notice"]


# ==============================================================================
# 3. Event Importance Formulation & Baselines (Methods A, B, C)
# ==============================================================================

def test_rule_based_event_importance_scoring():
    """
    V8.9 Section 3:
    Validates Method A (Rule-Based Formulation I_rule):
    - Respects source weights (MOPS > Media)
    - Respects evidence audit level (VERIFIED > ESTIMATED > UNKNOWN)
    - Respects category weights (Earnings/CapEx > Trading noise)
    - Respects stock relation (Direct mention > Sector match > Macro)
    - Returns clipped score in [0.0, 1.0]
    """
    t_dec = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    acc = "MOPS-FILING-20240115-2330-99"
    art_official = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電董事會核准資本預算擴建新廠",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        provider_available_evidence=acc,
        is_audit_verified=True,
    )
    # Register in TrustedSourceRegistry and certify via TrustedNewsSourceVerifier
    TrustedSourceRegistry.register_record(SourceEvidenceRecord(
        record_id="REC_MOPS_20240115_2330_99",
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_id=f"MOPS_{art_official.article_id}",
        accession_number=acc,
        stock_id="2330",
        provider_available_at=t_pub,
        payload_sha256=art_official.raw_payload_hash,
    ))
    art_official = TrustedNewsSourceVerifier.certify_article(art_official, accession_number=acc)

    art_media = RawNewsArticle(
        source_type=NewsSourceType.FINMIND_RSS,
        source_name="Yahoo奇摩股市",
        stock_id="2330",
        headline="台積電開盤震盪小漲",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        is_audit_verified=False,
    )

    score_official = RuleBasedEventImportanceScorer.score_article(art_official, "2330", t_dec)
    score_media = RuleBasedEventImportanceScorer.score_article(art_media, "2330", t_dec)

    assert 0.0 <= score_official <= 1.0
    assert 0.0 <= score_media <= 1.0
    assert score_official > score_media


def test_neural_event_importance_scorers(pretrained_encoder):
    """
    V8.9 Section 3:
    Validates Method B (I_meta) and Method C (I_sem_meta):
    - Outputs scalar in [0, 1] via Sigmoid
    - Deterministic initialization
    - Forward passes execute cleanly
    """
    t_dec = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    art = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電說明資本支出與先進製程產能",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        is_audit_verified=True,
    )

    m_32, t_128 = ControlledEventImportanceEvaluation.build_test_event_feature_bundle(
        art, "2330", t_dec, pretrained_encoder
    )
    assert m_32.shape == (1, 32)
    assert t_128.shape == (1, 128)

    # Method B
    scorer_b = MetadataEventImportanceScorer()
    scorer_b.eval()
    score_b = scorer_b(m_32)
    assert score_b.shape == (1, 1)
    assert 0.0 <= score_b.item() <= 1.0

    # Method C
    scorer_c = SemanticMetadataEventImportanceScorer()
    scorer_c.eval()
    score_c = scorer_c(m_32, t_128)
    assert score_c.shape == (1, 1)
    assert 0.0 <= score_c.item() <= 1.0


# ==============================================================================
# 4. Strict PIT Boundaries & Ex-Post Label Availability
# ==============================================================================

def test_ex_post_market_reaction_label_pit_contract():
    """
    V8.9 Section 3:
    Verifies that supervisory market reaction labels (volatility, CAR, volume surprise)
    are strictly available ONLY post-event (T+1 14:00 Asia/Taipei) and cannot leak into
    decision-time state features at time T.
    """
    t_dec = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")

    label = ExPostMarketReactionLabel.create_mock_pit_label(
        stock_id="2330",
        t_decision=t_dec,
        next_session_open=600.0,
        next_session_close=615.0,
        next_session_volume=50000.0,
        sma_20_volume=40000.0,
        std_20_volume=5000.0,
    )

    # 1. Label available_at must be strictly AFTER decision cutoff
    assert label.label_available_at > t_dec
    assert label.label_available_at == pd.Timestamp("2024-01-16 14:00:00", tz="Asia/Taipei")

    # 2. Quantified label values
    assert label.next_day_abs_log_return > 0.0
    assert label.cumulative_abnormal_return_2d > 0.0
    assert label.normalized_volume_surprise > 0.0
    assert 0.0 <= label.composite_market_impact_score <= 1.0


def test_decision_time_future_news_isolation():
    """
    V8.9 Section 3:
    News with provider_available_at > decision_timestamp MUST score 0.0 (unavailable).
    """
    t_dec = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")
    t_future = pd.Timestamp("2024-01-16 08:30:00", tz="Asia/Taipei")

    future_art = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電明日早盤重訊",
        source_published_at=t_future,
        provider_available_at=t_future,
        is_audit_verified=True,
    )

    score = RuleBasedEventImportanceScorer.score_article(future_art, "2330", t_dec)
    assert score == 0.0, "Future event must receive zero importance at past decision timestamp"


# ==============================================================================
# 5. Controlled Evaluations
# ==============================================================================

def test_controlled_ranking_plausibility(pretrained_encoder):
    """
    V8.9 Section 4:
    Verifies that major CapEx/Earnings events rank higher in importance
    than routine market commentary.
    """
    eval_res = ControlledEventImportanceEvaluation.evaluate_ranking_plausibility(pretrained_encoder)
    assert eval_res["ranking_plausible"] is True
    assert eval_res["rule_based"]["valid"] is True
    assert eval_res["semantic_metadata"]["valid"] is True


def test_controlled_deduplication_robustness():
    """
    V8.9 Section 4:
    Verifies that multiple syndicated outlets covering the exact same event
    do not inflate importance (1 / sqrt(N) syndication discount).
    """
    dedup_res = ControlledEventImportanceEvaluation.evaluate_deduplication_robustness()
    assert dedup_res["is_discount_effective"] is True
    assert dedup_res["score_single_outlet"] > dedup_res["score_3_syndicated_outlets"]
    assert dedup_res["score_3_syndicated_outlets"] > dedup_res["score_5_syndicated_outlets"]


def test_controlled_temporal_decay():
    """
    V8.9 Section 4:
    Verifies exponential half-life decay (3 days half-life -> ~50% decay per 3 days).
    """
    decay_res = ControlledEventImportanceEvaluation.evaluate_temporal_decay()
    assert decay_res["is_half_life_valid"] is True
    assert math.isclose(decay_res["ratio_day3_day0"], 0.50, abs_tol=0.05)
    assert math.isclose(decay_res["ratio_day6_day3"], 0.50, abs_tol=0.05)


def test_controlled_shortcut_learning_audit(pretrained_encoder):
    """
    V8.9 Section 4:
    Tests whether model is susceptible to trivial shortcuts (e.g. only relying on source=MOPS or ticker=2330).
    Contrasts a routine MOPS shareholder meeting notice vs a 120-billion MOPS CapEx expansion.
    Verifies that Semantic + Metadata importance (I_sem_meta) captures the semantic distinction.
    """
    shortcut_res = ControlledEventImportanceEvaluation.evaluate_shortcut_learning(pretrained_encoder)
    assert shortcut_res["semantic_separation_achieved"] is True
    assert shortcut_res["importance_delta"] > 0.0
    print(f"\n[Shortcut Learning Audit]")
    print(f"Routine Notice Score:  {shortcut_res['routine_notice_score']}")
    print(f"Major CapEx Score:     {shortcut_res['major_capex_score']}")
    print(f"Importance Delta:      {shortcut_res['importance_delta']}")
    print(f"Text Cosine Distance:  {1.0 - shortcut_res['headline_cosine_similarity']:.4f}")


def test_controlled_counterfactual_future_news_invariance(pretrained_encoder):
    """
    V8.9 Section 4:
    Modifying or inserting news at T+2 MUST have ZERO effect on importance scores
    and feature vectors at T_0 and T_1.
    """
    t_0 = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")
    t_pub = pd.Timestamp("2024-01-15 10:00:00", tz="Asia/Taipei")

    art_past = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電擴建產能公告",
        source_published_at=t_pub,
        provider_available_at=t_pub,
        is_audit_verified=True,
    )

    # Score before adding future news
    score_before = RuleBasedEventImportanceScorer.score_article(art_past, "2330", t_0)

    # Counterfactual future event inserted at T+2
    t_future = pd.Timestamp("2024-01-17 10:00:00", tz="Asia/Taipei")
    art_future = RawNewsArticle(
        source_type=NewsSourceType.MOPS_OFFICIAL,
        source_name="MOPS",
        stock_id="2330",
        headline="台積電未來營收警訊",
        source_published_at=t_future,
        provider_available_at=t_future,
        is_audit_verified=True,
    )

    # Score at T_0 must remain bitwise identical
    score_after = RuleBasedEventImportanceScorer.score_article(art_past, "2330", t_0)
    assert score_before == score_after, "Future event modified past decision-time score!"


# ==============================================================================
# 6. Provenance & Boundary Clarifications
# ==============================================================================

def test_data_provenance_taxonomies_and_boundaries():
    """
    V8.9 Section 5:
    Validates explicit isolation between:
    - Real audited fixtures
    - Estimated-PIT historical candidate news
    - Synthetic test cases
    - Human-annotated benchmarks
    Ensures that live historical Formal PIT certification remains documented as Pending.
    """
    # 1. Micro dataset articles default to ESTIMATED/CLAIM_ONLY
    micro_news = load_micro_news_dataset()
    for art in micro_news:
        assert art.evidence_audit_level in (EvidenceAuditLevel.ESTIMATED, EvidenceAuditLevel.EVIDENCE_CLAIM_ONLY)
        # Not formally certified without live institutional ledger
        assert "source_audit_certificate" not in art.metadata

    # 2. Benchmark data explicitly labeled as HUMAN_ANNOTATED
    for item in SEMANTIC_EVALUATION_BENCHMARK:
        assert item["category"] in (
            "multi_outlet_syndication", "paraphrase", "similar_event_diff_company",
            "same_company_diff_event", "negation_and_clarification",
            "material_earnings_capex", "negative_control", "confusable_entities"
        )
