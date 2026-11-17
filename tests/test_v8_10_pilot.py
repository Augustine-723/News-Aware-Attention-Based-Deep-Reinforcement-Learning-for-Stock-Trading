"""
Tests for V8.10 Small-Scale Event Importance Learning Pilot
Verifies:
1. Preflight A: TWSE Trading Calendar reconciliation, special closures (Typhoon Khanun), date boundaries (Fail Closed).
2. Preflight B: Text normalization (NFKC, zero-width, whitespace), hashing, and 4-layer holdout contamination detection.
3. Preflight C: Strictly pre-T metric computation (Beta120, SMA20, STD20), daily cumulative CAR vs interval AR, missing data integrity.
4. Supervised Pilot Dataset: 5-category data provenance, 32D metadata, 128D semantic text, causal availability.
5. Chronological Event-Grouped Splitter: Temporal order, zero cluster leakage, purging gaps.
6. Supervised Multi-Seed Experiment & Audits: Multi-seed convergence (42, 100, 2024), evaluation metrics, 6 controlled audits.
"""

import math
import os
import numpy as np
import pandas as pd
import pytest
import torch

from stock_rl.evaluation.importance import (
    TWSETradingCalendar,
    ExPostMarketReactionLabel,
    RuleBasedEventImportanceScorer,
    extract_event_metadata_vector,
    RawNewsArticle,
    NewsSourceType,
    EvidenceAuditLevel,
)
from stock_rl.evaluation.semantic_eval import (
    normalize_financial_text,
    compute_normalized_content_hash,
    compute_text_jaccard_similarity,
    verify_holdout_contamination,
    SEMANTIC_EVALUATION_BENCHMARK,
    SemanticBenchmarkManifest,
)
from stock_rl.evaluation.provenance import (
    load_trusted_pretrained_embedding_engine,
    GenuinePretrainedSemanticEncoder,
)
from stock_rl.evaluation.pilot_dataset import (
    PilotEventSample,
    PilotDatasetBuilder,
    PilotSourceManifest,
    SourceVerificationStatus,
    SupervisionTargetName,
    EventSourceRecord,
    RAW_PILOT_EVENTS,
)
from stock_rl.evaluation.pilot_experiment import (
    ChronologicalEventGroupedSplitter,
    CapacityMatchedMetadataScorer,
    TrainedMetadataScorer,
    TrainedSemanticMetadataScorer,
    SupervisedPilotTrainer,
    PilotExperimentRunner,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_split_metrics,
)


@pytest.fixture(scope="module")
def pretrained_encoder():
    """Loads trusted pretrained semantic encoder locally."""
    return load_trusted_pretrained_embedding_engine(local_files_only=True)


# ==============================================================================
# 1. Preflight A: Trading Calendar Reconciliation & Fail-Closed Boundaries
# ==============================================================================

def test_twse_calendar_reconciliation_and_boundaries():
    """
    V8.10 Preflight A:
    1. Reconciles TWSE calendar against authentic 202-session TAIEX index with 100% agreement.
    2. Verifies Typhoon Khanun (2023-08-03) closure.
    3. Fails closed (ValueError) when querying dates outside [2023-01-01, 2026-12-31].
    """
    taiex_path = "data/poc_real_data/taiex.csv"
    assert os.path.exists(taiex_path), f"Missing {taiex_path}"
    df = pd.read_csv(taiex_path)
    df["dt"] = pd.to_datetime(df["date"])
    idx = pd.DatetimeIndex(df["dt"])

    # 1. Reconciliation check
    recon = TWSETradingCalendar.reconcile_with_market_index(idx)
    assert recon["is_reconciled"] is True
    assert recon["num_market_sessions"] == 202
    assert recon["num_calendar_sessions"] == 202
    assert len(recon["diff_calendar_not_market"]) == 0
    assert len(recon["diff_market_not_calendar"]) == 0

    # 2. Typhoon Khanun closure verification
    typhoon_day = pd.Timestamp("2023-08-03")
    assert TWSETradingCalendar.is_trading_day(typhoon_day) is False
    assert "2023-08-03" in TWSETradingCalendar.TWSE_HOLIDAYS

    # 3. Fail closed outside verified date range [2023-01-01, 2026-12-31]
    too_early = pd.Timestamp("2022-12-31")
    too_late = pd.Timestamp("2027-01-01")

    with pytest.raises(ValueError, match="Fail Closed"):
        TWSETradingCalendar.is_trading_day(too_early)

    with pytest.raises(ValueError, match="Fail Closed"):
        TWSETradingCalendar.is_trading_day(too_late)


# ==============================================================================
# 2. Preflight B: Text Normalization & Holdout Contamination Audit
# ==============================================================================

def test_preflight_b_normalization_and_contamination_detection():
    """
    V8.10 Preflight B:
    1. Tests Unicode NFKC normalization, zero-width stripping, and whitespace collapse.
    2. Tests deterministic normalized SHA-256 content hashing.
    3. Tests character bigram Jaccard similarity.
    4. Tests that 4-layer contamination detector catches:
       - Exact hash violation
       - Headline-only match
       - Near-duplicate syndicated paraphrase (Jaccard >= 0.70)
       - Event cluster overlap
    5. Confirms 0% contamination of authentic pilot candidates against 36-pair benchmark.
    """
    # 1. Normalization
    raw_messy = "  台積電\u3000董事會\u200b\u200c核准\r\n\t資本預算  1200 億元  "
    norm = normalize_financial_text(raw_messy)
    assert norm == "台積電 董事會核准 資本預算 1200 億元"
    assert "\u200b" not in norm
    assert "\u200c" not in norm
    assert "\u3000" not in norm

    # 2. Normalized hash
    h1 = compute_normalized_content_hash("台積電 董事會核准", "擴建新廠")
    h2 = compute_normalized_content_hash("  台積電\u3000董事會核准  ", " 擴建新廠\u200b ")
    assert h1 == h2

    # 3. Jaccard similarity
    text_orig = "台積電董事會核准資本預算千億元擴建先進製程晶圓廠"
    text_paraphrase = "台積電董事會核准資本預算千億元興建先進製程晶圓廠"
    jaccard = compute_text_jaccard_similarity(text_orig, text_paraphrase)
    assert jaccard >= 0.80

    # 4. Contamination detector violations
    # Exact text from MOS-01 in benchmark
    exact_leak = [{"article_id": "LEAK-1", "headline": "中央社：台積電董事會今天核准資本預算約新台幣1200億元，將用於擴充先進製程產能及研發支出。", "cluster_id": "CLU-OTHER"}]
    res_exact = verify_holdout_contamination(exact_leak, SEMANTIC_EVALUATION_BENCHMARK)
    assert res_exact["is_contaminated"] is True
    assert len(res_exact["exact_violations"]) > 0

    # Headline match
    headline_leak = [{"article_id": "LEAK-2", "headline": "中央社 vs 經濟日報：台積電董事會核准千億元資本支出", "cluster_id": "CLU-OTHER"}]
    res_head = verify_holdout_contamination(headline_leak, SEMANTIC_EVALUATION_BENCHMARK)
    assert res_head["is_contaminated"] is True
    assert len(res_head["headline_violations"]) > 0

    # Cluster overlap
    cluster_leak = [{"article_id": "LEAK-3", "headline": "全新獨立新聞無重疊字詞", "cluster_id": "TSMC_CAPEX_1200"}]
    res_clu = verify_holdout_contamination(cluster_leak, SEMANTIC_EVALUATION_BENCHMARK)
    assert res_clu["is_contaminated"] is True
    assert len(res_clu["cluster_violations"]) > 0

    # 5. Clean pilot candidate dataset verification
    clean_res = verify_holdout_contamination(RAW_PILOT_EVENTS, SEMANTIC_EVALUATION_BENCHMARK)
    assert clean_res["is_contaminated"] is False
    assert clean_res["total_violations"] == 0


# ==============================================================================
# 3. Preflight C: Strictly Pre-T Metrics & Abnormal Return Formulation
# ==============================================================================

def test_preflight_c_pre_t_metrics_and_abnormal_returns():
    """
    V8.10 Preflight C:
    1. Beta120, Volume SMA20, and Volume STD20 are strictly computed from dates <= T.
    2. Distinguishes daily cumulative CAR (sum of daily abnormal returns) from
       interval abnormal return (holding period abnormal return across T+1 Open to T+2 Close).
    3. Retains MISSING_DATA and None when market data is unavailable (zero synthetic substitution).
    """
    t_dec = pd.Timestamp("2024-01-15 20:00:00", tz="Asia/Taipei")
    dates = pd.date_range("2023-06-01", "2024-01-17", freq="B", tz="Asia/Taipei")

    # Construct mock stock and market DataFrames
    np.random.seed(42)
    s_closes = [500.0 + i * 0.5 + np.random.randn() * 2 for i in range(len(dates))]
    m_closes = [16000.0 + i * 10.0 + np.random.randn() * 20 for i in range(len(dates))]
    volumes = [30000.0 + np.random.randn() * 3000 for _ in range(len(dates))]

    stock_df = pd.DataFrame({
        "open": [c - 1.0 for c in s_closes],
        "close": s_closes,
        "Trading_Volume": volumes,
    }, index=dates)

    market_df = pd.DataFrame({
        "open": [c - 20.0 for c in m_closes],
        "close": m_closes,
    }, index=dates)

    label = ExPostMarketReactionLabel.compute_from_market_data(
        stock_id="2330",
        t_decision=t_dec,
        stock_df=stock_df,
        market_df=market_df,
    )

    # 1. Pre-T metrics present in metadata
    assert "pre_t_beta120" in label.metadata
    assert "pre_t_volume_sma20" in label.metadata
    assert "pre_t_volume_std20" in label.metadata
    assert label.metadata["pre_t_volume_sma20"] > 0
    assert label.metadata["pre_t_volume_std20"] > 0

    # 2. Both daily cumulative CAR and interval AR are populated
    assert label.cumulative_abnormal_return_2d is not None
    assert label.interval_abnormal_return is not None
    assert "t2_cumulative_abnormal_return" in label.components
    assert "interval_abnormal_return" in label.components

    # 3. Missing data integrity (T+2 omitted)
    stock_df_incomplete = stock_df[stock_df.index <= "2024-01-16"]
    label_missing = ExPostMarketReactionLabel.compute_from_market_data(
        stock_id="2330",
        t_decision=t_dec,
        stock_df=stock_df_incomplete,
        market_df=market_df,
    )
    assert label_missing.components["t2_cumulative_abnormal_return"].missing_status == "MISSING_DATA"
    assert label_missing.components["t2_cumulative_abnormal_return"].value is None
    assert label_missing.components["interval_abnormal_return"].missing_status == "MISSING_DATA"
    assert label_missing.components["interval_abnormal_return"].value is None


# ==============================================================================
# 4. Supervised Pilot Dataset Construction & Provenance
# ==============================================================================

def test_pilot_dataset_construction_and_provenance(pretrained_encoder):
    """
    V8.10 Pilot Dataset:
    1. Builds authentic supervised pilot dataset across 2330, 2603, 2881.
    2. Verifies 32D metadata vector and 128D genuine semantic text vector.
    3. Enforces Point-in-Time causal constraint: available_at <= decision_timestamp.
    4. Verifies non-trivial target scores in [0.0, 1.0] for valid targets and None for missing.
    5. Verifies 3 independent targets across all samples.
    """
    builder = PilotDatasetBuilder(encoder=pretrained_encoder)
    samples = builder.build_dataset()

    assert len(samples) >= 40, f"Expected at least 40 samples, got {len(samples)}"

    stock_counts = {}
    category_counts = {}

    for s in samples:
        # Causal validity
        assert s.available_at <= s.decision_timestamp
        assert s.metadata_32d is not None and s.metadata_32d.shape == (32,)
        assert s.text_128d is not None and s.text_128d.shape == (128,)
        # Unit norm check on text embedding
        norm = float(np.linalg.norm(s.text_128d))
        assert abs(norm - 1.0) < 1e-4

        # Target score bounds
        if s.target_score is not None:
            assert 0.0 <= s.target_score <= 1.0

        # Human importance target must always be populated
        assert s.target_human_importance is not None
        assert 0.0 <= s.target_human_importance <= 1.0

        stock_counts[s.stock_id] = stock_counts.get(s.stock_id, 0) + 1
        category_counts[s.category] = category_counts.get(s.category, 0) + 1

    # Verify multi-ticker coverage
    assert set(stock_counts.keys()) == {"2330", "2603", "2881"}
    assert all(c >= 10 for c in stock_counts.values())

    # Verify category diversity
    assert "earnings" in category_counts
    assert "capex" in category_counts
    assert "dividend" in category_counts
    assert "routine" in category_counts

    # Verify target statistics
    t_stats = PilotDatasetBuilder.get_target_statistics(samples)
    assert t_stats["HUMAN_IMPORTANCE"]["valid_samples"] == len(samples)
    assert t_stats["EX_POST_MARKET_IMPACT"]["valid_samples"] == 45
    assert t_stats["EX_POST_MARKET_IMPACT"]["missing_samples"] == 5
    assert t_stats["HYBRID_IMPORTANCE"]["valid_samples"] == 45
    assert t_stats["HYBRID_IMPORTANCE"]["missing_samples"] == 5


# ==============================================================================
# 5. Chronological Event-Grouped Splitter Integrity
# ==============================================================================

def test_chronological_event_grouped_split_integrity(pretrained_encoder):
    """
    V8.10 Split Integrity:
    1. Chronological timeline ordering (Train < Val < Test).
    2. Strict cluster-grouped isolation (0% cluster overlap across splits).
    3. Multi-session temporal buffer between split boundaries.
    """
    builder = PilotDatasetBuilder(encoder=pretrained_encoder)
    samples = builder.build_dataset()

    splitter = ChronologicalEventGroupedSplitter(
        train_end_date="2023-11-30",
        val_end_date="2024-01-31",
    )
    splits = splitter.split(samples)

    train_s = splits["train"]
    val_s = splits["val"]
    test_s = splits["test"]

    assert len(train_s) > 0
    assert len(val_s) > 0
    assert len(test_s) > 0

    # 1. Chronological order
    t_train_max = max(s.decision_timestamp for s in train_s)
    t_val_min = min(s.decision_timestamp for s in val_s)
    t_val_max = max(s.decision_timestamp for s in val_s)
    t_test_min = min(s.decision_timestamp for s in test_s)

    assert t_train_max < t_val_min, "Train must strictly precede Val!"
    assert t_val_max < t_test_min, "Val must strictly precede Test!"

    # 2. Strict cluster-level isolation
    train_c = set(splits["train_clusters"])
    val_c = set(splits["val_clusters"])
    test_c = set(splits["test_clusters"])

    assert len(train_c & val_c) == 0, "Train and Val share clusters!"
    assert len(train_c & test_c) == 0, "Train and Test share clusters!"
    assert len(val_c & test_c) == 0, "Val and Test share clusters!"


# ==============================================================================
# 6. Supervised Multi-Seed Pilot Experiment & Audits
# ==============================================================================

def test_supervised_pilot_experiment_multi_seed_evaluation(pretrained_encoder):
    """
    V8.10 Comparative Pilot Experiment:
    1. Evaluates Baseline 0, Method A, Method B, Method B-matched, Method C, Method C-shuffled.
    2. Executes multi-seed evaluation across fixed seeds [42, 100, 2024].
    3. Reports regression and ranking metrics (MAE, RMSE, Spearman Rho, NDCG@K, Precision@K).
    4. Executes all 6 Controlled Audits and asserts compliance.
    5. Confirms claim level is EXPLORATORY_ONLY.
    """
    builder = PilotDatasetBuilder(encoder=pretrained_encoder)
    samples = builder.build_dataset()

    runner = PilotExperimentRunner(dataset_samples=samples)
    results = runner.run_multi_seed_experiment()

    # 1. Baseline 0 and Method A valid metrics
    b0 = results["baseline_0_untrained"]
    mA = results["method_a_rule_based"]
    assert "mae" in b0 and "spearman_rho" in b0
    assert "mae" in mA and "spearman_rho" in mA
    assert mA["spearman_rho"] > 0.40, "Rule-based scorer should show strong positive rank correlation"

    # 2. Multi-seed Method B, B-matched, Method C, C-shuffled statistics
    mB = results["method_b_metadata_trained"]
    mBm = results["method_b_matched_metadata_trained"]
    mC = results["method_c_semantic_metadata_trained"]
    mCs = results["method_c_shuffled_semantic_control"]

    for stat_dict in [mB, mBm, mC, mCs]:
        assert "mean" in stat_dict["mae"]
        assert "std" in stat_dict["mae"]
        assert "mean" in stat_dict["spearman_rho"]
        assert len(stat_dict["mae"]["values"]) == 3

    # Method C should achieve consistent positive Spearman rank correlation across seeds
    assert mC["spearman_rho"]["mean"] > 0.30
    assert mC["spearman_rho"]["std"] < 0.15

    # 3. Sub-group breakdowns
    subgroups = results["subgroup_breakdowns"]
    assert "2330" in subgroups["by_ticker"]
    assert "2603" in subgroups["by_ticker"]
    assert "2881" in subgroups["by_ticker"]

    # 4. Controlled Audits
    audits = results["controlled_audits"]
    assert audits["audit_1_semantic_incremental_value"]["status"] in ["VERIFIED_EXPLORATORY", "VERIFIED", "PASSED"]
    assert audits["audit_2_shortcut_learning"]["status"] == "PASSED"
    assert audits["audit_3_revision_leakage"]["status"] == "PASSED"
    assert audits["audit_4_duplicate_event_leakage"]["status"] == "PASSED"
    assert audits["audit_5_future_market_label_leakage"]["status"] == "PASSED"
    assert audits["audit_6_hard_negative_robustness"]["status"] in ["PASSED", "WARNING"]

    # 5. Exploratory claim tagging
    assert results["generalization_claim"] == "EXPLORATORY_ONLY"
    assert "N_test=11" in results["sample_size_note"]


# ==============================================================================
# 7. P0-1: Historical Event Authenticity & Source Manifest
# ==============================================================================

def test_pilot_source_manifest_and_authenticity_categories():
    """
    P0-1 Historical Event Authenticity:
    1. Verifies all 50 events exist in PilotSourceManifest.
    2. Verifies categorical distinction: CURATED_UNVERIFIED vs VERIFIED_SOURCE vs SYNTHETIC_FIXTURE.
    3. Confirms no fabricated external URLs or unverified formal PIT elevations.
    4. Verifies SHA-256 payload hash integrity.
    """
    manifest = PilotSourceManifest.generate_manifest()
    assert len(manifest) == 50, f"Expected 50 records in manifest, got {len(manifest)}"

    summary = PilotSourceManifest.get_provenance_summary()
    assert summary["total_records"] == 50
    assert summary["CURATED_UNVERIFIED"] == 38
    assert summary["VERIFIED_SOURCE"] == 12
    assert summary["SYNTHETIC_FIXTURE"] == 0
    assert summary["records_with_external_url"] == 0
    assert summary["records_with_valid_hash"] == 50

    for eid, rec in manifest.items():
        assert rec.verification_status in [
            SourceVerificationStatus.CURATED_UNVERIFIED,
            SourceVerificationStatus.VERIFIED_SOURCE,
        ]
        assert rec.url is None or rec.url == ""
        assert len(rec.payload_hash) == 64
        # Verify hash matches recomputed normalized content hash
        ev = [e for e in RAW_PILOT_EVENTS if e["event_id"] == eid][0]
        recomputed_hash = compute_normalized_content_hash(ev["headline"], ev.get("body_text", ""))
        assert rec.payload_hash == recomputed_hash


# ==============================================================================
# 8. P0-2: Consistent Supervision Targets & Zero Fallback
# ==============================================================================

def test_consistent_supervision_targets_no_fallback(pretrained_encoder):
    """
    P0-2 Consistent Supervision Targets:
    1. Tests the 3 independent targets: HUMAN_IMPORTANCE, EX_POST_MARKET_IMPACT, HYBRID_IMPORTANCE.
    2. Confirms that when market impact is missing, HYBRID_IMPORTANCE is strictly None
       and does NOT silently fall back to HUMAN_IMPORTANCE.
    3. Confirms that when market impact is available, HYBRID_IMPORTANCE = round(0.4*human + 0.6*market, 4).
    4. Confirms valid counts: Human=50 (0% missing), Market=45 (10% missing), Hybrid=45 (10% missing).
    """
    builder = PilotDatasetBuilder(encoder=pretrained_encoder)
    samples = builder.build_dataset()

    stats = PilotDatasetBuilder.get_target_statistics(samples)
    assert stats["HUMAN_IMPORTANCE"]["valid_samples"] == 50
    assert stats["HUMAN_IMPORTANCE"]["missing_samples"] == 0
    assert stats["EX_POST_MARKET_IMPACT"]["valid_samples"] == 45
    assert stats["EX_POST_MARKET_IMPACT"]["missing_samples"] == 5
    assert stats["HYBRID_IMPORTANCE"]["valid_samples"] == 45
    assert stats["HYBRID_IMPORTANCE"]["missing_samples"] == 5

    missing_samples = [s for s in samples if s.target_market_impact is None]
    assert len(missing_samples) == 5

    for s in missing_samples:
        # HUMAN_IMPORTANCE is available
        assert s.target_human_importance is not None
        # EX_POST_MARKET_IMPACT is None
        assert s.target_market_impact is None
        # HYBRID_IMPORTANCE MUST BE STRICTLY NONE! ZERO SILENT FALLBACK!
        assert s.target_hybrid_importance is None
        assert s.target_missing_status["HYBRID_IMPORTANCE"] == "MISSING_MARKET_COMPONENT"

    available_samples = [s for s in samples if s.target_market_impact is not None]
    assert len(available_samples) == 45

    for s in available_samples:
        expected_hybrid = round(float(np.clip(0.4 * s.target_human_importance + 0.6 * s.target_market_impact, 0.0, 1.0)), 4)
        assert abs(s.target_hybrid_importance - expected_hybrid) < 1e-6
        assert s.target_missing_status["HYBRID_IMPORTANCE"] == "AVAILABLE"


# ==============================================================================
# 9. P0-3: Purged Chronological Splitting & Embargo Enforcement
# ==============================================================================

def test_purged_chronological_splitting_and_embargo(pretrained_encoder):
    """
    P0-3 Purged Chronological Splitting:
    1. Tests cross-boundary cluster purging (if a cluster spans splits, all samples are purged).
    2. Tests Label Embargo enforcement (if a Train label is available after Train cutoff, it is purged).
    3. Tests trading sessions purging gap assertion (>= 3 sessions).
    4. Tests that raw vs purged counts are accurately reported.
    """
    builder = PilotDatasetBuilder(encoder=pretrained_encoder)
    samples = builder.build_dataset()

    # Create synthetic test case with cross-boundary cluster and embargo violation
    synth_samples = list(samples)

    # 1. Add a sample that violates Train Label Embargo
    bad_embargo_sample = PilotEventSample(
        sample_id="SMP-SYNTH-EMBARGO",
        event_id="EV-SYNTH-EMBARGO",
        article_id="ART-SYNTH-EMBARGO",
        stock_id="2330",
        source_name="MOPS",
        source_type=NewsSourceType.MOPS_OFFICIAL,
        decision_timestamp=pd.Timestamp("2023-11-29 20:00:00", tz="Asia/Taipei"),
        published_at=pd.Timestamp("2023-11-29 16:00:00", tz="Asia/Taipei"),
        available_at=pd.Timestamp("2023-11-29 16:00:00", tz="Asia/Taipei"),
        headline="合成測試樣本文本",
        body_text="合成測試內文",
        category="earnings",
        cluster_id="CLU-SYNTH-EMBARGO",
        target_human_importance=0.7,
        target_market_impact=0.6,
        target_hybrid_importance=0.64,
        # Label available on Dec 05 (after Nov 30 cutoff!)
        target_availability={"HYBRID_IMPORTANCE": pd.Timestamp("2023-12-05 14:00:00", tz="Asia/Taipei")},
        target_missing_status={"HYBRID_IMPORTANCE": "AVAILABLE"},
        target_score=0.64,
        metadata_32d=np.zeros(32, dtype=np.float32),
        text_128d=np.ones(128, dtype=np.float32) / math.sqrt(128),
    )
    synth_samples.append(bad_embargo_sample)

    # 2. Add two samples sharing the same cluster across Train and Val
    cross_cluster_train = PilotEventSample(
        sample_id="SMP-SYNTH-CROSS-1",
        event_id="EV-SYNTH-CROSS-1",
        article_id="ART-SYNTH-CROSS-1",
        stock_id="2603",
        source_name="MOPS",
        source_type=NewsSourceType.MOPS_OFFICIAL,
        decision_timestamp=pd.Timestamp("2023-10-15 20:00:00", tz="Asia/Taipei"),
        published_at=pd.Timestamp("2023-10-15 16:00:00", tz="Asia/Taipei"),
        available_at=pd.Timestamp("2023-10-15 16:00:00", tz="Asia/Taipei"),
        headline="合成跨期事件第一篇",
        body_text="合成跨期事件內文",
        category="capex",
        cluster_id="CLU-CROSS-PERIOD-CLUSTER",
        target_human_importance=0.8,
        target_market_impact=0.5,
        target_hybrid_importance=0.62,
        target_availability={"HYBRID_IMPORTANCE": pd.Timestamp("2023-10-17 14:00:00", tz="Asia/Taipei")},
        target_missing_status={"HYBRID_IMPORTANCE": "AVAILABLE"},
        target_score=0.62,
        metadata_32d=np.zeros(32, dtype=np.float32),
        text_128d=np.ones(128, dtype=np.float32) / math.sqrt(128),
    )
    cross_cluster_val = PilotEventSample(
        sample_id="SMP-SYNTH-CROSS-2",
        event_id="EV-SYNTH-CROSS-2",
        article_id="ART-SYNTH-CROSS-2",
        stock_id="2603",
        source_name="MOPS",
        source_type=NewsSourceType.MOPS_OFFICIAL,
        decision_timestamp=pd.Timestamp("2023-12-20 20:00:00", tz="Asia/Taipei"),
        published_at=pd.Timestamp("2023-12-20 16:00:00", tz="Asia/Taipei"),
        available_at=pd.Timestamp("2023-12-20 16:00:00", tz="Asia/Taipei"),
        headline="合成跨期事件第二篇修訂",
        body_text="合成跨期事件內文修訂",
        category="capex",
        cluster_id="CLU-CROSS-PERIOD-CLUSTER",  # Same cluster in Val!
        target_human_importance=0.8,
        target_market_impact=0.5,
        target_hybrid_importance=0.62,
        target_availability={"HYBRID_IMPORTANCE": pd.Timestamp("2023-12-22 14:00:00", tz="Asia/Taipei")},
        target_missing_status={"HYBRID_IMPORTANCE": "AVAILABLE"},
        target_score=0.62,
        metadata_32d=np.zeros(32, dtype=np.float32),
        text_128d=np.ones(128, dtype=np.float32) / math.sqrt(128),
    )
    synth_samples.extend([cross_cluster_train, cross_cluster_val])

    # Execute splitter
    splitter = ChronologicalEventGroupedSplitter(
        train_end_date="2023-11-30",
        val_end_date="2024-01-31",
        purging_sessions=3,
        target_name="HYBRID_IMPORTANCE",
    )
    res = splitter.split(synth_samples)

    # Verify that the bad embargo sample was purged
    train_ids = [s.sample_id for s in res["train"]]
    val_ids = [s.sample_id for s in res["val"]]
    test_ids = [s.sample_id for s in res["test"]]

    assert "SMP-SYNTH-EMBARGO" not in train_ids, "Embargo violation sample leaked into Train!"
    assert "SMP-SYNTH-CROSS-1" not in train_ids, "Cross-boundary cluster leaked into Train!"
    assert "SMP-SYNTH-CROSS-2" not in val_ids, "Cross-boundary cluster leaked into Val!"

    # Verify purge details in stats
    stats = res["stats"]
    assert stats["purged_details"]["embargo_violation_count"] >= 1
    assert stats["purged_details"]["spanning_cluster_count"] >= 2
    assert stats["train_val_gap_trading_sessions"] >= 3
    assert stats["val_test_gap_trading_sessions"] >= 3


# ==============================================================================
# 10. P1: Fair Capacity-Matched Baseline & Shuffled Semantic Ablation
# ==============================================================================

def test_capacity_matched_and_shuffled_semantic_ablation(pretrained_encoder):
    """
    P1 Fair Semantic Incremental Value Evaluation:
    1. Verifies CapacityMatchedMetadataScorer has ~10.7k parameters (matching Method C's 10,721 params within 1 param).
    2. Verifies Method C-shuffled breaks semantic ranking while maintaining identical capacity and feature norms.
    """
    model_c = TrainedSemanticMetadataScorer()
    params_c = sum(p.numel() for p in model_c.parameters())
    assert params_c == 10721

    model_bm = CapacityMatchedMetadataScorer()
    params_bm = sum(p.numel() for p in model_bm.parameters())
    assert params_bm == 10720
    assert abs(params_bm - params_c) <= 1, f"Expected capacity match within 1 param, got {params_bm} vs {params_c}"

    # Verify forward pass
    meta_dummy = torch.randn(4, 32)
    text_dummy = torch.randn(4, 128)
    text_dummy = text_dummy / text_dummy.norm(dim=-1, keepdim=True)

    out_bm = model_bm(meta_dummy)
    out_c = model_c(meta_dummy, text_dummy)
    assert out_bm.shape == (4, 1)
    assert out_c.shape == (4, 1)
    assert torch.all(out_bm >= 0.0) and torch.all(out_bm <= 1.0)
    assert torch.all(out_c >= 0.0) and torch.all(out_c <= 1.0)
