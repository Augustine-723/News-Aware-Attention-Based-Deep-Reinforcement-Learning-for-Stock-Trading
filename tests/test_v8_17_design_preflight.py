"""
Unit tests for V8.17 Design Validity Preflight & Audit Framework.
Verifies sample scope corrections with weekend rollover, latency identifiability metrics,
distinction between daily eligibility equality and full tensor equality,
synthetic boundary fixtures quarantine (including renamed fixtures),
robust fragility metrics, and chronological fold alignment.
"""

import pytest
import pandas as pd
from stock_rl.evaluation.pilot_dataset import (
    RAW_PILOT_EVENTS,
    PilotSourceManifest,
    PilotDatasetBuilder,
)
from stock_rl.evaluation.design_preflight import (
    DesignValidityPreflightAuditor,
    PreflightStatus,
)


@pytest.fixture
def twse_trading_calendar():
    df = pd.read_csv("data/poc_real_data/2330_ohlcv.csv")
    return sorted(df["date"].tolist())


def test_v8_17_p0_1_sample_scope_full_vs_q1(twse_trading_calendar):
    """
    P0-1 & P1-1:
    1. Re-computes time distribution, ticker distribution, and cluster counts.
    2. Separately evaluates Full Period (2023-06 to 2024-03) and 2024 Q1 (2024-01 to 2024-03).
    3. Confirms non-trading day event EV-2330-09 (2024-02-24, Saturday) rolls to 2024-02-26 (Monday).
    4. Confirms Full Period metrics:
       - 50 events, 45 clusters
       - original_event_calendar_days: 43
       - original_event_trading_days: 42
       - effective_decision_event_days: 43
       - recorded_no_new_event_trading_days: 159 (159/202)
    5. Confirms 2024 Q1 metrics:
       - 19 events, 16 clusters
       - original_event_calendar_days: 15
       - original_event_trading_days: 14
       - effective_decision_event_days: 15
       - recorded_no_new_event_trading_days: 41 (41/56)
    6. Confirms distinction between no new announcements and news memory slots.
    """
    audit = DesignValidityPreflightAuditor.audit_sample_scope(
        raw_events=RAW_PILOT_EVENTS,
        trading_calendar=twse_trading_calendar,
    )

    full = audit["full_period"]
    assert full["total_events"] == 50
    assert full["total_trading_days"] == 202
    assert full["total_stock_day_units"] == 606  # 202 * 3
    assert full["unique_clusters"] == 45
    assert full["original_event_calendar_days"] == 43
    assert full["original_event_trading_days"] == 42
    assert full["effective_decision_event_days"] == 43
    assert full["recorded_no_new_event_trading_days"] == 159
    assert full["stock_distribution"] == {"2330": 20, "2603": 16, "2881": 14}

    q1 = audit["q1_subperiod"]
    assert q1["total_events"] == 19
    assert q1["total_trading_days"] == 56
    assert q1["total_stock_day_units"] == 168  # 56 * 3
    assert q1["unique_clusters"] == 16
    assert q1["original_event_calendar_days"] == 15
    assert q1["original_event_trading_days"] == 14
    assert q1["effective_decision_event_days"] == 15
    assert q1["recorded_no_new_event_trading_days"] == 41
    assert q1["stock_distribution"] == {"2330": 7, "2603": 6, "2881": 6}

    # Non-trading day event verification
    nte = audit["non_trading_day_events"]
    assert len(nte) == 1
    assert nte[0]["event_id"] == "EV-2330-09"
    assert nte[0]["event_date"] == "2024-02-24"
    assert nte[0]["next_trading_day"] == "2024-02-26"

    # News memory retention note present
    assert "news_memory_retention_note" in audit
    assert "Recent 32 memory slots" in audit["news_memory_retention_note"]

    # Data sufficiency verdict
    assert audit["data_sufficiency_verdict"]["is_q1_alone_sufficient"] is False
    assert audit["data_sufficiency_verdict"]["recommended_period"] == "FULL_PERIOD_EXPLORATORY_ONLY"


def test_v8_17_p0_2_latency_unidentifiable_on_real_events(twse_trading_calendar):
    """
    P0-2, P1-3 & P1-4 & P1-B:
    1. Latest recorded pub_time_str in RAW_PILOT_EVENTS is 18:45:00.
    2. For tau in {0, 5, 15, 30, 60} min, latest available time is 18:45 + 60m = 19:45:00 <= 20:00:00 cutoff.
    3. Confirms changed_event_count=0, changed_stock_day_count=0, changed_decision_state_count=0.
    4. Formally classifies daily eligibility as DAILY_EVENT_ELIGIBILITY_IDENTICAL.
    5. Formally classifies full state tensor equality as FULL_STATE_EQUALITY_NOT_TESTED.
    6. Verifies metadata encoder inspection detects time-dependent feature E23 (Event Age).
    7. Tests verify_encoder=False correctly labels dynamic metadata audit as NOT_VERIFIED.
    """
    audit = DesignValidityPreflightAuditor.audit_latency_identifiability(
        raw_events=RAW_PILOT_EVENTS,
        trading_calendar=twse_trading_calendar,
        tau_list=[0, 5, 15, 30, 60],
        cutoff_time_str="20:00:00",
    )

    assert audit["audit_verdict"] == PreflightStatus.LATENCY_EFFECT_NOT_IDENTIFIABLE.value
    assert audit["daily_eligibility_verdict"] == PreflightStatus.DAILY_EVENT_ELIGIBILITY_IDENTICAL.value
    assert audit["state_tensor_equality_status"] == PreflightStatus.FULL_STATE_EQUALITY_NOT_TESTED.value

    assert audit["max_pub_time_str"] == "18:45:00"
    assert audit["changed_event_count"] == 0
    assert audit["changed_stock_day_count"] == 0
    assert audit["changed_decision_state_count"] == 0
    assert audit["deduplicated_changed_decision_dates_count"] == 0
    assert audit["sum_per_tau_differing_dates_count"] == 0

    for tau_comp in audit["tau_comparisons"].values():
        assert tau_comp["changed_events_count"] == 0
        assert tau_comp["changed_stock_days_count"] == 0
        assert tau_comp["differing_trading_days_count"] == 0

    # Metadata audit checks (P1-B & V8.18 P1-3)
    meta_audit = audit["metadata_features_audit"]
    assert meta_audit["state_tensor_equality_status"] == PreflightStatus.FULL_STATE_EQUALITY_NOT_TESTED.value
    assert meta_audit["raw_dict_contains_availability_timestamp"] is False
    assert meta_audit["contains_tau_dependent_decay_feature"] is True
    assert meta_audit["dynamic_metadata_audit_status"] == PreflightStatus.TIME_DEPENDENT_FEATURE_DETECTED.value
    assert meta_audit["numerical_contract_audit"] is not None
    assert (
        meta_audit["numerical_contract_audit"]["numerical_verification_status"]
        == PreflightStatus.METADATA_NUMERICAL_DIFFERENCE_VERIFIED.value
    )
    assert (
        meta_audit["numerical_contract_audit"]["static_dependency_status"]
        == PreflightStatus.STATIC_ENCODER_DEPENDENCY_CONFIRMED.value
    )

    # Standalone unverified encoder audit check
    unverified_meta = DesignValidityPreflightAuditor.audit_metadata_encoder_features(verify_encoder=False)
    assert unverified_meta["dynamic_metadata_audit_status"] == PreflightStatus.DYNAMIC_METADATA_AUDIT_NOT_VERIFIED.value
    assert unverified_meta["state_tensor_equality_status"] == PreflightStatus.FULL_STATE_EQUALITY_NOT_TESTED.value


def test_v8_17_p0_2_synthetic_boundary_fixtures_quarantined_and_active(twse_trading_calendar):
    """
    P0-2 Item 7, P1-3, P1-5 & P1-A:
    1. Creates synthetic boundary fixtures (19:15, 19:40, 19:55, 20:05).
    2. Confirms fixtures are explicitly tagged SYNTHETIC_FIXTURE_QUARANTINED.
    3. Verifies quarantine isolation check passes against clean empirical dataset.
    4. P1-A: Verifies quarantine breach detection when synthetic fixture is renamed but retains quarantine markers!
    5. Verifies pre-ingestion quarantine gate passes on clean candidate events.
    6. Confirms that on synthetic boundary fixtures, crossing 20:00 causes all 3 metrics to increment.
    """
    fixtures = DesignValidityPreflightAuditor.create_synthetic_boundary_fixtures(
        base_date="2024-03-15",
        stock_id="2881",
    )
    assert len(fixtures) == 4
    for f in fixtures:
        assert f["is_quarantined_test_fixture"] is True
        assert f["status"] == PreflightStatus.SYNTHETIC_FIXTURE_QUARANTINED.value

    # 1. Clean quarantine isolation verification
    iso_result = DesignValidityPreflightAuditor.verify_quarantine_isolation(
        synthetic_fixtures=fixtures,
        empirical_dataset=RAW_PILOT_EVENTS,
    )
    assert iso_result["quarantine_isolation_verified"] is True
    assert iso_result["overlap_count"] == 0

    # 2. Pre-ingestion validation method check
    pre_ingest = DesignValidityPreflightAuditor.validate_empirical_dataset_pre_ingestion(
        empirical_dataset=RAW_PILOT_EVENTS,
        synthetic_fixtures=fixtures,
    )
    assert pre_ingest["quarantine_pre_ingestion_verified"] is True

    # 3. Standard ID-overlap quarantine breach detection
    contaminated_set = list(RAW_PILOT_EVENTS) + [fixtures[0]]
    with pytest.raises(ValueError, match="QUARANTINE BREACH DETECTED"):
        DesignValidityPreflightAuditor.verify_quarantine_isolation(
            synthetic_fixtures=fixtures,
            empirical_dataset=contaminated_set,
        )

    # 4. P1-A: Renamed synthetic event breach detection (new event_id, but retains quarantine markers)
    renamed_fixture = dict(fixtures[0])
    renamed_fixture["event_id"] = "RENAMED-EMPIRICAL-LOOKALIKE-999"
    renamed_fixture["article_id"] = "ART-RENAMED-999"
    # Even with a brand-new non-overlapping ID, having is_quarantined_test_fixture=True must trigger rejection!
    renamed_contaminated_set = list(RAW_PILOT_EVENTS) + [renamed_fixture]
    with pytest.raises(ValueError, match="QUARANTINE BREACH DETECTED"):
        DesignValidityPreflightAuditor.validate_empirical_dataset_pre_ingestion(
            empirical_dataset=renamed_contaminated_set,
        )
    with pytest.raises(ValueError, match="QUARANTINE BREACH DETECTED"):
        DesignValidityPreflightAuditor.verify_quarantine_isolation(
            synthetic_fixtures=fixtures,
            empirical_dataset=renamed_contaminated_set,
        )

    # 5. Run latency audit on synthetic boundary fixtures
    synth_audit = DesignValidityPreflightAuditor.audit_latency_identifiability(
        raw_events=fixtures,
        trading_calendar=twse_trading_calendar,
        tau_list=[0, 5, 15, 30, 60],
        cutoff_time_str="20:00:00",
    )
    assert synth_audit["audit_verdict"] == PreflightStatus.LATENCY_EFFECT_IDENTIFIABLE.value
    assert synth_audit["daily_eligibility_verdict"] == PreflightStatus.DAILY_EVENT_ELIGIBILITY_CHANGED.value
    assert synth_audit["changed_event_count"] == 3
    assert synth_audit["changed_stock_day_count"] == 2
    assert synth_audit["changed_decision_state_count"] == 2
    assert synth_audit["sum_per_tau_differing_dates_count"] == 6


def test_v8_17_p1_3_robust_fragility_metrics():
    """
    P1-3:
    1. Uses Absolute Delta Sharpe as primary sensitivity metric.
    2. Handles baseline Sharpe <= 0 or near zero safely (FI set to None, flagged FI_UNRELIABLE).
    3. Calculates bounded FI when baseline Sharpe is reasonable (>= 0.20).
    """
    # Case 1: Healthy baseline Sharpe
    res_valid = DesignValidityPreflightAuditor.compute_delta_sharpe(
        sharpe_baseline=1.2,
        sharpe_perturbed=1.0,
        epsilon=0.20,
    )
    assert res_valid["absolute_delta_sharpe"] == 0.2
    assert res_valid["fragility_index"] == 0.1667
    assert res_valid["fi_status"] == "VALID_HEURISTIC_ESTIMATE"

    # Case 2: Near-zero baseline Sharpe (< 0.20)
    res_near_zero = DesignValidityPreflightAuditor.compute_delta_sharpe(
        sharpe_baseline=0.08,
        sharpe_perturbed=0.02,
        epsilon=0.20,
    )
    assert res_near_zero["absolute_delta_sharpe"] == 0.06
    assert res_near_zero["fragility_index"] is None
    assert res_near_zero["fi_status"] == PreflightStatus.FI_UNRELIABLE_NEAR_ZERO_OR_NEGATIVE.value

    # Case 3: Negative baseline Sharpe
    res_neg = DesignValidityPreflightAuditor.compute_delta_sharpe(
        sharpe_baseline=-0.4,
        sharpe_perturbed=-0.7,
        epsilon=0.20,
    )
    assert res_neg["absolute_delta_sharpe"] == 0.3
    assert res_neg["fragility_index"] is None
    assert res_neg["fi_status"] == PreflightStatus.FI_UNRELIABLE_NEAR_ZERO_OR_NEGATIVE.value


def test_v8_17_p1_2_and_p1_5_chronological_fold_alignment_and_leakage(twse_trading_calendar):
    """
    P1-2 & P1-5:
    1. Canonical Fold Alignment: Train 115 / Purge 10 / Val 25 / Embargo 5 / Test 47 = 202 days.
    2. Verifies date spans, trading days, event counts, and cluster counts per fold.
    3. Verifies Cluster Split Leakage Checked (zero shared clusters across Train, Val, Test).
    4. Confirms Validation fold event count is 4 (< 10), triggering INSUFFICIENT_EFFECTIVE_SAMPLE.
    5. Confirms exploratory design hypotheses justification for 10d purge and 5d embargo.
    6. Confirms adequacy status is explicitly labeled HYPOTHESIS_ONLY_UNPROVEN.
    """
    fold_audit = DesignValidityPreflightAuditor.audit_chronological_folds(
        raw_events=RAW_PILOT_EVENTS,
        trading_calendar=twse_trading_calendar,
    )

    assert fold_audit["fold_config_status"] == PreflightStatus.CHRONOLOGICAL_FOLD_CONFIG_VERIFIED.value
    assert fold_audit["total_calendar_trading_days"] == 202
    assert fold_audit["train_days_count"] == 115
    assert fold_audit["purge_days_count"] == 10
    assert fold_audit["val_days_count"] == 25
    assert fold_audit["embargo_days_count"] == 5
    assert fold_audit["test_days_count"] == 47

    folds = fold_audit["folds"]
    # Train Fold
    assert folds["TRAIN"]["start_date"] == "2023-06-01"
    assert folds["TRAIN"]["end_date"] == "2023-11-16"
    assert folds["TRAIN"]["trading_days"] == 115
    assert folds["TRAIN"]["events_count"] == 26
    assert folds["TRAIN"]["unique_clusters"] == 25

    # Purge Window
    assert folds["PURGE"]["start_date"] == "2023-11-17"
    assert folds["PURGE"]["end_date"] == "2023-11-30"
    assert folds["PURGE"]["trading_days"] == 10
    assert folds["PURGE"]["events_count"] == 2
    assert folds["PURGE"]["unique_clusters"] == 2

    # Validation Fold
    assert folds["VAL"]["start_date"] == "2023-12-01"
    assert folds["VAL"]["end_date"] == "2024-01-05"
    assert folds["VAL"]["trading_days"] == 25
    assert folds["VAL"]["events_count"] == 4
    assert folds["VAL"]["unique_clusters"] == 3

    # Embargo Window
    assert folds["EMBARGO"]["start_date"] == "2024-01-08"
    assert folds["EMBARGO"]["end_date"] == "2024-01-12"
    assert folds["EMBARGO"]["trading_days"] == 5
    assert folds["EMBARGO"]["events_count"] == 2
    assert folds["EMBARGO"]["unique_clusters"] == 1

    # Test Fold
    assert folds["TEST"]["start_date"] == "2024-01-15"
    assert folds["TEST"]["end_date"] == "2024-03-29"
    assert folds["TEST"]["trading_days"] == 47
    assert folds["TEST"]["events_count"] == 16
    assert folds["TEST"]["unique_clusters"] == 14

    # Cluster Split Leakage Verification
    leak_check = fold_audit["cluster_split_leakage_check"]
    assert leak_check["status"] == PreflightStatus.CLUSTER_SPLIT_LEAKAGE_CHECKED.value
    assert leak_check["cluster_leakage_detected"] is False
    assert len(leak_check["shared_clusters_train_val"]) == 0
    assert len(leak_check["shared_clusters_train_test"]) == 0
    assert len(leak_check["shared_clusters_val_test"]) == 0

    # Feasibility status & design hypothesis adequacy
    assert fold_audit["feasibility_status"] == PreflightStatus.INSUFFICIENT_EFFECTIVE_SAMPLE.value
    assert "design_hypotheses" in fold_audit
    assert fold_audit["design_hypotheses"]["adequacy_status"] == PreflightStatus.HYPOTHESIS_ONLY_UNPROVEN.value
    assert "HYPOTHESIS_ONLY_UNPROVEN" in fold_audit["design_hypotheses"]["justification"]


def test_v8_18_p1_3_metadata_encoder_numerical_contract():
    """
    V8.18 P1-3:
    1. Executes numerical contract testing of extract_event_metadata across tau in {0, 5, 15, 30, 60}.
    2. Verifies dimension E23 (Event Age) diff matches tau / 1440.0 days.
    3. Verifies other 23 dimensions remain 100% strictly identical (max diff == 0.0).
    4. Discloses pipeline tau integration gap (default data loading sets tau=0).
    5. Confirms bounded status:
       - STATIC_ENCODER_DEPENDENCY_CONFIRMED
       - METADATA_NUMERICAL_DIFFERENCE_VERIFIED
       - FULL_STATE_EQUALITY_NOT_TESTED
    """
    contract_res = DesignValidityPreflightAuditor.audit_metadata_encoder_numerical_contract(
        tau_list=[0, 5, 15, 30, 60]
    )

    assert contract_res["numerical_verification_status"] == PreflightStatus.METADATA_NUMERICAL_DIFFERENCE_VERIFIED.value
    assert contract_res["static_dependency_status"] == PreflightStatus.STATIC_ENCODER_DEPENDENCY_CONFIRMED.value
    assert contract_res["full_state_tensor_equality_status"] == PreflightStatus.FULL_STATE_EQUALITY_NOT_TESTED.value

    assert contract_res["all_e23_matched"] is True
    assert contract_res["all_other_dims_identical"] is True

    evals = contract_res["tau_evaluations"]
    assert len(evals) == 4

    for tau, key in [(5, "tau_5_vs_tau_0"), (15, "tau_15_vs_tau_0"), (30, "tau_30_vs_tau_0"), (60, "tau_60_vs_tau_0")]:
        eval_item = evals[key]
        assert eval_item["is_e23_match"] is True
        assert eval_item["is_other_23_dims_identical"] is True
        assert eval_item["max_diff_in_other_23_dims"] == 0.0
        assert eval_item["tau_minutes"] == tau

    # Pipeline integration gap disclosure
    disclosure = contract_res["pipeline_integration_disclosure"]
    assert disclosure["gap_identified"] is True
    assert "tau_minutes" in disclosure["description"]

    # Full state tensor boundary note
    assert "FULL_STATE_EQUALITY_NOT_TESTED" in contract_res["tensor_equality_boundary_note"]


def test_v8_18_p1_4_quarantine_ingestion_boundary_and_syn_preservation():
    """
    V8.18 P1-4:
    1. Confirms syndicated media articles with '-SYN' suffix (e.g. EV-2330-03-SYN)
       are genuine empirical records and NOT falsely quarantined.
    2. Confirms PilotSourceManifest.generate_manifest(enforce_quarantine=True) passes on clean data.
    3. Confirms PilotDatasetBuilder().build_dataset() enforces Quarantine Gate.
    4. Injects a renamed synthetic test fixture (event_id="RENAMED-SYNTH-999", is_quarantined_test_fixture=True)
       and verifies immediate ValueError at both Manifest generation and Dataset building.
    5. Verifies parameterized tau_minutes in build_dataset() propagates into sample.available_at.
    """
    # 1. Verify syndicated articles are present and valid
    syn_articles = [ev for ev in RAW_PILOT_EVENTS if "-SYN" in ev["event_id"]]
    assert len(syn_articles) == 3
    for syn_art in syn_articles:
        assert syn_art["source_type"] == "MEDIA_ARCHIVE"
        assert "is_quarantined_test_fixture" not in syn_art or not syn_art["is_quarantined_test_fixture"]

    # 2. PilotSourceManifest with enforce_quarantine=True succeeds on clean RAW_PILOT_EVENTS
    manifest = PilotSourceManifest.generate_manifest(RAW_PILOT_EVENTS, enforce_quarantine=True)
    assert len(manifest) == 50
    assert "EV-2330-03-SYN" in manifest
    assert "EV-2603-06-SYN" in manifest
    assert "EV-2881-06-SYN" in manifest

    # 3. PilotDatasetBuilder clean build
    builder = PilotDatasetBuilder()
    clean_subset = RAW_PILOT_EVENTS[:3]
    samples = builder.build_dataset(raw_events=clean_subset, tau_minutes=0)
    assert len(samples) == 3
    for s, ev in zip(samples, clean_subset):
        assert s.available_at == s.published_at

    # 4. Parameterized tau propagation
    samples_tau15 = builder.build_dataset(raw_events=clean_subset, tau_minutes=15)
    assert len(samples_tau15) == 3
    for s, ev in zip(samples_tau15, clean_subset):
        assert s.available_at == s.published_at + pd.Timedelta(minutes=15)

    # 5. Quarantine Breach Rejection on Renamed Synthetic Fixture
    synthetic_fixtures = DesignValidityPreflightAuditor.create_synthetic_boundary_fixtures()
    renamed_fixture = dict(synthetic_fixtures[0])
    renamed_fixture["event_id"] = "RENAMED-EMPIRICAL-LOOKALIKE-999"
    renamed_fixture["article_id"] = "ART-RENAMED-999"

    contaminated_events = list(clean_subset) + [renamed_fixture]

    # Must raise in PilotSourceManifest
    with pytest.raises(ValueError, match="QUARANTINE BREACH DETECTED"):
        PilotSourceManifest.generate_manifest(contaminated_events, enforce_quarantine=True)

    # Must raise in PilotDatasetBuilder
    with pytest.raises(ValueError, match="QUARANTINE BREACH DETECTED"):
        builder.build_dataset(raw_events=contaminated_events)


def test_v8_18_p1_1_unify_estimated_availability_provenance():
    """
    V8.18 P1-1:
    1. source_published_at always retains original publication time.
    2. estimated_provider_available_at is labeled as hypothesis.
    3. verified_provider_available_at is None for pilot data (unverified).
    4. Enforces time contract consistency across Sample, Manifest, and Metadata.
    5. Confirms metadata E23 matches (decision - available) formula.
    6. Verifies that Estimated Availability is strictly NOT elevated to Formal PIT.
    """
    builder = PilotDatasetBuilder()
    subset = RAW_PILOT_EVENTS[:5]
    manifest = PilotSourceManifest.generate_manifest(subset, tau_minutes=15)

    samples = builder.build_dataset(raw_events=subset, tau_minutes=15)
    assert len(samples) == 5

    for s in samples:
        m_rec = manifest[s.event_id]
        # 1. Publication timestamp integrity
        assert s.source_published_at == s.published_at
        assert s.source_published_at == m_rec.source_published_at

        # 2. Estimated provider available integrity
        assert s.estimated_provider_available_at == s.source_published_at + pd.Timedelta(minutes=15)
        assert s.estimated_provider_available_at == m_rec.estimated_provider_available_at

        # 3. Verified evidence absent -> None & is_formal_pit_validated is False
        assert s.verified_provider_available_at is None
        assert m_rec.verified_provider_available_at is None
        assert s.is_formal_pit_validated is False
        assert m_rec.is_formal_pit_validated is False
        assert s.provenance_category == "ESTIMATED_PIT_NEWS"

        # 4. Consistency contract passes cleanly
        contract_res = builder.validate_time_contract_consistency(s, manifest_rec=m_rec)
        assert contract_res["contract_valid"] is True

    # 5. Test contract breach: self-elevation to Formal PIT without verified receipt
    bad_sample = samples[0]
    bad_sample.is_formal_pit_validated = True
    with pytest.raises(ValueError, match="Formal PIT Violation"):
        builder.validate_time_contract_consistency(bad_sample)
    bad_sample.is_formal_pit_validated = False  # restore

    # 6. Test contract breach: Metadata E23 tampering
    orig_e23 = bad_sample.metadata_32d[22]
    bad_sample.metadata_32d[22] = 999.0
    with pytest.raises(ValueError, match="Metadata E23 Inconsistency"):
        builder.validate_time_contract_consistency(bad_sample)
    bad_sample.metadata_32d[22] = orig_e23  # restore


def test_v8_18_p1_2_trading_day_rollover_integration():
    """
    V8.18 P1-2:
    1. Empty raw_events=[] returns [] without loading full RAW_PILOT_EVENTS.
    2. EV-2330-09 (Saturday 2024-02-24 16:00:00) rolls to 2024-02-26 20:00:00.
    3. Next-Session Execution (T+1) and label dates align correctly:
       - t_decision = 2024-02-26 20:00:00 (Monday)
       - t1_day = 2024-02-27 (Tuesday)
       - t2_day = 2024-02-29 (Thursday, after 2/28 holiday)
    4. Evening filings past 20:00:00 roll to the next trading session.
    5. Tau pushing available time past 20:00:00 rolls to the next trading session.
    6. Boundary out-of-bounds filings at calendar end fail closed.
    """
    builder = PilotDatasetBuilder()

    # 1. Empty raw_events test
    assert builder.build_dataset(raw_events=[]) == []

    # 2. EV-2330-09 Saturday rollover test
    ev_09_list = [e for e in RAW_PILOT_EVENTS if e["event_id"] == "EV-2330-09"]
    samples_09 = builder.build_dataset(raw_events=ev_09_list)
    assert len(samples_09) == 1
    s_09 = samples_09[0]

    assert s_09.published_at == pd.Timestamp("2024-02-24 16:00:00", tz="Asia/Taipei")
    assert s_09.available_at == pd.Timestamp("2024-02-24 16:00:00", tz="Asia/Taipei")
    assert s_09.decision_timestamp == pd.Timestamp("2024-02-26 20:00:00", tz="Asia/Taipei")
    assert s_09.available_at <= s_09.decision_timestamp

    # Check label execution dates
    assert s_09.ex_post_label is not None
    assert s_09.ex_post_label.missing_status == "AVAILABLE"
    assert s_09.ex_post_label.t1_trading_date.strftime("%Y-%m-%d") == "2024-02-27"
    assert s_09.ex_post_label.t2_trading_date.strftime("%Y-%m-%d") == "2024-02-29"

    # 3. Evening announcement test (published after 20:00 on Friday 2024-03-15)
    evening_event = dict(RAW_PILOT_EVENTS[0])
    evening_event["event_id"] = "EV-TEST-EVENING"
    evening_event["date_str"] = "2024-03-15"
    evening_event["pub_time_str"] = "20:15:00"

    samples_evening = builder.build_dataset(raw_events=[evening_event])
    assert len(samples_evening) == 1
    assert samples_evening[0].decision_timestamp.strftime("%Y-%m-%d") == "2024-03-18"
    assert samples_evening[0].available_at <= samples_evening[0].decision_timestamp

    # 4. Tau pushes available time past 20:00 cutoff test (19:45:00 + 30m = 20:15:00)
    tau_push_event = dict(RAW_PILOT_EVENTS[0])
    tau_push_event["event_id"] = "EV-TEST-TAU-PUSH"
    tau_push_event["date_str"] = "2024-03-15"
    tau_push_event["pub_time_str"] = "19:45:00"

    samples_pushed = builder.build_dataset(raw_events=[tau_push_event], tau_minutes=30)
    assert len(samples_pushed) == 1
    assert samples_pushed[0].decision_timestamp.strftime("%Y-%m-%d") == "2024-03-18"
    assert samples_pushed[0].available_at <= samples_pushed[0].decision_timestamp

    # 5. Out of bounds at calendar end fails closed
    oob_event = dict(RAW_PILOT_EVENTS[0])
    oob_event["event_id"] = "EV-TEST-OOB"
    oob_event["date_str"] = "2024-03-29"
    oob_event["pub_time_str"] = "20:15:00"

    with pytest.raises(ValueError, match="FAIL_CLOSED_NO_NEXT_TRADING_SESSION"):
        builder.build_dataset(raw_events=[oob_event])


