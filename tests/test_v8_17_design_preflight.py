"""
Unit tests for V8.17 Design Validity Preflight & Audit Framework.
Verifies sample scope corrections with weekend rollover, latency identifiability metrics,
distinction between daily eligibility equality and full tensor equality,
synthetic boundary fixtures quarantine, robust fragility metrics, and chronological fold alignment.
"""

import pytest
import pandas as pd
from stock_rl.evaluation.pilot_dataset import RAW_PILOT_EVENTS
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
    P0-2, P1-3 & P1-4:
    1. Latest recorded pub_time_str in RAW_PILOT_EVENTS is 18:45:00.
    2. For tau in {0, 5, 15, 30, 60} min, latest available time is 18:45 + 60m = 19:45:00 <= 20:00:00 cutoff.
    3. Confirms changed_event_count=0, changed_stock_day_count=0, changed_decision_state_count=0.
    4. Formally classifies daily eligibility as DAILY_EVENT_ELIGIBILITY_IDENTICAL.
    5. Formally classifies full state tensor equality as FULL_STATE_EQUALITY_NOT_TESTED.
    6. Confirms metadata feature audit inspected features and verified availability_timestamp absence.
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

    # Metadata audit checks
    meta_audit = audit["metadata_features_audit"]
    assert meta_audit["state_tensor_equality_status"] == PreflightStatus.FULL_STATE_EQUALITY_NOT_TESTED.value
    assert meta_audit["contains_availability_timestamp"] is False


def test_v8_17_p0_2_synthetic_boundary_fixtures_quarantined_and_active(twse_trading_calendar):
    """
    P0-2 Item 7, P1-3 & P1-5:
    1. Creates synthetic boundary fixtures (19:15, 19:40, 19:55, 20:05).
    2. Confirms fixtures are explicitly tagged SYNTHETIC_FIXTURE_QUARANTINED.
    3. Verifies quarantine isolation check passes against empirical dataset.
    4. Verifies quarantine breach detection when synthetic fixture is contaminated into empirical set.
    5. Confirms that on synthetic boundary fixtures, crossing 20:00 causes all 3 metrics to increment:
       - changed_event_count > 0 (3 events)
       - changed_stock_day_count > 0 (2 stock days)
       - changed_decision_state_count > 0 (2 decision dates)
    """
    fixtures = DesignValidityPreflightAuditor.create_synthetic_boundary_fixtures(
        base_date="2024-03-15",
        stock_id="2881",
    )
    assert len(fixtures) == 4
    for f in fixtures:
        assert f["is_quarantined_test_fixture"] is True
        assert f["status"] == PreflightStatus.SYNTHETIC_FIXTURE_QUARANTINED.value

    # Quarantine isolation verification
    iso_result = DesignValidityPreflightAuditor.verify_quarantine_isolation(
        synthetic_fixtures=fixtures,
        empirical_dataset=RAW_PILOT_EVENTS,
    )
    assert iso_result["quarantine_isolation_verified"] is True
    assert iso_result["overlap_count"] == 0

    # Quarantine breach detection
    contaminated_set = list(RAW_PILOT_EVENTS) + [fixtures[0]]
    with pytest.raises(ValueError, match="QUARANTINE BREACH DETECTED"):
        DesignValidityPreflightAuditor.verify_quarantine_isolation(
            synthetic_fixtures=fixtures,
            empirical_dataset=contaminated_set,
        )

    # Run latency audit on synthetic boundary fixtures
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

    # Feasibility status
    assert fold_audit["feasibility_status"] == PreflightStatus.INSUFFICIENT_EFFECTIVE_SAMPLE.value
    assert "design_hypotheses" in fold_audit
    assert "10-day purge window and 5-day embargo window" in fold_audit["design_hypotheses"]["justification"]
