"""
Unit tests for V8.17 Design Validity Preflight & Audit Framework.
Verifies sample scope corrections, latency unidentifiability on real events,
synthetic boundary fixtures quarantine, robust fragility metrics, and fold feasibility.
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
    P0-1:
    1. Re-computes time distribution, ticker distribution, and cluster counts.
    2. Separately evaluates Full Period (2023-06 to 2024-03) and 2024 Q1 (2024-01 to 2024-03).
    3. Confirms non-trading day event EV-2330-09 (2024-02-24, Saturday) rolls to 2024-02-26 (Monday).
    4. Confirms Q1 contains 19 events, 15 distinct dates, 16 clusters, and 56 trading sessions.
    5. Confirms Full Period contains 50 events, 43 distinct dates, 45 clusters, and 202 trading sessions.
    """
    audit = DesignValidityPreflightAuditor.audit_sample_scope(
        raw_events=RAW_PILOT_EVENTS,
        trading_calendar=twse_trading_calendar,
    )

    full = audit["full_period"]
    assert full["total_events"] == 50
    assert full["total_trading_days"] == 202
    assert full["total_stock_day_units"] == 606  # 202 * 3
    assert full["unique_event_calendar_days"] == 43
    assert full["unique_clusters"] == 45
    assert full["stock_distribution"] == {"2330": 20, "2603": 16, "2881": 14}

    q1 = audit["q1_subperiod"]
    assert q1["total_events"] == 19
    assert q1["total_trading_days"] == 56
    assert q1["total_stock_day_units"] == 168  # 56 * 3
    assert q1["unique_event_calendar_days"] == 15
    assert q1["unique_clusters"] == 16
    assert q1["stock_distribution"] == {"2330": 7, "2603": 6, "2881": 6}

    # Non-trading day event verification
    nte = audit["non_trading_day_events"]
    assert len(nte) == 1
    assert nte[0]["event_id"] == "EV-2330-09"
    assert nte[0]["event_date"] == "2024-02-24"
    assert nte[0]["next_trading_day"] == "2024-02-26"

    # Data sufficiency verdict
    assert audit["data_sufficiency_verdict"]["is_q1_alone_sufficient"] is False
    assert audit["data_sufficiency_verdict"]["recommended_period"] == "FULL_PERIOD_EXPLORATORY_ONLY"


def test_v8_17_p0_2_latency_unidentifiable_on_real_events(twse_trading_calendar):
    """
    P0-2:
    1. Latest recorded pub_time_str in RAW_PILOT_EVENTS is 18:45:00.
    2. For tau in {0, 5, 15, 30, 60} min, latest available time is 18:45 + 60m = 19:45:00 <= 20:00:00 cutoff.
    3. Confirms changed_event_count=0, changed_stock_day_count=0, changed_decision_state_count=0.
    4. Formally classifies empirical latency effect as LATENCY_EFFECT_NOT_IDENTIFIABLE.
    """
    audit = DesignValidityPreflightAuditor.audit_latency_identifiability(
        raw_events=RAW_PILOT_EVENTS,
        trading_calendar=twse_trading_calendar,
        tau_list=[0, 5, 15, 30, 60],
        cutoff_time_str="20:00:00",
    )

    assert audit["audit_verdict"] == PreflightStatus.LATENCY_EFFECT_NOT_IDENTIFIABLE.value
    assert audit["max_pub_time_str"] == "18:45:00"
    assert audit["changed_event_count"] == 0
    assert audit["changed_stock_day_count"] == 0
    assert audit["changed_decision_state_count"] == 0

    for tau_comp in audit["tau_comparisons"].values():
        assert tau_comp["differing_trading_days_count"] == 0


def test_v8_17_p0_2_synthetic_boundary_fixtures_quarantined_and_active(twse_trading_calendar):
    """
    P0-2 Item 7:
    1. Creates synthetic boundary fixtures (19:15, 19:40, 19:55, 20:05).
    2. Confirms fixtures are explicitly tagged SYNTHETIC_FIXTURE_QUARANTINED.
    3. Confirms that on synthetic boundary fixtures, latency DOES shift decision state across cutoff.
    """
    fixtures = DesignValidityPreflightAuditor.create_synthetic_boundary_fixtures(
        base_date="2024-03-15",
        stock_id="2881",
    )
    assert len(fixtures) == 4
    for f in fixtures:
        assert f["is_quarantined_test_fixture"] is True
        assert f["status"] == PreflightStatus.SYNTHETIC_FIXTURE_QUARANTINED.value

    # Run latency audit on synthetic boundary fixtures
    synth_audit = DesignValidityPreflightAuditor.audit_latency_identifiability(
        raw_events=fixtures,
        trading_calendar=twse_trading_calendar,
        tau_list=[0, 5, 15, 30, 60],
        cutoff_time_str="20:00:00",
    )
    # Boundary fixtures at 19:40 and 19:15 cross 20:00 cutoff under tau=30 and tau=60!
    assert synth_audit["audit_verdict"] == PreflightStatus.LATENCY_EFFECT_IDENTIFIABLE.value
    assert synth_audit["changed_decision_state_count"] > 0


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


def test_v8_17_p1_5_chronological_fold_feasibility_insufficient_sample(twse_trading_calendar):
    """
    P1-5:
    1. Evaluates Walk-Forward feasibility with Purge (10d) and Embargo (5d).
    2. Reports Train (26 events), Purge (2), Val (4 events), Embargo (2), Test (16 events).
    3. Confirms Validation fold event count is 4 (< 10).
    4. Formally classifies fold sample feasibility as INSUFFICIENT_EFFECTIVE_SAMPLE.
    """
    fold_audit = DesignValidityPreflightAuditor.audit_chronological_folds(
        raw_events=RAW_PILOT_EVENTS,
        trading_calendar=twse_trading_calendar,
        train_ratio=0.57,
        val_ratio=0.12,
        purge_window_days=10,
        embargo_window_days=5,
    )

    assert fold_audit["feasibility_status"] == PreflightStatus.INSUFFICIENT_EFFECTIVE_SAMPLE.value
    val_events = fold_audit["folds"]["VAL"]["events_count"]
    assert val_events < 10
    assert fold_audit["total_calendar_trading_days"] == 202
