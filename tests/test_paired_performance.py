import pytest

from stock_rl.evaluation.paired_performance import compare


def sleeve(equity):
    return {"2330": {"session_dates": [f"2024-03-{i:02d}" for i in range(1, 8)],
                     "equity": equity}}


def test_identical_variable_returns_have_zero_delta_and_interval():
    curve = sleeve([100, 102, 103, 101, 102, 105, 104])
    report = compare(curve, curve, block_length=2, replicates=50)
    assert report["matched_return_sessions"] == 6
    assert report["delta_sharpe"] == 0
    assert report["delta_sharpe_ci95"] == [0, 0]


def test_unmatched_sessions_fail_closed():
    control = sleeve([100, 102, 103, 101, 102, 105, 104])
    treatment = sleeve([100, 103, 104, 102, 104, 105, 108])
    treatment["2330"]["session_dates"][-1] = "2024-03-08"
    with pytest.raises(ValueError, match="unmatched trading sessions"):
        compare(control, treatment, block_length=2)


def test_cash_has_undefined_sharpe():
    cash = sleeve([100] * 7)
    report = compare(cash, cash, block_length=2, replicates=50)
    assert report["status"] == "undefined_sharpe_zero_volatility"
    assert report["delta_sharpe_ci95"] is None
