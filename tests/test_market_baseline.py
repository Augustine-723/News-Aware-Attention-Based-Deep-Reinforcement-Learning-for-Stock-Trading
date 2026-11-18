import pytest

from stock_rl.config.schema import ACTION_CASH, ACTION_LONG
from stock_rl.evaluation.market_baseline import replay


def test_replay_with_real_cached_market_sessions():
    result = replay("2330")
    days = result["evaluation_sessions"]
    assert result["cached_sessions"] == 202
    assert result["decision_steps"] == len(days) - 1 == 55
    assert days[days.index("2024-02-05") + 1] == "2024-02-15"
    assert days[-1] == "2024-03-29"
    assert result["populated_event_slots"] == 0
    cash = result["policies"]["cash"]
    hold = result["policies"]["buy_and_hold"]
    assert cash["target_actions"] == [ACTION_CASH] * 55
    assert hold["target_actions"] == [ACTION_LONG] * 55
    assert cash["total_return"] == 0
    assert hold["total_return"] == pytest.approx(0.3321, abs=0.0001)
    assert hold["costs"]["commission"] > 0
    assert len(hold["daily_equity"]) == len(days)
