"""Unit tests for transaction friction hysteresis band."""
import numpy as np
import pytest


def act_hysteresis(q_cash, q_long, current_pos, theta):
    if current_pos == 1:
        return 0 if (q_cash - q_long > theta) else 1
    else:
        return 1 if (q_long > q_cash) else 0


def test_hysteresis_preserves_position_under_minor_drawdown():
    # Model in long position (pos=1), Q(cash) is slightly higher than Q(long) by 0.008
    q_cash = -0.100
    q_long = -0.108
    theta = 0.015
    # Without hysteresis, it would exit
    assert act_hysteresis(q_cash, q_long, 1, 0.0) == 0
    # With hysteresis theta=0.015, difference is 0.008 <= 0.015, so it holds
    assert act_hysteresis(q_cash, q_long, 1, theta) == 1


def test_hysteresis_allows_exit_on_large_conviction():
    # Model in long position (pos=1), Q(cash) is significantly higher by 0.025
    q_cash = -0.100
    q_long = -0.125
    theta = 0.015
    # Difference is 0.025 > 0.015, so it exits
    assert act_hysteresis(q_cash, q_long, 1, theta) == 0


def test_hysteresis_clean_entry():
    # Model in cash (pos=0), Q(long) > Q(cash) triggers entry regardless of theta
    q_cash = -0.120
    q_long = -0.110
    theta = 0.015
    assert act_hysteresis(q_cash, q_long, 0, theta) == 1

    # Q(long) <= Q(cash) stays in cash
    assert act_hysteresis(-0.100, -0.110, 0, theta) == 0
