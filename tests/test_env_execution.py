"""
Unit Tests for TWSE Execution Simulation and Point-in-Time Gym Environment
"""

import pytest
import numpy as np
from stock_rl.config.settings import TradingConfig
from stock_rl.config.schema import ACTION_CASH, ACTION_LONG
from stock_rl.env.execution import TWSEExecutionSimulator
from stock_rl.env.stock_trading_env import PointInTimeStockEnv


def test_twse_execution_buy_and_sell_taxes():
    config = TradingConfig()
    sim = TWSEExecutionSimulator(config)

    # 1. Buy: Cash -> Long
    initial_cash = 100_000.0
    cash_after_buy, shares, res_buy = sim.execute_target_position(
        current_position=ACTION_CASH,
        target_position=ACTION_LONG,
        current_cash=initial_cash,
        current_shares=0.0,
        prev_close_price=100.0,
        open_price=100.0,
    )
    assert res_buy.executed_action == ACTION_LONG
    assert res_buy.shares_traded > 0
    assert res_buy.tax_paid == 0.0  # Zero tax on buying in TWSE
    assert res_buy.commission_paid > 0.0

    # 2. Sell: Long -> Cash
    cash_after_sell, shares_zero, res_sell = sim.execute_target_position(
        current_position=ACTION_LONG,
        target_position=ACTION_CASH,
        current_cash=cash_after_buy,
        current_shares=shares,
        prev_close_price=100.0,
        open_price=100.0,
    )
    assert res_sell.executed_action == ACTION_CASH
    assert shares_zero == 0.0
    # TWSE Securities Transaction Tax: 0.3% deducted on sell
    expected_approx_tax = (shares * 100.0 * (1.0 - config.base_slippage_rate)) * 0.003
    assert abs(res_sell.tax_paid - expected_approx_tax) < 1.0


def test_limit_lock_rejection():
    config = TradingConfig()
    sim = TWSEExecutionSimulator(config)

    # Previous close = 100.0, open price = 110.0 (+10% limit up)
    # Buy order should be rejected due to limit-up lock
    cash, shares, res = sim.execute_target_position(
        current_position=ACTION_CASH,
        target_position=ACTION_LONG,
        current_cash=100_000.0,
        current_shares=0.0,
        prev_close_price=100.0,
        open_price=110.0,
    )
    assert res.order_rejected is True
    assert res.executed_action == ACTION_CASH


def test_point_in_time_env_step_cycle():
    env = PointInTimeStockEnv()
    obs, info = env.reset()
    assert info["portfolio_equity"] == 1_000_000.0

    # Take Long action
    obs, reward, terminated, truncated, step_info = env.step(ACTION_LONG)
    assert "portfolio_equity" in step_info
    assert "reward" in step_info
    assert not np.isnan(reward)
    assert not np.isnan(step_info["portfolio_equity"])
