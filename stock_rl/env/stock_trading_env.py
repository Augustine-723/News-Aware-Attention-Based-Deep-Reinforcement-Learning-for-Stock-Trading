"""
Point-in-Time Single-Stock Trading Gym Environment V8.0
Standardized gymnasium.Env implementing 20:00 decision cycle and mark-to-market equity log return.

Simulation Modeling Assumptions:
1. Continuous share proxy: Portfolio trades continuous share quantities as a high-fidelity proxy
   for 1,000-share board lots and intraday odd lots (officially permitted by TWSE since October 2020).
2. Execution mechanics: Next-day open price execution with broker commission discounts (0.1425% * 0.28),
   securities transaction tax (0.3% on sales), and linear bid-ask slippage.
3. Limit lock rejection: Limit-up buy orders and limit-down sell orders are rejected instantaneously at open.
   When rejected:
   - Rejected Buy: portfolio remains in Cash (0), holding days = 0, entry price = 0.0.
   - Rejected Sell: portfolio remains Long (1), holding days accumulate (+1), original entry price preserved.
4. Corporate Actions: Cash dividends and stock rights are accounted for via cash-adjusted total return pricing.
"""

from typing import Dict, Any, Tuple, Optional, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from stock_rl.config.schema import (
    ACTION_CASH,
    ACTION_LONG,
    NUM_ACTIONS,
    POSITION_STATE_DIM,
)
from stock_rl.config.settings import TradingConfig
from stock_rl.env.execution import TWSEExecutionSimulator, ExecutionResult
from stock_rl.data.synthetic_data import MultimodalBatch, generate_synthetic_batch


class PointInTimeStockEnv(gym.Env):
    """
    Simulates daily sequential trading for a single stock under TWSE market rules.
    Conforms to OpenAI Gymnasium interface.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices_df: Optional[np.ndarray] = None, # Array of [T, 4] for [Open, High, Low, Close]
        multimodal_batches: Optional[List[MultimodalBatch]] = None,
        config: TradingConfig = TradingConfig(),
        initial_capital: float = 1_000_000.0,
    ):
        super().__init__()
        self.config = config
        self.initial_capital = initial_capital
        self.simulator = TWSEExecutionSimulator(config)

        # Action space: 0 (Cash), 1 (Long)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # If data is not provided, generate synthetic episode of length 120
        if prices_df is None or multimodal_batches is None:
            self.episode_length = 120
            self._generate_synthetic_episode()
        else:
            self.prices_df = prices_df
            self.multimodal_batches = multimodal_batches
            self.episode_length = len(prices_df)

        # Internal state
        self.current_step: int = 0
        self.cash: float = initial_capital
        self.shares: float = 0.0
        self.current_position: int = ACTION_CASH
        self.entry_price: float = 0.0
        self.holding_days: int = 0
        self.portfolio_equity: float = initial_capital
        self.equity_history: List[float] = []

    def _generate_synthetic_episode(self):
        """Generates a realistic synthetic episode with prices and batch features."""
        rng = np.random.default_rng(123)
        prices = np.zeros((self.episode_length, 4))
        p = 100.0
        for t in range(self.episode_length):
            ret = rng.normal(0.0005, 0.015)
            p_open = p * (1.0 + rng.normal(0.0, 0.005))
            p_close = p_open * (1.0 + ret)
            p_high = max(p_open, p_close) * (1.0 + rng.uniform(0.001, 0.01))
            p_low = min(p_open, p_close) * (1.0 - rng.uniform(0.001, 0.01))
            prices[t] = [p_open, p_high, p_low, p_close]
            p = p_close
        self.prices_df = prices

        # Pre-generate multimodal batches for each timestep
        self.multimodal_batches = [
            generate_synthetic_batch(batch_size=1, seed=1000 + t)
            for t in range(self.episode_length)
        ]

    def _get_position_features(self, current_close: float) -> torch.Tensor:
        """
        Computes 4-d position state:
        [current_target_position, unrealized_pnl_pct, holding_duration_norm, distance_to_limit]
        """
        pos_action = float(self.current_position)
        if self.current_position == ACTION_LONG and self.entry_price > 0:
            unrealized_pnl = (current_close - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = 0.0

        holding_duration_norm = min(self.holding_days / 60.0, 1.0)

        # Price limit distance
        limit_dist = self.config.daily_price_limit_pct
        pos_vec = np.array([pos_action, unrealized_pnl, holding_duration_norm, limit_dist], dtype=np.float32)
        return torch.tensor(pos_vec).unsqueeze(0)  # [1, 4]

    def _get_current_observation(self) -> MultimodalBatch:
        batch = self.multimodal_batches[self.current_step]
        current_close = self.prices_df[self.current_step, 3]
        batch.position = self._get_position_features(current_close)
        return batch

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[MultimodalBatch, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0.0
        self.current_position = ACTION_CASH
        self.entry_price = 0.0
        self.holding_days = 0
        self.portfolio_equity = self.initial_capital
        self.equity_history = [self.portfolio_equity]

        obs = self._get_current_observation()
        info = {"portfolio_equity": self.portfolio_equity, "step": self.current_step}
        return obs, info

    def step(self, action: int) -> Tuple[MultimodalBatch, float, bool, bool, Dict[str, Any]]:
        """
        Executes decision made at 20:00 on day t:
        - Order executes at 09:00 on day t+1
        - Portfolio marked to market at 20:00 on day t+1
        - Returns: (obs, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        t = self.current_step
        if t >= self.episode_length - 1:
            # End of episode
            obs = self._get_current_observation()
            return obs, 0.0, False, True, {"portfolio_equity": self.portfolio_equity}

        prev_close_price = self.prices_df[t, 3]
        next_open_price = self.prices_df[t + 1, 0]
        next_close_price = self.prices_df[t + 1, 3]

        # Prior equity at 20:00 on day t
        V_prior = self.cash + (self.shares * prev_close_price)

        # 1. Holdings before next open (between 20:00 day t and 09:00 day t+1)
        # Any existing shares experience the overnight gap
        equity_pre_open = self.cash + (self.shares * next_open_price)

        # 2. Execution at 09:00 day t+1
        new_cash, new_shares, exec_res = self.simulator.execute_target_position(
            current_position=self.current_position,
            target_position=action,
            current_cash=self.cash,
            current_shares=self.shares,
            prev_close_price=prev_close_price,
            open_price=next_open_price,
        )

        self.cash = new_cash
        self.shares = new_shares
        self.current_position = exec_res.executed_action

        # Update entry price and holding days strictly based on executed_action and rejection state
        if exec_res.executed_action == ACTION_LONG:
            if exec_res.shares_traded > 0 and self.entry_price == 0.0:
                # Newly opened Long position
                self.entry_price = exec_res.executed_price
            elif exec_res.order_rejected:
                # Sell order rejected (e.g. limit-down lock): maintain original entry_price
                pass
            # Holding days accumulate while position is held
            self.holding_days += 1
        else:
            # Position is Cash: either already cash, successfully sold, or buy order rejected
            self.entry_price = 0.0
            self.holding_days = 0

        # 3. Mark-to-market at 20:00 day t+1
        V_next = self.cash + (self.shares * next_close_price)
        self.portfolio_equity = V_next
        self.equity_history.append(V_next)

        # 4. Reward: Transaction-cost-adjusted log return between consecutive decision times
        reward = float(np.log(max(V_next, 1e-4) / max(V_prior, 1e-4)))

        # Advance step
        self.current_step += 1
        terminated = False
        truncated = (self.current_step >= self.episode_length - 1)

        obs = self._get_current_observation()
        info = {
            "step": self.current_step,
            "portfolio_equity": self.portfolio_equity,
            "reward": reward,
            "commission": exec_res.commission_paid,
            "tax": exec_res.tax_paid,
            "slippage": exec_res.slippage_cost,
            "action": action,
            "executed_price": exec_res.executed_price,
        }

        return obs, reward, terminated, truncated, info
