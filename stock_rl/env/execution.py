"""
TWSE Market Execution Simulator V8.0
Simulates next-open execution, commission discounts, transaction taxes, slippage, and price limits.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from stock_rl.config.settings import TradingConfig


@dataclass
class ExecutionResult:
    executed_action: int          # Position after execution (0 or 1)
    executed_price: float         # Execution price per share
    shares_traded: float          # Number of shares bought (+) or sold (-)
    commission_paid: float        # Broker commission
    tax_paid: float               # Securities transaction tax (0.3% on sales)
    slippage_cost: float          # Cost attributed to market impact / slippage
    order_rejected: bool          # True if blocked by limit-up / limit-down / suspension


class TWSEExecutionSimulator:
    """
    Executes target position transitions under realistic Taiwan Stock Exchange (TWSE) rules.
    """
    def __init__(self, config: TradingConfig = TradingConfig()):
        self.config = config
        self.effective_commission_rate = config.twse_base_commission_rate * config.broker_discount
        self.tax_rate = config.twse_securities_tax_rate
        self.slippage_rate = config.base_slippage_rate
        self.price_limit_pct = config.daily_price_limit_pct

    def execute_target_position(
        self,
        current_position: int,       # 0 (Cash) or 1 (Long)
        target_position: int,        # 0 (Cash) or 1 (Long)
        current_cash: float,
        current_shares: float,
        prev_close_price: float,
        open_price: float,
        is_suspended: bool = False,
    ) -> Tuple[float, float, ExecutionResult]:
        """
        Executes order at open_price.
        Returns:
            new_cash: float
            new_shares: float
            execution_result: ExecutionResult
        """
        if is_suspended:
            # Cannot trade suspended stocks
            return current_cash, current_shares, ExecutionResult(
                executed_action=current_position,
                executed_price=0.0,
                shares_traded=0.0,
                commission_paid=0.0,
                tax_paid=0.0,
                slippage_cost=0.0,
                order_rejected=True,
            )

        # Check daily price limits relative to previous close (+-10%)
        upper_limit = prev_close_price * (1.0 + self.price_limit_pct)
        lower_limit = prev_close_price * (1.0 - self.price_limit_pct)

        # Check if limit locked
        if target_position == 1 and current_position == 0:
            # Buying: check if locked limit-up at open
            if open_price >= upper_limit - 1e-4:
                # Locked limit-up, cannot buy
                return current_cash, current_shares, ExecutionResult(
                    executed_action=0,
                    executed_price=0.0,
                    shares_traded=0.0,
                    commission_paid=0.0,
                    tax_paid=0.0,
                    slippage_cost=0.0,
                    order_rejected=True,
                )
        elif target_position == 0 and current_position == 1:
            # Selling: check if locked limit-down at open
            if open_price <= lower_limit + 1e-4:
                # Locked limit-down, cannot sell
                return current_cash, current_shares, ExecutionResult(
                    executed_action=1,
                    executed_price=0.0,
                    shares_traded=0.0,
                    commission_paid=0.0,
                    tax_paid=0.0,
                    slippage_cost=0.0,
                    order_rejected=True,
                )

        # No position change
        if target_position == current_position:
            return current_cash, current_shares, ExecutionResult(
                executed_action=current_position,
                executed_price=open_price,
                shares_traded=0.0,
                commission_paid=0.0,
                tax_paid=0.0,
                slippage_cost=0.0,
                order_rejected=False,
            )

        # Execute BUY: Cash (0) -> Long (1)
        if target_position == 1 and current_position == 0:
            effective_price = open_price * (1.0 + self.slippage_rate)
            slippage_cost_per_share = effective_price - open_price

            # Total cost per share = effective_price * (1 + commission)
            cost_per_share = effective_price * (1.0 + self.effective_commission_rate)
            shares_to_buy = current_cash / cost_per_share
            total_turnover = shares_to_buy * open_price
            commission = shares_to_buy * effective_price * self.effective_commission_rate
            total_slippage = shares_to_buy * slippage_cost_per_share

            new_cash = 0.0
            new_shares = shares_to_buy

            return new_cash, new_shares, ExecutionResult(
                executed_action=1,
                executed_price=effective_price,
                shares_traded=shares_to_buy,
                commission_paid=commission,
                tax_paid=0.0,  # No tax on buying
                slippage_cost=total_slippage,
                order_rejected=False,
            )

        # Execute SELL: Long (1) -> Cash (0)
        else:
            effective_price = open_price * (1.0 - self.slippage_rate)
            slippage_loss_per_share = open_price - effective_price

            gross_proceeds = current_shares * effective_price
            commission = gross_proceeds * self.effective_commission_rate
            tax = gross_proceeds * self.tax_rate  # 0.3% TWSE securities transaction tax
            total_slippage = current_shares * slippage_loss_per_share

            net_proceeds = gross_proceeds - commission - tax
            new_cash = current_cash + net_proceeds
            new_shares = 0.0

            return new_cash, new_shares, ExecutionResult(
                executed_action=0,
                executed_price=effective_price,
                shares_traded=-current_shares,
                commission_paid=commission,
                tax_paid=tax,
                slippage_cost=total_slippage,
                order_rejected=False,
            )
