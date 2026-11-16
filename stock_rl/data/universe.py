"""
Dynamic Point-in-Time Universe Selection for TWSE Listed Stocks V8.0
Maintains point-in-time constituent membership across rebalancing windows.
"""

from typing import Dict, List, Set
import pandas as pd
import numpy as np


class DynamicUniverseManager:
    """
    Manages historical dynamic universe of 100 TWSE stocks.
    Tracks point-in-time constituent status to prevent survivorship bias.
    """

    def __init__(self, target_size: int = 100):
        self.target_size = target_size
        # Map: rebalance_date -> list of stock IDs
        self._universe_history: Dict[pd.Timestamp, List[str]] = {}

    def register_rebalance(self, rebalance_date: pd.Timestamp, stock_ids: List[str]):
        """Registers universe constituents for a specific rebalance date."""
        if len(stock_ids) != self.target_size:
            # If fewer than target_size are available, keep existing count but log warning
            pass
        self._universe_history[rebalance_date] = list(stock_ids)

    def get_universe_as_of(self, query_timestamp: pd.Timestamp) -> List[str]:
        """
        Retrieves the universe constituents active as of query_timestamp.
        Uses the most recent rebalance date on or before query_timestamp.
        """
        valid_rebalances = [d for d in self._universe_history.keys() if d <= query_timestamp]
        if not valid_rebalances:
            raise KeyError(f"No historical universe rebalance found on or prior to {query_timestamp}")
        latest_rebalance = max(valid_rebalances)
        return self._universe_history[latest_rebalance]

    def is_member(self, stock_id: str, query_timestamp: pd.Timestamp) -> bool:
        """Checks if a stock is an active universe member on a given date."""
        universe = self.get_universe_as_of(query_timestamp)
        return stock_id in universe


def select_top_liquid_universe(
    daily_turnover_df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    lookback_days: int = 60,
    top_n: int = 100
) -> List[str]:
    """
    Selects top N stocks by median daily turnover over the prior 60 trading days.
    Strictly point-in-time: uses data up to rebalance_date only.
    """
    history = daily_turnover_df.loc[:rebalance_date].tail(lookback_days)
    median_turnover = history.median(axis=0).dropna()
    top_stocks = median_turnover.sort_values(ascending=False).head(top_n).index.tolist()
    return top_stocks
