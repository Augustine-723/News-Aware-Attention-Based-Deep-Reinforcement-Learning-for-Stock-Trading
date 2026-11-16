"""
Event-Driven Stock RL — Feature Schema & Dimension Specifications V8.0
Authoritative column ordering and tensor dimension contracts.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

# Sequence lengths
MARKET_WINDOW_LENGTH: int = 60  # 60 trading days
RECENT_EVENT_SLOTS: int = 32    # 32 recent events in the lookback window
ACTIVE_EVENT_SLOTS: int = 8     # 8 ongoing/active material event slots
TOTAL_EVENT_SLOTS: int = RECENT_EVENT_SLOTS + ACTIVE_EVENT_SLOTS  # 40 slots

# Feature Dimensions per Specification V8.0
PRICE_DIM: int = 12
TECHNICAL_DIM: int = 16
INSTITUTIONAL_DIM: int = 8
MACRO_DIM: int = 12
STOCK_CONTEXT_DIM: int = 16

EVENT_TEXT_DIM: int = 128
EVENT_METADATA_DIM: int = 24
EVENT_STOCK_RELATION_DIM: int = 8
EVENT_RAW_FEATURE_DIM: int = EVENT_TEXT_DIM + EVENT_METADATA_DIM + EVENT_STOCK_RELATION_DIM  # 160 dims

POSITION_STATE_DIM: int = 4

# Model Internal Dimensions
EVENT_PROJECTION_INTERMEDIATE_DIM: int = 256
EVENT_PROJECTION_OUTPUT_DIM: int = 128

MULTIMODAL_FUSION_DIM: int = 128
RL_STATE_DIM: int = MULTIMODAL_FUSION_DIM + POSITION_STATE_DIM  # 132 dims

# Multi-Task Horizons
RETURN_HORIZONS: List[int] = [1, 3, 5]  # Trading days
NUM_HORIZONS: int = len(RETURN_HORIZONS)

# Actions
ACTION_CASH: int = 0
ACTION_LONG: int = 1
NUM_ACTIONS: int = 2

# Authoritative feature column definitions
PRICE_FEATURES: List[str] = [
    "open_norm", "high_norm", "low_norm", "close_norm", "volume_norm",
    "turnover_norm", "vwap_norm", "daily_return", "intraday_range",
    "shadow_upper", "shadow_lower", "effective_spread_proxy"
]

TECHNICAL_FEATURES: List[str] = [
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "bb_upper_dist", "bb_lower_dist", "bb_width", "atr_14_norm",
    "obv_slope", "kdj_k", "kdj_d", "kdj_j",
    "mfi_14", "cci_14", "roc_10", "adx_14"
]

INSTITUTIONAL_FEATURES: List[str] = [
    "foreign_net_buy_shares_norm", "trust_net_buy_shares_norm", "dealer_net_buy_shares_norm",
    "total_institutional_flow_norm", "margin_long_change_norm", "margin_short_change_norm",
    "foreign_holding_ratio", "margin_utilization_ratio"
]

MACRO_FEATURES: List[str] = [
    "taiex_return", "taiex_vol_20", "usdtwd_fx_return", "tw_10y_yield_diff",
    "us_10y_yield_diff", "sox_semiconductor_return", "sp500_return", "brent_crude_return",
    "taiwan_interbank_rate", "macro_economic_leading_indicator", "tsmc_adr_premium", "vix_tw_proxy"
]

STOCK_CONTEXT_FEATURES: List[str] = [
    "industry_sector_id", "market_cap_tier", "historical_beta_1y", "annualized_vol_1y",
    "avg_daily_turnover_ratio", "pb_ratio_scaled", "pe_ratio_scaled", "dividend_yield_scaled",
    "debt_to_equity_scaled", "roe_ttm_scaled", "supply_chain_role", "foreign_ownership_cap",
    "margin_tradable_flag", "day_trading_eligible_flag", "weight_in_taiex", "governance_score_scaled"
]

EVENT_METADATA_FEATURES: List[str] = [
    "event_category_id", "source_credibility_tier", "event_sentiment_score", "sentiment_confidence",
    "event_age_fractional_days", "event_scope_tier", "is_regulatory_filing", "is_earnings_guidance",
    "is_supply_chain_disruption", "is_merger_acquisition", "is_litigation_investigation", "is_macro_policy",
    "is_capital_reduction", "is_dividend_announcement", "revision_count", "urgency_flag",
    "historical_source_accuracy", "title_body_consistency", "verified_entity_count", "is_breaking_news",
    "vendor_ingestion_delay_min", "document_word_count_bucket", "evidence_span_count", "active_flag"
]

EVENT_STOCK_RELATION_FEATURES: List[str] = [
    "entity_role_primary", "entity_role_supplier", "entity_role_customer", "entity_role_competitor",
    "co_occurrence_frequency_norm", "supply_chain_tier_distance", "revenue_exposure_percentage", "relation_confidence"
]

POSITION_STATE_FEATURES: List[str] = [
    "current_target_position",  # 0: Cash, 1: Long
    "unrealized_pnl_pct",       # normalized unrealized profit/loss percentage
    "holding_duration_days",    # normalized trading days since last position flip
    "distance_to_price_limit"   # normalized distance to +-10% TWSE daily limit
]


@dataclass(frozen=True)
class FeatureDimensions:
    price: int = PRICE_DIM
    technical: int = TECHNICAL_DIM
    institutional: int = INSTITUTIONAL_DIM
    macro: int = MACRO_DIM
    stock_context: int = STOCK_CONTEXT_DIM
    event_text: int = EVENT_TEXT_DIM
    event_metadata: int = EVENT_METADATA_DIM
    event_stock_relation: int = EVENT_STOCK_RELATION_DIM
    event_raw_feature: int = EVENT_RAW_FEATURE_DIM
    position_state: int = POSITION_STATE_DIM
    multimodal_fusion: int = MULTIMODAL_FUSION_DIM
    rl_state: int = RL_STATE_DIM
    num_horizons: int = NUM_HORIZONS
    num_actions: int = NUM_ACTIONS
