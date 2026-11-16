"""
Event-Driven Stock RL — Authoritative Feature Dictionary & Schema Specifications V8.0
Contains complete metadata: Identifier, Math Definition, Unit, Source, Availability Timing,
Missing-Value Policy, and Version.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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


@dataclass(frozen=True)
class FeatureField:
    feature_id: str
    name: str
    math_definition: str
    unit: str
    source: str
    available_timing: str
    missing_policy: str
    version: str = "V8.0"

    def is_available_at(self, record_timestamp: Optional[Any], decision_timestamp: Any) -> bool:
        """
        Validates historical availability: actual record timestamp must be <= decision_timestamp.
        Static available_timing documentation cannot substitute for point-in-time timestamp verification.
        """
        if record_timestamp is None:
            return False
        return record_timestamp <= decision_timestamp


# ==============================================================================
# Authoritative Feature Dictionaries V8.0
# ==============================================================================

PRICE_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("PRC_01", "open_norm", "(P_open - MA60) / Std60", "dimensionless", "TWSE", "Daily 20:00", "Forward-fill"),
    FeatureField("PRC_02", "high_norm", "(P_high - MA60) / Std60", "dimensionless", "TWSE", "Daily 20:00", "Forward-fill"),
    FeatureField("PRC_03", "low_norm", "(P_low - MA60) / Std60", "dimensionless", "TWSE", "Daily 20:00", "Forward-fill"),
    FeatureField("PRC_04", "close_norm", "(P_close - MA60) / Std60", "dimensionless", "TWSE", "Daily 20:00", "Forward-fill"),
    FeatureField("PRC_05", "volume_norm", "ln(1 + Vol) z-scored over 60 days", "dimensionless", "TWSE", "Daily 20:00", "Zero-fill"),
    FeatureField("PRC_06", "turnover_norm", "ln(1 + Turnover) z-scored over 60 days", "dimensionless", "TWSE", "Daily 20:00", "Zero-fill"),
    FeatureField("PRC_07", "vwap_norm", "(VWAP - P_close) / P_close", "ratio", "TWSE", "Daily 20:00", "Zero-fill"),
    FeatureField("PRC_08", "daily_return", "(P_close,t - P_close,t-1) / P_close,t-1", "ratio", "TWSE", "Daily 20:00", "Zero-fill"),
    FeatureField("PRC_09", "intraday_range", "(P_high - P_low) / P_open", "ratio", "TWSE", "Daily 20:00", "Zero-fill"),
    FeatureField("PRC_10", "shadow_upper", "(P_high - max(P_open, P_close)) / P_close", "ratio", "TWSE", "Daily 20:00", "Zero-fill"),
    FeatureField("PRC_11", "shadow_lower", "(min(P_open, P_close) - P_low) / P_close", "ratio", "TWSE", "Daily 20:00", "Zero-fill"),
    FeatureField("PRC_12", "effective_spread_proxy", "2 * sqrt(max(0, -cov(Delta P_t, Delta P_t-1))) / P_close", "ratio", "TWSE", "Daily 20:00", "Sector Median"),
]

TECHNICAL_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("TECH_01", "rsi_14", "(RSI_14 - 50) / 50", "bounded_ratio", "Technical Engine", "Daily 20:00", "Neutral (0.0)"),
    FeatureField("TECH_02", "macd_line", "(EMA12 - EMA26) / P_close", "ratio", "Technical Engine", "Daily 20:00", "Zero-fill"),
    FeatureField("TECH_03", "macd_signal", "EMA9(MACD) / P_close", "ratio", "Technical Engine", "Daily 20:00", "Zero-fill"),
    FeatureField("TECH_04", "macd_hist", "(MACD - Signal) / P_close", "ratio", "Technical Engine", "Daily 20:00", "Zero-fill"),
    FeatureField("TECH_05", "bb_upper_dist", "(BB_upper - P_close) / P_close", "ratio", "Technical Engine", "Daily 20:00", "Zero-fill"),
    FeatureField("TECH_06", "bb_lower_dist", "(P_close - BB_lower) / P_close", "ratio", "Technical Engine", "Daily 20:00", "Zero-fill"),
    FeatureField("TECH_07", "bb_width", "(BB_upper - BB_lower) / MA20", "ratio", "Technical Engine", "Daily 20:00", "Median"),
    FeatureField("TECH_08", "atr_14_norm", "ATR_14 / P_close", "ratio", "Technical Engine", "Daily 20:00", "Median"),
    FeatureField("TECH_09", "obv_slope", "10-day linear regression slope of OBV", "dimensionless", "Technical Engine", "Daily 20:00", "Zero-fill"),
    FeatureField("TECH_10", "kdj_k", "(K_9 - 50) / 50", "bounded_ratio", "Technical Engine", "Daily 20:00", "Neutral (0.0)"),
    FeatureField("TECH_11", "kdj_d", "(D_9 - 50) / 50", "bounded_ratio", "Technical Engine", "Daily 20:00", "Neutral (0.0)"),
    FeatureField("TECH_12", "kdj_j", "(J_9 - 50) / 50", "bounded_ratio", "Technical Engine", "Daily 20:00", "Neutral (0.0)"),
    FeatureField("TECH_13", "mfi_14", "(MFI_14 - 50) / 50", "bounded_ratio", "Technical Engine", "Daily 20:00", "Neutral (0.0)"),
    FeatureField("TECH_14", "cci_14", "CCI_14 / 200.0 clamped to [-1, 1]", "bounded_ratio", "Technical Engine", "Daily 20:00", "Neutral (0.0)"),
    FeatureField("TECH_15", "roc_10", "(P_close,t - P_close,t-10) / P_close,t-10", "ratio", "Technical Engine", "Daily 20:00", "Zero-fill"),
    FeatureField("TECH_16", "adx_14", "ADX_14 / 100.0", "ratio", "Technical Engine", "Daily 20:00", "Neutral (0.2)"),
]

INSTITUTIONAL_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("INST_01", "foreign_net_buy_shares_norm", "Foreign Net Buy Shares / Shares Outstanding", "ratio", "TWSE", "Daily 18:00 (<=20:00)", "Zero-fill"),
    FeatureField("INST_02", "trust_net_buy_shares_norm", "Investment Trust Net Buy Shares / Shares Outstanding", "ratio", "TWSE", "Daily 18:00 (<=20:00)", "Zero-fill"),
    FeatureField("INST_03", "dealer_net_buy_shares_norm", "Dealer Net Buy Shares / Shares Outstanding", "ratio", "TWSE", "Daily 18:00 (<=20:00)", "Zero-fill"),
    FeatureField("INST_04", "total_institutional_flow_norm", "Total 3-Institutional Net Buy / Shares Outstanding", "ratio", "TWSE", "Daily 18:00 (<=20:00)", "Zero-fill"),
    FeatureField("INST_05", "margin_long_change_norm", "Delta Margin Long Balance / Shares Outstanding", "ratio", "TWSE", "Daily 19:30 (<=20:00)", "Zero-fill"),
    FeatureField("INST_06", "margin_short_change_norm", "Delta Margin Short Balance / Shares Outstanding", "ratio", "TWSE", "Daily 19:30 (<=20:00)", "Zero-fill"),
    FeatureField("INST_07", "foreign_holding_ratio", "Foreign Shares Held / Total Shares Outstanding", "ratio", "TWSE", "Daily 18:00 (<=20:00)", "Forward-fill"),
    FeatureField("INST_08", "margin_utilization_ratio", "Margin Balance / Margin Quota", "ratio", "TWSE", "Daily 19:30 (<=20:00)", "Forward-fill"),
]

MACRO_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("MACRO_01", "taiex_return", "Delta ln(TAIEX)", "ratio", "TWSE", "Daily 14:00 (<=20:00)", "Zero-fill"),
    FeatureField("MACRO_02", "taiex_vol_20", "20-day annualized realized vol of TAIEX", "annualized_ratio", "TWSE", "Daily 14:00 (<=20:00)", "Median"),
    FeatureField("MACRO_03", "usdtwd_fx_return", "Delta ln(USD/TWD spot)", "ratio", "Taipei Forex", "Daily 16:30 (<=20:00)", "Zero-fill"),
    FeatureField("MACRO_04", "tw_10y_yield_diff", "Delta Taiwan 10Y sovereign yield", "bps", "TPEx", "Daily 17:00 (<=20:00)", "Zero-fill"),
    FeatureField("MACRO_05", "us_10y_yield_diff", "Delta US 10Y Treasury yield (lagged 1D)", "bps", "Federal Reserve / US", "T-1 Available <= 20:00", "Forward-fill"),
    FeatureField("MACRO_06", "sox_semiconductor_return", "SOX index daily return (lagged 1D)", "ratio", "NASDAQ", "T-1 Available <= 20:00", "Forward-fill"),
    FeatureField("MACRO_07", "sp500_return", "S&P 500 daily return (lagged 1D)", "ratio", "Cboe / S&P", "T-1 Available <= 20:00", "Forward-fill"),
    FeatureField("MACRO_08", "brent_crude_return", "Brent crude daily return (lagged 1D)", "ratio", "ICE", "T-1 Available <= 20:00", "Forward-fill"),
    FeatureField("MACRO_09", "taiwan_interbank_rate", "Taiwan CP 30-day interbank rate", "ratio", "CBC Taiwan", "Daily 17:00 (<=20:00)", "Forward-fill"),
    FeatureField("MACRO_10", "macro_economic_leading_indicator", "Delta NDC Composite Leading Indicator", "dimensionless", "NDC Taiwan", "Monthly (PIT release)", "Forward-fill"),
    FeatureField("MACRO_11", "tsmc_adr_premium", "(TSMC ADR / 5 * USD/TWD - TSMC Close) / TSMC Close", "ratio", "NYSE/TWSE", "T-1 Available <= 20:00", "Forward-fill"),
    FeatureField("MACRO_12", "vix_tw_proxy", "Taiwan implied/realized volatility index proxy", "annualized_ratio", "TAIFEX", "Daily 14:00 (<=20:00)", "Forward-fill"),
]

STOCK_CONTEXT_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("CTX_01", "industry_sector_id", "Normalized sector categorical index [0, 1]", "categorical", "TWSE Industry", "Static / Quarterly", "Static Default"),
    FeatureField("CTX_02", "market_cap_tier", "Market cap percentile rank [0, 1]", "rank", "TWSE", "Monthly Rebalance", "Forward-fill"),
    FeatureField("CTX_03", "historical_beta_1y", "250-day rolling beta against TAIEX", "ratio", "Market Data Engine", "Daily 20:00", "Market Neutral 1.0"),
    FeatureField("CTX_04", "annualized_vol_1y", "250-day annualized realized return vol", "annualized_ratio", "Market Data Engine", "Daily 20:00", "Median"),
    FeatureField("CTX_05", "avg_daily_turnover_ratio", "60-day median turnover / shares outstanding", "ratio", "Market Data Engine", "Daily 20:00", "Median"),
    FeatureField("CTX_06", "pb_ratio_scaled", "P/B ratio scaled by sector median", "ratio", "TWSE", "Daily 20:00", "Neutral 1.0"),
    FeatureField("CTX_07", "pe_ratio_scaled", "P/E ratio scaled by sector median", "ratio", "TWSE", "Daily 20:00", "Neutral 1.0"),
    FeatureField("CTX_08", "dividend_yield_scaled", "Trailing 12M cash dividend yield", "ratio", "TWSE", "Daily 20:00", "Zero-fill"),
    FeatureField("CTX_09", "debt_to_equity_scaled", "Total debt / Total equity scaled", "ratio", "MOPS Financials", "Quarterly PIT Release", "Forward-fill"),
    FeatureField("CTX_10", "roe_ttm_scaled", "Return on equity TTM scaled", "ratio", "MOPS Financials", "Quarterly PIT Release", "Forward-fill"),
    FeatureField("CTX_11", "supply_chain_role", "Category: Foundry=1, Fabless=2, OSAT=3, Component=4, Brand=5", "categorical", "Industry Research", "Static / Quarterly", "Neutral"),
    FeatureField("CTX_12", "foreign_ownership_cap", "Statutory foreign ownership limit [0, 1]", "ratio", "TWSE Regulations", "Static / Semi-Annual", "1.0 (No Cap)"),
    FeatureField("CTX_13", "margin_tradable_flag", "Binary flag: Margin trading eligible", "binary", "TWSE", "Daily 20:00", "0.0"),
    FeatureField("CTX_14", "day_trading_eligible_flag", "Binary flag: Day trading eligible", "binary", "TWSE", "Daily 20:00", "0.0"),
    FeatureField("CTX_15", "weight_in_taiex", "Stock weight in TAIEX benchmark [0, 1]", "ratio", "TWSE", "Monthly", "Zero-fill"),
    FeatureField("CTX_16", "governance_score_scaled", "TWSE Corporate Governance ranking percentile", "percentile", "TWSE Corporate Governance", "Annual PIT Release", "Median (0.5)"),
]

EVENT_METADATA_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("EV_META_01", "event_category_id", "Taxonomy: Earnings=1, Guidance=2, M&A=3, Legal=4, CapEx=5, Macro=6", "categorical", "LLM Extraction", "Event Release", "Categorical 0"),
    FeatureField("EV_META_02", "source_credibility_tier", "Exchange MOPS=1.0, Major News=0.8, Media Wire=0.6", "score", "Metadata Registry", "Event Release", "0.5"),
    FeatureField("EV_META_03", "event_sentiment_score", "LLM sentiment score [-1.0, +1.0]", "bounded_score", "LLM Extraction", "Event Release", "0.0"),
    FeatureField("EV_META_04", "sentiment_confidence", "LLM extraction confidence [0.0, 1.0]", "score", "LLM Extraction", "Event Release", "0.5"),
    FeatureField("EV_META_05", "event_age_fractional_days", "(T_decision - T_available) in fractional days", "days", "PIT Engine", "Event Release", "0.0"),
    FeatureField("EV_META_06", "event_scope_tier", "Company-specific=1.0, Supply-chain=0.7, Sector=0.4, Macro=0.1", "score", "LLM Extraction", "Event Release", "1.0"),
    FeatureField("EV_META_07", "is_regulatory_filing", "Binary indicator for MOPS material announcement", "binary", "Exchange Filing", "Event Release", "0.0"),
    FeatureField("EV_META_08", "is_earnings_guidance", "Binary indicator for quarterly revenue/margin guidance", "binary", "LLM Extraction", "Event Release", "0.0"),
    FeatureField("EV_META_09", "is_supply_chain_disruption", "Binary indicator for outage, fire, fab shortage", "binary", "LLM Extraction", "Event Release", "0.0"),
    FeatureField("EV_META_10", "is_merger_acquisition", "Binary indicator for tender offer, M&A, equity investment", "binary", "LLM Extraction", "Event Release", "0.0"),
    FeatureField("EV_META_11", "is_litigation_investigation", "Binary indicator for patent lawsuit, regulatory probe", "binary", "LLM Extraction", "Event Release", "0.0"),
    FeatureField("EV_META_12", "is_macro_policy", "Binary indicator for export restrictions, tariff change", "binary", "LLM Extraction", "Event Release", "0.0"),
    FeatureField("EV_META_13", "is_capital_reduction", "Binary indicator for cash capital reduction", "binary", "Exchange Filing", "Event Release", "0.0"),
    FeatureField("EV_META_14", "is_dividend_announcement", "Binary indicator for cash dividend distribution announcement", "binary", "Exchange Filing", "Event Release", "0.0"),
    FeatureField("EV_META_15", "revision_count", "Number of official revision amendments published", "count", "PIT Engine", "Event Release", "0.0"),
    FeatureField("EV_META_16", "urgency_flag", "Binary indicator for urgent breaking development", "binary", "News Vendor", "Event Release", "0.0"),
    FeatureField("EV_META_17", "historical_source_accuracy", "Empirical reliability score of publishing media outlet", "score", "Historical Analytics", "Event Release", "0.5"),
    FeatureField("EV_META_18", "title_body_consistency", "Cosine similarity between headline and extracted evidence", "score", "LLM Extraction", "Event Release", "1.0"),
    FeatureField("EV_META_19", "verified_entity_count", "Number of verified TWSE stock symbols matched", "count", "Entity Resolver", "Event Release", "1.0"),
    FeatureField("EV_META_20", "is_breaking_news", "Binary indicator: published during regular trading hours", "binary", "PIT Engine", "Event Release", "0.0"),
    FeatureField("EV_META_21", "vendor_ingestion_delay_min", "ln(1 + vendor delay minutes)", "dimensionless", "PIT Engine", "Event Release", "0.0"),
    FeatureField("EV_META_22", "document_word_count_bucket", "Word count bucket: Short=1, Medium=2, Long=3, Deep=4", "categorical", "Document Parser", "Event Release", "2.0"),
    FeatureField("EV_META_23", "evidence_span_count", "Count of extracted factual evidence sentence spans", "count", "LLM Extraction", "Event Release", "1.0"),
    FeatureField("EV_META_24", "active_flag", "Binary indicator: ongoing material event cycle (e.g. active tender)", "binary", "Event State Machine", "Daily 20:00", "0.0"),
]

EVENT_STOCK_RELATION_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("EV_REL_01", "entity_role_primary", "1.0 if stock is primary subject of event, else 0.0", "binary", "Entity Linker", "Event Release", "0.0"),
    FeatureField("EV_REL_02", "entity_role_supplier", "1.0 if stock is an upstream supplier to mentioned entity", "binary", "Knowledge Graph", "Event Release", "0.0"),
    FeatureField("EV_REL_03", "entity_role_customer", "1.0 if stock is downstream customer of mentioned entity", "binary", "Knowledge Graph", "Event Release", "0.0"),
    FeatureField("EV_REL_04", "entity_role_competitor", "1.0 if stock is direct horizontal market competitor", "binary", "Knowledge Graph", "Event Release", "0.0"),
    FeatureField("EV_REL_05", "co_occurrence_frequency_norm", "Rolling 1Y co-occurrence score in financial news", "score", "Knowledge Graph", "Event Release", "0.0"),
    FeatureField("EV_REL_06", "supply_chain_tier_distance", "1 / Tier distance (Tier 1 = 1.0, Tier 2 = 0.5, Tier 3 = 0.33)", "score", "Knowledge Graph", "Event Release", "0.0"),
    FeatureField("EV_REL_07", "revenue_exposure_percentage", "Estimated direct/indirect revenue exposure percentage", "ratio", "Supply Chain DB", "Event Release", "0.0"),
    FeatureField("EV_REL_08", "relation_confidence", "Confidence score of the linked entity-stock relationship", "score", "Knowledge Graph", "Event Release", "0.5"),
]

POSITION_STATE_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("POS_01", "current_target_position", "Target position state: 0.0 for Cash, 1.0 for Long", "state", "Gym Environment", "Decision 20:00", "0.0"),
    FeatureField("POS_02", "unrealized_pnl_pct", "(P_close,t - P_entry) / P_entry", "ratio", "Gym Environment", "Decision 20:00", "0.0"),
    FeatureField("POS_03", "holding_duration_days", "min(days_in_position / 60.0, 1.0)", "normalized_duration", "Gym Environment", "Decision 20:00", "0.0"),
    FeatureField("POS_04", "time_since_last_trade", "min(days_since_last_successful_trade / 60.0, 1.0)", "normalized_duration", "Gym Environment", "Decision 20:00", "1.0"),
]

# Stable Feature ID lists for authoritative indexing & verification
PRICE_FEATURE_IDS: List[str] = [f.feature_id for f in PRICE_FEATURE_FIELDS]
TECHNICAL_FEATURE_IDS: List[str] = [f.feature_id for f in TECHNICAL_FEATURE_FIELDS]
INSTITUTIONAL_FEATURE_IDS: List[str] = [f.feature_id for f in INSTITUTIONAL_FEATURE_FIELDS]
MACRO_FEATURE_IDS: List[str] = [f.feature_id for f in MACRO_FEATURE_FIELDS]
STOCK_CONTEXT_FEATURE_IDS: List[str] = [f.feature_id for f in STOCK_CONTEXT_FEATURE_FIELDS]
EVENT_METADATA_FEATURE_IDS: List[str] = [f.feature_id for f in EVENT_METADATA_FEATURE_FIELDS]
EVENT_STOCK_RELATION_FEATURE_IDS: List[str] = [f.feature_id for f in EVENT_STOCK_RELATION_FEATURE_FIELDS]
POSITION_STATE_FEATURE_IDS: List[str] = [f.feature_id for f in POSITION_STATE_FEATURE_FIELDS]

# Name lists for backward compatibility and index slicing
PRICE_FEATURES: List[str] = [f.name for f in PRICE_FEATURE_FIELDS]
TECHNICAL_FEATURES: List[str] = [f.name for f in TECHNICAL_FEATURE_FIELDS]
INSTITUTIONAL_FEATURES: List[str] = [f.name for f in INSTITUTIONAL_FEATURE_FIELDS]
MACRO_FEATURES: List[str] = [f.name for f in MACRO_FEATURE_FIELDS]
STOCK_CONTEXT_FEATURES: List[str] = [f.name for f in STOCK_CONTEXT_FEATURE_FIELDS]
EVENT_METADATA_FEATURES: List[str] = [f.name for f in EVENT_METADATA_FEATURE_FIELDS]
EVENT_STOCK_RELATION_FEATURES: List[str] = [f.name for f in EVENT_STOCK_RELATION_FEATURE_FIELDS]
POSITION_STATE_FEATURES: List[str] = [f.name for f in POSITION_STATE_FEATURE_FIELDS]

ALL_FEATURE_FIELDS: List[FeatureField] = (
    PRICE_FEATURE_FIELDS
    + TECHNICAL_FEATURE_FIELDS
    + INSTITUTIONAL_FEATURE_FIELDS
    + MACRO_FEATURE_FIELDS
    + STOCK_CONTEXT_FEATURE_FIELDS
    + EVENT_METADATA_FEATURE_FIELDS
    + EVENT_STOCK_RELATION_FEATURE_FIELDS
    + POSITION_STATE_FEATURE_FIELDS
)

FEATURE_ID_TO_FIELD: Dict[str, FeatureField] = {f.feature_id: f for f in ALL_FEATURE_FIELDS}
FEATURE_NAME_TO_FIELD: Dict[str, FeatureField] = {f.name: f for f in ALL_FEATURE_FIELDS}


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
