"""
Event-Driven Stock RL — Authoritative Feature Dictionary & Schema Specifications V8.0
Contains complete metadata: Identifier, Math Definition, Unit, Source, Availability Timing,
Missing-Value Policy, and Version.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional

class FeatureSchemaVersion(str, Enum):
    """
    Formal Feature Schema Versions.
    BASELINE_V8_0_ORIGINAL: Research formal baseline (v8.0-original) confirmed in ChatGPT V8 protocol.
    ALTERNATIVE_V8_1_CANDIDATE: Candidate market encoder ablation feature set (v8.1-candidate).
    """
    BASELINE_V8_0_ORIGINAL = "v8.0-original"
    ALTERNATIVE_V8_1_CANDIDATE = "v8.1-candidate"
    
    # Backward compatibility aliases
    BASELINE_V8_0 = "v8.0-original"
    ALTERNATIVE_V8_1 = "v8.1-candidate"

# Current active schema in runtime: v8.0-original is the Primary Research Baseline
CURRENT_ACTIVE_SCHEMA_VERSION: str = FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value

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

# Baseline Schema V8.0 Formal Specification (Locked per ChatGPT V8 Specification)
BASELINE_V8_0_PRICE_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("PRC_01", "return_1d", "(P_close,t - P_close,t-1) / P_close,t-1", "ratio", "TWSE", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("PRC_02", "return_5d", "(P_close,t - P_close,t-5) / P_close,t-5", "ratio", "TWSE", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("PRC_03", "return_20d", "(P_close,t - P_close,t-20) / P_close,t-20", "ratio", "TWSE", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("PRC_04", "overnight_gap", "(P_open,t - P_close,t-1) / P_close,t-1", "ratio", "TWSE", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("PRC_05", "open_close_return", "(P_close,t - P_open,t) / P_open,t", "ratio", "TWSE", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("PRC_06", "high_low_range", "(P_high,t - P_low,t) / P_open,t", "ratio", "TWSE", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("PRC_07", "close_location", "(P_close,t - P_low,t) / (P_high,t - P_low,t + 1e-7)", "ratio_0_1", "TWSE", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("PRC_08", "log_volume", "ln(1 + Volume_t)", "dimensionless", "TWSE", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("PRC_09", "volume_ratio_20d", "Volume_t / MA20(Volume)", "ratio", "TWSE", "Daily 20:00", "Neutral (1.0)", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("PRC_10", "turnover_ratio", "Volume_t / Shares_Outstanding", "ratio", "TWSE", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("PRC_11", "realized_volatility_20d", "Annualized 20-day realized volatility of daily log returns", "annualized_ratio", "TWSE", "Daily 20:00", "Sector Median", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("PRC_12", "log_trading_value", "ln(1 + Trading_Value_t)", "dimensionless", "TWSE", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
]

# Baseline Schema V8.0 Formal Specification: Technical (16 dims, T01-T16)
# Authoritative V8 Specification: T01 RSI14, T02 RSI7, T03 KDJ K, T04 KDJ D, T05 KDJ J,
# T06 MACD DIF, T07 MACD Signal, T08 MACD Histogram, T09 Close/EMA20-1, T10 Close/EMA60-1,
# T11 EMA20/EMA60-1, T12 ATR14/Close, T13 Bollinger %B, T14 Bollinger Bandwidth,
# T15 Normalized OBV Change 5D, T16 ADX14. (CCI-14 is excluded from Original V8).
BASELINE_V8_0_TECHNICAL_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("TECH_01", "rsi_14", "(RSI_14 - 50) / 50", "bounded_ratio", "Technical Engine", "Daily 20:00", "Neutral (0.0)", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_02", "rsi_7", "(RSI_7 - 50) / 50", "bounded_ratio", "Technical Engine", "Daily 20:00", "Neutral (0.0)", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_03", "kdj_k", "(K_9 - 50) / 50", "bounded_ratio", "Technical Engine", "Daily 20:00", "Neutral (0.0)", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_04", "kdj_d", "(D_9 - 50) / 50", "bounded_ratio", "Technical Engine", "Daily 20:00", "Neutral (0.0)", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_05", "kdj_j", "(3K - 2D - 50) / 50", "bounded_ratio", "Technical Engine", "Daily 20:00", "Neutral (0.0)", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_06", "macd_dif", "(EMA12 - EMA26) / P_close", "ratio", "Technical Engine", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_07", "macd_signal", "EMA9(MACD_DIF) / P_close", "ratio", "Technical Engine", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_08", "macd_hist", "(MACD_DIF - MACD_Signal) / P_close", "ratio", "Technical Engine", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_09", "close_ema20_ratio", "P_close / EMA20 - 1", "ratio", "Technical Engine", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_10", "close_ema60_ratio", "P_close / EMA60 - 1", "ratio", "Technical Engine", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_11", "ema20_ema60_ratio", "EMA20 / EMA60 - 1", "ratio", "Technical Engine", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_12", "atr14_close_ratio", "ATR_14 / P_close", "ratio", "Technical Engine", "Daily 20:00", "Median", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_13", "bollinger_percent_b", "(P_close - BB_lower) / (BB_upper - BB_lower + 1e-7)", "ratio", "Technical Engine", "Daily 20:00", "Neutral (0.5)", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_14", "bollinger_bandwidth", "(BB_upper - BB_lower) / MA20", "ratio", "Technical Engine", "Daily 20:00", "Median", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_15", "normalized_obv_change_5d", "(OBV_t - OBV_{t-5}) / (5 * MA20(Volume))", "dimensionless", "Technical Engine", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("TECH_16", "adx_14", "ADX_14 / 100.0", "ratio", "Technical Engine", "Daily 20:00", "Neutral (0.2)", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
]

# Baseline Schema V8.0 Formal Specification: Institutional (8 dims, I01-I08)
# Authoritative V8 Specification:
# I01: Foreign Net Buy / Average Volume 20D (denominator is MA20 Volume)
# I02: Investment Trust Net Buy / Average Volume 20D (denominator is MA20 Volume)
# I03: Dealer Net Buy / Average Volume 20D (denominator is MA20 Volume)
# I04: Foreign Net Buy 5D (accumulated 5D normalized by 5D average volume)
# I05: Total Institutional Net Buy 5D (accumulated 5D normalized by 5D average volume)
# I06: Margin Balance Change (normalized by MA20 Volume)
# I07: Short Balance Change (normalized by MA20 Volume)
# I08: Short / Margin Balance Ratio
BASELINE_V8_0_INSTITUTIONAL_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("INST_01", "foreign_net_buy_over_vol20d", "Foreign Net Buy Shares / MA20(Volume)", "ratio", "TWSE", "Daily 18:00 (<=20:00)", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("INST_02", "trust_net_buy_over_vol20d", "Investment Trust Net Buy Shares / MA20(Volume)", "ratio", "TWSE", "Daily 18:00 (<=20:00)", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("INST_03", "dealer_net_buy_over_vol20d", "Dealer Net Buy Shares / MA20(Volume)", "ratio", "TWSE", "Daily 18:00 (<=20:00)", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("INST_04", "foreign_net_buy_5d", "Sum(Foreign Net Buy Shares, 5D) / (5 * MA20(Volume))", "ratio", "TWSE", "Daily 18:00 (<=20:00)", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("INST_05", "total_institutional_net_buy_5d", "Sum(Total 3-Inst Net Buy Shares, 5D) / (5 * MA20(Volume))", "ratio", "TWSE", "Daily 18:00 (<=20:00)", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("INST_06", "margin_balance_change", "(Margin Balance_t - Margin Balance_{t-1}) / MA20(Volume)", "ratio", "TWSE", "Daily 19:30 (<=20:00)", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("INST_07", "short_balance_change", "(Short Balance_t - Short Balance_{t-1}) / MA20(Volume)", "ratio", "TWSE", "Daily 19:30 (<=20:00)", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("INST_08", "short_margin_balance_ratio", "Short Balance / (Margin Balance + 1e-7)", "ratio", "TWSE", "Daily 19:30 (<=20:00)", "Forward-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
]

# Baseline Schema V8.0 Formal Specification: Macro (12 dims, M01-M12)
# Authoritative V8 Specification:
# M01: TAIEX Return 1D
# M02: TAIEX Return 5D
# M03: TAIEX Volatility 20D
# M04: Sector Return 1D
# M05: Sector Relative Strength 20D
# M06: USD/TWD Return
# M07: S&P 500 Return (T-1 PIT <= 20:00)
# M08: NASDAQ Return (T-1 PIT <= 20:00)
# M09: SOX Return (T-1 PIT <= 20:00)
# M10: VIX Level (T-1 PIT <= 20:00)
# M11: US 10Y Yield (T-1 PIT <= 20:00)
# M12: US 10Y Yield Change (T-1 PIT <= 20:00)
# All cross-market features must strictly adhere to Historical Availability at Taiwan Time 20:00.
BASELINE_V8_0_MACRO_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("MACRO_01", "taiex_return_1d", "(TAIEX_t - TAIEX_{t-1}) / TAIEX_{t-1}", "ratio", "TWSE", "Daily 14:00 (<=20:00)", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("MACRO_02", "taiex_return_5d", "(TAIEX_t - TAIEX_{t-5}) / TAIEX_{t-5}", "ratio", "TWSE", "Daily 14:00 (<=20:00)", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("MACRO_03", "taiex_volatility_20d", "20-day annualized realized vol of TAIEX", "annualized_ratio", "TWSE", "Daily 14:00 (<=20:00)", "Median", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("MACRO_04", "sector_return_1d", "(SectorIndex_t - SectorIndex_{t-1}) / SectorIndex_{t-1}", "ratio", "TWSE Industry Index", "Daily 14:00 (<=20:00)", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("MACRO_05", "sector_relative_strength_20d", "SectorReturn_20D - TAIEX_Return_20D", "ratio", "TWSE Industry Index", "Daily 14:00 (<=20:00)", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("MACRO_06", "usdtwd_return", "(USDTWD_t - USDTWD_{t-1}) / USDTWD_{t-1}", "ratio", "Taipei Forex", "Daily 16:30 (<=20:00)", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("MACRO_07", "sp500_return", "S&P 500 daily return (T-1 PIT)", "ratio", "Cboe / S&P", "T-1 Available <= 20:00", "Forward-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("MACRO_08", "nasdaq_return", "NASDAQ Composite daily return (T-1 PIT)", "ratio", "NASDAQ", "T-1 Available <= 20:00", "Forward-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("MACRO_09", "sox_return", "SOX Semiconductor index daily return (T-1 PIT)", "ratio", "PHLX", "T-1 Available <= 20:00", "Forward-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("MACRO_10", "vix_level", "Cboe VIX index level / 100.0 (T-1 PIT)", "ratio", "Cboe", "T-1 Available <= 20:00", "Forward-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("MACRO_11", "us_10y_yield", "US 10Y Treasury sovereign yield / 100.0 (T-1 PIT)", "ratio", "Federal Reserve / US", "T-1 Available <= 20:00", "Forward-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("MACRO_12", "us_10y_yield_change", "Delta US 10Y Treasury yield bps / 100.0 (T-1 PIT)", "bps", "Federal Reserve / US", "T-1 Available <= 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
]

# Baseline Schema V8.0 Formal Specification: Stock Context (16 dims, C01-C16)
# Authoritative V8 Specification:
# C01–C08: Industry Embedding 8D (strictly trained on lagged historical data with independent versioning)
# C09: Log Market Capitalization
# C10: Log Average Trading Value 60D
# C11: Volatility Percentile
# C12: Liquidity Percentile
# C13: Beta 120D
# C14: Market Cap Percentile Within Industry
# C15: Listing Age
# C16: Recent Trading Suspension Indicator
BASELINE_V8_0_STOCK_CONTEXT_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("CTX_01", "industry_embedding_01", "Industry Embedding dimension 1 (Historical As-Of versioned)", "embedding", "Industry Representation", "Annual PIT", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_02", "industry_embedding_02", "Industry Embedding dimension 2 (Historical As-Of versioned)", "embedding", "Industry Representation", "Annual PIT", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_03", "industry_embedding_03", "Industry Embedding dimension 3 (Historical As-Of versioned)", "embedding", "Industry Representation", "Annual PIT", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_04", "industry_embedding_04", "Industry Embedding dimension 4 (Historical As-Of versioned)", "embedding", "Industry Representation", "Annual PIT", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_05", "industry_embedding_05", "Industry Embedding dimension 5 (Historical As-Of versioned)", "embedding", "Industry Representation", "Annual PIT", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_06", "industry_embedding_06", "Industry Embedding dimension 6 (Historical As-Of versioned)", "embedding", "Industry Representation", "Annual PIT", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_07", "industry_embedding_07", "Industry Embedding dimension 7 (Historical As-Of versioned)", "embedding", "Industry Representation", "Annual PIT", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_08", "industry_embedding_08", "Industry Embedding dimension 8 (Historical As-Of versioned)", "embedding", "Industry Representation", "Annual PIT", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_09", "log_market_capitalization", "ln(Market Capitalization_t)", "dimensionless", "TWSE", "Monthly Rebalance", "Forward-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_10", "log_avg_trading_value_60d", "ln(1 + MA60(Trading Value))", "dimensionless", "Market Data Engine", "Daily 20:00", "Zero-fill", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_11", "volatility_percentile", "Cross-sectional percentile rank of 60D volatility [0, 1]", "rank", "Market Data Engine", "Daily 20:00", "Median (0.5)", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_12", "liquidity_percentile", "Cross-sectional percentile rank of 60D turnover [0, 1]", "rank", "Market Data Engine", "Daily 20:00", "Median (0.5)", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_13", "beta_120d", "120-day rolling beta against TAIEX benchmark", "ratio", "Market Data Engine", "Daily 20:00", "Market Neutral (1.0)", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_14", "market_cap_percentile_within_industry", "Percentile rank of market cap within same industry [0, 1]", "rank", "TWSE", "Monthly Rebalance", "Median (0.5)", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_15", "listing_age", "min(days_since_ipo / (365.25 * 10), 1.0)", "normalized_duration", "TWSE", "Static / Daily 20:00", "1.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("CTX_16", "recent_trading_suspension_indicator", "1.0 if stock had trading halt in prior 20D, else 0.0", "binary", "TWSE", "Daily 20:00", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
]

# Baseline Schema V8.0 Formal Specification: Event Metadata (24 dims, E01-E24)
# Authoritative V8 Specification Layout:
# E01–E04: Actor / Institution Role Embedding (4D)
# E05–E12: Event Type Embedding (8D)
# E13–E16: Event Status Embedding (4D)
# E17: Evidence Traceability
# E18: Source Reliability Estimate
# E19: Source Independence
# E20: Novelty Input Estimate
# E21: Market Surprise (Pre-event prior implied volatility/move, NO post-event price reaction)
# E22: Quantified Magnitude
# E23: Event Age (Index 22)
# E24: Timestamp Reliability (Index 23)
BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("EV_META_01", "actor_role_embedding_01", "Actor / Institution Role Embedding dimension 1", "embedding", "LLM Entity Resolver", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_02", "actor_role_embedding_02", "Actor / Institution Role Embedding dimension 2", "embedding", "LLM Entity Resolver", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_03", "actor_role_embedding_03", "Actor / Institution Role Embedding dimension 3", "embedding", "LLM Entity Resolver", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_04", "actor_role_embedding_04", "Actor / Institution Role Embedding dimension 4", "embedding", "LLM Entity Resolver", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_05", "event_type_embedding_01", "Event Type Embedding dimension 1", "embedding", "LLM Event Classifier", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_06", "event_type_embedding_02", "Event Type Embedding dimension 2", "embedding", "LLM Event Classifier", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_07", "event_type_embedding_03", "Event Type Embedding dimension 3", "embedding", "LLM Event Classifier", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_08", "event_type_embedding_04", "Event Type Embedding dimension 4", "embedding", "LLM Event Classifier", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_09", "event_type_embedding_05", "Event Type Embedding dimension 5", "embedding", "LLM Event Classifier", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_10", "event_type_embedding_06", "Event Type Embedding dimension 6", "embedding", "LLM Event Classifier", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_11", "event_type_embedding_07", "Event Type Embedding dimension 7", "embedding", "LLM Event Classifier", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_12", "event_type_embedding_08", "Event Type Embedding dimension 8", "embedding", "LLM Event Classifier", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_13", "event_status_embedding_01", "Event Status Embedding dimension 1", "embedding", "LLM Status Parser", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_14", "event_status_embedding_02", "Event Status Embedding dimension 2", "embedding", "LLM Status Parser", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_15", "event_status_embedding_03", "Event Status Embedding dimension 3", "embedding", "LLM Status Parser", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_16", "event_status_embedding_04", "Event Status Embedding dimension 4", "embedding", "LLM Status Parser", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_17", "evidence_traceability", "Evidence Traceability score [0, 1]", "score", "LLM Extraction", "Event Release", "1.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_18", "source_reliability_estimate", "Source Reliability Estimate [0, 1]", "score", "Metadata Registry", "Event Release", "0.5", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_19", "source_independence", "Source Independence score [0, 1]", "score", "Metadata Registry", "Event Release", "1.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_20", "novelty_input_estimate", "Novelty Input Estimate prior score [0, 1]", "score", "Event Memory Engine", "Event Release", "0.5", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_21", "market_surprise", "Pre-event implied move vs prior historical volatility (strictly prior to event release, NO post-event price reaction)", "score", "Market Shock Engine", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_22", "quantified_magnitude", "Quantified financial / operational magnitude score [0, 1]", "score", "LLM Extraction", "Event Release", "0.5", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_23", "event_age", "(T_decision - T_available) in fractional days (E23)", "days", "PIT Engine", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_META_24", "timestamp_reliability", "Timestamp Reliability score [0, 1] (filing vs vendor PIT precision)", "score", "PIT Engine", "Event Release", "1.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
]

# Baseline Schema V8.0 Formal Specification: Event-Stock Relation (8 dims, R01-R08)
# Authoritative V8 Specification:
# R01: Direct Company Mention
# R02: Industry Match
# R03: Upstream Exposure
# R04: Downstream Exposure
# R05: Customer Relationship
# R06: Supplier Relationship
# R07: Geographic Exposure
# R08: Macro Factor Exposure
# All company relations and supply chain links must be Historical As-Of.
BASELINE_V8_0_EVENT_STOCK_RELATION_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("EV_REL_01", "direct_company_mention", "1.0 if stock symbol/name is explicitly mentioned, else 0.0", "binary", "Entity Resolver", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_REL_02", "industry_match", "1.0 if event entity shares same primary industry sector, else 0.0", "binary", "Industry KG", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_REL_03", "upstream_exposure", "Upstream supply chain exposure weight (Historical As-Of) [0, 1]", "score", "Supply Chain DB", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_REL_04", "downstream_exposure", "Downstream supply chain exposure weight (Historical As-Of) [0, 1]", "score", "Supply Chain DB", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_REL_05", "customer_relationship", "Customer relationship connection weight (Historical As-Of) [0, 1]", "score", "Supply Chain DB", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_REL_06", "supplier_relationship", "Supplier relationship connection weight (Historical As-Of) [0, 1]", "score", "Supply Chain DB", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_REL_07", "geographic_exposure", "Geographic / Regional operational exposure weight [0, 1]", "score", "Knowledge Graph", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("EV_REL_08", "macro_factor_exposure", "Macro economic factor exposure sensitivity score [0, 1]", "score", "Factor Model", "Event Release", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
]

# Baseline Schema V8.0 Formal Specification: Position State (4 dims, S01-S04)
BASELINE_V8_0_POSITION_STATE_FEATURE_FIELDS: List[FeatureField] = [
    FeatureField("POS_01", "current_target_position", "Target position state: 0.0 for Cash, 1.0 for Long", "state", "Gym Environment", "Decision 20:00", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("POS_02", "unrealized_pnl_pct", "(P_close,t - P_entry) / P_entry", "ratio", "Gym Environment", "Decision 20:00", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("POS_03", "holding_duration_days", "min(days_in_position / 60.0, 1.0)", "normalized_duration", "Gym Environment", "Decision 20:00", "0.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
    FeatureField("POS_04", "time_since_last_successful_trade", "min(days_since_last_successful_trade / 60.0, 1.0)", "normalized_duration", "Gym Environment", "Decision 20:00", "1.0", version=FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value),
]

BASELINE_V8_0_ALL_FEATURE_FIELDS: List[FeatureField] = (
    BASELINE_V8_0_PRICE_FEATURE_FIELDS
    + BASELINE_V8_0_TECHNICAL_FEATURE_FIELDS
    + BASELINE_V8_0_INSTITUTIONAL_FEATURE_FIELDS
    + BASELINE_V8_0_MACRO_FEATURE_FIELDS
    + BASELINE_V8_0_STOCK_CONTEXT_FEATURE_FIELDS
    + BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS
    + BASELINE_V8_0_EVENT_STOCK_RELATION_FEATURE_FIELDS
    + BASELINE_V8_0_POSITION_STATE_FEATURE_FIELDS
)

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
    FeatureField("EV_META_01", "category_earnings", "Binary indicator: earnings release, quarterly financial results", "binary", "Exchange MOPS", "Event Release", "0.0"),
    FeatureField("EV_META_02", "category_revenue_guidance", "Binary indicator: monthly revenue, guidance update", "binary", "Exchange MOPS", "Event Release", "0.0"),
    FeatureField("EV_META_03", "category_order_contract", "Binary indicator: major client order, commercial supply contract", "binary", "LLM Extraction", "Event Release", "0.0"),
    FeatureField("EV_META_04", "category_capacity_expansion", "Binary indicator: capex expansion, new fab, equipment purchase", "binary", "Exchange MOPS", "Event Release", "0.0"),
    FeatureField("EV_META_05", "category_merger_acquisition", "Binary indicator: tender offer, M&A, equity investment", "binary", "Exchange MOPS", "Event Release", "0.0"),
    FeatureField("EV_META_06", "category_litigation_regulatory", "Binary indicator: patent lawsuit, legal dispute, regulator probe", "binary", "Exchange MOPS", "Event Release", "0.0"),
    FeatureField("EV_META_07", "category_management_governance", "Binary indicator: board / executive change, insider holding change", "binary", "Exchange MOPS", "Event Release", "0.0"),
    FeatureField("EV_META_08", "category_macro_industry", "Binary indicator: macro policy, tariff, export controls, trade restriction", "binary", "News Vendor", "Event Release", "0.0"),
    FeatureField("EV_META_09", "source_credibility_tier", "Exchange MOPS=1.0, Major News=0.8, Media Wire=0.6, Other=0.4", "score", "Metadata Registry", "Event Release", "0.5"),
    FeatureField("EV_META_10", "event_sentiment_score", "LLM sentiment score [-1.0, +1.0]", "bounded_score", "LLM Extraction", "Event Release", "0.0"),
    FeatureField("EV_META_11", "sentiment_confidence", "LLM extraction confidence [0.0, 1.0]", "score", "LLM Extraction", "Event Release", "0.5"),
    FeatureField("EV_META_12", "event_scope_tier", "Company-specific=1.0, Supply-chain=0.7, Sector=0.4, Macro=0.1", "score", "LLM Extraction", "Event Release", "1.0"),
    FeatureField("EV_META_13", "event_age_fractional_days", "(T_decision - T_available) in fractional days", "days", "PIT Engine", "Event Release", "0.0"),
    FeatureField("EV_META_14", "historical_source_accuracy", "Empirical reliability score of publishing media outlet", "score", "Historical Analytics", "Event Release", "0.5"),
    FeatureField("EV_META_15", "title_body_consistency", "Cosine similarity between headline and extracted evidence", "score", "LLM Extraction", "Event Release", "1.0"),
    FeatureField("EV_META_16", "verified_entity_count", "Number of verified TWSE stock symbols matched", "count", "Entity Resolver", "Event Release", "1.0"),
    FeatureField("EV_META_17", "is_breaking_news", "Binary indicator: published during regular trading hours", "binary", "PIT Engine", "Event Release", "0.0"),
    FeatureField("EV_META_18", "revision_count", "Number of official revision amendments published", "count", "PIT Engine", "Event Release", "0.0"),
    FeatureField("EV_META_19", "urgency_flag", "Binary indicator for urgent breaking development", "binary", "News Vendor", "Event Release", "0.0"),
    FeatureField("EV_META_20", "vendor_ingestion_delay_min", "ln(1 + vendor delay minutes)", "dimensionless", "PIT Engine", "Event Release", "0.0"),
    FeatureField("EV_META_21", "document_word_count_bucket", "Word count bucket: Short=1, Medium=2, Long=3, Deep=4", "categorical", "Document Parser", "Event Release", "2.0"),
    FeatureField("EV_META_22", "evidence_span_count", "Count of extracted factual evidence sentence spans", "count", "LLM Extraction", "Event Release", "1.0"),
    FeatureField("EV_META_23", "is_regulatory_filing", "Binary indicator for MOPS material announcement", "binary", "Exchange Filing", "Event Release", "0.0"),
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

TABLE_NAME_TO_FIELDS: Dict[str, List[FeatureField]] = {
    "price": PRICE_FEATURE_FIELDS,
    "technical": TECHNICAL_FEATURE_FIELDS,
    "institutional": INSTITUTIONAL_FEATURE_FIELDS,
    "macro": MACRO_FEATURE_FIELDS,
    "stock_context": STOCK_CONTEXT_FEATURE_FIELDS,
    "event_metadata": EVENT_METADATA_FEATURE_FIELDS,
    "event_stock_relation": EVENT_STOCK_RELATION_FEATURE_FIELDS,
    "position_state": POSITION_STATE_FEATURE_FIELDS,
}

BASELINE_V8_0_TABLE_NAME_TO_FIELDS: Dict[str, List[FeatureField]] = {
    "price": BASELINE_V8_0_PRICE_FEATURE_FIELDS,
    "technical": BASELINE_V8_0_TECHNICAL_FEATURE_FIELDS,
    "institutional": BASELINE_V8_0_INSTITUTIONAL_FEATURE_FIELDS,
    "macro": BASELINE_V8_0_MACRO_FEATURE_FIELDS,
    "stock_context": BASELINE_V8_0_STOCK_CONTEXT_FEATURE_FIELDS,
    "event_metadata": BASELINE_V8_0_EVENT_METADATA_FEATURE_FIELDS,
    "event_stock_relation": BASELINE_V8_0_EVENT_STOCK_RELATION_FEATURE_FIELDS,
    "position_state": BASELINE_V8_0_POSITION_STATE_FEATURE_FIELDS,
}

PRICE_INDEX_BY_ID: Dict[str, int] = {f.feature_id: i for i, f in enumerate(PRICE_FEATURE_FIELDS)}
PRICE_INDEX_BY_NAME: Dict[str, int] = {f.name: i for i, f in enumerate(PRICE_FEATURE_FIELDS)}

TECHNICAL_INDEX_BY_ID: Dict[str, int] = {f.feature_id: i for i, f in enumerate(TECHNICAL_FEATURE_FIELDS)}
TECHNICAL_INDEX_BY_NAME: Dict[str, int] = {f.name: i for i, f in enumerate(TECHNICAL_FEATURE_FIELDS)}

INSTITUTIONAL_INDEX_BY_ID: Dict[str, int] = {f.feature_id: i for i, f in enumerate(INSTITUTIONAL_FEATURE_FIELDS)}
INSTITUTIONAL_INDEX_BY_NAME: Dict[str, int] = {f.name: i for i, f in enumerate(INSTITUTIONAL_FEATURE_FIELDS)}

MACRO_INDEX_BY_ID: Dict[str, int] = {f.feature_id: i for i, f in enumerate(MACRO_FEATURE_FIELDS)}
MACRO_INDEX_BY_NAME: Dict[str, int] = {f.name: i for i, f in enumerate(MACRO_FEATURE_FIELDS)}

STOCK_CONTEXT_INDEX_BY_ID: Dict[str, int] = {f.feature_id: i for i, f in enumerate(STOCK_CONTEXT_FEATURE_FIELDS)}
STOCK_CONTEXT_INDEX_BY_NAME: Dict[str, int] = {f.name: i for i, f in enumerate(STOCK_CONTEXT_FEATURE_FIELDS)}

EVENT_METADATA_INDEX_BY_ID: Dict[str, int] = {f.feature_id: i for i, f in enumerate(EVENT_METADATA_FEATURE_FIELDS)}
EVENT_METADATA_INDEX_BY_NAME: Dict[str, int] = {f.name: i for i, f in enumerate(EVENT_METADATA_FEATURE_FIELDS)}

EVENT_STOCK_RELATION_INDEX_BY_ID: Dict[str, int] = {f.feature_id: i for i, f in enumerate(EVENT_STOCK_RELATION_FEATURE_FIELDS)}
EVENT_STOCK_RELATION_INDEX_BY_NAME: Dict[str, int] = {f.name: i for i, f in enumerate(EVENT_STOCK_RELATION_FEATURE_FIELDS)}

POSITION_STATE_INDEX_BY_ID: Dict[str, int] = {f.feature_id: i for i, f in enumerate(POSITION_STATE_FEATURE_FIELDS)}
POSITION_STATE_INDEX_BY_NAME: Dict[str, int] = {f.name: i for i, f in enumerate(POSITION_STATE_FEATURE_FIELDS)}

GLOBAL_FEATURE_INDEX_BY_ID: Dict[str, int] = {}
GLOBAL_FEATURE_INDEX_BY_NAME: Dict[str, int] = {}
for fields in TABLE_NAME_TO_FIELDS.values():
    for i, f in enumerate(fields):
        GLOBAL_FEATURE_INDEX_BY_ID[f.feature_id] = i
        GLOBAL_FEATURE_INDEX_BY_NAME[f.name] = i


def get_feature_index(key: str, table_name: Optional[str] = None, schema_version: Optional[Any] = None) -> int:
    """
    Unified lookup interface: returns the 0-based column index of a feature in its tensor.
    Supports both v8.1-candidate (EV_META_13, etc.) and v8.0-original (E23, P01, etc.).
    Accepts standard shorthand codes (e.g. 'P07' -> 6, 'C01' -> 0, 'E23' -> 22, 'S04' -> 3).
    If table_name is specified, searches within that table.
    """
    import re
    # Standard shorthand matching (e.g. P07 -> 6, E23 -> 22, C01 -> 0, S04 -> 3)
    m = re.match(r"^([PTIMCERS])(\d{2})$", key)
    if m:
        return int(m.group(2)) - 1

    # Check baseline schema if requested
    if schema_version in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL, "v8.0-original", "v8.0"):
        if key in ("event_age_fractional_days", "event_age_days", "event_age"):
            return ORIGINAL_V8_EVENT_AGE_METADATA_INDEX
        if table_name is not None:
            table_lower = table_name.lower().strip()
            if table_lower in BASELINE_V8_0_TABLE_NAME_TO_FIELDS:
                fields = BASELINE_V8_0_TABLE_NAME_TO_FIELDS[table_lower]
                for i, f in enumerate(fields):
                    if f.feature_id == key or f.name == key:
                        return i
        for fields in BASELINE_V8_0_TABLE_NAME_TO_FIELDS.values():
            for i, f in enumerate(fields):
                if f.feature_id == key or f.name == key:
                    return i

    # Candidate / Default lookup
    if table_name is not None:
        table_lower = table_name.lower().strip()
        if table_lower not in TABLE_NAME_TO_FIELDS:
            raise KeyError(f"Unknown table_name '{table_name}'. Valid tables: {list(TABLE_NAME_TO_FIELDS.keys())}")
        fields = TABLE_NAME_TO_FIELDS[table_lower]
        for i, f in enumerate(fields):
            if f.feature_id == key or f.name == key:
                return i
        if table_lower == "event_metadata" and key in ("event_age", "event_age_days"):
            return CANDIDATE_V8_1_EVENT_AGE_METADATA_INDEX
        raise KeyError(f"Feature '{key}' not found in table '{table_name}'.")

    if key in GLOBAL_FEATURE_INDEX_BY_ID:
        return GLOBAL_FEATURE_INDEX_BY_ID[key]
    if key in GLOBAL_FEATURE_INDEX_BY_NAME:
        return GLOBAL_FEATURE_INDEX_BY_NAME[key]
    if key in ("event_age", "event_age_days"):
        return CANDIDATE_V8_1_EVENT_AGE_METADATA_INDEX
    
    # Fallback: check baseline table if not found in candidate
    for fields in BASELINE_V8_0_TABLE_NAME_TO_FIELDS.values():
        for i, f in enumerate(fields):
            if f.feature_id == key or f.name == key:
                return i

    raise KeyError(f"Feature '{key}' not found in any feature table.")


# Authoritative constants for Event Age metadata index:
ORIGINAL_V8_EVENT_AGE_METADATA_INDEX: int = 22  # E23 in v8.0-original
CANDIDATE_V8_1_EVENT_AGE_METADATA_INDEX: int = 12  # EV_META_13 in v8.1-candidate
EVENT_AGE_METADATA_INDEX: int = 12  # Active runtime candidate schema


def get_event_age_index(schema_version: Optional[Any] = None) -> int:
    """
    Unified lookup for Event Age index across schema versions:
    - BASELINE_V8_0_ORIGINAL ('v8.0-original'): E23 -> Index 22
    - ALTERNATIVE_V8_1_CANDIDATE ('v8.1-candidate'): EV_META_13 -> Index 12
    Defaults to CURRENT_ACTIVE_SCHEMA_VERSION if schema_version is None.
    """
    ver = schema_version if schema_version is not None else CURRENT_ACTIVE_SCHEMA_VERSION
    if ver in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL, "v8.0-original", "v8.0"):
        return ORIGINAL_V8_EVENT_AGE_METADATA_INDEX
    return CANDIDATE_V8_1_EVENT_AGE_METADATA_INDEX


def validate_schema_version(schema_version: str) -> bool:
    """
    Validates whether a schema version string is supported.
    Raises ValueError if unsupported.
    """
    valid = [v.value for v in FeatureSchemaVersion]
    if schema_version not in valid:
        raise ValueError(
            f"Unsupported schema version '{schema_version}'. Supported versions are: {valid}. "
            f"Silent mixing of schema versions is strictly prohibited."
        )
    return True


def get_schema_table_fields(
    table_name: str,
    schema_version: Optional[str] = None
) -> List[FeatureField]:
    """
    Returns the list of FeatureField definitions for the specified table and schema version.
    """
    ver = schema_version if schema_version is not None else CURRENT_ACTIVE_SCHEMA_VERSION
    table_lower = table_name.lower().strip()
    if ver in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL, "v8.0-original", "v8.0"):
        if table_lower not in BASELINE_V8_0_TABLE_NAME_TO_FIELDS:
            raise KeyError(f"Unknown table '{table_name}' in baseline schema.")
        return BASELINE_V8_0_TABLE_NAME_TO_FIELDS[table_lower]
    else:
        if table_lower not in TABLE_NAME_TO_FIELDS:
            raise KeyError(f"Unknown table '{table_name}' in candidate schema.")
        return TABLE_NAME_TO_FIELDS[table_lower]


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
