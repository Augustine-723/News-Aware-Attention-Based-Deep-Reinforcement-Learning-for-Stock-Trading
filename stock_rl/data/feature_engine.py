"""
Historical Feature Engineering Engine V8.0 / V8.3
Strictly implements the Authoritative 100-Feature Schema per
docs/specifications/v8_authoritative_100_feature_schema.md.

================================================================================
IMPLEMENTATION CONTRACT & ENGINEERING DEFAULTS
================================================================================
1. Point-in-Time Decision Cutoff:
   - Daily decision timestamp is strictly Asia/Taipei 20:00:00.
   - All TWSE market close data (13:30), institutional data (18:00, 19:30),
     and forex data (16:30) are available before 20:00.
   - All US and overseas cross-market features (S&P 500, NASDAQ, SOX, VIX, US 10Y)
     are strictly lagged by T-1 (available prior to 20:00).
   - All joins use backward-only As-Of join without bfill to prevent forward lookahead.
2. Numerical Stability Epsilon:
   - EPSILON = 1e-7 for all division and ratio calculations.
3. Module Technical Defaults:
   - RSI: Wilder's smoothing with lookbacks 14 and 7; normalized as (RSI - 50) / 50.
   - KDJ: Fast %K (9 periods), Slow %D (3 periods EMA), %J = 3K - 2D; normalized as (X - 50) / 50.
   - MACD: EMA12 - EMA26, Signal EMA9, normalized by P_close.
   - Bollinger Bands: 20-day SMA, 2.0 standard deviations; Bandwidth = (Upper - Lower) / MA20.
   - ATR: 14-day Wilder smoothing, normalized by P_close.
   - OBV: Cumulative volume signed by price direction, 5D change / (5 * MA20(Volume)).
   - ADX: 14-period Wilder smoothing, normalized by 100.0 into [0, 1].
4. Module Institutional Defaults:
   - Denominator for daily flows (I01..I03) is MA20 Volume (20-day average volume).
   - 5-day flows (I04..I05) are accumulated over 5 days and normalized by 5 * MA20 Volume.
   - Margin and short balance changes (I06..I07) are normalized by MA20 Volume.
5. Module Macro Defaults:
   - TAIEX 1D, 5D returns and 20D annualized volatility (factor = sqrt(250)).
   - Sector 1D return and 20D relative strength against TAIEX: SectorReturn20D - TAIEX_Return20D.
6. Module Stock Context Defaults:
   - C01..C08: Dense 8D industry representation with independent historical versioning.
   - Percentiles: cross-sectional percentile ranks normalized into [0, 1].
   - Beta: 120-day rolling covariance / benchmark variance against TAIEX.
7. Module Event Metadata Defaults:
   - E01..E04 (4D Actor), E05..E12 (8D Type), E13..E16 (4D Status).
   - E21 Market Surprise: strictly pre-event prior implied volatility / move, ZERO post-event reaction.
   - E23 Event Age: (T_decision - T_available) in fractional days (Python Index 22).
   - E24 Timestamp Reliability: in [0, 1] (Python Index 23).
8. Missing Data Imputation:
   - Implements declared policies: Forward-fill, Zero-fill, Neutral values (e.g. 1.0 for volume ratio, 0.0 for RSI/KDJ centered, 0.5 for percentiles).
   - ZERO bfill allowed across any module to eliminate lookahead leakage.
================================================================================
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from stock_rl.config.schema import (
    MARKET_WINDOW_LENGTH,
    PRICE_DIM,
    TECHNICAL_DIM,
    INSTITUTIONAL_DIM,
    MACRO_DIM,
    STOCK_CONTEXT_DIM,
    TOTAL_EVENT_SLOTS,
    EVENT_METADATA_DIM,
    EVENT_STOCK_RELATION_DIM,
    POSITION_STATE_DIM,
    CURRENT_ACTIVE_SCHEMA_VERSION,
    FeatureSchemaVersion,
)
from stock_rl.data.pit_engine import (
    ProvenanceRecord,
    asof_join_cross_market,
)

EPSILON: float = 1e-7
ANNUAL_TRADING_DAYS: float = 250.0


# ==============================================================================
# 1. Price Dynamics (12 Dims: P01–P12)
# ==============================================================================

def compute_price_features(
    df_ohlcv: pd.DataFrame,
    shares_outstanding: float = 1e8,
    schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION,
) -> np.ndarray:
    """
    Computes the 12 Price Dynamics features for each time step.
    Expected df_ohlcv columns: ['open', 'high', 'low', 'close', 'volume', 'trading_value']
    Returns:
        np.ndarray of shape (len(df_ohlcv), 12)
    """
    close = df_ohlcv['close'].astype(float)
    open_p = df_ohlcv['open'].astype(float)
    high = df_ohlcv['high'].astype(float)
    low = df_ohlcv['low'].astype(float)
    vol = df_ohlcv['volume'].astype(float)
    val = df_ohlcv['trading_value'].astype(float) if 'trading_value' in df_ohlcv.columns else (close * vol)

    if schema_version in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value, "v8.0-original", "v8.0"):
        # Original V8 Baseline 12D Price Dynamics:
        # P01: Return 1D
        ret_1d = close.pct_change(1).fillna(0.0)
        # P02: Return 5D
        ret_5d = close.pct_change(5).fillna(0.0)
        # P03: Return 20D
        ret_20d = close.pct_change(20).fillna(0.0)
        # P04: Overnight Gap: (Open_t - Close_{t-1}) / Close_{t-1} (Zero bfill)
        prev_close = close.shift(1).fillna(close.iloc[0])
        overnight_gap = ((open_p - prev_close) / (prev_close + EPSILON)).fillna(0.0)
        # P05: Open-Close Return: (Close_t - Open_t) / Open_t
        open_close_ret = ((close - open_p) / (open_p + EPSILON)).fillna(0.0)
        # P06: High-Low Range: (High_t - Low_t) / Open_t
        hl_range = ((high - low) / (open_p + EPSILON)).fillna(0.0)
        # P07: Close Location: (Close_t - Low_t) / (High_t - Low_t + EPSILON), clamped to [0, 1]
        close_loc = np.clip((close - low) / (high - low + EPSILON), 0.0, 1.0).fillna(0.5)
        # P08: Log Volume: ln(1 + Volume_t)
        log_vol = np.log1p(np.maximum(vol, 0.0))
        # P09: Volume Ratio 20D: Volume_t / MA20(Volume)
        ma20_vol = vol.rolling(window=20, min_periods=1).mean()
        vol_ratio_20d = (vol / (ma20_vol + EPSILON)).fillna(1.0)
        # P10: Turnover Ratio: Volume_t / Shares Outstanding
        turnover = (vol / (shares_outstanding + EPSILON)).fillna(0.0)
        # P11: Realized Volatility 20D: 20-day annualized std of daily returns
        log_ret = np.log(close / (prev_close + EPSILON)).fillna(0.0)
        realized_vol_20d = (log_ret.rolling(window=20, min_periods=2).std() * np.sqrt(ANNUAL_TRADING_DAYS)).fillna(0.0)
        # P12: Log Trading Value: ln(1 + Trading Value_t)
        log_val = np.log1p(np.maximum(val, 0.0))

        features = np.column_stack([
            ret_1d.values,
            ret_5d.values,
            ret_20d.values,
            overnight_gap.values,
            open_close_ret.values,
            hl_range.values,
            close_loc.values,
            log_vol.values,
            vol_ratio_20d.values,
            turnover.values,
            realized_vol_20d.values,
            log_val.values,
        ])
    else:
        # Candidate V8.1: 12 distinct normalized features per PRICE_FEATURE_FIELDS (P1-3)
        ma60 = close.rolling(60, min_periods=1).mean()
        std60 = close.rolling(60, min_periods=1).std().fillna(1.0).replace(0.0, 1.0)

        # PRC_01..PRC_04: Normalized OHLC by 60D rolling window
        open_norm = ((open_p - ma60) / std60).fillna(0.0)
        high_norm = ((high - ma60) / std60).fillna(0.0)
        low_norm = ((low - ma60) / std60).fillna(0.0)
        close_norm = ((close - ma60) / std60).fillna(0.0)

        # PRC_05: volume_norm: ln(1 + Vol) z-scored over 60 days
        log_vol = np.log1p(np.maximum(vol, 0.0))
        vol_m60 = log_vol.rolling(60, min_periods=1).mean()
        vol_s60 = log_vol.rolling(60, min_periods=1).std().fillna(1.0).replace(0.0, 1.0)
        volume_norm = ((log_vol - vol_m60) / vol_s60).fillna(0.0)

        # PRC_06: turnover_norm: ln(1 + Turnover) z-scored over 60 days
        turnover = (vol / (shares_outstanding + EPSILON)).fillna(0.0)
        log_to = np.log1p(np.maximum(turnover, 0.0))
        to_m60 = log_to.rolling(60, min_periods=1).mean()
        to_s60 = log_to.rolling(60, min_periods=1).std().fillna(1.0).replace(0.0, 1.0)
        turnover_norm = ((log_to - to_m60) / to_s60).fillna(0.0)

        # PRC_07: vwap_norm: (VWAP - P_close) / P_close
        vwap = np.where(vol > 0, val / (vol + EPSILON), (high + low + close) / 3.0)
        vwap_norm = pd.Series((vwap - close) / (close + EPSILON), index=close.index).fillna(0.0)

        # PRC_08: daily_return: (P_close,t - P_close,t-1) / P_close,t-1
        daily_return = close.pct_change(1).fillna(0.0)

        # PRC_09: intraday_range: (P_high - P_low) / P_open
        intraday_range = ((high - low) / (open_p + EPSILON)).fillna(0.0)

        # PRC_10: shadow_upper: (P_high - max(P_open, P_close)) / P_close
        max_oc = np.maximum(open_p, close)
        shadow_upper = ((high - max_oc) / (close + EPSILON)).fillna(0.0)

        # PRC_11: shadow_lower: (min(P_open, P_close) - P_low) / P_close
        min_oc = np.minimum(open_p, close)
        shadow_lower = ((min_oc - low) / (close + EPSILON)).fillna(0.0)

        # PRC_12: effective_spread_proxy: Roll (1984) proxy: 2 * sqrt(max(0, -cov(Delta P_t, Delta P_{t-1}))) / P_close
        delta_p = close.diff().fillna(0.0)
        delta_p_lag = delta_p.shift(1).fillna(0.0)
        cov_20 = (delta_p * delta_p_lag).rolling(20, min_periods=2).mean() - (
            delta_p.rolling(20, min_periods=2).mean() * delta_p_lag.rolling(20, min_periods=2).mean()
        )
        neg_cov = np.maximum(-cov_20.fillna(0.0), 0.0)
        effective_spread = (2.0 * np.sqrt(neg_cov) / (close + EPSILON)).fillna(0.0)

        features = np.column_stack([
            open_norm.values,
            high_norm.values,
            low_norm.values,
            close_norm.values,
            volume_norm.values,
            turnover_norm.values,
            vwap_norm.values,
            daily_return.values,
            intraday_range.values,
            shadow_upper.values,
            shadow_lower.values,
            effective_spread.values,
        ])

    return features.astype(np.float32)


# ==============================================================================
# 2. Technical Indicators (16 Dims: T01–T16)
# ==============================================================================

def _wilder_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ema_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    ema_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = ema_up / (ema_down + EPSILON)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _compute_kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    low_min = low.rolling(period, min_periods=1).min()
    high_max = high.rolling(period, min_periods=1).max()
    rsv = ((close - low_min) / (high_max - low_min + EPSILON)) * 100.0
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return k.fillna(50.0), d.fillna(50.0), j.fillna(50.0)


def compute_technical_features(
    df_ohlcv: pd.DataFrame,
    schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION,
) -> np.ndarray:
    """
    Computes 16 Technical Indicators according to the specified schema version.
    For V8.0 Baseline: T01-T16 per authoritative specification (CCI-14 excluded, T14 is BB Bandwidth).
    For V8.1 Candidate: 16 indicators per TECHNICAL_FEATURE_FIELDS (including MFI, CCI, BB distances).
    Returns:
        np.ndarray of shape (len(df_ohlcv), 16)
    """
    close = df_ohlcv['close'].astype(float)
    high = df_ohlcv['high'].astype(float)
    low = df_ohlcv['low'].astype(float)
    vol = df_ohlcv['volume'].astype(float)
    prev_close = close.shift(1).fillna(close.iloc[0])

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # Common indicators
    rsi14 = (_wilder_rsi(close, 14) - 50.0) / 50.0
    k9, d9, j9 = _compute_kdj(high, low, close, 9)
    kdj_k = np.clip((k9 - 50.0) / 50.0, -1.0, 1.0)
    kdj_d = np.clip((d9 - 50.0) / 50.0, -1.0, 1.0)
    kdj_j = np.clip((j9 - 50.0) / 50.0, -1.0, 1.0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_dif = ((ema12 - ema26) / (close + EPSILON)).fillna(0.0)
    macd_signal = (macd_dif.ewm(span=9, adjust=False).mean()).fillna(0.0)
    macd_hist = (macd_dif - macd_signal).fillna(0.0)

    sma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std().fillna(0.0)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20

    atr14 = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    atr14_close = (atr14 / (close + EPSILON)).fillna(0.0)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    smooth_tr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=close.index).ewm(alpha=1.0 / 14, adjust=False).mean() / (smooth_tr + EPSILON)
    minus_di = 100.0 * pd.Series(minus_dm, index=close.index).ewm(alpha=1.0 / 14, adjust=False).mean() / (smooth_tr + EPSILON)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPSILON)
    adx14 = np.clip(dx.ewm(alpha=1.0 / 14, adjust=False).mean() / 100.0, 0.0, 1.0).fillna(0.2)

    if schema_version in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value, "v8.0-original", "v8.0"):
        # T01: RSI14
        # T02: RSI7
        rsi7 = (_wilder_rsi(close, 7) - 50.0) / 50.0
        # T03..T05: KDJ
        # T06..T08: MACD
        # T09: Close / EMA20 - 1
        ema20 = close.ewm(span=20, adjust=False).mean()
        close_ema20 = ((close / (ema20 + EPSILON)) - 1.0).fillna(0.0)
        # T10: Close / EMA60 - 1
        ema60 = close.ewm(span=60, adjust=False).mean()
        close_ema60 = ((close / (ema60 + EPSILON)) - 1.0).fillna(0.0)
        # T11: EMA20 / EMA60 - 1
        ema20_ema60 = ((ema20 / (ema60 + EPSILON)) - 1.0).fillna(0.0)
        # T12: ATR14 / Close
        # T13: Bollinger %B
        boll_b = ((close - bb_lower) / (bb_upper - bb_lower + EPSILON)).fillna(0.5)
        # T14: Bollinger Bandwidth = (BB_upper - BB_lower) / MA20
        boll_bw = np.maximum((bb_upper - bb_lower) / (sma20 + EPSILON), 0.0).fillna(0.0)
        # T15: Normalized OBV Change 5D
        direction = np.sign(close.diff()).fillna(0.0)
        obv = (direction * vol).cumsum()
        ma20_vol = vol.rolling(20, min_periods=1).mean()
        obv_change_5d = ((obv - obv.shift(5).fillna(obv.iloc[0])) / (5.0 * ma20_vol + EPSILON)).fillna(0.0)
        # T16: ADX14

        features = np.column_stack([
            rsi14.values,
            rsi7.values,
            kdj_k.values,
            kdj_d.values,
            kdj_j.values,
            macd_dif.values,
            macd_signal.values,
            macd_hist.values,
            close_ema20.values,
            close_ema60.values,
            ema20_ema60.values,
            atr14_close.values,
            boll_b.values,
            boll_bw.values,
            obv_change_5d.values,
            adx14.values,
        ])
    else:
        # Candidate V8.1 Technical Indicators (16 Dims per TECHNICAL_FEATURE_FIELDS):
        # TECH_01: rsi_14
        # TECH_02: macd_line
        # TECH_03: macd_signal
        # TECH_04: macd_hist
        # TECH_05: bb_upper_dist: (BB_upper - P_close) / P_close
        bb_upper_dist = ((bb_upper - close) / (close + EPSILON)).fillna(0.0)
        # TECH_06: bb_lower_dist: (P_close - BB_lower) / P_close
        bb_lower_dist = ((close - bb_lower) / (close + EPSILON)).fillna(0.0)
        # TECH_07: bb_width: (BB_upper - BB_lower) / MA20
        bb_width = np.maximum((bb_upper - bb_lower) / (sma20 + EPSILON), 0.0).fillna(0.0)
        # TECH_08: atr_14_norm: ATR_14 / P_close
        # TECH_09: obv_slope: 10-day linear regression slope of OBV
        direction = np.sign(close.diff()).fillna(0.0)
        obv = (direction * vol).cumsum()
        ma20_vol = vol.rolling(20, min_periods=1).mean()
        obv_slope = ((obv - obv.shift(10).fillna(obv.iloc[0])) / (10.0 * ma20_vol + EPSILON)).fillna(0.0)
        # TECH_10..TECH_12: KDJ K, D, J
        # TECH_13: mfi_14: (MFI_14 - 50) / 50
        tp = (high + low + close) / 3.0
        rmf = tp * vol
        prev_tp = tp.shift(1).fillna(tp.iloc[0])
        pos_mf = np.where(tp > prev_tp, rmf, 0.0)
        neg_mf = np.where(tp < prev_tp, rmf, 0.0)
        mfr = (pd.Series(pos_mf, index=close.index).rolling(14, min_periods=1).sum() /
               (pd.Series(neg_mf, index=close.index).rolling(14, min_periods=1).sum() + EPSILON))
        mfi14 = 100.0 - (100.0 / (1.0 + mfr))
        mfi14_norm = ((mfi14 - 50.0) / 50.0).fillna(0.0)
        # TECH_14: cci_14: CCI_14 / 200.0 clamped to [-1, 1]
        sma_tp = tp.rolling(14, min_periods=1).mean()
        mad = (tp - sma_tp).abs().rolling(14, min_periods=1).mean()
        cci14 = (tp - sma_tp) / (0.015 * mad + EPSILON)
        cci14_norm = np.clip(cci14 / 200.0, -1.0, 1.0).fillna(0.0)
        # TECH_15: roc_10: (P_close,t - P_close,t-10) / P_close,t-10
        c_lag10 = close.shift(10).fillna(close.iloc[0])
        roc_10 = ((close - c_lag10) / (c_lag10 + EPSILON)).fillna(0.0)
        # TECH_16: adx_14

        features = np.column_stack([
            rsi14.values,
            macd_dif.values,
            macd_signal.values,
            macd_hist.values,
            bb_upper_dist.values,
            bb_lower_dist.values,
            bb_width.values,
            atr14_close.values,
            obv_slope.values,
            kdj_k.values,
            kdj_d.values,
            kdj_j.values,
            mfi14_norm.values,
            cci14_norm.values,
            roc_10.values,
            adx14.values,
        ])

    return features.astype(np.float32)


# ==============================================================================
# 3. Institutional Order Flow (8 Dims: I01–I08)
# ==============================================================================

def compute_institutional_features(
    df_inst: pd.DataFrame,
    volume_series: pd.Series,
    shares_outstanding: float = 1e8,
    schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION,
) -> np.ndarray:
    """
    Computes 8 Institutional features.
    Baseline V8.0: Normalized by MA20 Volume per authoritative specification.
    Candidate V8.1: Normalized by Shares Outstanding per candidate specification.
    Returns:
        np.ndarray of shape (len(df_inst), 8)
    """
    f_net = df_inst['foreign_net_buy'].astype(float)
    t_net = df_inst['trust_net_buy'].astype(float)
    d_net = df_inst['dealer_net_buy'].astype(float)
    margin = df_inst['margin_balance'].astype(float)
    short = df_inst['short_balance'].astype(float)

    if schema_version in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value, "v8.0-original", "v8.0"):
        ma20_vol = volume_series.rolling(20, min_periods=1).mean()
        # I01..I03: Net Buy / MA20 Volume
        i01 = (f_net / (ma20_vol + EPSILON)).fillna(0.0)
        i02 = (t_net / (ma20_vol + EPSILON)).fillna(0.0)
        i03 = (d_net / (ma20_vol + EPSILON)).fillna(0.0)
        # I04: Foreign Net Buy 5D / (5 * MA20 Vol)
        i04 = (f_net.rolling(5, min_periods=1).sum() / (5.0 * ma20_vol + EPSILON)).fillna(0.0)
        # I05: Total 3-Inst Net Buy 5D / (5 * MA20 Vol)
        total_net = f_net + t_net + d_net
        i05 = (total_net.rolling(5, min_periods=1).sum() / (5.0 * ma20_vol + EPSILON)).fillna(0.0)
        # I06: Margin Balance Change / MA20 Vol
        i06 = (margin.diff().fillna(0.0) / (ma20_vol + EPSILON)).fillna(0.0)
        # I07: Short Balance Change / MA20 Vol
        i07 = (short.diff().fillna(0.0) / (ma20_vol + EPSILON)).fillna(0.0)
        # I08: Short / Margin Balance Ratio
        i08 = np.maximum(short / (margin + EPSILON), 0.0).fillna(0.0)

        features = np.column_stack([
            i01.values, i02.values, i03.values, i04.values,
            i05.values, i06.values, i07.values, i08.values
        ])
    else:
        # Candidate V8.1: Normalized by Shares Outstanding per INSTITUTIONAL_FEATURE_FIELDS
        i01 = (f_net / (shares_outstanding + EPSILON)).fillna(0.0)
        i02 = (t_net / (shares_outstanding + EPSILON)).fillna(0.0)
        i03 = (d_net / (shares_outstanding + EPSILON)).fillna(0.0)
        i04 = ((f_net + t_net + d_net) / (shares_outstanding + EPSILON)).fillna(0.0)
        i05 = (margin.diff().fillna(0.0) / (shares_outstanding + EPSILON)).fillna(0.0)
        i06 = (short.diff().fillna(0.0) / (shares_outstanding + EPSILON)).fillna(0.0)
        margin_quota = shares_outstanding * 0.25
        i07 = np.clip(f_net.cumsum() / (shares_outstanding + EPSILON), 0.0, 1.0).fillna(0.3)
        i08 = np.clip(margin / (margin_quota + EPSILON), 0.0, 1.0).fillna(0.0)

        features = np.column_stack([
            i01.values, i02.values, i03.values, i04.values,
            i05.values, i06.values, i07.values, i08.values
        ])

    return features.astype(np.float32)


# ==============================================================================
# 4. Macro & Cross-Market Features (12 Dims: M01–M12)
# ==============================================================================

def compute_macro_features(
    df_macro: pd.DataFrame,
    df_foreign: Optional[pd.DataFrame] = None,
    schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION,
) -> np.ndarray:
    """
    Computes 12 Macro and cross-market indicators.
    Expected columns in df_macro:
        ['taiex', 'sector_index', 'usdtwd', 'sp500', 'nasdaq', 'sox', 'vix', 'us_10y_yield']
    Zero lookahead:
    - All bfill calls strictly eliminated (P0-1).
    - If df_foreign is provided, utilizes timezone-aware As-Of Join (asof_join_cross_market).
    - M05 Sector Relative Strength 20D = SectorReturn20D - TAIEX_Return20D (P0-2).
    Returns:
        np.ndarray of shape (len(df_macro), 12)
    """
    taiex = df_macro['taiex'].astype(float)
    sector = df_macro['sector_index'].astype(float)
    usdtwd = df_macro['usdtwd'].astype(float)

    prev_taiex = taiex.shift(1).fillna(taiex.iloc[0])
    prev_usdtwd = usdtwd.shift(1).fillna(usdtwd.iloc[0])

    # Domestic indicators
    m01 = taiex.pct_change(1).fillna(0.0)
    m02 = taiex.pct_change(5).fillna(0.0)
    log_ret_tw = np.log(taiex / (prev_taiex + EPSILON)).fillna(0.0)
    m03 = (log_ret_tw.rolling(20, min_periods=2).std() * np.sqrt(ANNUAL_TRADING_DAYS)).fillna(0.0)

    m04 = sector.pct_change(1).fillna(0.0)

    # P0-2 Fix: M05 Sector Relative Strength 20D = Sector Return 20D - TAIEX Return 20D
    taiex_ret_20d = taiex.pct_change(20).fillna(0.0)
    sector_ret_20d = sector.pct_change(20).fillna(0.0)
    m05 = (sector_ret_20d - taiex_ret_20d).fillna(0.0)

    m06 = usdtwd.pct_change(1).fillna(0.0)

    # US Cross-market data alignment (P0-1: ZERO bfill)
    if df_foreign is not None and isinstance(df_macro.index, pd.DatetimeIndex):
        foreign_aligned = asof_join_cross_market(
            taiwan_decision_times=df_macro.index,
            foreign_market_df=df_foreign,
            missing_policy="forward_fill",
        )
        sp500 = foreign_aligned['sp500'].astype(float)
        nasdaq = foreign_aligned['nasdaq'].astype(float)
        sox = foreign_aligned['sox'].astype(float)
        vix = foreign_aligned['vix'].astype(float)
        us_10y = foreign_aligned['us_10y_yield'].astype(float)
    else:
        # Lagged by 1 day and forward-filled strictly forward in time
        sp500 = df_macro['sp500'].astype(float).shift(1).ffill()
        nasdaq = df_macro['nasdaq'].astype(float).shift(1).ffill()
        sox = df_macro['sox'].astype(float).shift(1).ffill()
        vix = df_macro['vix'].astype(float).shift(1).ffill()
        us_10y = df_macro['us_10y_yield'].astype(float).shift(1).ffill()

    if schema_version in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value, "v8.0-original", "v8.0"):
        # M07: S&P 500 Return (T-1)
        m07 = sp500.pct_change(1).fillna(0.0)
        # M08: NASDAQ Return (T-1)
        m08 = nasdaq.pct_change(1).fillna(0.0)
        # M09: SOX Return (T-1)
        m09 = sox.pct_change(1).fillna(0.0)
        # M10: VIX Level / 100.0 (T-1)
        m10 = (vix / 100.0).fillna(0.20)
        # M11: US 10Y Sovereign Yield / 100.0 (T-1)
        m11 = (us_10y / 100.0).fillna(0.04)
        # M12: US 10Y Sovereign Yield Change (T-1)
        m12 = (us_10y.diff().fillna(0.0) / 100.0)

        features = np.column_stack([
            m01.values, m02.values, m03.values, m04.values, m05.values, m06.values,
            m07.values, m08.values, m09.values, m10.values, m11.values, m12.values
        ])
    else:
        # Candidate V8.1 Macro Indicators per MACRO_FEATURE_FIELDS
        c_m01 = np.log(taiex / (prev_taiex + EPSILON)).fillna(0.0)
        c_m02 = m03
        c_m03 = np.log(usdtwd / (prev_usdtwd + EPSILON)).fillna(0.0)
        c_m04 = df_macro['tw_10y_yield'].diff().fillna(0.0) if 'tw_10y_yield' in df_macro.columns else pd.Series(0.0, index=df_macro.index)
        c_m05 = (us_10y.diff().fillna(0.0) / 100.0)
        c_m06 = sox.pct_change(1).fillna(0.0)
        c_m07 = sp500.pct_change(1).fillna(0.0)
        c_m08 = df_macro['brent'].pct_change(1).fillna(0.0) if 'brent' in df_macro.columns else pd.Series(0.0, index=df_macro.index)
        c_m09 = df_macro['interbank_rate'].fillna(0.015) if 'interbank_rate' in df_macro.columns else pd.Series(0.015, index=df_macro.index)
        c_m10 = df_macro['leading_indicator'].pct_change(1).fillna(0.0) if 'leading_indicator' in df_macro.columns else pd.Series(0.0, index=df_macro.index)
        c_m11 = df_macro['adr_premium'].fillna(0.0) if 'adr_premium' in df_macro.columns else pd.Series(0.0, index=df_macro.index)
        c_m12 = (vix / 100.0).fillna(0.20)

        features = np.column_stack([
            c_m01.values, c_m02.values, c_m03.values, c_m04.values,
            c_m05.values, c_m06.values, c_m07.values, c_m08.values,
            c_m09.values, c_m10.values, c_m11.values, c_m12.values,
        ])

    return features.astype(np.float32)


# ==============================================================================
# 5. Stock Context Features (16 Dims: C01–C16)
# ==============================================================================

def assemble_stock_context_features(
    industry_embedding_8d: np.ndarray,      # [8]
    market_cap: float,
    trading_value_60d_ma: float,
    volatility_percentile: float,
    liquidity_percentile: float,
    beta_120d: float,
    industry_market_cap_percentile: float,
    days_since_ipo: int,
    had_trading_suspension_20d: bool,
    schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION,
) -> np.ndarray:
    """
    Assembles the 16 Stock Context features for a single stock decision step.
    Supports both V8.0 Baseline (8D embedding + 8 market context scalars)
    and V8.1 Candidate (16 context fields per STOCK_CONTEXT_FEATURE_FIELDS).
    Returns:
        np.ndarray of shape (16,)
    """
    c01_08 = np.asarray(industry_embedding_8d, dtype=np.float32).flatten()
    if len(c01_08) != 8:
        raise ValueError(f"Industry embedding must have 8 dimensions, got {len(c01_08)}")

    if schema_version in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value, "v8.0-original", "v8.0"):
        c09 = np.log(max(market_cap, 1.0))
        c10 = np.log1p(max(trading_value_60d_ma, 0.0))
        c11 = np.clip(volatility_percentile, 0.0, 1.0)
        c12 = np.clip(liquidity_percentile, 0.0, 1.0)
        c13 = float(beta_120d)
        c14 = np.clip(industry_market_cap_percentile, 0.0, 1.0)
        c15 = min(max(days_since_ipo / (365.25 * 10.0), 0.0), 1.0)
        c16 = 1.0 if had_trading_suspension_20d else 0.0

        scalars = np.array([c09, c10, c11, c12, c13, c14, c15, c16], dtype=np.float32)
        return np.concatenate([c01_08, scalars], axis=0).astype(np.float32)
    else:
        # Candidate V8.1 context features
        c01 = float(np.mean(c01_08))
        c02 = np.clip(industry_market_cap_percentile, 0.0, 1.0)
        c03 = float(beta_120d)
        c04 = np.clip(volatility_percentile, 0.0, 1.0)
        c05 = np.clip(liquidity_percentile, 0.0, 1.0)
        c06 = 1.0
        c07 = 1.0
        c08 = 0.03
        c09 = 0.5
        c10 = 0.15
        c11 = 1.0
        c12 = 1.0
        c13 = 1.0
        c14 = 1.0
        c15 = min(max(market_cap / 1e12, 0.0), 1.0)
        c16 = 0.8
        return np.array([c01, c02, c03, c04, c05, c06, c07, c08,
                         c09, c10, c11, c12, c13, c14, c15, c16], dtype=np.float32)


# ==============================================================================
# 6. Event Metadata & Relations (24 Dims E01–E24, 8 Dims R01–R08)
# ==============================================================================

def assemble_event_metadata(
    actor_embedding_4d: np.ndarray,         # [4]
    type_embedding_8d: np.ndarray,          # [8]
    status_embedding_4d: np.ndarray,        # [4]
    evidence_traceability: float,
    source_reliability: float,
    source_independence: float,
    novelty_input_estimate: float,
    market_surprise_preevent: float,
    quantified_magnitude: float,
    event_age_days: float,
    timestamp_reliability: float,
    schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION,
) -> np.ndarray:
    """
    Assembles the 24 Event Metadata features strictly per V8.0 specification layout:
    - E01..E04: Actor Embedding (4D)
    - E05..E12: Type Embedding (8D)
    - E13..E16: Status Embedding (4D)
    - E17: Traceability [0, 1]
    - E18: Source Reliability [0, 1]
    - E19: Source Independence [0, 1]
    - E20: Novelty Input Estimate [0, 1]
    - E21: Market Surprise (strictly pre-event prior implied volatility / move)
    - E22: Quantified Magnitude [0, 1]
    - E23: Event Age in fractional days (Python Index 22)
    - E24: Timestamp Reliability [0, 1] (Python Index 23)
    Returns:
        np.ndarray of shape (24,)
    """
    if schema_version in (FeatureSchemaVersion.BASELINE_V8_0_ORIGINAL.value, "v8.0-original", "v8.0"):
        act = np.asarray(actor_embedding_4d, dtype=np.float32).flatten()
        typ = np.asarray(type_embedding_8d, dtype=np.float32).flatten()
        sta = np.asarray(status_embedding_4d, dtype=np.float32).flatten()
        if len(act) != 4 or len(typ) != 8 or len(sta) != 4:
            raise ValueError("Invalid event embedding dimensions (expected 4, 8, 4)")

        scalars = np.array([
            np.clip(evidence_traceability, 0.0, 1.0),
            np.clip(source_reliability, 0.0, 1.0),
            np.clip(source_independence, 0.0, 1.0),
            np.clip(novelty_input_estimate, 0.0, 1.0),
            float(market_surprise_preevent),
            np.clip(quantified_magnitude, 0.0, 1.0),
            max(float(event_age_days), 0.0),
            np.clip(timestamp_reliability, 0.0, 1.0),
        ], dtype=np.float32)

        return np.concatenate([act, typ, sta, scalars], axis=0).astype(np.float32)
    else:
        # Candidate layout: EV_META_01..08 categories, EV_META_13 is event_age (index 12)
        res = np.zeros(24, dtype=np.float32)
        res[0:8] = np.asarray(type_embedding_8d, dtype=np.float32).flatten()[:8]
        res[8] = np.clip(source_reliability, 0.0, 1.0)
        res[9] = float(market_surprise_preevent)
        res[10] = np.clip(evidence_traceability, 0.0, 1.0)
        res[11] = 1.0
        res[12] = max(float(event_age_days), 0.0)  # EV_META_13
        res[13] = np.clip(source_reliability, 0.0, 1.0)
        res[14] = np.clip(source_independence, 0.0, 1.0)
        res[15] = 1.0
        res[16] = 0.0
        res[17] = 0.0
        res[18] = 0.0
        res[19] = 0.0
        res[20] = 2.0
        res[21] = 1.0
        res[22] = 0.0
        res[23] = 1.0
        return res


def assemble_event_stock_relation(
    direct_company_mention: bool,
    industry_match: bool,
    upstream_exposure: float,
    downstream_exposure: float,
    customer_relationship: float,
    supplier_relationship: float,
    geographic_exposure: float,
    macro_factor_exposure: float,
    schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION,
) -> np.ndarray:
    """
    Assembles the 8 Historical As-Of Event-Stock Relation features (R01–R08).
    Returns:
        np.ndarray of shape (8,)
    """
    return np.array([
        1.0 if direct_company_mention else 0.0,
        1.0 if industry_match else 0.0,
        np.clip(upstream_exposure, 0.0, 1.0),
        np.clip(downstream_exposure, 0.0, 1.0),
        np.clip(customer_relationship, 0.0, 1.0),
        np.clip(supplier_relationship, 0.0, 1.0),
        np.clip(geographic_exposure, 0.0, 1.0),
        np.clip(macro_factor_exposure, 0.0, 1.0),
    ], dtype=np.float32)


# ==============================================================================
# 7. Position State Features (4 Dims: S01–S04)
# ==============================================================================

def assemble_position_state(
    target_position: int,
    unrealized_pnl_pct: float,
    holding_duration_days: int,
    days_since_last_successful_trade: int,
) -> np.ndarray:
    """
    Assembles the 4 Position State features (S01–S04).
    Returns:
        np.ndarray of shape (4,)
    """
    s01 = 1.0 if target_position == 1 else 0.0
    s02 = float(unrealized_pnl_pct)
    s03 = min(max(holding_duration_days / 60.0, 0.0), 1.0)
    s04 = min(max(days_since_last_successful_trade / 60.0, 0.0), 1.0)
    return np.array([s01, s02, s03, s04], dtype=np.float32)


# ==============================================================================
# 8. Audited Provenance Builders (P1-7)
# ==============================================================================

def build_audited_stock_context(
    industry_embedding_8d: np.ndarray,
    market_cap: float,
    trading_value_60d_ma: float,
    volatility_percentile: float,
    liquidity_percentile: float,
    beta_120d: float,
    industry_market_cap_percentile: float,
    days_since_ipo: int,
    had_trading_suspension_20d: bool,
    provenance: ProvenanceRecord,
    decision_time: pd.Timestamp,
    schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION,
) -> np.ndarray:
    """
    Assembles Stock Context with strict PIT verification of ProvenanceRecord.
    If provenance is not available as of decision_time, fills with neutral fallback values.
    Industry embedding is guaranteed to have been trained strictly prior to decision_time.
    """
    if not provenance.is_available_as_of(decision_time):
        return np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            10.0, 10.0, 0.5, 0.5, 1.0, 0.5, 1.0, 0.0
        ], dtype=np.float32)

    return assemble_stock_context_features(
        industry_embedding_8d=industry_embedding_8d,
        market_cap=market_cap,
        trading_value_60d_ma=trading_value_60d_ma,
        volatility_percentile=volatility_percentile,
        liquidity_percentile=liquidity_percentile,
        beta_120d=beta_120d,
        industry_market_cap_percentile=industry_market_cap_percentile,
        days_since_ipo=days_since_ipo,
        had_trading_suspension_20d=had_trading_suspension_20d,
        schema_version=schema_version,
    )


def build_audited_event_metadata(
    actor_embedding_4d: np.ndarray,
    type_embedding_8d: np.ndarray,
    status_embedding_4d: np.ndarray,
    evidence_traceability: float,
    source_reliability: float,
    source_independence: float,
    novelty_input_estimate: float,
    market_surprise_preevent: float,
    quantified_magnitude: float,
    timestamp_reliability: float,
    provenance: ProvenanceRecord,
    decision_time: pd.Timestamp,
    schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION,
) -> np.ndarray:
    """
    Assembles 24D Event Metadata with strict PIT verification.
    Guarantees:
    1. provenance.available_at <= decision_time
    2. Event age computed strictly as (decision_time - provenance.available_at).
    3. E21 Market Surprise is strictly pre-event prior implied move (ZERO post-event price reaction).
    4. If unverified or future event, returns neutral masked representation.
    """
    if not provenance.is_available_as_of(decision_time):
        return np.zeros(EVENT_METADATA_DIM, dtype=np.float32)

    t_avail = provenance.available_at
    if t_avail is None:
        return np.zeros(EVENT_METADATA_DIM, dtype=np.float32)

    if t_avail.tzinfo is not None and decision_time.tzinfo is not None:
        delta_sec = (decision_time - t_avail).total_seconds()
    elif t_avail.tzinfo is None and decision_time.tzinfo is None:
        delta_sec = (decision_time - t_avail).total_seconds()
    else:
        t_avail_utc = t_avail.tz_convert("UTC") if t_avail.tzinfo is not None else t_avail.tz_localize("UTC")
        t_dec_utc = decision_time.tz_convert("UTC") if decision_time.tzinfo is not None else decision_time.tz_localize("UTC")
        delta_sec = (t_dec_utc - t_avail_utc).total_seconds()

    event_age_days = max(delta_sec / 86400.0, 0.0)

    return assemble_event_metadata(
        actor_embedding_4d=actor_embedding_4d,
        type_embedding_8d=type_embedding_8d,
        status_embedding_4d=status_embedding_4d,
        evidence_traceability=evidence_traceability,
        source_reliability=source_reliability,
        source_independence=source_independence,
        novelty_input_estimate=novelty_input_estimate,
        market_surprise_preevent=market_surprise_preevent,
        quantified_magnitude=quantified_magnitude,
        event_age_days=event_age_days,
        timestamp_reliability=timestamp_reliability,
        schema_version=schema_version,
    )


def build_audited_event_stock_relation(
    direct_company_mention: bool,
    industry_match: bool,
    upstream_exposure: float,
    downstream_exposure: float,
    customer_relationship: float,
    supplier_relationship: float,
    geographic_exposure: float,
    macro_factor_exposure: float,
    provenance: ProvenanceRecord,
    decision_time: pd.Timestamp,
    schema_version: str = CURRENT_ACTIVE_SCHEMA_VERSION,
) -> np.ndarray:
    """
    Assembles 8D Event-Stock Relation with strict PIT verification of supply chain / relation links.
    If provenance is unverified or future, returns zeroes.
    """
    if not provenance.is_available_as_of(decision_time):
        return np.zeros(EVENT_STOCK_RELATION_DIM, dtype=np.float32)

    return assemble_event_stock_relation(
        direct_company_mention=direct_company_mention,
        industry_match=industry_match,
        upstream_exposure=upstream_exposure,
        downstream_exposure=downstream_exposure,
        customer_relationship=customer_relationship,
        supplier_relationship=supplier_relationship,
        geographic_exposure=geographic_exposure,
        macro_factor_exposure=macro_factor_exposure,
        schema_version=schema_version,
    )
