# Authoritative V8.0 100-Feature Schema Specification

**Document Version**: V8.2-Final  
**Target Git Branch**: `fix/v8-audit-patch`  
**Repository Location**: `docs/specifications/v8_authoritative_100_feature_schema.md`  
**Baseline Identifier**: `v8.0-original`  
**Ablation Identifier**: `v8.1-candidate`  

---

## 1. Overview & Architectural Principles

This document defines the formal, immutable 100-dimensional feature specification for the Event-Driven Stock Deep Reinforcement Learning System (`v8.0-original`).

### Core Rules:
1. **Point-in-Time (PIT) Strictness**: All features computed at decision time $t$ (daily at 20:00 Taiwan Time) must strictly depend on data known and available at or before $t$. Cross-market US features (S&P 500, NASDAQ, SOX, US 10Y, VIX) are lagged to $T-1$ available prior to 20:00.
2. **Temporal Isolation of Embeddings**: Industry embeddings (`C01`–`C08`) and Knowledge Graph relations (`R01`–`R08`) must be strictly Historical As-Of, preventing future corporate actions or graph links from leaking into historical decision states.
3. **No Lookahead in Event Features**: Event Metadata `E21` (`market_surprise`) measures pre-event implied volatility/move against historical distribution; it **strictly never** incorporates post-event price reaction.
4. **Dynamic Event Age Indexing**: In `v8.0-original`, Event Age is `E23` (**Python index 22**), resolved dynamically via `get_event_age_index()` and `get_feature_index()`. In Candidate `v8.1-candidate`, Event Age is `EV_META_13` (**Python index 12**).

---

## 2. Authoritative 100-Feature Catalog (Original V8 Baseline)

### Module 1: Price Dynamics (12 Dims, `PRC_01`–`PRC_12` / `P01`–`P12`)

| ID | Shorthand | Feature Name | Mathematical Definition | Unit | Source | Availability | Missing Policy |
|---|---|---|---|---|---|---|---|
| `PRC_01` | `P01` | `return_1d` | $(P_{close,t} - P_{close,t-1}) / P_{close,t-1}$ | `ratio` | TWSE | Daily 20:00 | Zero-fill |
| `PRC_02` | `P02` | `return_5d` | $(P_{close,t} - P_{close,t-5}) / P_{close,t-5}$ | `ratio` | TWSE | Daily 20:00 | Zero-fill |
| `PRC_03` | `P03` | `return_20d` | $(P_{close,t} - P_{close,t-20}) / P_{close,t-20}$ | `ratio` | TWSE | Daily 20:00 | Zero-fill |
| `PRC_04` | `P04` | `overnight_gap` | $(P_{open,t} - P_{close,t-1}) / P_{close,t-1}$ | `ratio` | TWSE | Daily 20:00 | Zero-fill |
| `PRC_05` | `P05` | `open_close_return` | $(P_{close,t} - P_{open,t}) / P_{open,t}$ | `ratio` | TWSE | Daily 20:00 | Zero-fill |
| `PRC_06` | `P06` | `high_low_range` | $(P_{high,t} - P_{low,t}) / P_{open,t}$ | `ratio` | TWSE | Daily 20:00 | Zero-fill |
| `PRC_07` | `P07` | `close_location` | $(P_{close,t} - P_{low,t}) / (P_{high,t} - P_{low,t} + \epsilon)$ | `ratio_0_1` $[0, 1]$ | TWSE | Daily 20:00 | Zero-fill |
| `PRC_08` | `P08` | `log_volume` | $\ln(1 + \text{Volume}_t)$ | `dimensionless` | TWSE | Daily 20:00 | Zero-fill |
| `PRC_09` | `P09` | `volume_ratio_20d` | $\text{Volume}_t / MA_{20}(\text{Volume})$ | `ratio` | TWSE | Daily 20:00 | Neutral (1.0) |
| `PRC_10` | `P10` | `turnover_ratio` | $\text{Volume}_t / \text{Shares Outstanding}$ | `ratio` | TWSE | Daily 20:00 | Zero-fill |
| `PRC_11` | `P11` | `realized_volatility_20d` | $\sqrt{\frac{250}{19}\sum_{k=0}^{19}(r_{t-k}-\bar{r})^2}$ | `annualized_ratio` | TWSE | Daily 20:00 | Sector Median |
| `PRC_12` | `P12` | `log_trading_value` | $\ln(1 + \text{Trading Value}_t)$ | `dimensionless` | TWSE | Daily 20:00 | Zero-fill |

---

### Module 2: Technical Indicators (16 Dims, `TECH_01`–`TECH_16` / `T01`–`T16`)

*(CCI-14 is strictly excluded from Original V8; T14 is Bollinger Bandwidth)*

| ID | Shorthand | Feature Name | Mathematical Definition | Unit | Source | Availability | Missing Policy |
|---|---|---|---|---|---|---|---|
| `TECH_01` | `T01` | `rsi_14` | $(RSI_{14} - 50) / 50$ | `bounded_ratio` $[-1, 1]$ | Technical Engine | Daily 20:00 | Neutral (0.0) |
| `TECH_02` | `T02` | `rsi_7` | $(RSI_7 - 50) / 50$ | `bounded_ratio` $[-1, 1]$ | Technical Engine | Daily 20:00 | Neutral (0.0) |
| `TECH_03` | `T03` | `kdj_k` | $(K_9 - 50) / 50$ | `bounded_ratio` $[-1, 1]$ | Technical Engine | Daily 20:00 | Neutral (0.0) |
| `TECH_04` | `T04` | `kdj_d` | $(D_9 - 50) / 50$ | `bounded_ratio` $[-1, 1]$ | Technical Engine | Daily 20:00 | Neutral (0.0) |
| `TECH_05` | `T05` | `kdj_j` | $(3K - 2D - 50) / 50$ | `bounded_ratio` $[-1, 1]$ | Technical Engine | Daily 20:00 | Neutral (0.0) |
| `TECH_06` | `T06` | `macd_dif` | $(EMA_{12} - EMA_{26}) / P_{close}$ | `ratio` | Technical Engine | Daily 20:00 | Zero-fill |
| `TECH_07` | `T07` | `macd_signal` | $EMA_9(\text{MACD\_DIF}) / P_{close}$ | `ratio` | Technical Engine | Daily 20:00 | Zero-fill |
| `TECH_08` | `T08` | `macd_hist` | $(\text{MACD\_DIF} - \text{MACD\_Signal}) / P_{close}$ | `ratio` | Technical Engine | Daily 20:00 | Zero-fill |
| `TECH_09` | `T09` | `close_ema20_ratio` | $P_{close} / EMA_{20} - 1$ | `ratio` | Technical Engine | Daily 20:00 | Zero-fill |
| `TECH_10` | `T10` | `close_ema60_ratio` | $P_{close} / EMA_{60} - 1$ | `ratio` | Technical Engine | Daily 20:00 | Zero-fill |
| `TECH_11` | `T11` | `ema20_ema60_ratio` | $EMA_{20} / EMA_{60} - 1$ | `ratio` | Technical Engine | Daily 20:00 | Zero-fill |
| `TECH_12` | `T12` | `atr14_close_ratio` | $ATR_{14} / P_{close}$ | `ratio` | Technical Engine | Daily 20:00 | Median |
| `TECH_13` | `T13` | `bollinger_percent_b` | $(P_{close} - BB_{lower}) / (BB_{upper} - BB_{lower} + \epsilon)$ | `ratio` | Technical Engine | Daily 20:00 | Neutral (0.5) |
| `TECH_14` | `T14` | `bollinger_bandwidth` | $(BB_{upper} - BB_{lower}) / MA_{20}$ | `ratio` | Technical Engine | Daily 20:00 | Median |
| `TECH_15` | `T15` | `normalized_obv_change_5d` | $(OBV_t - OBV_{t-5}) / (5 \cdot MA_{20}(\text{Volume}))$ | `dimensionless` | Technical Engine | Daily 20:00 | Zero-fill |
| `TECH_16` | `T16` | `adx_14` | $ADX_{14} / 100.0$ | `ratio` $[0, 1]$ | Technical Engine | Daily 20:00 | Neutral (0.2) |

---

### Module 3: Institutional Order Flow (8 Dims, `INST_01`–`INST_08` / `I01`–`I08`)

*(I01–I03 denominator is strictly MA20 Volume)*

| ID | Shorthand | Feature Name | Mathematical Definition | Unit | Source | Availability | Missing Policy |
|---|---|---|---|---|---|---|---|
| `INST_01` | `I01` | `foreign_net_buy_over_vol20d` | $\text{ForeignNetBuyShares}_t / MA_{20}(\text{Volume})$ | `ratio` | TWSE | Daily 18:00 (<=20:00) | Zero-fill |
| `INST_02` | `I02` | `trust_net_buy_over_vol20d` | $\text{TrustNetBuyShares}_t / MA_{20}(\text{Volume})$ | `ratio` | TWSE | Daily 18:00 (<=20:00) | Zero-fill |
| `INST_03` | `I03` | `dealer_net_buy_over_vol20d` | $\text{DealerNetBuyShares}_t / MA_{20}(\text{Volume})$ | `ratio` | TWSE | Daily 18:00 (<=20:00) | Zero-fill |
| `INST_04` | `I04` | `foreign_net_buy_5d` | $\sum_{k=0}^4 \text{ForeignNetBuy}_{t-k} / (5 \cdot MA_{20}(\text{Volume}))$ | `ratio` | TWSE | Daily 18:00 (<=20:00) | Zero-fill |
| `INST_05` | `I05` | `total_institutional_net_buy_5d` | $\sum_{k=0}^4 \text{TotalInstNetBuy}_{t-k} / (5 \cdot MA_{20}(\text{Volume}))$ | `ratio` | TWSE | Daily 18:00 (<=20:00) | Zero-fill |
| `INST_06` | `I06` | `margin_balance_change` | $(\text{MarginBalance}_t - \text{MarginBalance}_{t-1}) / MA_{20}(\text{Volume})$ | `ratio` | TWSE | Daily 19:30 (<=20:00) | Zero-fill |
| `INST_07` | `I07` | `short_balance_change` | $(\text{ShortBalance}_t - \text{ShortBalance}_{t-1}) / MA_{20}(\text{Volume})$ | `ratio` | TWSE | Daily 19:30 (<=20:00) | Zero-fill |
| `INST_08` | `I08` | `short_margin_balance_ratio` | $\text{ShortBalance}_t / (\text{MarginBalance}_t + \epsilon)$ | `ratio` | TWSE | Daily 19:30 (<=20:00) | Forward-fill |

---

### Module 4: Macro & Market Environment (12 Dims, `MACRO_01`–`MACRO_12` / `M01`–`M12`)

| ID | Shorthand | Feature Name | Mathematical Definition | Unit | Source | Availability | Missing Policy |
|---|---|---|---|---|---|---|---|
| `MACRO_01` | `M01` | `taiex_return_1d` | $(TAIEX_t - TAIEX_{t-1}) / TAIEX_{t-1}$ | `ratio` | TWSE | Daily 14:00 (<=20:00) | Zero-fill |
| `MACRO_02` | `M02` | `taiex_return_5d` | $(TAIEX_t - TAIEX_{t-5}) / TAIEX_{t-5}$ | `ratio` | TWSE | Daily 14:00 (<=20:00) | Zero-fill |
| `MACRO_03` | `M03` | `taiex_volatility_20d` | 20-day annualized realized vol of TAIEX | `annualized_ratio` | TWSE | Daily 14:00 (<=20:00) | Median |
| `MACRO_04` | `M04` | `sector_return_1d` | $(Sector_t - Sector_{t-1}) / Sector_{t-1}$ | `ratio` | TWSE Industry Index | Daily 14:00 (<=20:00) | Zero-fill |
| `MACRO_05` | `M05` | `sector_relative_strength_20d` | $SectorReturn_{20D} - TAIEX\_Return_{20D}$ | `ratio` | TWSE Industry Index | Daily 14:00 (<=20:00) | Zero-fill |
| `MACRO_06` | `M06` | `usdtwd_return` | $(USDTWD_t - USDTWD_{t-1}) / USDTWD_{t-1}$ | `ratio` | Taipei Forex | Daily 16:30 (<=20:00) | Zero-fill |
| `MACRO_07` | `M07` | `sp500_return` | S&P 500 daily return (T-1 PIT) | `ratio` | Cboe / S&P | T-1 Available <= 20:00 | Forward-fill |
| `MACRO_08` | `M08` | `nasdaq_return` | NASDAQ Composite daily return (T-1 PIT) | `ratio` | NASDAQ | T-1 Available <= 20:00 | Forward-fill |
| `MACRO_09` | `M09` | `sox_return` | SOX Semiconductor daily return (T-1 PIT) | `ratio` | PHLX | T-1 Available <= 20:00 | Forward-fill |
| `MACRO_10` | `M10` | `vix_level` | Cboe VIX index level / 100.0 (T-1 PIT) | `ratio` | Cboe | T-1 Available <= 20:00 | Forward-fill |
| `MACRO_11` | `M11` | `us_10y_yield` | US 10Y sovereign yield / 100.0 (T-1 PIT) | `ratio` | Federal Reserve / US | T-1 Available <= 20:00 | Forward-fill |
| `MACRO_12` | `M12` | `us_10y_yield_change` | Delta US 10Y yield bps / 100.0 (T-1 PIT) | `bps` | Federal Reserve / US | T-1 Available <= 20:00 | Zero-fill |

---

### Module 5: Stock Context & Cross-Sectional Factors (16 Dims, `CTX_01`–`CTX_16` / `C01`–`C16`)

| ID | Shorthand | Feature Name | Mathematical Definition | Unit | Source | Availability | Missing Policy |
|---|---|---|---|---|---|---|---|
| `CTX_01` | `C01` | `industry_embedding_01` | Industry Embedding dim 1 (Historical As-Of versioned) | `embedding` | Industry Representation | Annual PIT | 0.0 |
| `CTX_02` | `C02` | `industry_embedding_02` | Industry Embedding dim 2 (Historical As-Of versioned) | `embedding` | Industry Representation | Annual PIT | 0.0 |
| `CTX_03` | `C03` | `industry_embedding_03` | Industry Embedding dim 3 (Historical As-Of versioned) | `embedding` | Industry Representation | Annual PIT | 0.0 |
| `CTX_04` | `C04` | `industry_embedding_04` | Industry Embedding dim 4 (Historical As-Of versioned) | `embedding` | Industry Representation | Annual PIT | 0.0 |
| `CTX_05` | `C05` | `industry_embedding_05` | Industry Embedding dim 5 (Historical As-Of versioned) | `embedding` | Industry Representation | Annual PIT | 0.0 |
| `CTX_06` | `C06` | `industry_embedding_06` | Industry Embedding dim 6 (Historical As-Of versioned) | `embedding` | Industry Representation | Annual PIT | 0.0 |
| `CTX_07` | `C07` | `industry_embedding_07` | Industry Embedding dim 7 (Historical As-Of versioned) | `embedding` | Industry Representation | Annual PIT | 0.0 |
| `CTX_08` | `C08` | `industry_embedding_08` | Industry Embedding dim 8 (Historical As-Of versioned) | `embedding` | Industry Representation | Annual PIT | 0.0 |
| `CTX_09` | `C09` | `log_market_capitalization` | $\ln(\text{Market Capitalization}_t)$ | `dimensionless` | TWSE | Monthly Rebalance | Forward-fill |
| `CTX_10` | `C10` | `log_avg_trading_value_60d` | $\ln(1 + MA_{60}(\text{Trading Value}))$ | `dimensionless` | Market Data Engine | Daily 20:00 | Zero-fill |
| `CTX_11` | `C11` | `volatility_percentile` | Cross-sectional rank of 60D volatility $[0, 1]$ | `rank` | Market Data Engine | Daily 20:00 | Median (0.5) |
| `CTX_12` | `C12` | `liquidity_percentile` | Cross-sectional rank of 60D turnover $[0, 1]$ | `rank` | Market Data Engine | Daily 20:00 | Median (0.5) |
| `CTX_13` | `C13` | `beta_120d` | 120-day rolling beta against TAIEX | `ratio` | Market Data Engine | Daily 20:00 | Market Neutral (1.0) |
| `CTX_14` | `C14` | `market_cap_percentile_within_industry` | Market cap percentile rank within industry $[0, 1]$ | `rank` | TWSE | Monthly Rebalance | Median (0.5) |
| `CTX_15` | `C15` | `listing_age` | $\min(\text{days\_since\_ipo} / (365.25 \cdot 10), 1.0)$ | `normalized_duration` | TWSE | Static / Daily 20:00 | 1.0 |
| `CTX_16` | `C16` | `recent_trading_suspension_indicator` | 1.0 if trading halt in prior 20D, else 0.0 | `binary` | TWSE | Daily 20:00 | 0.0 |

---

### Module 6: Event Metadata (24 Dims, `EV_META_01`–`EV_META_24` / `E01`–`E24`)

*(Authoritative Layout: 4D Actor + 8D Type + 4D Status + 8 Scalars; E21 Market Surprise has NO post-event reaction)*

| ID | Shorthand | Feature Name | Description / Formula | Unit | Source | Availability | Missing Policy |
|---|---|---|---|---|---|---|---|
| `EV_META_01` | `E01` | `actor_role_embedding_01` | Actor / Institution Role Embedding dim 1 | `embedding` | LLM Entity Resolver | Event Release | 0.0 |
| `EV_META_02` | `E02` | `actor_role_embedding_02` | Actor / Institution Role Embedding dim 2 | `embedding` | LLM Entity Resolver | Event Release | 0.0 |
| `EV_META_03` | `E03` | `actor_role_embedding_03` | Actor / Institution Role Embedding dim 3 | `embedding` | LLM Entity Resolver | Event Release | 0.0 |
| `EV_META_04` | `E04` | `actor_role_embedding_04` | Actor / Institution Role Embedding dim 4 | `embedding` | LLM Entity Resolver | Event Release | 0.0 |
| `EV_META_05` | `E05` | `event_type_embedding_01` | Event Type Embedding dim 1 | `embedding` | LLM Event Classifier | Event Release | 0.0 |
| `EV_META_06` | `E06` | `event_type_embedding_02` | Event Type Embedding dim 2 | `embedding` | LLM Event Classifier | Event Release | 0.0 |
| `EV_META_07` | `E07` | `event_type_embedding_03` | Event Type Embedding dim 3 | `embedding` | LLM Event Classifier | Event Release | 0.0 |
| `EV_META_08` | `E08` | `event_type_embedding_04` | Event Type Embedding dim 4 | `embedding` | LLM Event Classifier | Event Release | 0.0 |
| `EV_META_09` | `E09` | `event_type_embedding_05` | Event Type Embedding dim 5 | `embedding` | LLM Event Classifier | Event Release | 0.0 |
| `EV_META_10` | `E10` | `event_type_embedding_06` | Event Type Embedding dim 6 | `embedding` | LLM Event Classifier | Event Release | 0.0 |
| `EV_META_11` | `E11` | `event_type_embedding_07` | Event Type Embedding dim 7 | `embedding` | LLM Event Classifier | Event Release | 0.0 |
| `EV_META_12` | `E12` | `event_type_embedding_08` | Event Type Embedding dim 8 | `embedding` | LLM Event Classifier | Event Release | 0.0 |
| `EV_META_13` | `E13` | `event_status_embedding_01` | Event Status Embedding dim 1 | `embedding` | LLM Status Parser | Event Release | 0.0 |
| `EV_META_14` | `E14` | `event_status_embedding_02` | Event Status Embedding dim 2 | `embedding` | LLM Status Parser | Event Release | 0.0 |
| `EV_META_15` | `E15` | `event_status_embedding_03` | Event Status Embedding dim 3 | `embedding` | LLM Status Parser | Event Release | 0.0 |
| `EV_META_16` | `E16` | `event_status_embedding_04` | Event Status Embedding dim 4 | `embedding` | LLM Status Parser | Event Release | 0.0 |
| `EV_META_17` | `E17` | `evidence_traceability` | Evidence Traceability score $[0, 1]$ | `score` | LLM Extraction | Event Release | 1.0 |
| `EV_META_18` | `E18` | `source_reliability_estimate` | Source Reliability Estimate $[0, 1]$ | `score` | Metadata Registry | Event Release | 0.5 |
| `EV_META_19` | `E19` | `source_independence` | Source Independence score $[0, 1]$ | `score` | Metadata Registry | Event Release | 1.0 |
| `EV_META_20` | `E20` | `novelty_input_estimate` | Novelty Input Estimate prior score $[0, 1]$ | `score` | Event Memory Engine | Event Release | 0.5 |
| `EV_META_21` | `E21` | `market_surprise` | Pre-event implied move vs prior historical volatility (strictly prior to release, NO post-event reaction) | `score` | Market Shock Engine | Event Release | 0.0 |
| `EV_META_22` | `E22` | `quantified_magnitude` | Quantified impact magnitude $[0, 1]$ | `score` | LLM Extraction | Event Release | 0.5 |
| `EV_META_23` | `E23` | `event_age` | $(T_{decision} - T_{available})$ in fractional days (**Python Index 22**) | `days` | PIT Engine | Event Release | 0.0 |
| `EV_META_24` | `E24` | `timestamp_reliability` | Timestamp Reliability score $[0, 1]$ (**Python Index 23**) | `score` | PIT Engine | Event Release | 1.0 |

---

### Module 7: Event–Stock Relation (8 Dims, `EV_REL_01`–`EV_REL_08` / `R01`–`R08`)

*(All links and supply chain exposures must be Historical As-Of)*

| ID | Shorthand | Feature Name | Mathematical Definition | Unit | Source | Availability | Missing Policy |
|---|---|---|---|---|---|---|---|
| `EV_REL_01` | `R01` | `direct_company_mention` | 1.0 if stock symbol/name explicitly mentioned, else 0.0 | `binary` | Entity Resolver | Event Release | 0.0 |
| `EV_REL_02` | `R02` | `industry_match` | 1.0 if entity shares primary sector, else 0.0 | `binary` | Industry KG | Event Release | 0.0 |
| `EV_REL_03` | `R03` | `upstream_exposure` | Upstream supply chain exposure weight (Historical As-Of) $[0, 1]$ | `score` | Supply Chain DB | Event Release | 0.0 |
| `EV_REL_04` | `R04` | `downstream_exposure` | Downstream supply chain exposure weight (Historical As-Of) $[0, 1]$ | `score` | Supply Chain DB | Event Release | 0.0 |
| `EV_REL_05` | `R05` | `customer_relationship` | Customer relationship weight (Historical As-Of) $[0, 1]$ | `score` | Supply Chain DB | Event Release | 0.0 |
| `EV_REL_06` | `R06` | `supplier_relationship` | Supplier relationship weight (Historical As-Of) $[0, 1]$ | `score` | Supply Chain DB | Event Release | 0.0 |
| `EV_REL_07` | `R07` | `geographic_exposure` | Geographic / Regional operational exposure weight $[0, 1]$ | `score` | Knowledge Graph | Event Release | 0.0 |
| `EV_REL_08` | `R08` | `macro_factor_exposure` | Macro economic factor sensitivity $[0, 1]$ | `score` | Factor Model | Event Release | 0.0 |

---

### Module 8: Agent Position State (4 Dims, `POS_01`–`POS_04` / `S01`–`S04`)

| ID | Shorthand | Feature Name | Mathematical Definition | Unit | Source | Availability | Missing Policy |
|---|---|---|---|---|---|---|---|
| `POS_01` | `S01` | `current_target_position` | Target position state: 0.0 for Cash, 1.0 for Long | `state` | Gym Environment | Decision 20:00 | 0.0 |
| `POS_02` | `S02` | `unrealized_pnl_pct` | $(P_{close,t} - P_{entry}) / P_{entry}$ | `ratio` | Gym Environment | Decision 20:00 | 0.0 |
| `POS_03` | `S03` | `holding_duration_days` | $\min(\text{days\_in\_position} / 60.0, 1.0)$ | `normalized_duration` | Gym Environment | Decision 20:00 | 0.0 |
| `POS_04` | `S04` | `time_since_last_successful_trade` | $\min(\text{days\_since\_last\_successful\_trade} / 60.0, 1.0)$ | `normalized_duration` | Gym Environment | Decision 20:00 | 1.0 |

---

## 3. Summary Dimension Audit

$$\begin{aligned}
\text{Total Feature Dimension} &= \text{Price}(12) + \text{Technical}(16) + \text{Institutional}(8) + \text{Macro}(12) \\
&\quad + \text{Stock Context}(16) + \text{Event Metadata}(24) + \text{Event-Stock Relation}(8) + \text{Position State}(4) \\
&= 100 \text{ Dimensions}
\end{aligned}$$
