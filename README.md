# Event-Driven Stock RL — Point-in-Time Research System V8.3

A point-in-time, event-driven, multimodal deep reinforcement learning research framework for Taiwan Stock Exchange (TWSE) listed common stocks, built in compliance with Specification V8.0 / V8.3.

---

## 1. Schema Activation & Dual-Version Architecture

- **Primary Research Baseline**: `v8.0-original` (`CURRENT_ACTIVE_SCHEMA_VERSION = "v8.0-original"`).
- **Market Encoder Ablation**: `v8.1-candidate` (`"v8.1-candidate"`).
- **Strict Incompatibility Guard**: Datasets, `MultimodalBatch`, `MultimodalReplayBuffer`, and Checkpoints strictly record `schema_version`. Silent cross-schema mixing is strictly forbidden and raises explicit `ValueError` exceptions.
- **Dynamic Event Age Index**:
  - `v8.0-original`: Event Age is `E23` (**Python index 22**), resolved dynamically via `get_event_age_index()`.
  - `v8.1-candidate`: Event Age is `EV_META_13` (**Python index 12**).

---

## 2. Authoritative 100-Feature Schema (`v8.0-original`)

$$\begin{aligned}
\text{Total Dimension} &= \text{Price}(12) + \text{Technical}(16) + \text{Institutional}(8) + \text{Macro}(12) \\
&\quad + \text{Stock Context}(16) + \text{Event Metadata}(24) + \text{Event-Stock Relation}(8) + \text{Position State}(4) \\
&= 100 \text{ Dimensions}
\end{aligned}$$

1. **Price Dynamics (12D, `P01`–`P12`)**: `return_1d`, `return_5d`, `return_20d`, `overnight_gap`, `open_close_return`, `high_low_range`, `close_location` ($[0, 1]$), `log_volume`, `volume_ratio_20d`, `turnover_ratio`, `realized_volatility_20d`, `log_trading_value`.
2. **Technical Indicators (16D, `T01`–`T16`)**: `rsi_14`, `rsi_7`, `kdj_k`, `kdj_d`, `kdj_j`, `macd_dif`, `macd_signal`, `macd_hist`, `close_ema20_ratio`, `close_ema60_ratio`, `ema20_ema60_ratio`, `atr14_close_ratio`, `bollinger_percent_b`, `bollinger_bandwidth` ($\ge 0$), `normalized_obv_change_5d`, `adx_14` ($[0, 1]$). *(CCI-14 is strictly excluded from Original V8; T14 is Bollinger Bandwidth)*.
3. **Institutional Order Flow (8D, `I01`–`I08`)**: `foreign_net_buy_over_vol20d`, `trust_net_buy_over_vol20d`, `dealer_net_buy_over_vol20d`, `foreign_net_buy_5d`, `total_institutional_net_buy_5d`, `margin_balance_change`, `short_balance_change`, `short_margin_balance_ratio`. *(I01–I03 denominator is strictly MA20 Volume)*.
4. **Macro & Cross-Market (12D, `M01`–`M12`)**: `taiex_return_1d`, `taiex_return_5d`, `taiex_volatility_20d`, `sector_return_1d`, `sector_relative_strength_20d`, `usdtwd_return`, `sp500_return` (T-1), `nasdaq_return` (T-1), `sox_return` (T-1), `vix_level` (T-1), `us_10y_yield` (T-1), `us_10y_yield_change` (T-1). *(All US cross-market features strictly lagged T-1 for availability at Taiwan 20:00)*.
5. **Stock Context (16D, `C01`–`C16`)**: `industry_embedding_01`..`08` (dense 8D versioned embedding), `log_market_capitalization`, `log_avg_trading_value_60d`, `volatility_percentile`, `liquidity_percentile`, `beta_120d`, `market_cap_percentile_within_industry`, `listing_age`, `recent_trading_suspension_indicator`.
6. **Event Metadata (24D, `E01`–`E24`)**: `actor_role_embedding_01`..`04` (4D), `event_type_embedding_01`..`08` (8D), `event_status_embedding_01`..`04` (4D), `evidence_traceability`, `source_reliability_estimate`, `source_independence`, `novelty_input_estimate`, `market_surprise` (strictly pre-event implied move vs prior historical vol; **zero post-event price reaction**), `quantified_magnitude`, `event_age` (E23 at Python index 22), `timestamp_reliability` (E24 at Python index 23).
7. **Event–Stock Relation (8D, `R01`–`R08`)**: `direct_company_mention`, `industry_match`, `upstream_exposure`, `downstream_exposure`, `customer_relationship`, `supplier_relationship`, `geographic_exposure`, `macro_factor_exposure` (Historical As-Of).
8. **Position State (4D, `S01`–`S04`)**: `current_target_position`, `unrealized_pnl_pct`, `holding_duration_days`, `time_since_last_successful_trade`.

---

## 3. System Architecture

```
stock_rl/
├── config/
│   ├── schema.py           # Dual-version schema, authoritative 100-feature catalog, dynamic indices
│   └── settings.py         # Trading costs (TWSE fees/taxes), hyperparameters, dates
├── data/
│   ├── feature_engine.py   # Historical 100-feature engineering & implementation contract
│   ├── normalizer.py       # PIT FeatureNormalizer (fit on train only, frozen for val/test)
│   ├── pit_engine.py       # Point-in-time guard, multi-timestamp tracking, boundary purging
│   ├── universe.py         # Historical dynamic 100-stock selection (survivorship-free)
│   ├── synthetic_data.py   # Dual-version multimodal batch generator with causal event structure
│   └── tensor_validator.py # Strict tensor dimensions, NaN/Inf, mask, and range contract validator
├── models/
│   ├── market_encoder.py   # GRU encoders for Price, Technical, Institutional, and Macro
│   ├── event_module.py     # Event Projection (160->256->128), Age encoding, NRE token, Temporal Attention
│   ├── cross_modal.py      # Cross-Modal Attention fusing market tokens & event token into 128-d
│   ├── multi_task_heads.py # Relevance, Novelty, 3-Horizon Return & Risk heads + Uncertainty Loss
│   ├── dueling_dqn.py      # 132-d RL state, Dueling Double DQN (shared backbone, value & advantage streams)
│   └── full_model.py       # End-to-end unified model coordinating encoders, heads, and DQN
├── env/
│   ├── execution.py        # TWSE simulation (open execution, commission, 0.3% tax, slippage, +-10% limits)
│   └── stock_trading_env.py# Point-in-Time Gymnasium trading environment (20:00 decision cycle, equity log return)
├── rl/
│   ├── replay_buffer.py    # Experience replay buffer with schema version guard & tensor collation
│   └── agent.py            # Dueling Double DQN agent with consistent target networks & schema checkpoints
├── training/
│   ├── stage1_pretrain.py  # Stage 1: Multi-task representation pretraining (schema-aware)
│   ├── stage2_rl.py        # Stage 2: Frozen-encoder Dueling Double DQN training (schema-aware)
│   └── stage3_joint.py     # Stage 3: End-to-end joint fine-tuning with momentum migration (schema-aware)
└── evaluation/
    ├── metrics.py          # Sharpe, Sortino, Calmar, Max Drawdown, Turnover, cost-adjusted returns
    ├── walk_forward.py     # Purged boundary walk-forward validator with schema tagging
    └── ablations.py        # Baseline vs Candidate ablation framework & comparative study
```

---

## 4. Running Tests & Verification

```bash
# Run the complete test suite
pytest tests/ -v

# Run the V8.3 Schema Migration test suite
pytest tests/test_v8_3_schema_migration.py -v
```

## Reproduce the 2024 Q1 market-only study

From the repository root, install the dependencies in `requirements.txt`, then run:

```bash
python -m stock_rl.evaluation.market_baseline > market_baseline.json
```

The JSON includes input digests, evaluation sessions, target actions, execution costs and
equity curves. See the [research handoff](docs/reports/v8_20_project_closeout.md) for
the observed results, test setup and historical evidence requirements.
