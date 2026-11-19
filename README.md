# 台股 News-Aware 深度強化學習研究

目前的研究問題是：在相同價量資訊、交易成本與執行規則下，加入法人買賣超，會如何改變模型的持股決策與淨報酬？V8.40 已完成三檔台股的四月留出期測試。新聞分支另保留公告查證、事件時間與文字對照實驗。

## 研究進度與資料索引（截至 V8.40）

目前可執行三檔台股的歷史市場與籌碼實驗。每日台北時間 20:00 決策，下一交易日開盤執行，各股獨立選擇持股或現金。

| 資料或實驗 | 現有範圍與用途 | 檔案與結果 |
|---|---|---|
| 市場與籌碼 | 2330、2603、2881；2023-06 至 2024-03 共 202 個交易日。含價量、法人、融資融券、產業與總體資料 | [市場資料](data/poc_real_data/)、[法人與融資融券消融實驗](docs/reports/v8_39_flow_margin_ablation.md) |
| 2024-04 留出期 | 20 個交易日、19 個已實現報酬日。4 月 1 日首次決策，4 月 2–30 日評估凍結模型 | [實驗報告與重現指令](docs/reports/v8_40_april_holdout.md)、[逐股動作與權益](docs/results/v8_40_april_holdout.json)、[22 份來源回應的網址與雜湊](docs/evidence/april_market_sources.json) |
| 均線與籌碼交互門檻 | 以決策日收盤與 20 日均價分組，分析 52 交易日（2–4 月）之持股與超額報酬 | [交互門檻實驗報告](docs/reports/v8_41_trend_institutional_interaction.md)、[分組結果](docs/results/v8_41_trend_interaction.json) |
| 官方公告 | 2024 Q1 三檔各一筆 MOPS 樣本，可核對申報時間與來源；台積電熊本案另有 SEC 申報證據。同一事件跨來源核對時須去重 | [MOPS 證據](docs/evidence/mops_three_stock_q1_sample.json)、[SEC 證據](data/official_disclosures/sec_2330_20240206_evidence.json) |
| 探索性新聞 | 16 筆候選事件，訓練／驗證／測試為 10／3／3。文字實驗使用事後撰寫的摘要，替換成相同句子後，三個種子的測試動作皆相同 | [文字與時間敏感度](docs/reports/v8_37_text_sensitivity.md) |

**重現範圍。** 四月的 GitHub 紀錄提供來源雜湊、模型雜湊、逐股結果與重跑指令。重跑需要取得相符的原始回應，並備妥或依報告重新訓練模型權重。公告申報時間與資料商首次可得時間屬於不同欄位；歷史供應商可得時間憑證仍為 0，正式新聞 PIT 回測尚待補足。

**最新探索性分析。** [均線與籌碼交互門檻分析](docs/reports/v8_41_trend_institutional_interaction.md)：用決策日收盤與過去 20 個交易日均價（含當日）分組，檢驗 52 個評估交易日（二至四月）。結果顯示在「股價站在 20 日均線之上」時，加入法人買賣超在三個種子皆帶來穩定正超額報酬（平均 +4.89 pp）；而在「跌破 20 日均線」時，法人籌碼容易滯後殺跌而拖累報酬（平均 −2.91 pp）。長榮在站上均線後由法人籌碼鎖定主升段，是超額報酬的關鍵來源。

---

## 重現四月籌碼實驗

從 repository 根目錄執行。前兩次訓練分別產生對照組與法人買賣超組的模型權重；若已保存相符權重，可直接執行最後一行。四月資料若未快取，需要網路連線取得原始回應。

```bash
python -m pip install -r requirements.txt
python -m stock_rl.evaluation.market_drl_pilot --omit-institutional --checkpoints checkpoints/v8_38 > no_institutional.json
python -m stock_rl.evaluation.market_drl_pilot --institutional-only flow --checkpoints checkpoints/v8_39_flow > flow.json
python -m stock_rl.evaluation.april_holdout april_source_cache checkpoints/v8_38 checkpoints/v8_39_flow > april_holdout.json
```

三組模型使用同一批股票、同一段四月行情，差異反映訓練種子帶來的策略變動。下表是三檔股票以相同初始資金配置後的淨報酬。

| 訓練種子 | 移除全部籌碼特徵 | 僅加入法人買賣超 | 差額（百分點） |
|---|---:|---:|---:|
| 42 | 4.54% | 1.89% | −2.65 |
| 100 | 0.00% | 0.27% | +0.27 |
| 2024 | −0.87% | 3.59% | +4.46 |

不同種子的差額方向仍有分歧。種子 2024 的法人組只交易長榮，這也是下一輪要逐股拆解損益的原因。

## 技術規格

以下為 V8.0／V8.3 架構規格，現行實驗沿用原始特徵定義。

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
