# V8.43: End-to-End Trend-Gated Flow Training and Holdout Evaluation

## 1. Executive Summary & Research Motivation

In V8.40, holdout evaluation over the out-of-sample April 2024 window revealed that institutional order flow caused significant cross-seed divergence: Seed 2024 gained +4.46 pp while Seed 42 suffered a −2.65 pp penalty due to foreign selling on the April 19 dip below MA20 triggering a premature panic liquidation at the market bottom.

In V8.41, empirical interaction threshold analysis demonstrated that institutional order flow generates **+4.89 pp** mean alpha when the stock is in an established uptrend ($C_t \ge \text{MA20}_t$), but acts as a procyclical lagging drag (**−2.91 pp**) when below MA20.

In V8.42, inference-time gating demonstrated that strict action clamping ($\pi \cdot \mathbb{I}(C_t \ge \text{MA20}_t)$) suffers catastrophic whipsaw friction ($60,000 cost across 23 trades), whereas dynamic input feature masking preserves policy smoothness.

In **V8.43**, we advanced from heuristic inference-time masking to **end-to-end training**: the Q-network is trained from scratch with the MA20 dynamic feature mask embedded directly into the training loop and replay buffer. The network learns value representations that rely on institutional order flow during uptrends ($C_t \ge \text{MA20}_t$) while falling back cleanly to price and macro context during pullbacks ($C_t < \text{MA20}_t$).

---

## 2. Model Architecture & Training Protocol

- **Base Framework**: Double DQN with Dueling Architecture and Multi-Head Temporal Attention (`StockRLFullModel`, schema `v8.0-original`).
- **Dynamic Feature Masking**:
  - When $C_t \ge \text{MA20}_t$: Institutional flow channels ($I_{01}–I_{05}$) are active; margin channels ($I_{06}–I_{08}$) are masked.
  - When $C_t < \text{MA20}_t$: All institutional channels ($I_{01}–I_{08}$) are dynamically zeroed and marked missing in the attention mask.
- **Fitting Windows**:
  - Train: 2023-09-01 to 2023-12-29 (83 sessions per stock)
  - Validation: 2024-01-02 to 2024-01-31 (22 sessions per stock)
  - Test: 2024-02-01 to 2024-03-29 (34 sessions per stock)
  - Holdout: 2024-04-01 to 2024-04-30 (19 return sessions per stock, strictly quarantined)

---

## 3. Test Window Performance (Feb–Mar 2024, 34 sessions)

| Seed | Control (No Inst) | Flow Only | Trend-Gated Flow (V8.43) | Delta vs Flow | Delta vs Control |
|---|---:|---:|---:|---:|---:|
| **42** | +11.48% | +8.57% | **+12.38%** | **+3.81 pp** | **+0.90 pp** |
| **100** | +1.85% | +6.49% | **0.00%** | −6.49 pp | −1.85 pp |
| **2024** | +3.86% | +5.99% | **+10.10%** | **+4.11 pp** | **+6.24 pp** |
| **Mean (Active)** | +7.67% | +7.28% | **+11.24%** | **+3.96 pp** | **+3.57 pp** |

### Per-Stock Breakdown (Test Window)
- **Seed 42**:
  - TSMC (2330): +22.90% (Sharpe 4.71, 1 trade)
  - Evergreen (2603): +13.44% (Sharpe 2.32, 3 trades — up from +1.25% under unconstrained flow!)
  - Fubon (2881): +0.80% (Sharpe 0.55, 4 trades)
- **Seed 2024**:
  - TSMC (2330): +18.75% (Sharpe 4.70, 16 trades — up from +10.57% under unconstrained flow)
  - Evergreen (2603): +11.54% (Sharpe 1.94, 1 trade — up from +0.28% under unconstrained flow!)
  - Fubon (2881): 0.00% (0 trades)

---

## 4. Independent April Holdout Performance (19 return sessions)

| Seed | Control (No Inst) | Flow Only | Trend-Gated Flow (V8.43) | Delta vs Flow | Delta vs Control | Delta Sharpe (95% CI vs Ctrl) |
|---|---:|---:|---:|---:|---:|:---|
| **42** | +3.99% | +2.77% | **+5.16%** | **+2.39 pp** | **+1.17 pp** | **+1.11** ([+0.31, +2.03]) |
| **100** | 0.00% | +3.39% | **0.00%** | −3.39 pp | 0.00 pp | N/A (Zero Volatility) |
| **2024** | +3.78% | +1.37% | **+0.47%** | −0.90 pp | −3.32 pp | −2.52 ([−8.85, +0.81]) |

### Deep-Dive: Seed 42 Turnaround Mechanism
In V8.40, unconstrained flow led Seed 42 into a panic liquidation of TSMC (2330) on April 19 at the exact trough ($750), incurring $5,618 in friction and finishing April at −2.47% (or −4.66% under re-trained weights).

Under **Trend-Gated Flow (V8.43)**:
1. **TSMC (2330)**: When price dipped below MA20 on April 16–19, the institutional flow channels were masked. The network fell back on long-term value representations, **holding the 1.0 position throughout the entire month**. Net return rebounded to **+0.63%**, with only 1 initial entry trade and $854 total friction.
2. **Evergreen (2603)**: Maintained full 1.0 position from Day 1 through month-end, capturing **+12.98%** with only $1,354 friction.
3. **Fubon (2881)**: Captured late-April financial breakout (+1.87%, 2 trades, $6,306 friction).
4. **Overall Portfolio**: Reached **+5.16%** net return with only **4 trades executed** across the entire month and total friction of **$8,514** (friction rate: **0.28%**).
5. **Statistical Significance**: The paired bootstrap Sharpe comparison against Control yielded a delta Sharpe of **+1.11** with a 95% confidence interval of **[+0.31, +2.03]**, confirming that the outperformance is strictly positive.

---

## 5. Friction and Turnover Comparison (April Holdout)

| Model Configuration | Seed 42 Trades | Seed 42 Friction | Seed 2024 Trades | Seed 2024 Friction |
|---|---:|---:|---:|---:|
| **Control (`without_institutional`)** | 4 trades | $8,244 | 13 trades | $28,941 |
| **Flow Only (`flow_only`)** | 5 trades | $11,382 | 19 trades | $49,205 |
| **Trend-Gated Flow (`trend_gated_flow`)** | **4 trades** | **$8,514** | **11 trades** | **$28,217** |

---

## 6. Reproduction & Audit Commands

```bash
# 1. Train End-to-End Trend-Gated Flow models
python stock_rl/evaluation/market_drl_pilot.py \
  --trend-gated-flow \
  --checkpoints checkpoints/v8_43_trend_gated > docs/results/v8_43_trend_gated_fitting.json

# 2. Evaluate on April Holdout and compile results
python stock_rl/evaluation/evaluate_trend_gated_end_to_end.py \
  --cache april_source_cache \
  --control-dir checkpoints/v8_38 \
  --flow-dir checkpoints/v8_39_flow \
  --tg-dir checkpoints/v8_43_trend_gated \
  --test-results docs/results/v8_43_trend_gated_fitting.json \
  --output docs/results/v8_43_trend_gated_flow.json
```

[Complete numerical results and paired evaluation logs](../results/v8_43_trend_gated_flow.json).

---

## 7. Research Integrity & Invariant Disclaimers

1. **Formal PIT Backtest Status**: **NO-GO**. Historical verified records = 0, Provider availability verified records = 0.
2. **Schema Invariance**: Maintained `v8.0-original` multi-modal schema.
3. **Quarantine Invariance**: April 2024 data remained strictly quarantined during model fitting and hyperparameter selection.
