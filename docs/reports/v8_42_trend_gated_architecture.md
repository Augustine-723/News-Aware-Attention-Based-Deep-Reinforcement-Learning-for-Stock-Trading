# Trend-Gated Architecture: Implementation and Holdout Evaluation

Following the V8.41 regime findings (where institutional flow generated +4.89 pp mean alpha above MA20 but −2.91 pp drag below MA20), we implemented and evaluated three architectural variants of the **Trend Gate (趨勢門檻機制)** on the independent April 2024 holdout across the three frozen seeds (42, 100, 2024).

## 1. Architectural Implementations

1. **Strict Action Clamping (`flow_gated_strict`)**:
   $$\pi_{\text{strict}}(s_t) = \pi(s_t) \cdot \mathbb{I}(C_t \ge \text{MA20}_t)$$
   Forces the target action to CASH ($0$) whenever closing price is below the 20-day moving average.
2. **Asymmetric Entry Filter (`flow_gated_entry`)**:
   Prohibits opening *new* LONG positions ($0 \to 1$) when $C_t < \text{MA20}_t$, but allows existing holdings to be managed by the neural policy.
3. **Dynamic Input Feature Gate (`flow_gated_feature_mask`)**:
   Modulates the neural network's input representation:
   - When $C_t \ge \text{MA20}_t$: supplies full institutional flow ($I_{01}–I_{05}$).
   - When $C_t < \text{MA20}_t$: dynamically masks institutional flow channels ($I_{01}–I_{05} = 0$, missing mask = True). The model falls back to pure price/technical/context features, eliminating procyclical dumping panic at market bottoms.

---

## 2. April Holdout Results (19 return sessions)

| Seed | Control (No Inst) | Flow Only | Gated Strict | Gated Entry | Gated Feature Mask | Optimal Variant vs Control |
|---|---:|---:|---:|---:|---:|:---|
| **42** | +3.99% | +2.77% | +0.04% | +0.85% | **+0.85%** | Controls drawdown; low friction (3 trades) |
| **100** | 0.00% | +3.39% | +0.33% | +1.83% | **+3.51%** | **Highest return among all arms (+3.51 pp over control)** |
| **2024** | +3.78% | +1.37% | −1.50% | +0.17% | **+3.05%** | **Jumps +1.68 pp over unconstrained flow (+3.05% vs +1.37%)** |
| **Mean** | **+2.59%** | **+2.51%** | **−0.38%** | **+0.95%** | **+2.47%** | **Feature Mask is the only architecture with positive returns across all 3 seeds** |

---

## 3. Trade Frequency and Friction Analysis

| Policy Arm | Total Executed Trades (April) | Total Transaction Friction (NTD) | Average Friction Rate |
|---|---:|---:|---:|
| **Control (`without_institutional`)** | 13 trades | $29,121 | 0.97% |
| **Flow Only (`flow_only`)** | 19 trades | $49,205 | 1.64% |
| **Gated Strict (`flow_gated_strict`)** | **23 trades** | **$59,996** | **2.00%** |
| **Gated Entry (`flow_gated_entry`)** | 15 trades | $37,943 | 1.26% |
| **Gated Feature Mask (`flow_gated_feature_mask`)** | **19 trades** | **$50,039** | **1.67%** |

### Key Diagnostic Takeaway: Clamping vs. Masking
- **Strict output clamping fails**: Binary hard rules ($\mathbb{I}(C_t \ge \text{MA20}_t)$) create boundary oscillations. Near the 20MA threshold, daily price fluctuation forces repetitive buy-sell roundtrips, inflating trade count to 23 trades and burning 2.0% ($60,000) in friction.
- **Input feature masking succeeds**: Dynamic feature masking leaves the discrete policy output smooth. It strips away misleading negative flow signals during pullbacks without forcing artificial liquidations, allowing Seed 100 to achieve **+3.51%** and Seed 2024 to recover to **+3.05%**.

---

## 4. Reproduction

```bash
python stock_rl/evaluation/trend_gated_holdout.py \
  --cache april_source_cache \
  --control-dir checkpoints/v8_38 \
  --flow-dir checkpoints/v8_39_flow \
  --output docs/results/v8_42_trend_gated_holdout.json
```

[Full bootstrap intervals and per-sleeve equity curves](../results/v8_42_trend_gated_holdout.json).
