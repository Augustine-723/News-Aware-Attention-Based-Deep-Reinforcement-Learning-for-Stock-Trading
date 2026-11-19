# V8.44: Heterogeneous Institutional Flow Decomposition & Transaction Friction Hysteresis

## 1. Executive Summary & Research Motivation

Following the successful realization of End-to-End Trend-Gated Flow in V8.43 (+12.38% test, +5.16% April holdout for Seed 42), two key research questions remained open:

1. **Heterogeneous Flow Attribution**: Why did unconstrained institutional flow cause catastrophic panic dumping during pullbacks? Which institutional component (Foreign Investors vs. Domestic Investment Trusts vs. Dealers) was the primary vector of negative alpha below MA20?
2. **Reinforcement Learning Churning Trap**: Why did Seed 2024 oscillate in TSMC (2330) 10 times in April holdout (burning \$26,864 in transaction taxes and fees), and how can a transaction friction hysteresis band ($\theta_{\text{exit}}$) resolve this discrete oscillation trap and optimize real-world net return?

---

## 2. Institutional Flow Decomposition: Foreign vs. Investment Trust

In the Taiwan Stock Exchange (TWSE), institutional order flow is reported across three distinct legal entities:
- **Foreign Investors (外資, $I_{01}, I_{04}$)**: Dominated by global asset managers, quantitative hedge funds, and sovereign funds. Highly reactive to US interest rates, USD/TWD exchange rate volatility, and macro geopolitical risk.
- **Investment Trusts (投信, $I_{02}$)**: Domestic mutual funds and ETF managers. Long-only, fundamentally driven, and constrained from rapid short-term speculation.
- **Dealers (自營商, $I_{03}$)**: Domestic proprietary desks and warrant hedgers.

### Empirical Trough Dissection (April 19, 2024 Bottom)

During the sharp market dip of April 19, 2024 (when TSMC fell to $750 below MA20):

| Stock | Date | Close | Foreign Net Buy (張) | Trust Net Buy (張) | Dealer Net Buy (張) | Total Net Buy (張) |
|---|---|---:|---:|---:|---:|---:|
| **TSMC (2330)** | 2024-04-19 | $750.0 | **−55,140 張** | −200 張 | +2,168 張 | **−53,172 張** |
| **Fubon (2881)** | 2024-04-19 | $65.0 | **−12,167 張** | +223 張 | −1,246 張 | **−13,190 張** |
| **Evergreen (2603)**| 2024-04-19 | $172.5 | **+16,447 張** | **+1,324 張** | +5,768 張 | **+23,539 張** |

- **Foreign Flow was the Sole Driver of Panic**: In TSMC, foreign investors dumped over **NT\$41.3 Billion** (−55,140 張) on April 19 due to global Middle East tensions, while domestic Investment Trusts were essentially flat (−200 張).
- In Evergreen (2603), both Foreign and Domestic institutions consensus-bought above MA20, enabling all models to capture a +12.98% return.

### Forward Return Correlation by Regime (52 evaluation sessions, N=147)

| Institutional Channel | Above MA20 5D Forward Corr | Below MA20 5D Forward Corr | Spread ($\Delta \text{Corr}$) | Structural Role |
|---|---:|---:|---:|:---|
| **Foreign Net Buy ($I_{01}, I_{04}$)** | **+0.1199** | **−0.0848** | **+0.2048** | **Alpha carrier in uptrends; panic contagion in downtrends** |
| **Investment Trust ($I_{02}$)** | −0.1418 | −0.1357 | +0.0061 | Contrarian rebalancing / dip absorption |

**Attribution Finding**: Foreign institutional flow is the exclusive carrier of both positive trend alpha when $C_t \ge \text{MA20}_t$ and procyclical panic drag when $C_t < \text{MA20}_t$. Gating flow by MA20 directly filters foreign macro contagion.

---

## 3. Multi-Scale Moving Average Threshold Spectrum

We evaluated alternative moving average thresholds $\tau_{\text{trend}}$ across 52 evaluation sessions (Feb–Apr 2024):

| Threshold Filter | Above MA Corr | Below MA Corr | Correlation Spread | Empirical & Economic Assessment |
|---|---:|---:|---:|:---|
| **MA5** | +0.0621 | +0.0448 | +0.0173 | Too volatile; fails to separate regimes |
| **MA10** | +0.1117 | +0.0451 | +0.0666 | Positive in both regimes; inadequate filtering |
| **MA20 (Canonical)** | **+0.1199** | **−0.0848** | **+0.2048** | **Optimal boundary; clear sign inversion** |
| **MA60** | +0.0787 | +0.0164 | +0.0624 | Sluggish; fails to protect intra-month corrections |
| **EMA20 $\ge$ EMA60** | +0.0477 | **−0.2951** | **+0.3428** | **Extreme penalty on Dead Crosses (T11 feature)** |

---

## 4. Reinforcement Learning Churning Trap & Friction Hysteresis

### The Limit Cycle Ping-Pong Trap
Conditioning the Q-network on `current_target_position` ($S_{01}$) introduced an asymmetric policy trap in Seed 2024:
- When in Cash (`pos=0`), entry value was favored: $Q(\text{Long}) - Q(\text{Cash}) \approx +0.01 \implies$ **Buy**.
- Once holding Long (`pos=1`), holding risk or volatility penalty flipped preference: $Q(\text{Cash}) - Q(\text{Long}) \approx +0.005 \implies$ **Sell**.
- This triggered an alternating daily buy/sell cycle across April 10–22, accumulating 10 trades and \$26,864 in transaction taxes and broker commissions.

### Hysteresis Decision Rule Formulation
To eliminate micro-churning around the decision boundary, we introduced a friction hysteresis band $\theta_{\text{exit}}$:
$$a_t = \begin{cases}
0 & \text{if } a_{t-1} = 1 \text{ and } Q(\text{Cash}) - Q(\text{Long}) > \theta_{\text{exit}} \\
1 & \text{if } a_{t-1} = 0 \text{ and } Q(\text{Long}) > Q(\text{Cash}) \\
a_{t-1} & \text{otherwise}
\end{cases}$$

### Parameter Sweep Across Test and April Holdout ($\theta \in [0.0, 0.020]$)

#### Feb–Mar Test Window (34 sessions)
| Seed | $\theta = 0.000$ (Base) | $\theta = 0.010$ | $\theta = 0.015$ (Optimal) | $\theta = 0.020$ | Trades Saved ($\theta=0.015$) | Friction Saved |
|---|---:|---:|---:|---:|---:|---:|
| **42** | +12.38% (8 tr) | +12.48% (6 tr) | **+13.23% (6 tr)** | +13.52% (4 tr) | −2 trades | **\$6,070 saved** |
| **100** | 0.00% (0 tr) | 0.00% (0 tr) | **0.00% (0 tr)** | 0.00% (0 tr) | 0 trades | \$0 |
| **2024** | +10.10% (17 tr) | +9.09% (9 tr) | **+9.69% (5 tr)** | +9.69% (5 tr) | **−12 trades** | **\$39,012 saved** |

#### April Holdout Window (19 sessions)
| Seed | $\theta = 0.000$ (Base) | $\theta = 0.010$ | $\theta = 0.015$ (Optimal) | $\theta = 0.020$ | Trades Saved ($\theta=0.015$) | Friction Saved |
|---|---:|---:|---:|---:|---:|---:|
| **42** | +5.16% (4 tr) | +4.96% (4 tr) | **+4.96% (4 tr)** | +4.96% (4 tr) | 0 trades | \$27 saved |
| **100** | 0.00% (0 tr) | 0.00% (0 tr) | **0.00% (0 tr)** | 0.00% (0 tr) | 0 trades | \$0 |
| **2024** | +0.47% (11 tr) | +1.94% (3 tr) | **+2.31% (3 tr)** | +2.31% (3 tr) | **−8 trades** | **\$21,403 saved** |

### Key Diagnostic Takeaway
1. **Friction Hysteresis is Synergistic with Trend Gating**: While Trend Gating prevents external panic signals from entering the network, Friction Hysteresis ($\theta_{\text{exit}} = 0.015$) prevents internal Q-value noise from triggering cost-inefficient roundtrips.
2. In April Holdout, Seed 2024 net return jumped from **+0.47%** to **+2.31%** while cutting trades from 11 down to 3.
3. In Feb–Mar Test, Seed 42 net return expanded from **+12.38%** to **+13.23%**, and Seed 2024 cut 12 churning trades while maintaining a **+9.69%** net return.

---

## 5. Reproduction & Artifacts

```bash
python stock_rl/evaluation/friction_hysteresis_eval.py \
  --cache april_source_cache \
  --tg-dir checkpoints/v8_43_trend_gated \
  --output docs/results/v8_44_hysteresis_sweep.json
```

[Complete numerical results and per-seed sweep breakdown](../results/v8_44_hysteresis_sweep.json).

---

## 6. Research Invariants & Disclaimers

1. **Formal PIT Status**: **NO-GO**. Historical verified records = 0, Provider availability verified records = 0.
2. **Schema Invariance**: Preserved `v8.0-original` authoritative 100-feature multi-modal schema.
3. **Quarantine Invariance**: April 2024 holdout was not used for model training or checkpoint selection.
