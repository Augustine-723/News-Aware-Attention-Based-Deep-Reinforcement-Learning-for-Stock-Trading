# Incremental value of the institutional feature block

The three-stock V8.22 M0 already included I01–I08. For the matched control, zero all eight institutional channels and mark their mask missing before fitting the train-only normalizer. Retrain with the same three seeds, dates, next-open contract, costs and validation checkpoint rule. Both arms have empty event inputs. This comparison measures the whole block and its mask; it does not isolate individual investor categories or market regimes.

| Seed | Omitted, net | Retained, net | Retained minus omitted (pp) | Net Sharpe difference |
|---|---:|---:|---:|---:|
| 42 | 11.48% | 3.85% | -7.63 | +1.00 |
| 100 | 1.85% | 5.70% | +3.84 | +2.19 |
| 2024 | 3.86% | 3.85% | -0.01 | -0.004 |

The three seeds share 33 return dates. Bootstrap intervals for the first two Sharpe differences were undefined in some replicates. The third interval spans zero. The observed incremental return was seed-dependent.

```bash
python -m stock_rl.evaluation.market_drl_pilot --omit-institutional --checkpoints checkpoints/v8_38 > no_institutional.json
python -m stock_rl.evaluation.institutional_comparison no_institutional.json
```
