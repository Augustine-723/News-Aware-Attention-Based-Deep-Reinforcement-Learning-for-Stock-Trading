# Split institutional inputs into flow and margin blocks

Six retrained policies used the original v8.0 feature schema and V8.22 validation and execution rules. Flow retains I01–I05 (three institutional investor net flows and five-day aggregates). Margin retains I06–I08 (margin and short balance changes, short-to-margin balance ratio). The omitted channels are zeroed and their missing indicators set before fitting normalization. News slots are empty throughout.

| Seed | Neither | Flow | Margin | Both |
|---|---:|---:|---:|---:|
| 42 | 11.48% | 8.57% | 2.47% | 3.85% |
| 100 | 1.85% | 6.49% | 0.00% | 5.70% |
| 2024 | 3.86% | 5.99% | 6.32% | 3.85% |

All values are cost-adjusted equal-initial-capital returns on the same 33 dates. Flow exceeded the combined block for every seed, but did not beat the no-block control for seed 42. All defined paired Sharpe difference intervals against the no-block control included zero. Repeated reuse of these dates makes model rankings exploratory.

Reproduce for each MODE (flow or margin):

```bash
MODE=flow
python -m stock_rl.evaluation.market_drl_pilot --institutional-only "$MODE" --checkpoints "checkpoints/v8_39_$MODE" > "$MODE.json"
python -m stock_rl.evaluation.institutional_comparison docs/results/v8_38_no_institutional.json --treatment "$MODE.json"
```
