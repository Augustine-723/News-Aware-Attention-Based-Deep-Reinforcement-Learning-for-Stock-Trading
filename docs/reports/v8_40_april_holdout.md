# April 2024 frozen-checkpoint holdout

We fixed the V8.38 no-institutional and V8.39 flow-only checkpoints before collecting Taiwan data for the next calendar month. No refit, validation selection or normalization update used April. Three stock sleeves start in cash with equal capital. April 1 is the first decision; April 2–30 supply 19 realized return sessions. This is a retrospective extension using estimated daily availability.

Seventeen FinMind stock, sector, TAIEX and FX responses each contain all 20 Taiwan trading dates. For the US inputs we used five FRED series: SP500, NASDAQCOM, NASDAQSOX, VIXCLS and DGS10. The original US equity feed used Yahoo; its April download was rate-limited. The March history is retained through March 29 and the April US rows, including April 1–5, are reconstructed from FRED. The frozen normalizers and 60-session history are reused. [Source URLs and exact byte hashes](../evidence/april_market_sources.json).

| Seed | No institutional | Flow only | Difference (pp) | Sharpe difference and descriptive 95% interval |
|---|---:|---:|---:|---|
| 42 | 4.54% | 1.89% | −2.65 | −1.09 [−3.91, 1.98] |
| 100 | 0.00% | 0.27% | +0.27 | Undefined: control stayed in cash |
| 2024 | −0.87% | 3.59% | +4.46 | +5.71 [2.68, 10.72] |

For seed 2024 the flow-only policy traded only 2603 (four position changes), whereas the control changed positions 19 times across all three stocks. The positive interval applies only to that seed and short window. The direction of the effect remains seed-dependent.

Reproduce with the original v8.38 and v8.39 checkpoints (or regenerate them using their reports):

```bash
python -m stock_rl.evaluation.april_holdout april_source_cache checkpoints/v8_38 checkpoints/v8_39_flow
```

[Checkpoint digests, sleeve actions, equity, costs and matched bootstrap comparisons](../results/v8_40_april_holdout.json).
