# Frozen checkpoint comparison on two January validation cases

Three market-only V8.22 checkpoints used their own training-period normalizers and the same stock-specific market inputs for each M0/M1 pair. M1 retains event presence and age. All event text channels were zero. This replay uses the estimated timing and quarantine decisions from V8.28.

At the January 18 20:00 Taipei decision for 2603, the maximum absolute difference across the two action Q values was 0.006180 (seed 42), 0.013963 (100) and 0.006142 (2024). All three models chose cash for both inputs. All nine comparisons without a visible event (2603 on January 17 and 2881 on both dates, across three seeds) had exactly equal Q values.

These validation cases measure sensitivity of fixed market-only models to an event input. They provide no estimate of the return from training a news-aware policy. Full precision values and checkpoint hashes are in the [result](../results/v8_29_real_news_inference.json).

Run with the existing V8.22 checkpoints:

```bash
python -m stock_rl.evaluation.news_timing_case --checkpoint checkpoints/v8_22/m0-seed-*.pt
```
