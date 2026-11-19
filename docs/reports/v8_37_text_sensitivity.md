# Retrospective event text and availability sensitivity

We used 16 newly authored Chinese notes keyed by the frozen V8.33 article URLs (10 train, 3 validation, 3 test). These retrospective notes use topics reported at publication. Their authors had access to the earlier study results. Actual contemporary wording and delivery remain unknown. No publisher body was used as training text.

The existing verified BGE encoder and fixed 512-to-128 projection augmented M1 timing inputs. Structured event features remained masked. With the V8.22 protocol, each of three seeds was refit on each treatment and compared on the same 33 test return sessions. The placebo uses the same Chinese sentence for every event. The delay treatment retrains and tests with first eligibility shifted forward one trading decision, preserving the publication timestamp.

| Seed | Market only | Timing only | Notes | Constant sentence | Delayed notes |
|---|---:|---:|---:|---:|---:|
| 42 | 3.85% | 8.98% | 8.98% | 8.98% | 9.37% |
| 100 | 5.70% | 3.35% | 0.00% | 0.00% | 0.00% |
| 2024 | 3.85% | 6.94% | 3.85% | 3.85% | 7.46% |

The distinct and constant notes produced identical test actions on all stocks and seeds. Delaying availability changed net returns by 0.39, 0.00 and 3.62 percentage points respectively. Both defined paired bootstrap intervals for the delay's Sharpe difference included zero. The imposed delay is one sensitivity scenario. Neither it nor the constant-text control estimates the effect of every possible missing version, article or receipt. Test-period reuse and retrospective selection limit inference. Formal PIT and full-text M3 remain unevaluated.

Reproduce with the V8.22 environment and pinned BGE snapshot (see stock_rl/evaluation/provenance.py). The cache root is configurable with --cache-dir.

```bash
python -m stock_rl.evaluation.timing_policy_pilot --text-mode notes > docs/results/v8_37_notes.json
python -m stock_rl.evaluation.timing_policy_pilot --text-mode constant > docs/results/v8_37_constant.json
python -m stock_rl.evaluation.timing_policy_pilot --text-mode notes --delay-session > docs/results/v8_37_delayed.json
python -m stock_rl.evaluation.summary_sensitivity
```
