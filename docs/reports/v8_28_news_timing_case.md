# Historical news timing replay

Replay two January 18 validation-period cases against each stock's market feed.

```bash
python -m stock_rl.evaluation.news_timing_case
python -m pytest -q tests/test_news_timing_case.py
```

The runner reads the frozen V8.27 publisher evidence. For this two-case experiment, it requires matching article paragraphs and a single timezone-qualified OG publication timestamp on the FinMind index date. A date conflict excludes the article. Accepted candidates use the project's five-minute estimated lag. This assumption supplies no evidence about actual historical delivery.

At 20:00 Taipei on January 17 both stocks had zero visible events. At the January 18 decision, 2603 had one visible event and 2881 had zero. Both snapshots mapped execution to January 19 at 09:00. The M1 input retains event presence and age. Its text channels are zero. The replay uses no article text, learned text vectors, trading actions or performance estimates.

[Inputs and decision times](../results/v8_28_news_timing_case.json) · [Frozen publisher evidence](../evidence/finmind_stock_day_sample.json)
