# Public news metadata pilot

Run from the repository root with Python 3.10+ and timezone data.

```bash
python -m stock_rl.evaluation.public_news_probe 2023-10-19 2024-01-18 2024-02-06
python -m stock_rl.evaluation.finmind_news_index 2330 2024-02-06
```

[Sample responses summarized as metadata](../evidence/public_news_sample.json) contain the query URLs, observed retrieval timestamps, per-response raw-byte SHA-256 and source locators. Raw text is excluded. The probes have no training or event-store side effects.

Cnyes returned 47, 68 and 48 category entries respectively. We sampled only the first ten listed entries per date. Filtering locally on non-paywalled rows, explicit 2330/2603/2881 stock tags and publication by 20:00 Taipei yielded 0, 0 and 1 matches. The service returned a later time in a trial query ending at 20:00, so the adapter requests full-day pages and filters individual timestamps. Zero matches describe sampled pages only. These pages carry no historical traffic rankings.

On 2024-02-06 article 5447931 (2330, 18:47:17 Taipei) reported the decision on the second Japanese fab. FinMind returned 17 index rows for 2330. Rows 9, 14 and 16 (zero-based) independently identify coverage of the same event by topic. No FinMind title contained the full Cnyes headline. FinMind timestamps lack explicit offsets in this response. Topic correspondence establishes neither identical text nor historical availability.

February 6 was outside the project's trading sessions. A publication-time check is only a candidate screen. The available model inputs still require the project's trading calendar, data-use rights and version-specific availability evidence. Post-cutoff publications may enter later decisions under the existing calendar rules. This small pilot does not estimate full stock-day coverage, validate provider receipts, or measure strategy returns.
