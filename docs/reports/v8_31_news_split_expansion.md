# Three-stock split coverage

For the missing Fubon train and validation cells, we selected the first Anue Cnyes result from each of two targeted FinMind queries (2023-11-17 and 2024-01-10). They returned 66 and 33 index rows. For TSMC validation we reused the existing 2024-01-18 FinMind snapshot and selected its first Liberty Times Finance result. Each publisher page supplied a matching headline, article paragraphs and an explicit publication timestamp. The TSMC story reports a market rumor. Its inclusion records publication, with no verification of the underlying claim.

All nine stock/split cells now contain one estimated first-decision day. Fubon's January 10 publication at 20:13:04 Taipei maps to the January 11 decision and the January 12 open. The five-minute lag is assumed. Publisher pages were retrieved retrospectively. Full-text usage rights, historical article versions and provider availability remain unverified. Sampling targeted missing cells and cannot estimate the frequency of news. Training has only three candidate stock-days out of 246 possible; evidence remains insufficient for a strategy-performance claim.

Reproduce offline:

```bash
python -m stock_rl.evaluation.news_split_coverage
```

[Publisher evidence](../evidence/fubon_tsmc_split_sample.json) · [Full ledger](../results/v8_31_news_split_coverage.json)
