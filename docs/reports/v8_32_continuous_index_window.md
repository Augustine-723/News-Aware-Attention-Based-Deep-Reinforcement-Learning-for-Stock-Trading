# FinMind coverage for the first five training sessions

We selected the first five sessions of the frozen training split without consulting article counts. FinMind returned a schema-valid stock-day response for all 15 queries. Thirteen contained rows. The September 6 and 7 Evergreen responses contained zero rows. This establishes two observed empty index responses; other outlets or unindexed articles may still exist.

| Stock | Indexed sessions | Total index rows |
|---|---:|---:|
| 2330 | 5/5 | 141 |
| 2603 | 3/5 | 24 |
| 2881 | 5/5 | 51 |

The counters are raw index rows. Repeated links, syndication and cross-stock mentions can occur. Retrieval stamps and response hashes refer to this retrospective query. No articles from this window have been admitted by this measurement to the historical training ledger. Weekend coverage is outside this five-session experiment; weekend news can affect later decisions.

Reproduce the live query:

```bash
python -m stock_rl.evaluation.news_index_window 2023-09-01
```

[Recorded stock-day responses summarized as metadata](../evidence/finmind_train_week_index.json)
