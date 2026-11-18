# FinMind stock-day article access pilot

We reused the three dates from the public news pilot. For each date and stock we queried TaiwanStockNews and counted the raw rows. For each nonempty result we selected data[0] before looking at the publisher, read the embedded HTTPS publisher URL in the Google News RSS link, followed ordinary public redirects, and inspected the returned HTML title and article paragraphs. No fallback article was substituted. The selection is deterministic and small. Duplicate or syndicated index rows have not been collapsed.

| FinMind date | 2330 | 2603 | 2881 |
|---|---:|---:|---:|
| 2023-10-19 | 72 | 0 | 3 |
| 2024-01-18 | 92 | 5 | 6 |
| 2024-02-06 | 17 | 0 | 9 |

Seven nonempty stock-days yielded seven HTTP 200 responses. Six contained a matching title and identifiable article paragraphs. The Sinotrade sample had a matching title but body text was not confirmed. Of the seven probes, four also showed a timezone-qualified timestamp on the same local day (OG publication time or article time element). The Business Today OG publication date is January 21 for the January 18 indexed item. Quarantine its timestamp until the version history is reconciled. The February 6 date is a market holiday. No sampled item has a historical provider receipt.

To rerun the index for any cell from the repository root:

```bash
python -m stock_rl.evaluation.finmind_news_index 2603 2024-01-18
```

[Per-cell evidence](../evidence/finmind_stock_day_sample.json) records the exact API, hashes, first row locator, publisher URL, final HTTP status and URL, observed retrieval time, time fields and manual body finding. To repeat the publisher review, open the recorded URL and compare the article heading and paragraphs against the first FinMind row. Public availability today does not establish historical text identity or retention rights. This test measures present-day access for seven selected links only.
