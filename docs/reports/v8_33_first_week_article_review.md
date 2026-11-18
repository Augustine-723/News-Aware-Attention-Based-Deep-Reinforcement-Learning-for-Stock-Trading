# First training week publisher checks

All 15 downloaded FinMind responses matched the frozen SHA-256 values. Across 216 rows, 149 distinct index URLs yielded 144 exact destination keys after extracting cleartext HTTPS links embedded in Google News RSS tokens. Opaque URLs and URL variants remain separate. Story-level deduplication is unresolved.

For each indexed stock-day, select the first title naming the stock with a link to the supported Liberty Times Finance or Cnyes hosts. Eight distinct URLs qualified. Seven had matching titles, readable article paragraphs and an unambiguous same-day publication time. One TSMC page had a title and timestamp but its article paragraphs were unconfirmed. The other 136 destination keys did not receive publisher review. Sampling by publisher and title limits representativeness.

All seven passed the five-minute estimated lag before the same day's 20:00 cutoff, covering seven stock-days on four distinct decision dates. Rights and contemporaneous version/availability evidence remain unresolved. No article body is included in Git.

The offline selection can be reproduced with private index responses named YYYY-MM-DD-STOCK.json matching the V8.32 manifest:

```bash
python -m stock_rl.evaluation.news_week_dedup /path/to/private/index
python -m stock_rl.evaluation.news_split_coverage
```
