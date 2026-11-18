# First-decision coverage for historical news candidates

Two additional stock-day queries for 2603 (December 18, 2023 and March 15, 2024) returned 32 and 19 FinMind index rows. We checked the first Liberty Times Finance row in each response against the public publisher page. Both articles name 2603 and include an explicit publisher timestamp. Raw article text stays outside Git. Selection was targeted, so these counts cannot estimate daily news prevalence.

Run the offline ledger from the repository root:

```bash
python -m stock_rl.evaluation.news_split_coverage
```

Coverage counts distinct candidate URLs and distinct first eligible decision dates per stock. The denominator excludes each split's final session because V8.22 cannot execute a new action inside the split after that session. The report uses publisher time plus five minutes, including holiday rollover. Missing historical versions, rights and delivery receipts still preclude formal M3 evaluation. Unsampled dates remain unknown.

[New source evidence](../evidence/evergreen_split_sample.json) · [Candidate ledger and coverage](../results/v8_30_news_split_coverage.json)
