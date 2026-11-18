# V8.19 historical disclosure handoff

Baseline 4c216f6ce89abff10409452ce3d29866fe0b923d on fix/v8-audit-patch had a clean checkout. Existing market CSVs are under data/poc_real_data and the existing OpenAPI specimen and evidence manifest under data/official_disclosures. The 50 exploratory records retain their previous research classification.

## Acquisition on 2026-11-18

| Source | Observation | Decision |
|---|---|---|
| [TWSE OpenAPI](https://openapi.twse.com.tw/v1/opendata/t187ap04_L) | HTTP 200, 94 JSON records, all filing dates 1151117. Raw SHA-256 949d1b51b851737030c47c6bfe97af6f6143d34247ab45ac543ae9f78e83394d | Zero target-quarter records in this response |
| [Dataset 18415](https://data.gov.tw/dataset/18415) | Daily CSV, free, Open Government Data License v1. CSV link returned a security page. No Q1 export verified. | Request archived extract from TWSE or the listed Securities and Futures Bureau contact |
| [MOPS](https://mops.twse.com.tw/mops/web/ajax_t05st01) | Single January 2024 query for 2330 returned the security restriction page | Stop interactive acquisition |
| [TEJ product list](https://www.tejwin.com/en/all-product-list/) | Taiwan daily corporate announcements full-text lists coverage from January 2013, serial, announcement time, subject and content | Check campus license and field definitions with library/TEJ |

Historical source verification 0. Historical provider receipts 0. Real empirical replay, Formal PIT and large-scale training remain NO-GO. The current-day response is evidence about the endpoint only.

## Private intake

From repository root with project dependencies installed, run the independent synthetic example:

```bash
python -m stock_rl.evaluation.historical_import tests/fixtures/v8_19/synthetic.csv \
  --fixture --source-url https://example.invalid/synthetic \
  --observed-at 2026-11-18T12:00:00+08:00 --dry-run
```

Expected summary: one matched candidate, zero verified. Replace the example path, URL and observed time with the actual values and omit --fixture for a real export. UTF-8 CSV (BOM supported) or JSON array must contain 公司代號、發言日期、發言時間、主旨、說明. Use original bytes. A vendor must export matching field names for this adapter. Use --licensed-export for a vendor file. Scope includes calendar dates January through March 2024 and stocks 2330, 2603, 2881. Filing time needs seconds. Each output locator uses a zero-based logical record index excluding any CSV header.

Dry run emits text-free metadata. To retain normalized text privately, add --output PATH --license-document TERMS --expected-sha256 DIGEST and remove --dry-run. Choose a new file in a private directory outside Git. Review retention rights first. The terms digest records the document the operator supplied. Independent source certification and license approval are still required. Keep original bytes and any official delivery receipt privately alongside the output. The estimate uses the existing collector's five-minute hypothesis. Importing a local file cannot establish historical delivery time.

Python read_snapshot returns existing EvidenceManifestEntry objects consumable by execute_minimal_pipeline_integration. Its formal mode rejects candidates without provider receipts. Candidate JSONL is staging data. The existing tensor smoke path uses placeholder market context. Empirical pilot labels require a separate approved research dataset.

## Source access checklist

1. Ask TWSE/SFB for 1–5 original Q1 rows across the three stocks, archive URL or official delivery reference, original serial/revision, filing timezone and seconds, and any first-publication or delivery log keyed to the same record.
2. Alternatively ask the campus library and TEJ for the named product, a sample export with original serial, subject, article and dates, and first user-visible vendor distribution time with receipt. Confirm the distinction between ingress, export and distribution.
3. Confirm private raw retention, academic analysis, derived-vector publication and redistribution separately. Retain the written terms. An export supporting only filing time may support a later estimated-time study after authentication and licensing.
4. Reconcile the sample byte digest, row locator, timestamps and official delivery evidence before promotion. Leave formal availability empty without a historical provider receipt. Inquiry drafts remain unsent.

## Regression evidence

The temporal and import subset passed 14 tests, covering the existing rollover and tau checks, exact timestamp import, digest mismatch, fixture classification and formal rejection. The previously failing V8.10 multi-seed assertion was reproduced on the clean baseline using the pinned BGE weights. Its test now computes the expected exploratory verdict from all three control results. The research scoring rule is unchanged.
Full suite: 259 passed, 2 warnings (150.16 seconds), CPU PyTorch 2.5.1 and the pinned BGE model.
