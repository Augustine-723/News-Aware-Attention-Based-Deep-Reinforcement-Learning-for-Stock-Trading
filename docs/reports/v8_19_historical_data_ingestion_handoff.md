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

## Follow-up acquisition — first alternative-channel historical disclosure

One 2330 Form 6-K was acquired from EDGAR. The accession index identifies TSMC, the primary HTML and the February 6 acceptance time. The issuer's Chinese release independently corroborates the JASM Kumamoto expansion. The committed evidence JSON under data/official_disclosures records the URLs, byte lengths, SHA-256 values and observed file completion times. Raw HTML stays outside Git. The SEC webmaster FAQ describes EDGAR public filing content as free to access and reuse.

Count one historical issuer filing for SEC source-quality research. Count zero verified MOPS filings and zero provider receipts. SEC acceptance converts to 20:55:14 in Taipei, after the decision cutoff. First dissemination time and any earlier MOPS posting remain unknown. Keep this specimen outside the TWSE-schema importer and empirical replay. The next useful comparison is the corresponding original MOPS filing and its actual publication time.

To reproduce the two hash checks from the repository root, use a private directory. A changed server response requires inspection.

```bash
set -e
evidence=data/official_disclosures/sec_2330_20240206_evidence.json
private=$(mktemp -d)
for key in source_index primary_document; do
  url=$(jq -r ".$key.url" "$evidence")
  digest=$(jq -r ".$key.sha256" "$evidence")
  curl --fail --silent --show-error "$url" -o "$private/$key"
  printf '%s  %s\n' "$digest" "$private/$key" | sha256sum --check
done
```

## Taiwan cutoff evidence for the same event

CNA's article 202402060275 displays February 6, 18:48 Taipei as publication and February 7, 14:58 as its update. This places publisher-reported coverage 72 minutes before the February 6 decision. The currently accessible article is a later revision. Its precise original text and delivery to a research feed remain unverified. Only citation metadata and a digest of the observed response are recorded. This is corroboration of the existing event, with no additional disclosure counted.

The issuer page for release 3105 displays the date only. Its HTML contains a 1970 datePublished, a dateModified of 20:46:32 Taipei and a date-display element with a UTC time equivalent to 20:00 Taipei. These fields do not establish first publication. No matching official MOPS row with a filing timestamp was located. Keep the SEC, press and MOPS clocks separate. Public-report chronology is ready for a case study. Empirical text replay still needs a timestamped original version and provider receipt.

## Codex and Antigravity working handoff

Codex ran DesignValidityPreflightAuditor.get_effective_decision_day against each of taiex.csv and 2330_ohlcv.csv. Both calendars jump from February 5 to February 15. For calendar sensitivity only, both February 6 clock inputs (18:48 and 20:55:14 Taipei, tau=0) resolve to February 15 at 20:00 and the next session opens February 16. February 6 had no decision sample in this builder. This corrects the earlier description of an evening decision and does not certify either timestamp as provider availability.

Antigravity completed the local source search at c1ed3ec. See [the reviewed evidence record](../evidence/ev_2330_jasm_expansion_20240206.md). Codex independently matched both retained EDGAR files to their recorded SHA-256 values and checked the investment and process figures against the filing. Counts: one SEC filing, zero verified MOPS originals and zero historical provider receipts. No pilot promotion. Next external dependency: an authorized historical export with original version, permissions and, for formal replay, version-linked distribution evidence.
