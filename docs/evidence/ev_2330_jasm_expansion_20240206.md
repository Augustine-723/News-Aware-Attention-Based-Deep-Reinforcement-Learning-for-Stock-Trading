# Event Evidence: TSMC (2330) JASM Kumamoto Second-Fab Expansion — 2024-02-06

**Research Status**: AUTHORIZED_CHANNEL_PARTIAL  
**Formal PIT Eligible**: NO — SEC acceptance time (20:55:14 Taipei) is after the daily decision cutoff  
**MOPS Original**: NOT LOCATED IN LOCAL SOURCES  
**Historical Verified Count**: 0 (this event does not change that counter)  
**Pilot Event**: No existing pilot entry for 2024-02-06 expansion; EV-2330-09 covers the separate Feb-24 opening ceremony  

---

## 1. Source Survey — Local Authorized Sources Searched

| Source | Coverage | 2024-02-06 / 2330 Result | Evidence Level |
|---|---|---|---|
| `data/poc_real_data/mops_announcements.json` | 115 records, all ROC 115 (Nov 2026) | **0 records for 2330 or 2024-Q1** | `NOT_VERIFIED` |
| `data/official_disclosures/twse_openapi_raw_response_snapshot.json` | 94 records, all filing date 1151117 | **0 records for 2330 or 2024** | `NOT_VERIFIED` |
| `data/official_disclosures/evidence_manifest_v8_14.json` | 1 entry, 2026-11-17, stock 2881 | **0 matches** | `NOT_VERIFIED` |
| `data/poc_real_data/historical_news_micro.json` | 10 articles, all Jan 2024 | **0 articles for 2024-02-06** | `NOT_VERIFIED` |
| `stock_rl/evaluation/pilot_dataset.py` → `RAW_PILOT_EVENTS` | 50 curated events | **EV-2330-09** = Feb-24 opening only; no Feb-06 entry | `NOT_VERIFIED` |
| SEC EDGAR (network, public, free reuse) | TSMC 6-K filings | **1 matching filing** — see §2 | `OFFICIALLY_DOCUMENTED` |

**Conclusion**: No MOPS original filing or provider receipt for 2024-02-06 is available in the local research environment. The only authorized historical source located is SEC EDGAR Form 6-K, committed by Codex.

---

## 2. SEC EDGAR Form 6-K — Evidence Record

This evidence is committed in `data/official_disclosures/sec_2330_20240206_evidence.json`.

### Source Identity

| Field | Value |
|---|---|
| **Stock** | TSMC (2330 / NYSE: TSM) |
| **Source Channel** | SEC EDGAR — Form 6-K (Foreign Private Issuer Current Report) |
| **Accession Number** | `0001046179-24-000012` |
| **CIK** | 1046179 |
| **Source ID (Locator)** | `Document Format Files / sequence 1` (primary HTML document) |
| **Revision** | No amendment; this is the original 6-K, not a 6-K/A |

### Timestamps

| Field | Value | Semantic |
|---|---|---|
| **SEC Accepted At (UTC−5)** | 2024-02-06T07:55:14−05:00 | Moment EDGAR accepted the filing; this is the official SEC record time |
| **SEC Accepted At (Taipei, UTC+8)** | 2024-02-06T**20:55:14+08:00** | Taipei-equivalent; **after** the daily decision cutoff (20:00 Taipei) |
| **Timezone** | SEC timestamps are US Eastern Time (EST, UTC−5 in February) | No DST ambiguity |
| **MOPS Filing Time** | **Unknown** — not located in any local source | Cannot establish MOPS pre-market availability |
| **CNA Press Report** | 2024-02-06T18:48+08:00 (displayed minute-precision) | 72 min before SEC acceptance; original version unverified |
| **Issuer Page (pr.tsmc.com/chinese/news/3105)** | HTML `dateModified` = 20:46:32 Taipei; `datePublished` = 1970 epoch (placeholder) | Does not establish first publication |
| **Effective Decision Day (calendar)** | 2024-02-06 is in the CNY holiday gap (Feb 6–14 no TSE trading); next session = **2024-02-15** | Per `DesignValidityPreflightAuditor.get_effective_decision_day` on both taiex.csv and 2330_ohlcv.csv |

### File Hashes and Locators

| Document | URL | Bytes | SHA-256 | Observed Write Completed |
|---|---|---|---|---|
| **Source index** | https://www.sec.gov/Archives/edgar/data/1046179/000104617924000012/0001046179-24-000012-index.htm | 7,365 | `c09a5e86ab008098e1bc142747fb68a748b2df91095f3bdb90cb584223d501ca` | 2026-11-18T08:10:49.598061Z |
| **Primary document** | https://www.sec.gov/Archives/edgar/data/1046179/000104617924000012/tsm-boardx20240206x6kx2.htm | 30,248 | `b4ed2843c3401d2cbd8cf1afc203e34f358854746fae8809ff90da52d53c8b33` | 2026-11-18T08:10:55.798052Z |

> [!NOTE]
> SHA-256 values record the byte content observed on 2026-11-18. EDGAR may update an index page over time; re-downloading may produce a different index digest. The 6-K document itself is immutable after acceptance. Use `sha256sum --check` per the reproduction script in the handoff doc to verify independently.

### Content Summary

The 6-K announces: (1) JASM (TSMC's Japan fab subsidiary) Phase 1 (12/16nm) construction completion and planned production start by end of 2024; (2) board approval for a second Kumamoto fab (3/5nm class); investment up to JPY 1 trillion for both phases, with SONY and Toyota as co-investors.

This is **corroborative** of EV-2330-09 (opening ceremony), not identical. These are distinct corporate events:

| Event | Date | Description | Evidence |
|---|---|---|---|
| **Second fab board decision** | 2024-02-06 | Board approves second Kumamoto fab | SEC 6-K (this document) |
| **JASM grand opening** | 2024-02-24 | First fab opening ceremony | EV-2330-09 (CNA press archive) |

### Retention Rights

- **SEC EDGAR content**: Per [SEC Webmaster FAQ](https://www.sec.gov/about/webmaster-frequently-asked-questions), public filing content is freely available for access and reuse.
- **Raw HTML files**: Kept outside Git per project private-storage policy.
- **This evidence record** (committed JSON + this markdown): Public summary only; no raw licensed text retained.

---

## 3. Authorized Re-verification Script

From the repository root with network access (requires `curl` and `jq`):

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

---

## 4. Limitations and Open Gaps

| Gap | Description | Required to Close |
|---|---|---|
| **MOPS Filing Time** | Unknown — no MOPS record for 2024-02-06 2330 found locally | TWSE Dataset 18415 Q1 export, or MOPS historical search |
| **Provider Receipt** | No data vendor feed receipt for this event | TEJ historical corporate announcement product (campus license) |
| **Formal PIT** | SEC acceptance 20:55 Taipei > 20:00 decision cutoff; even if MOPS time were known, needs provider receipt | Remains NO-GO |
| **Earlier MOPS Posting** | CNA 18:48 suggests announcement may have been filed before SEC acceptance; unverified | MOPS historical full-text with timestamp |
| **Original Text Version** | CNA article has a 14:58 next-day update; original version not confirmed | MOPS or TEJ original timestamped record |
| **Provider Distribution Time** | Distinct from filing time; unknown for all vendors | TEJ feed ingress log; not available publicly |

---

## 5. Evidence Counts (Unchanged)

- **Historical Source Verified**: 0 (SEC 6-K is `OFFICIALLY_DOCUMENTED` for reuse rights, but it cannot establish MOPS or provider availability time)
- **Provider Availability Verified**: 0
- **Formal PIT**: NO-GO
- **Pilot Event Count**: 50 (unchanged — 2024-02-06 expansion event is distinct from EV-2330-09)

---

## 6. Next Steps for Authorized Acquisition

1. **Route A — TWSE/SFB Archive Request**: Request 1–5 original Q1 rows for 2330 from Dataset 18415 (draft inquiry: `docs/inquiries/twse_historical_disclosure_inquiry.md`).
2. **Route B — TEJ Campus License**: Request TEJ Corporate Announcements full-text product via campus library for 2024 Q1 (draft inquiry: `docs/inquiries/tej_historical_disclosure_inquiry.md`). Confirm: serial number, subject, body, filing timestamp (seconds), and first vendor distribution time.
3. **Route C — MOPS Direct Historical Search**: Requires network access and may hit anti-bot restrictions; documented as `UNAVAILABLE_OR_BLOCKED` per V8.16 findings.

*Last updated: 2026-11-18 | Branch: fix/v8-audit-patch*
