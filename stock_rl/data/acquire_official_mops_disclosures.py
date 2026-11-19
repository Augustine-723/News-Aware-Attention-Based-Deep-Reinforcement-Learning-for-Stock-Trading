"""Polite and legally compliant acquisition of authentic MOPS corporate disclosures (2023-09 to 2024-04)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from lxml import html
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "poc_real_data"
OUTPUT_DIR = ROOT / "data" / "official_disclosures"
ENDPOINT = "https://mopsov.twse.com.tw/mops/web/ajax_t05st01"

STOCKS = ["2330", "2603", "2881"]
# 2023-09 to 2024-04
MONTHS = [
    (112, 9, "train"),
    (112, 10, "train"),
    (112, 11, "train"),
    (112, 12, "train"),
    (113, 1, "validation"),
    (113, 2, "test"),
    (113, 3, "test"),
    (113, 4, "holdout"),
]


def load_trading_calendar():
    taiex_csv = DATA_DIR / "taiex.csv"
    df = pd.read_csv(taiex_csv, usecols=["date"], parse_dates=["date"])
    dates = list(df["date"].dt.normalize())

    # Include April 2024 dates from april_source_cache if present
    april_json = ROOT / "april_source_cache" / "taiex.json"
    if april_json.exists():
        raw_apr = json.loads(april_json.read_text())
        apr_dates = [pd.Timestamp(d["date"]).normalize() for d in raw_apr.get("data", [])]
        dates.extend(apr_dates)

    # Include the next trading session after April (2024-05-02)
    dates.append(pd.Timestamp("2024-05-02"))

    calendar = pd.DatetimeIndex(sorted(set(dates)))
    return calendar


def resolve_effective_decision_day(filing_dt, calendar):
    """
    Resolve the next effective trading decision point (at 20:00 Taipei).
    If filing is before or at 20:00 on a trading day, effective decision day is that day 20:00.
    If filing is after 20:00 or on a non-trading day (weekend/holiday), rollover to next trading day 20:00.
    If no next trading day exists in the calendar, returns None (unevaluable).
    """
    ts = pd.Timestamp(filing_dt).tz_localize(None)
    filing_date = ts.normalize()
    is_trading_day = (filing_date in calendar)
    is_before_cutoff = (filing_dt.hour < 20 or (filing_dt.hour == 20 and filing_dt.minute == 0 and filing_dt.second == 0))

    if is_trading_day and is_before_cutoff:
        effective_day = filing_date
    else:
        valid_days = calendar[calendar > filing_date]
        if len(valid_days) == 0:
            return None
        effective_day = valid_days[0]

    eff_str = effective_day.strftime("%Y-%m-%d") + "T20:00:00+08:00"
    eff_ts = pd.Timestamp(eff_str)
    if eff_ts < filing_dt:
        raise ValueError(
            f"Strict Causality Violation: effective decision {eff_str} is earlier than filing {filing_dt.isoformat()}"
        )

    return eff_str


def resolve_split_from_effective_day(effective_day_str):
    """
    Assign split partition strictly based on effective decision day:
    - Train: through 2023-12-29 20:00:00
    - Validation: 2024-01-02 20:00:00 to 2024-01-31 20:00:00
    - Test: 2024-02-01 20:00:00 to 2024-03-29 20:00:00
    - Holdout: 2024-04-01 20:00:00 to 2024-04-30 20:00:00
    - Unevaluable: beyond 2024-04-30 or None
    """
    if effective_day_str is None:
        return "unevaluable"
    day = effective_day_str[:10]
    if day <= "2023-12-29":
        return "train"
    elif "2024-01-02" <= day <= "2024-01-31":
        return "validation"
    elif "2024-02-01" <= day <= "2024-03-29":
        return "test"
    elif "2024-04-01" <= day <= "2024-04-30":
        return "holdout"
    else:
        return "unevaluable"


def fetch_monthly_disclosures(stock, year_roc, month, split, calendar):
    m_str = f"{month:02d}"
    fields = {
        "step": "1",
        "firstin": "1",
        "off": "1",
        "TYPEK": "all",
        "co_id": stock,
        "year": str(year_roc),
        "month": m_str,
        "b_date": "01",
        "e_date": "31",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
        "Referer": "https://mopsov.twse.com.tw/mops/web/t05st01",
    }
    req = Request(ENDPOINT, data=urlencode(fields).encode(), headers=headers)
    with urlopen(req, timeout=30) as resp:
        raw = resp.read()

    tree = html.fromstring(raw.decode("utf-8", errors="replace"))
    rows = tree.xpath("//tr[td/input[@onclick]]")
    disclosures = []

    for tr in rows:
        cells = [c.text_content().strip() for c in tr.xpath("./td")]
        if len(cells) < 5:
            continue
        # cells: [stock, name, roc_date, time, headline]
        roc_date = cells[2]
        spoke_time = cells[3]
        headline = " ".join(cells[4].split())

        # Extract sequence from onclick script
        script = tr.xpath("./td/input/@onclick")
        seq_no = None
        if script:
            m = re.search(r"t05st01_fm\.seq_no\.value='(\d+)'", script[0])
            if m:
                seq_no = int(m.group(1))

        # Parse Gregorian date
        parts = roc_date.split("/")
        year_ce = int(parts[0]) + 1911
        date_iso = f"{year_ce}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        filing_dt = datetime.fromisoformat(f"{date_iso}T{spoke_time}").replace(tzinfo=ZoneInfo("Asia/Taipei"))

        effective_decision = resolve_effective_decision_day(filing_dt, calendar)
        effective_split = resolve_split_from_effective_day(effective_decision)

        disclosures.append({
            "stock_id": stock,
            "seq_no": seq_no,
            "roc_date": roc_date,
            "spoke_date": date_iso,
            "spoke_time": spoke_time,
            "filing_datetime": filing_dt.isoformat(),
            "headline": headline,
            "is_after_20_00": (filing_dt.hour >= 20 and not (filing_dt.hour == 20 and filing_dt.minute == 0 and filing_dt.second == 0)),
            "effective_decision_day": effective_decision,
            "split": effective_split,
            "source": "TWSE_MOPS_OFFICIAL",
        })

    return disclosures, hashlib.sha256(raw).hexdigest()


def realign_existing_dataset(path: Path, calendar) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    disclosures = data["disclosures"]
    earlier_count = 0
    for d in disclosures:
        f_dt = datetime.fromisoformat(d["filing_datetime"])
        eff = resolve_effective_decision_day(f_dt, calendar)
        split = resolve_split_from_effective_day(eff)
        d["effective_decision_day"] = eff
        d["split"] = split
        if eff is not None and pd.Timestamp(eff) < f_dt:
            earlier_count += 1

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Realigned {len(disclosures)} disclosures in {path}")
    print(f"Sanity Check: Earlier than filing count = {earlier_count}")
    df_res = pd.DataFrame(disclosures)
    print("\n" + "=" * 65)
    print("Realigned Summary by Stock and Split:")
    print(pd.crosstab(df_res["stock_id"], df_res["split"], margins=True))
    print("=" * 65)
    return earlier_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "mops_official_disclosures_2023_2024.json")
    parser.add_argument("--realign", action="store_true", help="Realign existing output file without network queries")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calendar = load_trading_calendar()

    if args.realign:
        realign_existing_dataset(args.output, calendar)
        return

    all_disclosures = []
    receipts = []

    print(f"Starting compliant MOPS disclosure acquisition for stocks: {STOCKS}")
    for stock in STOCKS:
        for year_roc, month, split in MONTHS:
            ce_year = year_roc + 1911
            print(f"Querying {stock} for {ce_year}-{month:02d} (ROC {year_roc})...", end=" ", flush=True)
            try:
                items, digest = fetch_monthly_disclosures(stock, year_roc, month, split, calendar)
                all_disclosures.extend(items)
                receipts.append({
                    "stock": stock,
                    "year_ce": ce_year,
                    "month": month,
                    "count": len(items),
                    "raw_sha256": digest,
                })
                print(f"found {len(items)} items (hash: {digest[:8]}...)")
            except Exception as e:
                print(f"ERROR: {e}")
            time.sleep(1.0) # Polite rate limit

    payload = {
        "metadata": {
            "source": "Taiwan Stock Exchange MOPS Official Listed Company Material Disclosures",
            "endpoint": ENDPOINT,
            "scope": "2330, 2603, 2881 from 2023-09 to 2024-04",
            "total_disclosures": len(all_disclosures),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "compliance": "Securities and Exchange Act Public Disclosures; Rate-Limited Compliant Query; Zero-Bypass",
        },
        "query_receipts": receipts,
        "disclosures": all_disclosures,
    }

    raw_json = json.dumps(payload, indent=2, ensure_ascii=False)
    args.output.write_text(raw_json)
    print(f"\nSuccessfully acquired {len(all_disclosures)} official disclosures!")
    print(f"Saved to: {args.output}")

    # Summary by stock and split
    df_res = pd.DataFrame(all_disclosures)
    print("\n" + "=" * 65)
    print("Summary by Stock and Split:")
    print(pd.crosstab(df_res["stock_id"], df_res["split"], margins=True))
    print("=" * 65)


if __name__ == "__main__":
    main()
