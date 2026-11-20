"""Inspect TWSE snapshot receipts as of a Taipei decision cutoff."""
import argparse
import hashlib
import json
import math
from datetime import date, datetime, time
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")
TARGETS = ("2330", "2603", "2881")
BASE = "https://openapi.twse.com.tw/v1/"
SOURCES = {
    BASE + "opendata/t187ap04_L": ("disclosures", "公司代號", "發言日期"),
    BASE + "exchangeReport/STOCK_DAY_ALL": ("market_daily", "Code", "Date"),
}


def source_day(value):
    value = str(value)
    if len(value) != 7 or not value.isdecimal():
        raise ValueError("Expected a seven-digit ROC date")
    return date(int(value[:3]) + 1911, int(value[3:5]), int(value[5:]))


def inspect_snapshots(paths, decision_day):
    cutoff = datetime.combine(decision_day, time(20), TAIPEI)
    events = {}
    quotes = {stock: set() for stock in TARGETS}
    receipts = []
    for path in paths:
        with ZipFile(path) as archive:
            if len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError("Duplicate archive member")
            sources = json.loads(archive.read("manifest.json"))["sources"]
            if isinstance(sources, dict):
                sources = sources.values()
            seen = set()
            for record in sources:
                if record["url"] in seen:
                    raise ValueError("Duplicate source receipt")
                seen.add(record["url"])
                kind, code, date_field = SOURCES[record["url"]]
                receipt = dict(archive=Path(path).name, source=kind, status=record["status"])
                receipts.append(receipt)
                if record["status"] != "success":
                    receipt["reason"] = record.get("reason", "source reported failure")
                    continue
                observed = datetime.fromisoformat(record.get("observed_retrieved_at") or record["observed_at"])
                if observed.utcoffset() is None or record["http_status"] != 200:
                    raise ValueError("Missing timezone or unsuccessful HTTP status")
                observed = observed.astimezone(TAIPEI)
                raw = archive.read(record.get("archive_member", kind + ".json"))
                digest = hashlib.sha256(raw).hexdigest()
                if digest != (record.get("raw_sha256") or record["sha256"]):
                    raise ValueError("Raw response SHA-256 mismatch")
                rows = json.loads(raw)
                if not isinstance(rows, list) or len(rows) != record.get("response_count", record.get("rows")):
                    raise ValueError("Invalid response array or row count")
                dates = sorted({r[date_field] for r in rows})
                counts = {s: sum(r[code] == s for r in rows) for s in TARGETS}
                if dates != sorted(record["source_dates"]) or counts != record["target_counts"]:
                    raise ValueError("Source dates or target counts differ from receipt")
                receipt.update(observed_at=observed.isoformat(), by_cutoff=observed <= cutoff,
                               source_dates=[source_day(d).isoformat() for d in dates], target_counts=counts)
                if observed > cutoff:
                    continue
                for index, r in enumerate(rows):
                    stock = r[code]
                    if stock not in TARGETS:
                        continue
                    day = source_day(r[date_field])
                    if day > observed.date():
                        raise ValueError("Source date after observation")
                    if kind == "market_daily":
                        values = [r[key].strip().replace(",", "") for key in ("OpeningPrice", "ClosingPrice")]
                        if any(v in ("", "--", "-") for v in values):
                            continue
                        prices = [float(v) for v in values]
                        if all(math.isfinite(x) and x > 0 for x in prices):
                            quotes[stock].add(day)
                        continue
                    stamp = datetime.combine(day, datetime.strptime(r["發言時間"].replace(":", "").zfill(6), "%H%M%S").time(), TAIPEI)
                    if stamp > observed:
                        raise ValueError("Disclosure timestamp after observation")
                    # Exclude the daily export date so repeated observations of identical content match.
                    payload = {k.strip(): v for k, v in r.items() if k.strip() != "出表日期"}
                    key = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                    if key not in events or observed.isoformat() < events[key]["observed_at"]:
                        events[key] = dict(stock=stock, filing_at=stamp.isoformat(), observed_at=observed.isoformat(),
                                           raw_sha256=digest, row_index=index)
            for url in sorted(SOURCES.keys() - seen):
                receipts.append(dict(archive=Path(path).name, source=SOURCES[url][0], status="missing_receipt"))
    return dict(cutoff=cutoff.isoformat(), receipts=receipts, disclosure_versions=events,
                market={stock: dict(source_days=sorted(d.isoformat() for d in days),
                                    decision_day_quote=decision_day in days) for stock, days in quotes.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-day", required=True, type=date.fromisoformat)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_snapshots(args.archives, args.decision_day), ensure_ascii=False, indent=2))
