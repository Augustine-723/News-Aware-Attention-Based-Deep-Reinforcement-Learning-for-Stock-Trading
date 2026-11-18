"""Measure FinMind index coverage over five consecutive local market sessions."""
import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .finmind_news_index import STOCKS, inspect_index

CALENDAR = Path(__file__).resolve().parents[2] / "data/poc_real_data/taiex.csv"


def fetch(url):
    with urlopen(url, timeout=25) as response:
        return response.read()


def measure(stock, day, reader=fetch):
    url = ("https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockNews"
           f"&data_id={stock}&start_date={day}")
    row = dict(stock=stock, day=day.isoformat(), url=url, status="unknown", rows=None)
    try:
        raw = reader(url)
        row["raw_sha256"] = hashlib.sha256(raw).hexdigest()
        entries = inspect_index(raw, stock, day)
        row.update(status="indexed" if entries else "empty_index", rows=len(entries),
                   distinct_links=len({e["link_sha256"] for e in entries}))
    except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        row["error_type"] = type(exc).__name__
    row["observed_at"] = datetime.now(timezone.utc).isoformat()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", type=date.fromisoformat)
    args = parser.parse_args()
    with CALENDAR.open() as stream:
        sessions = sorted({date.fromisoformat(r["date"]) for r in csv.DictReader(stream)})
    window = [d for d in sessions if d >= args.start][:5]
    if len(window) != 5 or window[0] != args.start:
        parser.error("start must be a session with five available sessions")
    results = [measure(stock, day) for day in window for stock in sorted(STOCKS)]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
