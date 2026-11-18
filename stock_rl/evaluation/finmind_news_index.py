"""Inspect one stock-day of the public FinMind TaiwanStockNews index."""
import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from urllib.request import urlopen

from .public_news_probe import STOCKS


def inspect_index(raw, stock, day):
    payload = json.loads(raw)
    if payload["status"] != 200 or not isinstance(payload["data"], list):
        raise ValueError("invalid news index response")
    entries = []
    for i, row in enumerate(payload["data"]):
        timestamp = datetime.fromisoformat(row["date"])
        if row["stock_id"] != stock or timestamp.date() != day:
            raise ValueError("unexpected stock or date")
        entries.append(dict(source_locator=f"data[{i}]", publisher=row["source"],
                            source_time_as_reported=row["date"],
                            link_sha256=hashlib.sha256(row["link"].encode()).hexdigest()))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock", choices=sorted(STOCKS))
    parser.add_argument("day", type=date.fromisoformat)
    args = parser.parse_args()
    url = ("https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockNews"
           f"&data_id={args.stock}&start_date={args.day}")
    with urlopen(url, timeout=25) as response:
        raw = response.read()
    print(json.dumps(dict(source_url=url, observed_retrieved_at=datetime.now(timezone.utc).isoformat(),
                          raw_response_sha256=hashlib.sha256(raw).hexdigest(),
                          stock_id=args.stock, day=args.day.isoformat(),
                          entries=inspect_index(raw, args.stock, args.day),
                          verified_provider_available_at=None, formal_pit=False),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
