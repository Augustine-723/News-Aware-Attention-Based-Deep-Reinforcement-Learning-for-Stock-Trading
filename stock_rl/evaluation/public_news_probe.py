"""Sample two public Cnyes listing pages per date and emit only research metadata."""
import argparse
import hashlib
import json
from datetime import date, datetime, time, timezone
from urllib.request import urlopen
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")
STOCKS = {"2330", "2603", "2881"}


def listing_url(day, page):
    start = datetime.combine(day, time.min, TAIPEI)
    return ("https://api.cnyes.com/media/api/v1/newslist/category/tw_stock"
            f"?startAt={int(start.timestamp())}&endAt={int(start.replace(hour=23, minute=59, second=59).timestamp())}"
            f"&limit=5&page={page}")


def inspect_page(raw, day, observed, page_number=1):
    if observed.tzinfo is None:
        raise ValueError("timezone required")
    page = json.loads(raw)["items"]
    rows = page["data"]
    if page["current_page"] != page_number or not isinstance(rows, list) or len(rows) > 5:
        raise ValueError("unexpected page")
    matches = []
    for index, row in enumerate(rows):
        published = datetime.fromtimestamp(row["publishAt"], TAIPEI)
        if published.date() != day or published > observed:
            raise ValueError("publication outside requested interval")
        tags = row["stock"] or []
        if not isinstance(tags, list):
            raise ValueError("expected stock tags")
        stocks = sorted(STOCKS.intersection(tags))
        if stocks and row["payment"] == 0 and published.time() <= time(20):
            matches.append(dict(news_id=row["newsId"], stock_ids=stocks,
                                source_locator=f"items.data[{index}]",
                                source_published_at=published.isoformat(),
                                published_by_2000=published.time() <= time(20),
                                verified_provider_available_at=None))
    return dict(day=day.isoformat(), source_url=listing_url(day, page_number),
                observed_retrieved_at=observed.isoformat(),
                raw_response_sha256=hashlib.sha256(raw).hexdigest(),
                category_total=page["total"], sampled=len(rows), matches=matches,
                license_status="unverified", formal_pit=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dates", nargs="+", type=date.fromisoformat)
    args = parser.parse_args()
    if len(args.dates) > 6:
        parser.error("pilot supports at most six dates per invocation")
    for day in dict.fromkeys(args.dates):
        for page in (1, 2):
            with urlopen(listing_url(day, page), timeout=25) as response:
                raw = response.read()
            result = inspect_page(raw, day, datetime.now(timezone.utc), page)
            print(json.dumps(result), flush=True)
            if result["sampled"] < 5 or result["category_total"] <= page * 5:
                break


if __name__ == "__main__":
    main()
