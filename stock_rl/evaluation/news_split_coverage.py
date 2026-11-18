"""Measure first eligible decision coverage for frozen estimated news candidates."""
import json
from pathlib import Path

import pandas as pd

from .news_timing_case import candidate

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads((ROOT / path).read_text())


def build_report():
    protocol = read_json("configs/v8_22_study.json")
    calendar = pd.DatetimeIndex(pd.read_csv(ROOT / "data/poc_real_data/taiex.csv",
                                             parse_dates=["date"]).date)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("invalid trading calendar")
    sessions = calendar.tz_localize(protocol["timezone"])
    decisions = sessions + pd.Timedelta(hours=protocol["decision_hour"])
    ledger = []
    for name in ("finmind_stock_day_sample", "evergreen_split_sample", "fubon_tsmc_split_sample"):
        for row in read_json(f"docs/evidence/{name}.json"):
            if "probe" not in row:
                continue
            event, status = candidate(row)
            record = dict(stock=row["stock"], url=row["probe"]["url"], status=status,
                          evidence=name, verified_provider_available_at=None, license_status="unverified")
            if event:
                record.update(source_published_at=event["provenance"].source_published_at.isoformat(),
                              estimated_available_at=event["provenance"].available_at.isoformat())
            ledger.append(record)
    for page in read_json("docs/evidence/public_news_sample.json")["cnyes_pages"]:
        for match in page["matches"]:
            published = pd.Timestamp(match["source_published_at"])
            for stock in match["stock_ids"]:
                ledger.append(dict(stock=stock, url=f"https://news.cnyes.com/news/id/{match['news_id']}",
                                   evidence="public_news_sample", status="estimated_candidate",
                                   source_published_at=published.isoformat(),
                                   estimated_available_at=(published + pd.Timedelta(minutes=5)).isoformat(),
                                   verified_provider_available_at=None, license_status="unverified"))
    counts = []
    for phase, (start, end) in protocol["splits"].items():
        window = decisions[(calendar >= start) & (calendar <= end)]
        for stock in protocol["stocks"]:
            days = set()
            urls = set()
            for record in ledger:
                if record["stock"] != stock or "estimated_available_at" not in record:
                    continue
                idx = decisions.searchsorted(pd.Timestamp(record["estimated_available_at"]))
                if idx >= len(decisions) or decisions[idx] not in window[:-1]:
                    continue
                record.update(phase=phase, first_decision=decisions[idx].isoformat(),
                              next_open=(sessions[idx+1] + pd.Timedelta(hours=9)).isoformat())
                urls.add(record["url"])
                days.add(record["first_decision"])
            counts.append(dict(stock=stock, phase=phase, executable_decisions=len(window)-1,
                               candidate_articles=len(urls), first_decision_days=len(days)))
    return dict(coverage=counts, ledger=ledger, verified_provider_receipts=0)


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
