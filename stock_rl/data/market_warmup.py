"""Verify FinMind price/flow receipts against TAIEX sessions before a decision."""
import argparse
from collections import defaultdict
from datetime import date, datetime, time
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .prospective_snapshot import TAIPEI, TARGETS

GROUPS = {"Dealer_self", "Dealer_Hedging", "Foreign_Dealer_Self", "Foreign_Investor", "Investment_Trust"}


def load_rows(folder, manifest, stem, stock, dataset, cutoff):
    name = stem + ".json"
    receipt = manifest[name]
    parsed = urlsplit(receipt["url"])
    query = parse_qs(parsed.query)
    if (parsed.scheme, parsed.netloc, parsed.path) != ("https", "api.finmindtrade.com", "/api/v4/data"):
        raise ValueError(f"Unexpected source for {name}")
    if query.get("dataset") != [dataset] or query.get("data_id") != [stock]:
        raise ValueError(f"Unexpected query for {name}")
    observed = datetime.fromisoformat(receipt["observed_at"])
    if observed.utcoffset() is None or observed > cutoff:
        raise ValueError(f"{name} unavailable at cutoff")
    body = (folder / name).read_bytes()
    if hashlib.sha256(body).hexdigest() != receipt["sha256"] or receipt["http_status"] != 200:
        raise ValueError(f"Invalid receipt for {name}")
    payload = json.loads(body)
    if payload["status"] != 200 or not isinstance(payload["data"], list) or not payload["data"]:
        raise ValueError(f"Missing data for {name}")
    for row in payload["data"]:
        day = date.fromisoformat(row["date"])
        if row["stock_id"] != stock or day > observed.astimezone(TAIPEI).date():
            raise ValueError(f"Unexpected stock or date in {name}")
    return payload["data"]


def index_rows(rows, flow=False):
    indexed = defaultdict(dict)
    for row in rows:
        day = row["date"]
        key = row["name"] if flow else "quote"
        if key in indexed[day]:
            raise ValueError(f"Duplicate {key} on {day}")
        fields = ("buy", "sell") if flow else ("open", "max", "min", "close", "Trading_Volume", "Trading_money")
        values = [row[f] for f in fields]
        if not all(isinstance(n, (int, float)) and math.isfinite(n) and n >= 0 for n in values):
            raise ValueError(f"Invalid numeric value on {day}")
        if not flow and not (0 < row["min"] <= min(row["open"], row["close"]) <=
                             max(row["open"], row["close"]) <= row["max"]):
            raise ValueError(f"Invalid OHLC on {day}")
        indexed[day][key] = row
    return indexed


def inspect_window(folder, decision_day):
    cutoff = datetime.combine(decision_day, time(20), TAIPEI)
    manifest = json.loads((folder / "manifest.json").read_text())
    calendar = load_rows(folder, manifest, "taiex", "TAIEX", "TaiwanStockPrice", cutoff)
    dates = sorted(row["date"] for row in calendar)
    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate TAIEX session")
    stocks = {}
    for stock in TARGETS:
        price = index_rows(load_rows(folder, manifest, stock+"_ohlcv", stock, "TaiwanStockPrice", cutoff))
        flow = index_rows(load_rows(folder, manifest, stock+"_inst", stock, "TaiwanStockInstitutionalInvestorsBuySell", cutoff), True)
        complete = {d for d in dates if d in price and GROUPS <= flow.get(d, {}).keys()}
        windows = [dates[i-59:i+1] for i in range(59, len(dates)) if set(dates[i-59:i+1]) <= complete]
        window = windows[-1] if windows else []
        stocks[stock] = dict(last_price=max(price), last_flow=max(flow), complete_sessions=len(complete),
                             warmup_start=window[0] if window else None, warmup_end=window[-1] if window else None,
                             decision_window_complete=bool(window and window[-1] == decision_day.isoformat()))
    return dict(cutoff=cutoff.isoformat(), calendar_end=dates[-1], stocks=stocks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-day", required=True, type=date.fromisoformat)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_window(args.folder, args.decision_day), indent=2))
