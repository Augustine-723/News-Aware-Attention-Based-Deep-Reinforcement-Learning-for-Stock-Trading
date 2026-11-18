"""Reproduce exact-link counts and select one publisher check per stock-day."""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

MANIFEST = Path(__file__).resolve().parents[2] / "docs/evidence/finmind_train_week_index.json"
NAMES = {"2330": "台積電", "2603": "長榮", "2881": "富邦金"}


def publisher_url(link):
    parsed = urlsplit(link)
    if parsed.hostname != "news.google.com":
        return link
    token = parsed.path.split("/")[-1]
    try:
        payload = base64.urlsafe_b64decode(token + "===")
    except ValueError:
        return link
    match = re.search(rb'https://[!-~]+', payload)
    return match[0].decode("ascii") if match else link


def summarize(directory):
    links = set()
    destinations = set()
    selected = []
    count = 0
    for cell in json.loads(MANIFEST.read_text()):
        day, stock = cell["day"], cell["stock"]
        raw = (directory / f"{day}-{stock}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != cell["raw_sha256"]:
            raise ValueError(f"snapshot changed: {day}/{stock}")
        rows = json.loads(raw)["data"]
        chosen = False
        for i, row in enumerate(rows):
            count += 1
            link = row["link"]
            url = publisher_url(link)
            links.add(link)
            destinations.add(url)
            if chosen or NAMES[stock] not in row["title"]:
                continue
            if urlsplit(url).hostname not in {"ec.ltn.com.tw", "m.cnyes.com", "news.cnyes.com"}:
                continue
            selected.append(dict(date=day, stock=stock, index=i, url=url))
            chosen = True
    return dict(index_rows=count, distinct_index_links=len(links),
                distinct_destination_keys=len(destinations), selected=selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="private YYYY-MM-DD-STOCK.json index files")
    print(json.dumps(summarize(parser.parse_args().directory), indent=2))
