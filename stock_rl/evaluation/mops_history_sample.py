"""Look up one historical listed-company filing and emit metadata only."""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from lxml import html

ENDPOINT = "https://mopsov.twse.com.tw/mops/web/ajax_t05st01"


def request(fields):
    with urlopen(ENDPOINT, data=urlencode(fields).encode(), timeout=40) as response:
        raw = response.read()
    tree = html.fromstring(raw.decode("utf-8", errors="replace"))
    return tree, dict(params=fields, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                      observed_at=datetime.now(timezone.utc).isoformat())


def lookup(stock, day, sequence):
    roc = f"{day.year-1911}/{day:%m/%d}"
    listing, index_receipt = request(dict(step="1", firstin="1", off="1", TYPEK="all",
                                         co_id=stock, year=str(day.year-1911), month=f"{day:%m}",
                                         b_date=f"{day:%d}", e_date=f"{day:%d}"))
    matches = []
    for tr in listing.xpath('//tr[td/input[@onclick]]'):
        cells = [cell.text_content().strip() for cell in tr.xpath('./td')]
        script = tr.xpath('./td/input/@onclick')[0]
        fields = dict(re.findall(r"t05st01_fm\.(\w+)\.value='([^']*)'", script))
        if fields.get("seq_no") == str(sequence) and fields.get("TYPEK") == "sii":
            if (cells[0] != stock or cells[2] != roc or fields.get("co_id") != stock
                    or fields.get("spoke_date") != day.strftime("%Y%m%d")
                    or fields.get("spoke_time") != cells[3].replace(":", "")):
                raise ValueError("listing identity mismatch")
            matches.append((cells, fields))
    if len(matches) != 1:
        raise ValueError("expected exactly one matching official listing row")
    cells, fields = matches[0]
    detail, detail_receipt = request(dict(firstin="1", step="2", **fields))
    labels = {}
    issuer = False
    for tr in detail.xpath("//tr"):
        values = [" ".join(cell.text_content().split()) for cell in tr.xpath("./td")]
        if len(values) == 1 and re.search(rf"上市公司\)\s*{stock}\s", values[0]):
            issuer = True
        if len(values) > 1:
            labels.update(zip(values[::2], values[1::2]))
    expected = {"序號": str(sequence), "發言日期": roc, "發言時間": cells[3],
                "主旨": " ".join(cells[4].split())}
    if not issuer or any(labels.get(k) != v for k, v in expected.items()) or not labels.get("說明"):
        raise ValueError("unconfirmed detail identity or missing body")
    published = datetime.fromisoformat(f"{day}T{cells[3]}").replace(tzinfo=ZoneInfo("Asia/Taipei"))
    return dict(stock_id=stock, sequence=sequence, headline=labels["主旨"],
                filing_datetime=published.isoformat(), endpoint=ENDPOINT,
                listing=index_receipt, detail=detail_receipt, body_chars=len(labels["說明"]),
                verified_provider_available_at=None, original_revision_verified=False,
                license_status="unverified", formal_pit=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock", choices=["2330", "2603", "2881"])
    parser.add_argument("day", type=date.fromisoformat)
    parser.add_argument("sequence", type=int)
    args = parser.parse_args()
    print(json.dumps(lookup(args.stock, args.day, args.sequence), indent=2, ensure_ascii=False))
