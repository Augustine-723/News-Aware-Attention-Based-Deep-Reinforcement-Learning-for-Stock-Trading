import hashlib
import json
from datetime import date
from zipfile import ZipFile

import pytest

from stock_rl.data.prospective_snapshot import BASE, inspect_snapshots


def snapshot(path, observed, corrupt=False):
    row = {"發言日期": "1151117", "發言時間": "93001", "公司代號": "2330", "主旨 ": "synthetic"}
    raw = json.dumps([row]).encode()
    receipt = dict(status="success", url=BASE+"opendata/t187ap04_L", http_status=200,
                   observed_at=observed, sha256="bad" if corrupt else hashlib.sha256(raw).hexdigest(),
                   rows=1, source_dates=["1151117"], target_counts={"2330": 1, "2603": 0, "2881": 0})
    with ZipFile(path, "x") as z:
        z.writestr("manifest.json", json.dumps({"sources": {"disclosures": receipt}}))
        z.writestr("disclosures.json", raw)
    return path


def test_cutoff_first_observation_and_missing_market(tmp_path):
    early = snapshot(tmp_path/"a.zip", "2026-11-18T20:00:00+08:00")
    late = snapshot(tmp_path/"b.zip", "2026-11-18T20:00:01+08:00")
    cutoff = date(2026, 11, 18)
    assert not inspect_snapshots([late], cutoff)["disclosure_versions"]
    result = inspect_snapshots([late, early, early], cutoff)
    event, = result["disclosure_versions"].values()
    assert event["observed_at"] == "2026-11-18T20:00:00+08:00"
    assert event["filing_at"] == "2026-11-17T09:30:01+08:00"
    assert any(r["status"] == "missing_receipt" for r in result["receipts"])
    assert not result["market"]["2330"]["decision_day_quote"]


def test_corrupt_payload_rejected(tmp_path):
    path = snapshot(tmp_path/"c.zip", "2026-11-18T18:00:00+08:00", corrupt=True)
    with pytest.raises(ValueError, match="SHA-256"):
        inspect_snapshots([path], date(2026, 11, 18))


def test_current_receipt_format_stale_market_and_failed_disclosures(tmp_path):
    path = tmp_path/"market.zip"
    raw = json.dumps([dict(Code="2881", Date="1151119", OpeningPrice="100", ClosingPrice="101")]).encode()
    quote = dict(status="success", url=BASE+"exchangeReport/STOCK_DAY_ALL", http_status=200,
                 observed_retrieved_at="2026-11-20T15:00:00+08:00", raw_sha256=hashlib.sha256(raw).hexdigest(),
                 archive_member="q.json", response_count=1, source_dates=["1151119"],
                 target_counts={"2330": 0, "2603": 0, "2881": 1})
    failed = dict(status="failed", url=BASE+"opendata/t187ap04_L", reason="timeout")
    with ZipFile(path, "x") as z:
        z.writestr("manifest.json", json.dumps({"sources": [quote, failed]}))
        z.writestr("q.json", raw)
    result = inspect_snapshots([path], date(2026, 11, 20))
    assert result["market"]["2881"] == dict(source_days=["2026-11-19"], decision_day_quote=False)
    assert result["receipts"][1]["reason"] == "timeout"
    assert "target_counts" not in result["receipts"][1]
    earlier = inspect_snapshots([path], date(2026, 11, 19))
    assert earlier["market"]["2881"]["source_days"] == []
