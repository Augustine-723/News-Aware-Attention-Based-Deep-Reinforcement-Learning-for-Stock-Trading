from datetime import date, timedelta

import pytest

from stock_rl.data import market_warmup as warm


def test_missing_flow_session_breaks_window(tmp_path, monkeypatch):
    days = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(61)]
    price = [dict(date=d, open=1, close=2, min=1, max=2, Trading_Volume=100, Trading_money=150) for d in days]
    flow = [dict(date=d, name=n, buy=1, sell=0) for d in days for n in warm.GROUPS]
    (tmp_path/"manifest.json").write_text("{}")
    monkeypatch.setattr(warm, "load_rows", lambda folder, manifest, stem, stock, dataset, cutoff:
                        flow if stem.endswith("_inst") else price)
    day = date.fromisoformat(days[-1])
    assert warm.inspect_window(tmp_path, day)["stocks"]["2330"]["decision_window_complete"]
    flow.pop()
    out = warm.inspect_window(tmp_path, day)["stocks"]["2330"]
    assert not out["decision_window_complete"] and out["warmup_end"] == days[-2]
    flow.pop(20)
    assert warm.inspect_window(tmp_path, day)["stocks"]["2330"]["warmup_end"] is None


def test_late_receipt_rejected_before_reading_payload(tmp_path):
    receipt = dict(url="https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id=2330",
                   observed_at="2026-11-20T20:00:01+08:00")
    cutoff = warm.datetime.fromisoformat("2026-11-20T20:00:00+08:00")
    with pytest.raises(ValueError, match="cutoff"):
        warm.load_rows(tmp_path, {"price.json": receipt}, "price", "2330", "TaiwanStockPrice", cutoff)
