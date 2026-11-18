import json
from datetime import date, datetime, timezone

import pytest

from stock_rl.evaluation.public_news_probe import inspect_page


def test_cutoff_and_quarantine():
    day = date(2024, 2, 6)
    observed = datetime(2025, 1, 1, tzinfo=timezone.utc)
    stamp = int(datetime(2024, 2, 6, 12, tzinfo=timezone.utc).timestamp())
    rows = [dict(stock=["2330"], newsId=i, publishAt=stamp+i, payment=0)
            for i in range(1)]
    raw = json.dumps(dict(items=dict(current_page=1, total=48, data=rows))).encode()
    result = inspect_page(raw, day, observed)
    assert [row["published_by_2000"] for row in result["matches"]] == [True]
    assert all(row["verified_provider_available_at"] is None for row in result["matches"])
    assert result["formal_pit"] is False
    with pytest.raises(ValueError, match="interval"):
        inspect_page(raw, date(2024, 2, 7), observed)
    rows[0]["publishAt"] += 1
    late = json.dumps(dict(items=dict(current_page=1, total=48, data=rows))).encode()
    assert inspect_page(late, day, observed)["matches"] == []


def test_finmind_index_preserves_source_clock_and_checks_stock():
    from stock_rl.evaluation.finmind_news_index import inspect_index

    day = date(2024, 2, 6)
    row = dict(stock_id="2330", date="2024-02-06 10:00:00",
               link="https://example.org/synthetic", source="fixture")
    raw = json.dumps(dict(status=200, data=[row])).encode()
    assert inspect_index(raw, "2330", day)[0]["source_time_as_reported"] == row["date"]
    with pytest.raises(ValueError, match="stock or date"):
        inspect_index(raw, "2603", day)
