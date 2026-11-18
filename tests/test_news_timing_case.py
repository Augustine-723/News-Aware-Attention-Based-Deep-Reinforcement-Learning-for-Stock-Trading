from stock_rl.evaluation.news_timing_case import build_report


def test_real_evidence_cases_at_decision_time():
    cases = {row["stock"]: row for row in build_report()}
    shipping = cases["2603"]
    assert shipping["estimated_available_at"] == "2024-01-18T17:15:16+08:00"
    assert [r["visible_event_ids"] for r in shipping["observations"]] == [
        [], ["https://m.cnyes.com/news/id/5433463"]]
    financial = cases["2881"]
    assert financial["status"] == "index_and_publisher_dates_conflict"
    assert financial["estimated_available_at"] is None
    assert all(not r["visible_event_ids"] for r in financial["observations"])
    for case in cases.values():
        assert case["verified_provider_available_at"] is None
        assert case["observations"][-1]["next_open"] == "2024-01-19T09:00:00+08:00"
        assert all(r["text_nonzeros"] == 0 for r in case["observations"])
