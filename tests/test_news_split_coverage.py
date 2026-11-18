from stock_rl.evaluation.news_split_coverage import build_report


def test_estimated_split_coverage_and_holiday_rollover():
    report = build_report()
    shipping = [r for r in report["coverage"] if r["stock"] == "2603"]
    assert [r["candidate_articles"] for r in shipping] == [1, 1, 1]
    assert [r["executable_decisions"] for r in shipping] == [82, 21, 33]
    holiday = next(r for r in report["ledger"] if r["url"].endswith("/5447931"))
    assert holiday["first_decision"] == "2024-02-15T20:00:00+08:00"
    assert holiday["next_open"] == "2024-02-16T09:00:00+08:00"
    assert all("phase" not in r for r in report["ledger"] if r["status"] != "estimated_candidate")
