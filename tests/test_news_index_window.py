from datetime import date

from stock_rl.evaluation.news_index_window import measure


def test_zero_requires_successful_index_response():
    day = date(2023, 9, 1)
    empty = measure("2881", day, lambda url: b'{"status":200,"data":[]}')
    failed = measure("2881", day, lambda url: b'{"status":429,"data":[]}')
    assert (empty["status"], empty["rows"]) == ("empty_index", 0)
    assert (failed["status"], failed["rows"]) == ("unknown", None)
