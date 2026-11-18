from datetime import date

from lxml import html
import pytest

from stock_rl.evaluation import mops_history_sample as sample


@pytest.mark.parametrize("issuer,accepted", [("2330", True), ("2881", False)])
def test_details_cross_checked_against_listing(monkeypatch, issuer, accepted):
    fields = dict(seq_no="3", TYPEK="sii", co_id="2330", spoke_date="20240206", spoke_time="185524")
    script = ";".join(f"document.t05st01_fm.{key}.value='{value}'" for key, value in fields.items())
    listing = (f'<table><tr><td>2330</td><td>測試</td><td>113/02/06</td><td>18:55:24</td>'
               f'<td>合成案例</td><td><input onclick="{script}"></td></tr></table>')
    detail = (f"<table><tr><td>(上市公司) {issuer} 測試</td></tr>"
              "<tr><td>序號</td><td>3</td><td>發言日期</td><td>113/02/06</td>"
              "<td>發言時間</td><td>18:55:24</td></tr><tr><td>主旨</td><td>合成案例</td></tr>"
              "<tr><td>說明</td><td>測試說明</td></tr></table>")
    pages = iter([listing, detail])
    monkeypatch.setattr(sample, "request", lambda params: (html.fromstring(next(pages)), {}))
    if accepted:
        result = sample.lookup("2330", date(2024, 2, 6), 3)
        assert result["body_chars"] == 4 and result["formal_pit"] is False
    else:
        with pytest.raises(ValueError, match="unconfirmed detail"):
            sample.lookup("2330", date(2024, 2, 6), 3)
