from datetime import date
from pathlib import Path
import shutil
import pytest
from fetch import live_pages

FIXTURES = Path(__file__).parent / "fixtures"
TARGET_DATE = date(2026, 11, 24)


@pytest.fixture
def patched_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(live_pages, "CACHE", tmp_path)
    day_dir = tmp_path / "2026-11-24"
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def test_pure_helpers():
    assert live_pages._hd(TARGET_DATE) == "20261124"
    assert live_pages._f("18.5℃") == pytest.approx(18.5)
    assert live_pages._f("-0.5") == pytest.approx(-0.5)
    assert live_pages._f("欠場") is None
    assert live_pages._f("") is None

    order = live_pages._odds3t_order()
    assert len(order) == 120
    assert len(set(order)) == 120
    # 1st row across 6 columns: 1-2-3, 2-1-3, 3-1-2, 4-1-2, 5-1-2, 6-1-2
    assert order[0] == "1-2-3"
    assert order[1] == "2-1-3"
    assert order[6] == "1-2-4"


def test_venues_today_and_has_file(patched_cache):
    assert live_pages.has_file(TARGET_DATE, "index.html") is False
    assert live_pages.venues_today(TARGET_DATE) == []

    shutil.copyfile(FIXTURES / "index.html", patched_cache / "index.html")
    assert live_pages.has_file(TARGET_DATE, "index.html") is True
    assert live_pages.venues_today(TARGET_DATE) == ["01", "12"]


def test_deadlines(patched_cache):
    assert live_pages.deadlines(TARGET_DATE, "01") == {}
    shutil.copyfile(FIXTURES / "racelist_01.html", patched_cache / "racelist_01.html")
    # 3R has "-" so it is excluded; 1R, 2R, 4R are returned with 1-based index keys
    assert live_pages.deadlines(TARGET_DATE, "01") == {
        1: "10:30",
        2: "10:58",
        4: "11:55",
    }


def test_odds3t(patched_cache):
    # Generate 120 synthetic oddsPoint cells where 1-2-3 is 12.5, 2-1-3 is scratched ("欠場"), etc.
    order = live_pages._odds3t_order()
    tds = []
    for i, combo in enumerate(order):
        if combo == "2-1-3":
            tds.append('<td class="oddsPoint">欠場</td>')
        else:
            tds.append(f'<td class="oddsPoint">{10.0 + i * 0.5:.1f}</td>')
    html = f"<html><body><table><tr>{''.join(tds)}</tr></table></body></html>"
    (patched_cache / "odds3t_01_1.html").write_text(html, encoding="utf-8")

    res = live_pages.odds3t(TARGET_DATE, "01", 1)
    assert len(res) == 120
    assert res["1-2-3"] == pytest.approx(10.0)
    assert res["2-1-3"] is None
    assert res["1-2-4"] == pytest.approx(10.0 + 6 * 0.5)

    # Fewer than 120 cells returns {}
    (patched_cache / "odds3tfinal_01_1.html").write_text(
        '<html><body><td class="oddsPoint">10.0</td></body></html>', encoding="utf-8"
    )
    assert live_pages.odds3t(TARGET_DATE, "01", 1, final=True) == {}


def test_beforeinfo(patched_cache):
    shutil.copyfile(FIXTURES / "beforeinfo_01_1.html", patched_cache / "beforeinfo_01_1.html")
    info = live_pages.beforeinfo(TARGET_DATE, "01", 1)

    assert len(info["boats"]) == 2
    assert info["boats"][0] == {
        "boat_no": 1,
        "weight": pytest.approx(52.0),
        "exhibit_time": pytest.approx(6.75),
        "tilt": pytest.approx(-0.5),
        "propeller": "新",
        "parts": "ピストン×2",
    }
    assert info["boats"][1]["boat_no"] == 2
    assert info["boats"][1]["parts"] == "なし"

    assert info["weather"] == {
        "weather": "晴",
        "wind_dir_code": 5,
        "wind_speed": pytest.approx(3.0),
        "wave_height": pytest.approx(2.0),
        "air_temp": pytest.approx(18.5),
        "water_temp": pytest.approx(17.0),
    }

    assert info["start_exhibit"] == [
        {"course": 1, "boat_no": 1, "st_text": ".12"},
        {"course": 2, "boat_no": 2, "st_text": "F.01"},
    ]
