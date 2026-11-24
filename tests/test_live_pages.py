from datetime import date
from pathlib import Path
import shutil
import subprocess
import pytest
from fetch import live_pages

FIXTURES = Path(__file__).parent / "fixtures"
TARGET_DATE = date(2026, 11, 24)

# Hard-coded 120-cell 3-trifecta table in boatrace.jp DOM order (20 rows x 6 columns).
# Column c (0..5) = 1st-place boat (c+1); Row r (0..19) = r-th (2nd, 3rd) pair in lexicographical order.
EXPECTED_ODDS3T_TABLE = [
    # Row 0
    ("1-2-3", 10.0), ("2-1-3", None), ("3-1-2", 11.0), ("4-1-2", 11.5), ("5-1-2", 12.0), ("6-1-2", 12.5),
    # Row 1
    ("1-2-4", 13.0), ("2-1-4", 13.5), ("3-1-4", 14.0), ("4-1-3", 14.5), ("5-1-3", 15.0), ("6-1-3", 15.5),
    # Row 2
    ("1-2-5", 16.0), ("2-1-5", 16.5), ("3-1-5", 17.0), ("4-1-5", 17.5), ("5-1-4", 18.0), ("6-1-4", 18.5),
    # Row 3
    ("1-2-6", 19.0), ("2-1-6", 19.5), ("3-1-6", 20.0), ("4-1-6", 20.5), ("5-1-6", 21.0), ("6-1-5", 21.5),
    # Row 4
    ("1-3-2", 22.0), ("2-3-1", 22.5), ("3-2-1", 23.0), ("4-2-1", 23.5), ("5-2-1", 24.0), ("6-2-1", 24.5),
    # Row 5
    ("1-3-4", 25.0), ("2-3-4", 25.5), ("3-2-4", 26.0), ("4-2-3", 26.5), ("5-2-3", 27.0), ("6-2-3", 27.5),
    # Row 6
    ("1-3-5", 28.0), ("2-3-5", 28.5), ("3-2-5", 29.0), ("4-2-5", 29.5), ("5-2-4", 30.0), ("6-2-4", 30.5),
    # Row 7
    ("1-3-6", 31.0), ("2-3-6", 31.5), ("3-2-6", 32.0), ("4-2-6", 32.5), ("5-2-6", 33.0), ("6-2-5", 33.5),
    # Row 8
    ("1-4-2", 34.0), ("2-4-1", 34.5), ("3-4-1", 35.0), ("4-3-1", 35.5), ("5-3-1", 36.0), ("6-3-1", 36.5),
    # Row 9
    ("1-4-3", 37.0), ("2-4-3", 37.5), ("3-4-2", 38.0), ("4-3-2", 38.5), ("5-3-2", 39.0), ("6-3-2", 39.5),
    # Row 10
    ("1-4-5", 40.0), ("2-4-5", 40.5), ("3-4-5", 41.0), ("4-3-5", 41.5), ("5-3-4", 42.0), ("6-3-4", 42.5),
    # Row 11
    ("1-4-6", 43.0), ("2-4-6", 43.5), ("3-4-6", 44.0), ("4-3-6", 44.5), ("5-3-6", 45.0), ("6-3-5", 45.5),
    # Row 12
    ("1-5-2", 46.0), ("2-5-1", 46.5), ("3-5-1", 47.0), ("4-5-1", 47.5), ("5-4-1", 48.0), ("6-4-1", 48.5),
    # Row 13
    ("1-5-3", 49.0), ("2-5-3", 49.5), ("3-5-2", 50.0), ("4-5-2", 50.5), ("5-4-2", 51.0), ("6-4-2", 51.5),
    # Row 14
    ("1-5-4", 52.0), ("2-5-4", 52.5), ("3-5-4", 53.0), ("4-5-3", 53.5), ("5-4-3", 54.0), ("6-4-3", 54.5),
    # Row 15
    ("1-5-6", 55.0), ("2-5-6", 55.5), ("3-5-6", 56.0), ("4-5-6", 56.5), ("5-4-6", 57.0), ("6-4-5", 57.5),
    # Row 16
    ("1-6-2", 58.0), ("2-6-1", 58.5), ("3-6-1", 59.0), ("4-6-1", 59.5), ("5-6-1", 60.0), ("6-5-1", 60.5),
    # Row 17
    ("1-6-3", 61.0), ("2-6-3", 61.5), ("3-6-2", 62.0), ("4-6-2", 62.5), ("5-6-2", 63.0), ("6-5-2", 63.5),
    # Row 18
    ("1-6-4", 64.0), ("2-6-4", 64.5), ("3-6-4", 65.0), ("4-6-3", 65.5), ("5-6-3", 66.0), ("6-5-3", 66.5),
    # Row 19
    ("1-6-5", 67.0), ("2-6-5", 67.5), ("3-6-5", 68.0), ("4-6-5", 68.5), ("5-6-4", 69.0), ("6-5-4", 69.5),
]


@pytest.fixture(autouse=True)
def block_subprocess(monkeypatch):
    def _fail(*args, **kwargs):
        raise RuntimeError("External subprocess calls (rclone) are forbidden in unit tests")

    monkeypatch.setattr(subprocess, "run", _fail)
    monkeypatch.setattr(subprocess, "Popen", _fail)


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
    assert order == [combo for combo, _ in EXPECTED_ODDS3T_TABLE]


def test_missing_cache_files_do_not_call_subprocess(patched_cache):
    # When cache files are absent, readers must quietly return empty/False without invoking rclone/subprocess
    assert live_pages.has_file(TARGET_DATE, "index.html") is False
    assert live_pages.venues_today(TARGET_DATE) == []
    assert live_pages.deadlines(TARGET_DATE, "01") == {}
    assert live_pages.odds3t(TARGET_DATE, "01", 1, final=False) == {}
    assert live_pages.odds3t(TARGET_DATE, "01", 1, final=True) == {}
    assert live_pages.beforeinfo(TARGET_DATE, "01", 1) == {}

    # Verify the guard fixture itself triggers if sync_raw or file_times is called
    with pytest.raises(RuntimeError, match="External subprocess calls"):
        live_pages.sync_raw(TARGET_DATE)
    with pytest.raises(RuntimeError, match="External subprocess calls"):
        live_pages.file_times(TARGET_DATE)


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


def test_odds3t_independent_order_and_final_switch(patched_cache):
    # Build HTML directly from the independent EXPECTED_ODDS3T_TABLE (without calling _odds3t_order)
    pre_tds = [
        '<td class="oddsPoint">欠場</td>' if val is None else f'<td class="oddsPoint">{val:.1f}</td>'
        for _, val in EXPECTED_ODDS3T_TABLE
    ]
    pre_html = f"<html><body><table><tr>{''.join(pre_tds)}</tr></table></body></html>"
    (patched_cache / "odds3t_01_1.html").write_text(pre_html, encoding="utf-8")

    # Build a distinct final odds HTML with +100.0 offset (and 2-1-3 numeric, 6-5-4 scratched)
    final_tds = []
    for combo, val in EXPECTED_ODDS3T_TABLE:
        if combo == "2-1-3":
            final_tds.append('<td class="oddsPoint">213.0</td>')
        elif combo == "6-5-4":
            final_tds.append('<td class="oddsPoint">欠場</td>')
        else:
            final_tds.append(f'<td class="oddsPoint">{val + 100.0:.1f}</td>')
    final_html = f"<html><body><table><tr>{''.join(final_tds)}</tr></table></body></html>"
    (patched_cache / "odds3tfinal_01_1.html").write_text(final_html, encoding="utf-8")

    res_pre = live_pages.odds3t(TARGET_DATE, "01", 1, final=False)
    assert len(res_pre) == 120

    # Assert row/column boundaries and interior cells against hard-coded expected values
    # Row 0 boundaries (first col, second col, last col)
    assert res_pre["1-2-3"] == pytest.approx(10.0)
    assert res_pre["2-1-3"] is None
    assert res_pre["6-1-2"] == pytest.approx(12.5)
    # Row 1 boundaries (first col, last col)
    assert res_pre["1-2-4"] == pytest.approx(13.0)
    assert res_pre["6-1-3"] == pytest.approx(15.5)
    # Row 3 -> Row 4 transition (where 2nd boat advances from 2 to 3 in col 0)
    assert res_pre["1-2-6"] == pytest.approx(19.0)
    assert res_pre["6-1-5"] == pytest.approx(21.5)
    assert res_pre["1-3-2"] == pytest.approx(22.0)
    assert res_pre["2-3-1"] == pytest.approx(22.5)
    assert res_pre["6-2-1"] == pytest.approx(24.5)
    # Last row (Row 19) boundaries (first col, interior col, last col)
    assert res_pre["1-6-5"] == pytest.approx(67.0)
    assert res_pre["4-6-5"] == pytest.approx(68.5)
    assert res_pre["6-5-4"] == pytest.approx(69.5)

    # Full table verification against independent fixture
    for combo, expected_val in EXPECTED_ODDS3T_TABLE:
        if expected_val is None:
            assert res_pre[combo] is None
        else:
            assert res_pre[combo] == pytest.approx(expected_val)

    # Check final=True reads odds3tfinal_01_1.html and final=False reads odds3t_01_1.html
    res_final = live_pages.odds3t(TARGET_DATE, "01", 1, final=True)
    assert len(res_final) == 120
    assert res_final["1-2-3"] == pytest.approx(110.0)
    assert res_final["2-1-3"] == pytest.approx(213.0)
    assert res_final["6-1-2"] == pytest.approx(112.5)
    assert res_final["1-6-5"] == pytest.approx(167.0)
    assert res_final["6-5-4"] is None

    # Truncated file (< 120 cells) returns {}
    (patched_cache / "odds3tfinal_01_2.html").write_text(
        '<html><body><td class="oddsPoint">10.0</td></body></html>', encoding="utf-8"
    )
    assert live_pages.odds3t(TARGET_DATE, "01", 2, final=True) == {}


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
