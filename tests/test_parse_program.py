from datetime import date
from pathlib import Path
import pytest
from fetch.parse_program import parse_program_text, _slice_boat_line

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_program_text_valid_and_malformed_skipped():
    raw_text = (FIXTURES / "program_sample.txt").read_text(encoding="utf-8")
    target_date = date(2026, 11, 24)
    records = parse_program_text(raw_text, target_date)

    # 6 valid boats in 1R; preamble, short lines, non-digit racer_id, and invalid grade are skipped
    assert len(records) == 6
    assert [r.boat_no for r in records] == [1, 2, 3, 4, 5, 6]
    assert [r.racer_id for r in records] == [9001, 9002, 9003, 9004, 9005, 9006]

    b1 = records[0]
    assert b1.race_id == "20261124_01_01"
    assert b1.date == "2026-11-24"
    assert b1.venue_code == "01"
    assert b1.race_no == 1
    assert b1.race_title == "予選"
    assert b1.distance == 1800
    assert b1.age == 30
    assert b1.branch == "東京"
    assert b1.weight == pytest.approx(52.0)
    assert b1.grade == "A1"
    assert b1.nat_win_rate == pytest.approx(7.50)
    assert b1.nat_2ren_rate == pytest.approx(60.00)
    assert b1.loc_win_rate == pytest.approx(8.10)
    assert b1.loc_2ren_rate == pytest.approx(65.50)
    assert b1.motor_no == 11
    # Concatenated motor_2ren_rate ("40.00") and 3-digit boat_num ("101") without space
    assert b1.motor_2ren_rate == pytest.approx(40.00)
    assert b1.boat_num == 101
    assert b1.boat_2ren_rate == pytest.approx(35.00)

    b6 = records[5]
    assert b6.boat_no == 6
    assert b6.racer_id == 9006
    assert b6.branch == "広島"
    assert b6.grade == "B2"
    assert b6.motor_no == 16
    assert b6.boat_num == 26
    assert b6.boat_2ren_rate == pytest.approx(22.50)


def test_slice_boat_line_edge_cases():
    # Too short (< 64 bytes in cp932)
    assert _slice_boat_line("1 9001テストA 30東京52A1") is None
    # Invalid boat number 7
    assert _slice_boat_line("7 9001テストA 30東京52A1 7.50 60.00 8.10 65.50 11 40.00101 35.00") is None
    # Unencodable in cp932
    assert _slice_boat_line("1 9001🛶テスト 30東京52A1 7.50 60.00 8.10 65.50 11 40.00101 35.00") is None
    # Non-numeric float field
    assert _slice_boat_line("1 9001テストA 30東京52A1 XX.X 60.00 8.10 65.50 11 40.00101 35.00") is None
