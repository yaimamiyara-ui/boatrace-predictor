from datetime import date
from pathlib import Path
import pytest
from fetch.parse_results import parse_results_text, _parse_race_time, _parse_st

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_race_time():
    assert _parse_race_time("1.48.5") == pytest.approx(108.5)
    assert _parse_race_time("2.01.0") == pytest.approx(121.0)
    assert _parse_race_time(".") is None
    assert _parse_race_time("") is None
    assert _parse_race_time(None) is None
    assert _parse_race_time("1.48") is None


def test_parse_st():
    assert _parse_st("0.12", "01") == (pytest.approx(0.12), "NORMAL")
    assert _parse_st("F0.02", "F") == (pytest.approx(-0.02), "F")
    assert _parse_st("0.03", "F") == (pytest.approx(-0.03), "F")
    assert _parse_st(".", "F") == (None, "F")
    assert _parse_st("L", "L") == (None, "L")
    assert _parse_st("0.10", "K0") == (None, "K")
    assert _parse_st(".", "05") == (None, "NORMAL")


def test_parse_results_text_normal_f_l_k_and_payouts():
    raw_text = (FIXTURES / "results_sample.txt").read_text(encoding="utf-8")
    target_date = date(2026, 11, 24)
    boats, payouts = parse_results_text(raw_text, target_date)

    assert len(payouts) == 3
    assert len(boats) == 12

    # Race 1: normal race
    p1 = payouts[0]
    assert p1.race_id == "20261124_01_01"
    assert p1.winning_trick == "逃げ"
    assert p1.refund_boats == ""
    assert p1.trifecta_comb == "1-2-3"
    assert p1.trifecta_payout == 1250
    assert p1.trifecta_popularity == 4
    assert p1.exacta_comb == "1-2"
    assert p1.exacta_payout == 430
    assert p1.exacta_popularity == 2
    assert p1.is_aborted is False

    r1_boats = [b for b in boats if b.race_no == 1]
    assert [b.finish_order for b in r1_boats] == [1, 2, 3, 4, 5, 6]
    assert r1_boats[0].winning_trick == "逃げ"
    assert r1_boats[1].winning_trick == ""
    assert r1_boats[0].st_time == pytest.approx(0.12)
    assert r1_boats[0].st_status == "NORMAL"
    assert r1_boats[0].race_time_sec == pytest.approx(108.5)
    assert r1_boats[0].weather == "晴"
    assert r1_boats[0].wind_dir == "北東"
    assert r1_boats[0].wind_speed == 3
    assert r1_boats[0].wave_height == 2
    assert r1_boats[4].race_time_sec is None

    # Race 2: F (boat 1), L (boat 5), and scratched K0 (boat 6)
    p2 = payouts[1]
    assert p2.race_id == "20261124_01_02"
    assert p2.winning_trick == "差し"
    assert p2.refund_boats == "1,5,6"
    assert p2.trifecta_comb == "2-3-4"
    assert p2.trifecta_payout == 4800
    assert p2.trifecta_popularity == 18

    r2_by_boat = {b.boat_no: b for b in boats if b.race_no == 2}
    assert r2_by_boat[2].finish_order == 1
    assert r2_by_boat[2].winning_trick == "差し"

    b_f = r2_by_boat[1]
    assert b_f.finish_order is None
    assert b_f.finish_status == "F"
    assert b_f.st_status == "F"
    assert b_f.st_time == pytest.approx(-0.02)

    b_l = r2_by_boat[5]
    assert b_l.finish_order is None
    assert b_l.finish_status == "L"
    assert b_l.st_status == "L"
    assert b_l.st_time is None

    b_k = r2_by_boat[6]
    assert b_k.finish_order is None
    assert b_k.finish_status == "K0"
    assert b_k.st_status == "K"
    assert b_k.st_time is None
    assert b_k.exhibit_time is None
    assert b_k.course is None

    # Race 3: aborted race fallback header
    p3 = payouts[2]
    assert p3.race_id == "20261124_01_03"
    assert p3.is_aborted is True
