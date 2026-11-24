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


def test_parse_results_weather_header_then_void_race():
    # Race 1 has a winning trick ("逃げ") and normal payouts.
    # Race 2 has a normal header with weather, followed by a separate "不成立" line,
    # 5 flying boats (F), and 1 boat (boat 6) that did not foul ("00") and only lacks a finish order.
    raw_text = "\n".join([
        "01KBGN",
        " 1R 予選 H1800m 晴 風 北東 3m 波 2cm",
        "着 艇 登番 選 手 名 モーター ボート 展示 進入 スタートタイミング レースタイム 逃げ",
        " 01 1 9001 テスト選手A 11 21 6.75 1 0.12 1.48.5",
        " 02 2 9002 テスト選手B 12 22 6.80 2 0.15 1.49.8",
        " 03 3 9003 テスト選手C 13 23 6.82 3 0.18 1.51.0",
        " 04 4 9004 テスト選手D 14 24 6.79 4 0.21 1.52.3",
        " 05 5 9005 テスト選手E 15 25 6.85 5 0.19 .",
        " 06 6 9006 テスト選手F 16 26 6.90 6 0.25 .",
        " 3連単 1-2-3 1250 人気 4",
        " 2連単 1-2 430 人気 2",
        " 2R 予選 H1800m 雨 風 北 6m 波 5cm",
        " レース不成立",
        " F  1 9021 テスト選手M 51 61 6.70 1 F0.03 .",
        " F  2 9022 テスト選手N 52 62 6.72 2 F0.01 .",
        " F  3 9023 テスト選手O 53 63 6.74 3 F0.02 .",
        " F  4 9024 テスト選手P 54 64 6.76 4 F0.04 .",
        " F  5 9025 テスト選手Q 55 65 6.78 5 F0.01 .",
        " 00 6 9026 テスト選手R 56 66 6.81 6 0.15 .",
        "01KEND",
    ])
    target_date = date(2026, 11, 24)
    boats, payouts = parse_results_text(raw_text, target_date)

    assert len(payouts) == 2
    p2 = payouts[1]
    assert p2.race_id == "20261124_01_02"
    assert p2.is_aborted is True
    # No winning trick carried over from Race 1
    assert p2.winning_trick == ""
    # Empty payouts
    assert p2.trifecta_comb == ""
    assert p2.trifecta_payout == 0
    assert p2.trifecta_popularity is None
    assert p2.exacta_comb == ""
    assert p2.exacta_payout == 0
    assert p2.exacta_popularity is None
    # Boats 1..5 (F) are in refund_boats; boat 6 ("00", NORMAL st_status) is not
    assert p2.refund_boats == "1,2,3,4,5"

    r2_boats = {b.boat_no: b for b in boats if b.race_no == 2}
    assert len(r2_boats) == 6

    for bno in (1, 2, 3, 4, 5):
        b = r2_boats[bno]
        assert b.finish_order is None
        assert b.finish_status == "F"
        assert b.st_status == "F"
        assert b.st_time is not None and b.st_time < 0
        assert b.winning_trick == ""
        assert b.weather == "雨"
        assert b.wind_dir == "北"
        assert b.wind_speed == 6
        assert b.wave_height == 5

    # Boat 6 did not foul: lacks a finish order (00 -> None), but has NORMAL st_status and is not refunded
    b6 = r2_boats[6]
    assert b6.racer_id == 9026
    assert b6.finish_order is None
    assert b6.finish_status == "00"
    assert b6.st_status == "NORMAL"
    assert b6.st_time == pytest.approx(0.15)
    assert b6.exhibit_time == pytest.approx(6.81)
    assert b6.course == 6
    assert b6.winning_trick == ""
    assert b6.weather == "雨"
    assert b6.wind_dir == "北"
    assert b6.wind_speed == 6
    assert b6.wave_height == 5
