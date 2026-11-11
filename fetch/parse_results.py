import re
import unicodedata
from datetime import date
from typing import List, Optional, Tuple
from common.schema import ResultBoatRecord, PayoutRaceRecord

# レースヘッダ＋気象行 (NFKC正規化後):
# 例: " 1R 予選 H1800m 晴 風 北東 3m 波 2cm"
# TODO: 要確認 - 順延・進入固定表記や無風("無風 0m")時のスペース位置
RACE_WEATHER_RE = re.compile(
    r"^\s*(\d{1,2})R\s+(.+?)\s+H(\d{4})m\s+([^\s]+)\s+風\s+([^\s]+)\s+(\d+)m\s+波\s+(\d+)cm"
)

# 成績艇行 (NFKC正規化後):
# 例: " 01 1 4320 峰 竜太 12 34 6.75 1 0.12 1.48.5"
# TODO: 要確認 - 欠場・失格時に展示タイムや進入コース、STが空欄になる場合のトークン構成
RESULT_BOAT_RE = re.compile(
    r"^\s*(0[1-6]|[FLKS]\d?)\s+"     # 1: 着順コード
    r"([1-6])\s+"                    # 2: 艇番
    r"(\d{4})\s+"                    # 3: 登録番号
    r".+?\s+"                        # 選手名
    r"(\d{1,3})\s+"                  # 4: モーター番号
    r"(\d{1,3})\s+"                  # 5: ボート番号
    r"(\d+\.\d+|\.+|-+)?\s*"         # 6: 展示タイム
    r"([1-6]|\.)?\s*"                # 7: 進入コース
    r"([FL]?\d+\.\d+|\.+|-+)?\s*"    # 8: スタートタイミング
    r"([0-9.]+)?"                    # 9: レースタイム (例: 1.48.5)
)

TRIFECTA_RE = re.compile(r"3連単\s+([1-6]-[1-6]-[1-6])\s+(\d+)\s+人気\s*(\d+)?")
EXACTA_RE = re.compile(r"2連単\s+([1-6]-[1-6])\s+(\d+)\s+人気\s*(\d+)?")


def _parse_race_time(time_str: Optional[str]) -> Optional[float]:
    """'1.48.5' -> 108.5秒 に変換"""
    if not time_str:
        return None
    parts = time_str.strip(".").split(".")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) * 0.1
    return None


def _parse_st(st_raw: Optional[str], finish_code: str) -> Tuple[Optional[float], str]:
    """ST文字列と着順コードから (st_time, st_status) を返す。Fはマイナス値に変換"""
    if finish_code.startswith("K"):
        return None, "K"
    if finish_code.startswith("L"):
        return None, "L"
    if not st_raw or st_raw.startswith(".") or st_raw.startswith("-"):
        return None, "F" if finish_code.startswith("F") else "NORMAL"

    if st_raw.startswith("F"):
        val = float(st_raw[1:])
        return -val, "F"
    if st_raw.startswith("L"):
        return None, "L"
    try:
        val = float(st_raw)
        if finish_code.startswith("F"):
            return -val, "F"
        return val, "NORMAL"
    except ValueError:
        return None, "NORMAL"


def parse_results_text(
    raw_text: str, target_date: date
) -> Tuple[List[ResultBoatRecord], List[PayoutRaceRecord]]:
    """Kファイル解凍テキストをパースして (ResultBoatRecordリスト, PayoutRaceRecordリスト) を返す"""
    boat_records: List[ResultBoatRecord] = []
    payout_records: List[PayoutRaceRecord] = []

    date_str = target_date.strftime("%Y-%m-%d")
    date_compact = target_date.strftime("%Y%m%d")

    current_venue: Optional[str] = None
    current_race_no: Optional[int] = None
    current_weather = ""
    current_wind_dir = ""
    current_wind_speed: Optional[int] = None
    current_wave: Optional[int] = None
    current_trick = ""

    current_race_boats: List[ResultBoatRecord] = []
    current_payout: Optional[PayoutRaceRecord] = None

    def flush_race():
        nonlocal current_race_boats, current_payout
        if current_payout is not None:
            refund_list = [
                str(b.boat_no)
                for b in current_race_boats
                if b.st_status in ("F", "L", "K")
            ]
            current_payout.refund_boats = ",".join(sorted(set(refund_list)))
            current_payout.winning_trick = current_trick
            for b in current_race_boats:
                if b.finish_order == 1:
                    b.winning_trick = current_trick
            boat_records.extend(current_race_boats)
            payout_records.append(current_payout)
        current_race_boats = []
        current_payout = None

    for raw_line in raw_text.splitlines():
        norm_line = unicodedata.normalize("NFKC", raw_line.rstrip("\r\n"))

        # 場ブロック開始 (例: 01KBGN)
        m_venue = re.match(r"^(\d{2})KBGN", norm_line.strip())
        if m_venue:
            flush_race()
            current_venue = m_venue.group(1)
            current_race_no = None
            continue

        # 場ブロック終了 (例: 01KEND)
        if re.match(r"^(\d{2})KEND", norm_line.strip()):
            flush_race()
            current_venue = None
            current_race_no = None
            continue

        if not current_venue:
            continue

        # レースヘッダ・気象行
        m_weather = RACE_WEATHER_RE.match(norm_line)
        if m_weather:
            flush_race()
            current_race_no = int(m_weather.group(1))
            current_weather = m_weather.group(4)
            current_wind_dir = m_weather.group(5)
            current_wind_speed = int(m_weather.group(6))
            current_wave = int(m_weather.group(7))
            current_trick = ""
            race_id = f"{date_compact}_{current_venue}_{current_race_no:02d}"
            current_payout = PayoutRaceRecord(
                race_id=race_id,
                date=date_str,
                venue_code=current_venue,
                race_no=current_race_no,
            )
            continue

        # 気象情報が欠落または特殊表記のレースヘッダフォールバック (中止・不成立等)
        m_race_fallback = re.match(r"^\s*(\d{1,2})R\s+", norm_line)
        if m_race_fallback and "H" in norm_line and "m" in norm_line:
            flush_race()
            current_race_no = int(m_race_fallback.group(1))
            current_weather = ""
            current_wind_dir = ""
            current_wind_speed = None
            current_wave = None
            current_trick = ""
            race_id = f"{date_compact}_{current_venue}_{current_race_no:02d}"
            current_payout = PayoutRaceRecord(
                race_id=race_id,
                date=date_str,
                venue_code=current_venue,
                race_no=current_race_no,
                is_aborted=("中止" in norm_line or "不成立" in norm_line),
            )
            continue

        if current_race_no is None or current_payout is None:
            continue

        # 決まり手：実ファイルでは列見出し行の末尾にある
        # 例（NFKC後）: "着 艇 登番 選 手 名 モーター ボート 展示 進入 スタートタイミング レースタイム 逃げ"
        if "レースタイム" in norm_line:
            current_trick = norm_line.split("レースタイム", 1)[1].strip()
            continue

        # 決まり手行 (例: "決まり手 逃げ")
        if "決まり手" in norm_line:
            parts = norm_line.split("決まり手")
            if len(parts) > 1:
                current_trick = parts[1].strip()
            continue

        # 艇成績行
        m_boat = RESULT_BOAT_RE.match(norm_line)
        if m_boat:
            finish_code = m_boat.group(1).strip()
            boat_no = int(m_boat.group(2))
            racer_id = int(m_boat.group(3))
            motor_no = int(m_boat.group(4))
            boat_num = int(m_boat.group(5))

            ex_raw = m_boat.group(6)
            exhibit_time = None
            if ex_raw and re.match(r"^\d+\.\d+$", ex_raw):
                exhibit_time = float(ex_raw)

            course_raw = m_boat.group(7)
            course = int(course_raw) if (course_raw and course_raw.isdigit()) else None

            st_raw = m_boat.group(8)
            st_time, st_status = _parse_st(st_raw, finish_code)

            finish_order = int(finish_code) if finish_code.isdigit() else None
            race_time_sec = _parse_race_time(m_boat.group(9))

            race_id = f"{date_compact}_{current_venue}_{current_race_no:02d}"
            current_race_boats.append(
                ResultBoatRecord(
                    race_id=race_id,
                    date=date_str,
                    venue_code=current_venue,
                    race_no=current_race_no,
                    boat_no=boat_no,
                    racer_id=racer_id,
                    motor_no=motor_no,
                    boat_num=boat_num,
                    exhibit_time=exhibit_time,
                    course=course,
                    st_time=st_time,
                    st_status=st_status,
                    finish_order=finish_order,
                    finish_status=finish_code,
                    race_time_sec=race_time_sec,
                    weather=current_weather,
                    wind_dir=current_wind_dir,
                    wind_speed=current_wind_speed,
                    wave_height=current_wave,
                )
            )
            continue

        # 払戻行: 3連単
        m_tri = TRIFECTA_RE.search(norm_line)
        if m_tri:
            current_payout.trifecta_comb = m_tri.group(1)
            current_payout.trifecta_payout = int(m_tri.group(2))
            if m_tri.group(3):
                current_payout.trifecta_popularity = int(m_tri.group(3))
            continue

        # 払戻行: 2連単
        m_exa = EXACTA_RE.search(norm_line)
        if m_exa:
            current_payout.exacta_comb = m_exa.group(1)
            current_payout.exacta_payout = int(m_exa.group(2))
            if m_exa.group(3):
                current_payout.exacta_popularity = int(m_exa.group(3))
            continue

    flush_race()
    return boat_records, payout_records
