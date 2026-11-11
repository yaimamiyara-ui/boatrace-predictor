import re
import unicodedata
from datetime import date
from typing import List
from common.schema import ProgramBoatRecord

VENUE_BLOCK_RE = re.compile(r"^(\d{2})BBGN$", re.MULTILINE)

# 1艇行の正規表現:
# 例: "1 4320峰 竜太39佐賀52A1 8.45 65.12 8.90 70.00 12 45.20 34 38.10 ..."
# TODO: 要確認 - 氏名4文字の後ろにスペースが入らない固定長連結パターンのため、
# 数字境界で年齢(2桁)・支部(漢字2〜3文字)・体重(2桁)・級別(A1/A2/B1/B2)を抽出
BOAT_LINE_RE = re.compile(
    r"^\s*([1-6])\s+"                # 1: 枠番
    r"(\d{4})"                       # 2: 登録番号(4桁)
    r"[^\d]+?"                       # 選手名(非数字)
    r"(\d{2})"                       # 3: 年齢(2桁)
    r"([^\d\s]{2,3})"                # 4: 支部(2〜3文字)
    r"(\d{2})"                       # 5: 体重(2桁)
    r"([AB][12])\s+"                 # 6: 級別
    r"(\d+\.\d+)\s+"                 # 7: 全国勝率
    r"(\d+\.\d+)\s+"                 # 8: 全国2連率
    r"(\d+\.\d+)\s+"                 # 9: 当地勝率
    r"(\d+\.\d+)\s+"                 # 10: 当地2連率
    r"(\d+)\s+"                      # 11: モーター番号
    r"(\d+\.\d+)\s+"                 # 12: モーター2連率
    r"(\d+)\s+"                      # 13: ボート番号
    r"(\d+\.\d+)"                    # 14: ボート2連率
)


def _normalize_line(line: str) -> str:
    return line.rstrip("\r\n")


def parse_program_text(raw_text: str, target_date: date) -> List[ProgramBoatRecord]:
    """Bファイル解凍テキストをパースしてProgramBoatRecordのリストを返す"""
    records: List[ProgramBoatRecord] = []
    date_str = target_date.strftime("%Y-%m-%d")
    date_compact = target_date.strftime("%Y%m%d")

    current_venue = None
    current_race_no = None
    current_title = ""
    current_distance = 1800

    for raw_line in raw_text.splitlines():
        line = _normalize_line(raw_line)
        norm_line = unicodedata.normalize("NFKC", line)

        # 場ブロック開始判定 (例: 01BBGN, 24BBGN)
        m_venue = re.match(r"^(\d{2})BBGN", norm_line.strip())
        if m_venue:
            current_venue = m_venue.group(1)
            current_race_no = None
            continue

        if not current_venue:
            continue

        # レースヘッダ判定 (例: " 1R 予選 H1800m 電話投票締切予定10:30")
        m_race = re.match(r"^\s*(\d{1,2})R\s+(.+?)\s+H(\d{4})m", norm_line)
        if m_race:
            current_race_no = int(m_race.group(1))
            current_title = m_race.group(2).strip()
            current_distance = int(m_race.group(3))
            continue

        if current_race_no is None:
            continue

        # 艇行判定 (NFKC正規化後の文字列でマッチング)
        m_boat = BOAT_LINE_RE.match(norm_line)
        if m_boat:
            boat_no = int(m_boat.group(1))
            race_id = f"{date_compact}_{current_venue}_{current_race_no:02d}"
            records.append(
                ProgramBoatRecord(
                    race_id=race_id,
                    date=date_str,
                    venue_code=current_venue,
                    race_no=current_race_no,
                    race_title=current_title,
                    distance=current_distance,
                    boat_no=boat_no,
                    racer_id=int(m_boat.group(2)),
                    age=int(m_boat.group(3)),
                    branch=m_boat.group(4),
                    weight=float(m_boat.group(5)),
                    grade=m_boat.group(6),
                    nat_win_rate=float(m_boat.group(7)),
                    nat_2ren_rate=float(m_boat.group(8)),
                    loc_win_rate=float(m_boat.group(9)),
                    loc_2ren_rate=float(m_boat.group(10)),
                    motor_no=int(m_boat.group(11)),
                    motor_2ren_rate=float(m_boat.group(12)),
                    boat_num=int(m_boat.group(13)),
                    boat_2ren_rate=float(m_boat.group(14)),
                )
            )

    return records
