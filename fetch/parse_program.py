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

        # 艇行判定：Shift-JISのバイト位置で切り出す（固定長）。
        # 実ファイル確認（2026-11-10）で「モーター2連率」と「ボート番号」が
        # 空白なしで連結される行（例: 36.52106）があったため、正規表現ではなく桁位置で読む。
        fields = _slice_boat_line(line)
        if fields:
            race_id = f"{date_compact}_{current_venue}_{current_race_no:02d}"
            records.append(
                ProgramBoatRecord(
                    race_id=race_id,
                    date=date_str,
                    venue_code=current_venue,
                    race_no=current_race_no,
                    race_title=current_title,
                    distance=current_distance,
                    **fields,
                )
            )

    return records


# Bファイル艇行のバイト位置（cp932）。実ファイルで確認済み:
# "1 3876中辻崇人49福岡52A1 7.22 57.78 8.37 78.43 21 36.52106 25.87"
_B_COLS = {
    "boat_no": (0, 1),
    "racer_id": (2, 6),
    "name": (6, 14),
    "age": (14, 16),
    "branch": (16, 20),
    "weight": (20, 22),
    "grade": (22, 24),
    "nat_win_rate": (24, 29),
    "nat_2ren_rate": (29, 35),
    "loc_win_rate": (35, 40),
    "loc_2ren_rate": (40, 46),
    "motor_no": (46, 49),
    "motor_2ren_rate": (49, 55),
    "boat_num": (55, 58),
    "boat_2ren_rate": (58, 64),
}


def _slice_boat_line(line: str):
    """艇行なら各項目のdictを返す。艇行でなければNone"""
    try:
        b = line.encode("cp932")
    except UnicodeEncodeError:
        return None
    if len(b) < 64 or b[0:1] not in (b"1", b"2", b"3", b"4", b"5", b"6") or b[1:2] != b" ":
        return None

    def col(key: str) -> str:
        s, e = _B_COLS[key]
        return b[s:e].decode("cp932", errors="replace").strip()

    try:
        racer_id = col("racer_id")
        grade = col("grade")
        if not (racer_id.isdigit() and len(racer_id) == 4 and grade in ("A1", "A2", "B1", "B2")):
            return None
        return dict(
            boat_no=int(col("boat_no")),
            racer_id=int(racer_id),
            age=int(col("age")),
            branch=unicodedata.normalize("NFKC", col("branch")).replace(" ", ""),
            weight=float(col("weight")),
            grade=grade,
            nat_win_rate=float(col("nat_win_rate")),
            nat_2ren_rate=float(col("nat_2ren_rate")),
            loc_win_rate=float(col("loc_win_rate")),
            loc_2ren_rate=float(col("loc_2ren_rate")),
            motor_no=int(col("motor_no")),
            motor_2ren_rate=float(col("motor_2ren_rate")),
            boat_num=int(col("boat_num")),
            boat_2ren_rate=float(col("boat_2ren_rate")),
        )
    except ValueError:
        return None
