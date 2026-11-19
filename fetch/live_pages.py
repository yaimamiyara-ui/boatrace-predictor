"""公式サイト（boatrace.jp）の当日ページ（締切時刻・3連単オッズ・直前情報）を読む。

GitHub Actionsからは公式サイトにアクセスしない。自宅PCの pc/fetch_day.ps1 が少ない回数だけ取得して
Googleドライブの data/raw/live/YYYY-MM-DD/ に置いたHTMLを、sync_raw() で手元に写してから読む。
"""
import json
import re
import subprocess
from datetime import date, datetime
from itertools import permutations
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

RAW_ROOT = "gdrive:boatrace-predictor/data/raw"
CACHE = Path("cache") / "raw"


def raw_dir(d: date) -> Path:
    return CACHE / f"{d:%Y-%m-%d}"


def sync_raw(d: date) -> Path:
    """その日の生ファイルをGoogleドライブから手元に写す"""
    local = raw_dir(d)
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rclone", "copy", f"{RAW_ROOT}/live/{d:%Y-%m-%d}", str(local), "--transfers", "8"],
                   capture_output=True, text=True)
    return local


def file_times(d: date) -> dict[str, datetime]:
    """{ファイル名: 取得時刻（Googleドライブ上の更新時刻）}"""
    r = subprocess.run(["rclone", "lsjson", f"{RAW_ROOT}/live/{d:%Y-%m-%d}"], capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    out = {}
    for it in json.loads(r.stdout or "[]"):
        try:
            out[it["Name"]] = datetime.fromisoformat(it["ModTime"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            pass
    return out


def has_file(d: date, name: str) -> bool:
    f = raw_dir(d) / name
    return f.exists() and f.stat().st_size > 0


def _file(d: date, name: str) -> Optional[BeautifulSoup]:
    if not has_file(d, name):
        return None
    return BeautifulSoup((raw_dir(d) / name).read_bytes(), "lxml")


def _hd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _f(text: str) -> Optional[float]:
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(m.group()) if m else None


def venues_today(d: date) -> list[str]:
    """当日開催の場コード一覧"""
    soup = _file(d, "index.html")
    if soup is None:
        return []
    codes = set()
    for a in soup.select('a[href*="raceindex"]'):
        m = re.search(r"jcd=(\d{2})", a.get("href", ""))
        if m:
            codes.add(m.group(1))
    return sorted(codes)


def deadlines(d: date, jcd: str) -> dict[int, str]:
    """{レース番号: 'HH:MM'} 締切予定時刻"""
    soup = _file(d, f"racelist_{jcd}.html")
    if soup is None:
        return {}
    for tr in soup.select("table tr"):
        cells = tr.find_all("td")
        if cells and "締切予定時刻" in cells[0].get_text():
            times = [c.get_text(strip=True) for c in cells[1:]]
            return {i + 1: t for i, t in enumerate(times) if re.match(r"^\d{1,2}:\d{2}$", t)}
    return {}


# 3連単オッズ表の並び：列＝1着(1〜6)、行＝(2着, 3着)の組を2着昇順・3着昇順で20行。
# 実ページ（2026-11-12）で確認：1列目1行目=1-2-3、2列目1行目=2-1-3、1列目2行目=1-2-4。
def _odds3t_order() -> list[str]:
    order = [None] * 120
    for first in range(1, 7):
        pairs = [(s, t) for s, t in permutations([b for b in range(1, 7) if b != first], 2)]
        pairs.sort()
        for row, (s, t) in enumerate(pairs):
            order[row * 6 + (first - 1)] = f"{first}-{s}-{t}"
    return order


_ODDS3T_ORDER = _odds3t_order()


def odds3t(d: date, jcd: str, rno: int, final: bool = False) -> dict[str, Optional[float]]:
    """{'1-2-3': 23.0, ...} 120通り。欠場などで売られない目はNone"""
    soup = _file(d, f"odds3t{'final' if final else ''}_{jcd}_{rno}.html")
    if soup is None:
        return {}
    cells = soup.select("td.oddsPoint")
    if len(cells) != 120:
        return {}
    out = {}
    for combo, td in zip(_ODDS3T_ORDER, cells):
        v = _f(td.get_text())
        out[combo] = v
    return out


def beforeinfo(d: date, jcd: str, rno: int) -> dict:
    """直前情報。{'boats': [{boat_no, weight, exhibit_time, tilt, parts}], 'weather': {...}, 'start_exhibit': [...]}"""
    soup = _file(d, f"beforeinfo_{jcd}_{rno}.html")
    if soup is None:
        return {}
    boats = []
    table = None
    for t in soup.find_all("table"):
        txt = t.get_text()
        if "展示" in txt and "チルト" in txt:
            table = t
            break
    if table is not None:
        for tb in table.find_all("tbody"):
            tr = tb.find("tr")
            if tr is None:
                continue
            c = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(c) < 8 or not c[0].isdigit():
                continue
            boats.append(
                {
                    "boat_no": int(c[0]),
                    "weight": _f(c[3]),
                    "exhibit_time": _f(c[4]),
                    "tilt": _f(c[5]),
                    "propeller": c[6],
                    "parts": c[7],
                }
            )

    weather = {}
    for unit in soup.select(".weather1_bodyUnit"):
        cls = " ".join(unit.get("class", []))
        txt = unit.get_text(" ", strip=True)
        if "is-weather" in cls:
            weather["weather"] = txt
        elif "is-windDirection" in cls:
            img = unit.find(class_=re.compile(r"is-wind\d+"))
            if img:
                m = re.search(r"is-wind(\d+)", " ".join(img.get("class", [])))
                weather["wind_dir_code"] = int(m.group(1)) if m else None
        elif "is-wind" in cls:
            weather["wind_speed"] = _f(txt)
        elif "is-wave" in cls:
            weather["wave_height"] = _f(txt)
        elif "is-direction" in cls:
            weather["air_temp"] = _f(txt)
        elif "is-waterTemperature" in cls:
            weather["water_temp"] = _f(txt)

    # スタート展示：進入順に艇番とSTが並ぶ（展示前は空）
    start_exhibit = []
    for course, el in enumerate(soup.select(".table1_boatImage1"), start=1):
        num = el.select_one(".table1_boatImage1Number")
        st = el.select_one(".table1_boatImage1Time")
        if num and num.get_text(strip=True).isdigit():
            st_txt = st.get_text(strip=True) if st else ""
            start_exhibit.append(
                {
                    "course": course,
                    "boat_no": int(num.get_text(strip=True)),
                    "st_text": st_txt,
                }
            )
    return {"boats": boats, "weather": weather, "start_exhibit": start_exhibit}
