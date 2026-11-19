"""公式の日別ファイル（番組表B・競走成績K、LZH圧縮）を読む。

GitHub Actionsからは公式サイトにアクセスしない（2026-11-18にサイト側から403で遮断されたため）。
自宅PCの pc/fetch_day.ps1 が1日1回ダウンロードしてGoogleドライブの data/raw/lzh/ に置いたファイルを読む。
"""
import io
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import lhafile

RAW_ROOT = "gdrive:boatrace-predictor/data/raw"


def download_and_extract_lzh(target_date: date, file_type: str) -> Optional[str]:
    """指定日のB(番組表)またはK(競走成績)を、Googleドライブの生ファイルから読んでテキストで返す。
    まだ置かれていなければ None（関数名は以前のまま）。"""
    t = file_type.lower()
    if t not in ("b", "k"):
        raise ValueError(f"Invalid file_type: {file_type}")
    name = f"{t}{target_date:%y%m%d}.lzh"
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / name
        r = subprocess.run(["rclone", "copyto", f"{RAW_ROOT}/lzh/{name}", str(local)], capture_output=True, text=True)
        if r.returncode != 0 or not local.exists() or local.stat().st_size == 0:
            return None
        lzh = lhafile.Lhafile(io.BytesIO(local.read_bytes()))
        names = lzh.namelist()
        if not names:
            return None
        return lzh.read(names[0]).decode("cp932", errors="replace")
