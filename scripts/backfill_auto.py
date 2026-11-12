"""過去データの自動継続取得（Claude追加）。

1時間ごとに起動し、まだ取り終えていない月を新しい順に探して取得する。
取り終えた月は gdrive:boatrace-predictor/data/_done/YYYY-MM という印を置き、次回からは飛ばす。
制限時間（既定45分）を過ぎたら、新しい月には手を付けずに終了し、次回の起動で続きを取る。
"""
import argparse
import subprocess
import sys
import time
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.backfill import run_backfill
from storage.gdrive import GDRIVE_ROOT

JST = timezone(timedelta(hours=9))
DONE_DIR = f"{GDRIVE_ROOT}/data/_done"
FIRST_MONTH = date(2012, 1, 1)


def done_months() -> set[str]:
    res = subprocess.run(["rclone", "lsf", DONE_DIR], capture_output=True, text=True)
    if res.returncode != 0:
        return set()
    return {line.strip().rstrip("/") for line in res.stdout.splitlines() if line.strip()}


def mark_done(ym: str) -> None:
    subprocess.run(["rclone", "touch", f"{DONE_DIR}/{ym}"], check=True, capture_output=True, text=True)


def all_months(last_day: date) -> list[str]:
    """FIRST_MONTH から last_day を含む月まで、新しい順"""
    months = []
    y, m = last_day.year, last_day.month
    while date(y, m, 1) >= FIRST_MONTH:
        months.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return months


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget-min", type=float, default=45)
    a = p.parse_args()

    started = time.monotonic()
    yesterday = (datetime.now(JST) - timedelta(days=1)).date()
    current_month = f"{yesterday.year:04d}-{yesterday.month:02d}"
    done = done_months()
    todo = [ym for ym in all_months(yesterday) if ym not in done]
    print(f"[AUTO] done_months={len(done)} remaining_months={len(todo)}", flush=True)

    for ym in todo:
        if (time.monotonic() - started) / 60 > a.budget_min:
            print("[AUTO] 制限時間に達したため終了（次回続きから）", flush=True)
            break
        y, m = map(int, ym.split("-"))
        start = date(y, m, 1)
        end = min(date(y, m, monthrange(y, m)[1]), yesterday)
        run_backfill(start.isoformat(), end.isoformat())
        # 今月はまだ日が残っているので印を付けない（毎晩の取得と次回の自動取得で埋める）
        if ym != current_month:
            mark_done(ym)
            print(f"[AUTO] {ym} 完了", flush=True)

    left = len([ym for ym in all_months(yesterday) if ym not in done_months()])
    print(f"[AUTO] 残り {left} か月", flush=True)


if __name__ == "__main__":
    main()
