"""当日データの記録（Claude追加）。

mode=schedule : 当日の開催場と締切時刻を取得して保存
mode=prerace  : 締切まで残り3〜25分のレースについて、3連単オッズと直前情報を保存（未記録のレースだけ）
mode=final    : 当日の全レースの確定3連単オッズを保存（夜に実行）

保存先（Googleドライブ）:
  data/live/schedule/YYYY/YYYY-MM-DD.parquet     race_id, venue_code, race_no, deadline
  data/live/odds_pre/YYYY/YYYY-MM-DD.parquet     race_id, combo, odds, fetched_at, minutes_to_deadline
  data/live/beforeinfo/YYYY/YYYY-MM-DD.parquet   race_id, boat_no, weight, exhibit_time, tilt, ... , fetched_at
  data/live/odds_final/YYYY/YYYY-MM-DD.parquet   race_id, combo, odds, fetched_at
"""
import argparse
import json
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fetch import live_pages
from storage.gdrive import GDRIVE_ROOT

JST = timezone(timedelta(hours=9))
WINDOW_MIN = (3, 25)  # 締切まで残り何分のレースを対象にするか


def _remote(kind: str, d: date) -> str:
    return f"{GDRIVE_ROOT}/data/live/{kind}/{d:%Y}/{d:%Y-%m-%d}.parquet"


def load_day(kind: str, d: date) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "x.parquet"
        r = subprocess.run(["rclone", "copyto", _remote(kind, d), str(local)], capture_output=True, text=True)
        if r.returncode == 0 and local.exists():
            return pd.read_parquet(local)
    return pd.DataFrame()


def append_day(kind: str, d: date, new: pd.DataFrame) -> int:
    if new.empty:
        return 0
    old = load_day(kind, d)
    df = pd.concat([old, new], ignore_index=True) if not old.empty else new
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "x.parquet"
        df.to_parquet(local, index=False)
        subprocess.run(["rclone", "copyto", str(local), _remote(kind, d)], check=True, capture_output=True, text=True)
    return len(new)


def race_id(d: date, jcd: str, rno: int) -> str:
    return f"{d:%Y%m%d}_{jcd}_{rno:02d}"


def build_schedule(d: date) -> pd.DataFrame:
    rows = []
    for jcd in live_pages.venues_today(d):
        for rno, hhmm in live_pages.deadlines(d, jcd).items():
            rows.append({"race_id": race_id(d, jcd, rno), "venue_code": jcd, "race_no": rno, "deadline": hhmm})
    return pd.DataFrame(rows)


def get_schedule(d: date) -> pd.DataFrame:
    sch = load_day("schedule", d)
    if sch.empty:
        sch = build_schedule(d)
        if not sch.empty:
            append_day("schedule", d, sch)
    return sch


def mode_schedule(d: date) -> None:
    sch = load_day("schedule", d)
    if sch.empty:
        sch = build_schedule(d)
        append_day("schedule", d, sch)
    print(f"[SCHEDULE] {d} venues={sch['venue_code'].nunique() if not sch.empty else 0} races={len(sch)}", flush=True)


def mode_prerace(d: date, now: datetime) -> None:
    sch = get_schedule(d)
    if sch.empty:
        print("[PRERACE] 本日の開催なし", flush=True)
        return
    done = load_day("odds_pre", d)
    done_ids = set(done["race_id"].unique()) if not done.empty else set()

    targets = []
    for r in sch.itertuples():
        h, m = map(int, r.deadline.split(":"))
        dl = datetime(d.year, d.month, d.day, h, m, tzinfo=JST)
        left = (dl - now).total_seconds() / 60
        if WINDOW_MIN[0] < left <= WINDOW_MIN[1] and r.race_id not in done_ids:
            targets.append((r, dl))

    odds_rows, info_rows = [], []
    for r, dl in targets:
        fetched = datetime.now(JST)
        odds = live_pages.odds3t(d, r.venue_code, r.race_no)
        info = live_pages.beforeinfo(d, r.venue_code, r.race_no)
        left = round((dl - fetched).total_seconds() / 60, 1)
        for combo, v in odds.items():
            odds_rows.append({"race_id": r.race_id, "combo": combo, "odds": v,
                              "fetched_at": fetched.isoformat(timespec="seconds"), "minutes_to_deadline": left})
        w = info.get("weather", {}) if info else {}
        se = {x["boat_no"]: x for x in info.get("start_exhibit", [])} if info else {}
        for b in (info.get("boats", []) if info else []):
            s = se.get(b["boat_no"], {})
            info_rows.append({
                "race_id": r.race_id, **b,
                "exhibit_course": s.get("course"), "exhibit_st_text": s.get("st_text"),
                "weather": w.get("weather"), "wind_speed": w.get("wind_speed"),
                "wind_dir_code": w.get("wind_dir_code"), "wave_height": w.get("wave_height"),
                "air_temp": w.get("air_temp"), "water_temp": w.get("water_temp"),
                "fetched_at": fetched.isoformat(timespec="seconds"), "minutes_to_deadline": left,
            })

    n1 = append_day("odds_pre", d, pd.DataFrame(odds_rows))
    n2 = append_day("beforeinfo", d, pd.DataFrame(info_rows))
    n3 = record_recommendations(d, pd.DataFrame(odds_rows))
    print(f"[PRERACE] races={len(targets)} odds_rows={n1} beforeinfo_rows={n2} recommendations={n3}", flush=True)


STAKE_YEN = 100


def record_recommendations(d: date, odds_df: pd.DataFrame) -> int:
    """朝の予想（1着確率）と締切前オッズから期待値を計算し、推奨買い目を保存する。
    v0の確率の正確さを確かめるまでは記録のみで、Discord通知はしない。"""
    if odds_df.empty:
        return 0
    from model import v0
    from storage.gdrive import read_remote_parquet
    pred = read_remote_parquet(f"predictions/{d:%Y}/{d:%Y-%m-%d}.parquet")
    if pred.empty:
        return 0
    rows = []
    for rid, g in odds_df.groupby("race_id"):
        pr = pred[pred["race_id"] == rid]
        if pr.empty:
            continue
        r = pr.iloc[0]
        p = [float(r[f"p{i}"]) for i in range(1, 7)]
        import numpy as np
        probs = v0.harville(np.array(p))
        odds = dict(zip(g["combo"], g["odds"]))
        for pick in v0.select_picks(probs, odds):
            rows.append({
                "race_id": rid, "model_ver": r["model_ver"], "data_window": r["data_window"],
                **pick, "stake": STAKE_YEN, "odds_fetched_at": g["fetched_at"].iloc[0],
                "minutes_to_deadline": g["minutes_to_deadline"].iloc[0],
            })
    return append_day_rel(f"recommendations/{d:%Y}/{d:%Y-%m-%d}.parquet", pd.DataFrame(rows))


def append_day_rel(rel: str, new: pd.DataFrame) -> int:
    from storage.gdrive import read_remote_parquet, write_remote_parquet
    if new.empty:
        return 0
    old = read_remote_parquet(rel)
    write_remote_parquet(pd.concat([old, new], ignore_index=True) if not old.empty else new, rel)
    return len(new)


def mode_final(d: date) -> None:
    sch = get_schedule(d)
    done = load_day("odds_final", d)
    done_ids = set(done["race_id"].unique()) if not done.empty else set()
    rows = []
    for r in sch.itertuples():
        if r.race_id in done_ids:
            continue
        fetched = datetime.now(JST).isoformat(timespec="seconds")
        for combo, v in live_pages.odds3t(d, r.venue_code, r.race_no).items():
            rows.append({"race_id": r.race_id, "combo": combo, "odds": v, "fetched_at": fetched})
    n = append_day("odds_final", d, pd.DataFrame(rows))
    print(f"[FINAL] races={len(sch) - len(done_ids)} odds_rows={n}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["schedule", "prerace", "final"])
    p.add_argument("--date", default=None, help="YYYY-MM-DD（省略時はJSTの今日）")
    a = p.parse_args()
    now = datetime.now(JST)
    d = datetime.strptime(a.date, "%Y-%m-%d").date() if a.date else now.date()
    if a.mode == "schedule":
        mode_schedule(d)
    elif a.mode == "prerace":
        mode_prerace(d, now)
    else:
        mode_final(d)


if __name__ == "__main__":
    main()
