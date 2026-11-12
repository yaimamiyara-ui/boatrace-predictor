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
    n3 = record_recommendations(d, pd.DataFrame(odds_rows), pd.DataFrame(info_rows))
    print(f"[PRERACE] races={len(targets)} odds_rows={n1} beforeinfo_rows={n2} recommendations={n3}", flush=True)


STAKE_YEN = 100


def _load_json(remote_rel: str):
    """Googleドライブの data/ 以下のJSONを読む。無ければ None"""
    import json
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "x.json"
        r = subprocess.run(["rclone", "copyto", f"{GDRIVE_ROOT}/data/{remote_rel}", str(f)], capture_output=True, text=True)
        if r.returncode == 0 and f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    return None


def _active() -> dict:
    import json
    return json.loads((Path(__file__).resolve().parent.parent / "model" / "active.json").read_text(encoding="utf-8"))


def pre_predictions(d: date, info_df: pd.DataFrame, active: dict) -> pd.DataFrame:
    """直前情報ありモデル（v1p）で、対象レースの1着確率を出し直す。使えないときは空の表を返す"""
    from model import features, v1
    from storage.gdrive import read_remote_parquet
    if info_df.empty or not active.get("use_pre"):
        return pd.DataFrame()
    ds = d.isoformat()
    feat = read_remote_parquet(f"live/features/{d:%Y}/{ds}.parquet")
    rates = _load_json(f"live/features/{d:%Y}/{ds}_rates.json")
    if feat.empty or rates is None:
        return pd.DataFrame()
    feat = feat[feat["race_id"].isin(set(info_df["race_id"]))]
    # 展示タイムが6艇そろったレースだけ更新する
    ok = info_df.groupby("race_id")["exhibit_time"].apply(lambda s: s.notna().sum() == 6)
    feat = feat[feat["race_id"].isin(set(ok[ok].index))]
    if feat.empty:
        return pd.DataFrame()
    pre = info_df.rename(columns={"exhibit_course": "course"})[["race_id", "boat_no", "exhibit_time", "course", "wind_speed", "wave_height"]]
    pre = pre.assign(course=pre["course"].fillna(pre["boat_no"]))
    feat = features.add_pre(feat, pre, rates=rates)
    window = active.get("pre_window", "10y")
    model_dir = Path("cache") / "models" / "v1p" / window
    model_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rclone", "copy", f"{GDRIVE_ROOT}/data/models/v1p/{window}", str(model_dir)], check=True, capture_output=True)
    booster, T, _ = v1.load(model_dir)
    return v1.predict_frame(booster, T, feat, window, model_ver="v1p", has_beforeinfo=True)


def record_recommendations(d: date, odds_df: pd.DataFrame, info_df: pd.DataFrame | None = None) -> int:
    """予想（1着確率）と締切前オッズから期待値を計算し、推奨買い目を保存する。
    直前情報ありモデルが有効なら、その確率を優先する。市場に勝てることを確かめるまでは記録のみで、通知はしない。"""
    if odds_df.empty:
        return 0
    import numpy as np
    from model import market, ordering, v0
    from storage.gdrive import read_remote_parquet
    pred = read_remote_parquet(f"predictions/{d:%Y}/{d:%Y-%m-%d}.parquet")
    if pred.empty:
        return 0
    active = _active()
    try:
        pre = pre_predictions(d, info_df if info_df is not None else pd.DataFrame(), active)
    except Exception as e:  # 直前情報での更新に失敗しても、朝の予想で続ける
        print(f"[PRERACE] 直前情報での更新をスキップ: {type(e).__name__}: {e}", flush=True)
        pre = pd.DataFrame()
    if not pre.empty:
        append_day_rel(f"predictions_pre/{d:%Y}/{d:%Y-%m-%d}.parquet", pre)
    # 3連単確率の補正表（無ければHarville）
    def load_table(key):
        params = _load_json(f"models/ordering/{active[key]}.json") if active.get(key) else None
        return params.get("table") if params else None

    table_base = load_table("ordering")
    table_pre = load_table("pre_ordering") if not pre.empty else None

    rows = []
    for rid, g in odds_df.groupby("race_id"):
        src = pre[pre["race_id"] == rid] if not pre.empty else pd.DataFrame()
        pr = src if not src.empty else pred[pred["race_id"] == rid]
        if pr.empty:
            continue
        r = pr.iloc[0]
        used_pre = not src.empty
        table = (table_pre or table_base) if used_pre else table_base
        okey = ("pre_ordering" if (used_pre and table_pre) else "ordering")
        p = np.array([float(r[f"p{i}"]) for i in range(1, 7)])
        odds = {c: (None if pd.isna(o) else float(o)) for c, o in zip(g["combo"], g["odds"])}
        # 欠場艇（その艇を含む目がすべて売られていない）がいれば、確率を0にして残りで割り直す
        out = market.scratched_boats(odds)
        if len(out) >= 4:   # 3連単が成立しない／オッズ未発売
            continue
        if out:
            p = market.renormalize(p, out)
        probs = ordering.trifecta_probs_table(p, table) if table else v0.harville(p)
        for pick in v0.select_picks(probs, odds):
            rows.append({
                "race_id": rid, "model_ver": r["model_ver"], "data_window": r["data_window"],
                "ordering": active.get(okey, "") if table else "harville",
                **pick, "stake": STAKE_YEN, "scratched": ",".join(map(str, sorted(out))), "odds_fetched_at": g["fetched_at"].iloc[0],
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
