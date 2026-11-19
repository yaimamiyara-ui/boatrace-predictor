"""当日データの記録（Claude実装）。公式サイトにはアクセスせず、自宅PCがGoogleドライブに置いた生ファイルを読む。

mode=schedule : 開催場と締切時刻を保存し、締切前オッズを取るレース（1日20、無作為）を決めて監視リストを書く
mode=prerace  : 監視リストのレースのうち、自宅PCが直前情報とオッズを取得済みのものを保存し、推奨を記録（1レース1回）
mode=final    : 監視リストのレースの確定3連単オッズを保存

保存先（Googleドライブ data/live/）:
  schedule/YYYY/YYYY-MM-DD.parquet     race_id, venue_code, race_no, deadline, watch
  odds_pre/YYYY/YYYY-MM-DD.parquet     race_id, combo, odds, fetched_at, minutes_to_deadline
  beforeinfo/YYYY/YYYY-MM-DD.parquet   race_id, boat_no, weight, exhibit_time, tilt, ... , fetched_at
  odds_final/YYYY/YYYY-MM-DD.parquet   race_id, combo, odds, fetched_at
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


WATCH_PER_DAY = 20          # 1日に締切前オッズを取るレース数（公式サイトへのアクセスを抑えるため）
WATCH_FROM = "08:50"        # これより前に締切のレースは対象外（PCが監視リストを受け取る時間が要るため）


def build_schedule(d: date) -> pd.DataFrame:
    """自宅PCが置いた index.html / racelist_*.html から、当日の全レースの締切時刻を作る"""
    rows = []
    for jcd in live_pages.venues_today(d):
        for rno, hhmm in live_pages.deadlines(d, jcd).items():
            rows.append({"race_id": race_id(d, jcd, rno), "venue_code": jcd, "race_no": rno, "deadline": hhmm})
    return pd.DataFrame(rows)


def choose_watch(sch: pd.DataFrame, d: date) -> pd.Series:
    """締切前オッズを取るレースを無作為に選ぶ（日付で固定した乱数。偏りなく検証するため）"""
    ok = sch[sch["deadline"].str.zfill(5) >= WATCH_FROM]
    pick = ok.sample(n=min(WATCH_PER_DAY, len(ok)), random_state=int(f"{d:%Y%m%d}")) if len(ok) else ok
    return sch["race_id"].isin(set(pick["race_id"]))


def get_schedule(d: date) -> pd.DataFrame:
    sch = load_day("schedule", d)
    if not sch.empty and "watch" not in sch.columns:   # 旧形式（全レース取得していた頃）の予定表
        sch["watch"] = False
    return sch


def mode_schedule(d: date) -> None:
    sch = get_schedule(d)
    if not sch.empty:
        print(f"[SCHEDULE] {d} 作成済み races={len(sch)} watch={int(sch['watch'].sum())}", flush=True)
        return
    live_pages.sync_raw(d)
    sch = build_schedule(d)
    if sch.empty or not all(live_pages.has_file(d, f"racelist_{v}.html") for v in live_pages.venues_today(d)):
        print(f"[SCHEDULE] {d} 自宅PCからの開催情報がまだありません（次回もう一度確認）", flush=True)
        return
    sch["watch"] = choose_watch(sch, d)
    append_day("schedule", d, sch)
    # 自宅PCが読む監視リスト
    w = sch[sch["watch"]].sort_values("deadline")
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "watchlist.csv"
        w.assign(jcd=w["venue_code"], rno=w["race_no"])[["jcd", "rno", "deadline"]].to_csv(f, index=False, encoding="utf-8")
        subprocess.run(["rclone", "copyto", str(f), f"{live_pages.RAW_ROOT}/live/{d:%Y-%m-%d}/watchlist.csv"],
                       check=True, capture_output=True, text=True)
    print(f"[SCHEDULE] {d} venues={sch['venue_code'].nunique()} races={len(sch)} watch={len(w)}", flush=True)


def _deadline(d: date, hhmm: str) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime(d.year, d.month, d.day, h, m, tzinfo=JST)


def mode_prerace(d: date, now: datetime) -> None:
    sch = get_schedule(d)
    if sch.empty:
        print("[PRERACE] 本日の予定がまだありません", flush=True)
        return
    live_pages.sync_raw(d)
    times = live_pages.file_times(d)
    done = load_day("odds_pre", d)
    done_ids = set(done["race_id"].unique()) if not done.empty else set()

    targets = []
    for r in sch[sch["watch"]].itertuples():
        of, bf = f"odds3t_{r.venue_code}_{r.race_no}.html", f"beforeinfo_{r.venue_code}_{r.race_no}.html"
        if r.race_id not in done_ids and live_pages.has_file(d, of) and live_pages.has_file(d, bf):
            targets.append((r, _deadline(d, r.deadline), times.get(of)))

    odds_rows, info_rows = [], []
    for r, dl, got in targets:
        info = live_pages.beforeinfo(d, r.venue_code, r.race_no)
        fetched = got.astimezone(JST) if got else datetime.now(JST)   # 自宅PCが取得した時刻
        odds = live_pages.odds3t(d, r.venue_code, r.race_no)
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
    n_ex = int(info_df.groupby("race_id")["exhibit_time"].apply(lambda s: s.notna().sum() == 6).sum()) if info_df is not None and not info_df.empty else 0
    print(f"[PRERACE] 直前情報で更新したレース={len(pre)}（展示タイムがそろっていたレース={n_ex}、対象={odds_df['race_id'].nunique()}）", flush=True)
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
    if sch.empty:
        print("[FINAL] 予定なし", flush=True)
        return
    live_pages.sync_raw(d)
    times = live_pages.file_times(d)
    done = load_day("odds_final", d)
    done_ids = set(done["race_id"].unique()) if not done.empty else set()
    rows, n = [], 0
    for r in sch[sch["watch"]].itertuples():
        name = f"odds3tfinal_{r.venue_code}_{r.race_no}.html"
        if r.race_id in done_ids or not live_pages.has_file(d, name):
            continue
        got = times.get(name)
        fetched = (got.astimezone(JST) if got else datetime.now(JST)).isoformat(timespec="seconds")
        n += 1
        for combo, v in live_pages.odds3t(d, r.venue_code, r.race_no, final=True).items():
            rows.append({"race_id": r.race_id, "combo": combo, "odds": v, "fetched_at": fetched})
    k = append_day("odds_final", d, pd.DataFrame(rows))
    print(f"[FINAL] races={n} odds_rows={k}", flush=True)


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
