"""自動答え合わせ（Claude実装）。毎朝、前日分を答え合わせし、集計してダッシュボードとDiscordの要約を出す。

保存先:
  data/settled/predictions/YYYY/YYYY-MM-DD.parquet
  data/settled/recommendations/YYYY/YYYY-MM-DD.parquet
  data/dashboard/index.html
"""
import argparse
import html
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.evaluate import calibration_table, ece, settle_predictions, settle_recommendations
from scripts.backfill import run_backfill
from storage.gdrive import GDRIVE_ROOT, CACHE_DIR, read_category, read_remote_parquet, sync_category, write_remote_parquet

JST = timezone(timedelta(hours=9))


def load_all_settled(kind: str) -> pd.DataFrame:
    local = CACHE_DIR / "settled" / kind
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rclone", "copy", f"{GDRIVE_ROOT}/data/settled/{kind}", str(local)], capture_output=True, text=True)
    files = sorted(local.glob("*/*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def summarize_recs(r: pd.DataFrame) -> dict:
    if r.empty:
        return {"n": 0, "stake": 0, "payout": 0, "roi": None, "hit": 0}
    stake, payout = r["stake"].sum(), r["payout"].sum()
    return {"n": len(r), "stake": int(stake), "payout": int(payout),
            "roi": payout / stake if stake else None, "hit": int((r["outcome"] == "hit").sum())}


def build_dashboard(live: pd.DataFrame, recs: pd.DataFrame, bt: pd.DataFrame) -> str:
    def sec_model(name, s):
        if s.empty:
            return f"<h2>{name}</h2><p>まだデータがありません</p>"
        cal = calibration_table(s)
        rows = "".join(
            f"<tr><td>{r.pred:.3f}</td><td>{r.actual:.3f}</td><td>{int(r.n)}</td></tr>" for r in cal.itertuples()
        )
        return (f"<h2>{name}</h2><p>レース数 {len(s):,}　LogLoss {s['logloss'].mean():.4f}　"
                f"ブライアスコア {s['brier'].mean():.4f}　ECE {ece(cal):.4f}</p>"
                f"<details><summary>確率の当たり具合（20区間）</summary><table><tr><th>予想確率</th><th>実際の1着率</th><th>艇数</th></tr>{rows}</table></details>")

    rec_html = "<h2>推奨買い目（記録のみ・1点100円換算）</h2>"
    if recs.empty:
        rec_html += "<p>まだ記録がありません</p>"
    else:
        recs = recs.assign(month=recs["race_id"].str[:6])
        t = recs.groupby("month").apply(lambda g: pd.Series(summarize_recs(g))).reset_index()
        rec_html += "<table><tr><th>月</th><th>点数</th><th>的中</th><th>投資</th><th>払戻</th><th>回収率</th></tr>" + "".join(
            f"<tr><td>{r.month}</td><td>{r.n}</td><td>{r.hit}</td><td>{r.stake:,}</td><td>{r.payout:,}</td>"
            f"<td>{'' if r.roi is None else f'{r.roi:.1%}'}</td></tr>" for r in t.itertuples()) + "</table>"
        bins = pd.cut(recs["ev"], [1.15, 1.3, 1.5, 2, 3, 1e9], right=False)
        t2 = recs.groupby(bins, observed=True).apply(lambda g: pd.Series(summarize_recs(g))).reset_index()
        rec_html += "<h3>期待値の帯ごと</h3><table><tr><th>期待値</th><th>点数</th><th>的中</th><th>回収率</th></tr>" + "".join(
            f"<tr><td>{html.escape(str(r.ev))}</td><td>{r.n}</td><td>{r.hit}</td>"
            f"<td>{'' if r.roi is None else f'{r.roi:.1%}'}</td></tr>" for r in t2.itertuples()) + "</table>"

    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html lang="ja"><meta charset="utf-8"><title>ボートレース予想 ダッシュボード</title>
<style>body{{font-family:sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ccc;padding:4px 10px;text-align:right}}</style>
<h1>ボートレース予想 ダッシュボード</h1><p>更新 {now}</p>
{sec_model("運用中の予想（v0）", live)}{rec_html}{sec_model("過去検証（v0・その時点までのデータだけで予想）", bt)}
</html>"""


def notify_discord(text: str) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    requests.post(url, json={"content": text[:1900]}, timeout=20)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD（省略時はJSTの前日）")
    a = p.parse_args()
    d = datetime.strptime(a.date, "%Y-%m-%d").date() if a.date else (datetime.now(JST) - timedelta(days=1)).date()
    ds = d.isoformat()

    run_backfill(ds, ds)  # 前日の成績がまだ無ければ取得
    sync_category("results", [d.year])
    sync_category("payouts", [d.year])
    results = read_category("results", ds, ds)
    payouts = read_category("payouts", ds, ds)

    pred = read_remote_parquet(f"predictions/{d:%Y}/{ds}.parquet")
    sp = settle_predictions(pred, results) if not pred.empty else pd.DataFrame()
    if not sp.empty:
        write_remote_parquet(sp, f"settled/predictions/{d:%Y}/{ds}.parquet")

    recs = read_remote_parquet(f"recommendations/{d:%Y}/{ds}.parquet")
    sr = settle_recommendations(recs, payouts) if not recs.empty else pd.DataFrame()
    if not sr.empty:
        write_remote_parquet(sr, f"settled/recommendations/{d:%Y}/{ds}.parquet")

    live = load_all_settled("predictions")
    allrecs = load_all_settled("recommendations")
    bt = read_remote_parquet("backtest/v0/settled.parquet")
    page = build_dashboard(live, allrecs, bt)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "index.html"
        f.write_text(page, encoding="utf-8")
        subprocess.run(["rclone", "copyto", str(f), f"{GDRIVE_ROOT}/data/dashboard/index.html"], check=True, capture_output=True)

    day = summarize_recs(sr)
    tot = summarize_recs(allrecs)
    lines = [f"【答え合わせ {ds}】"]
    if not sp.empty:
        lines.append(f"予想 {len(sp)}レース　LogLoss {sp['logloss'].mean():.3f}　ブライア {sp['brier'].mean():.3f}")
    else:
        lines.append("予想の記録なし")
    if day["n"]:
        lines.append(f"推奨 {day['n']}点（的中{day['hit']}）回収率 {day['roi']:.0%}")
    else:
        lines.append("推奨なし")
    if tot["n"]:
        lines.append(f"累計 推奨{tot['n']}点 回収率 {tot['roi']:.0%}")
    lines.append("※v0は精度確認中のため、推奨は記録のみです")
    notify_discord("\n".join(lines))
    print("[SETTLE] " + " / ".join(lines[1:]))


if __name__ == "__main__":
    main()
