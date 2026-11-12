"""v0の過去検証（Claude実装）：過去の各レースを「その時点までのデータだけ」で予想し、答え合わせする。

- コース別1着率は各月の初日時点（前日まで）の直近3年で計算
- alpha は検証開始日より前の3か月で決め、期間中は固定
保存先: data/backtest/v0/settled.parquet
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model import v0
from model.evaluate import calibration_table, ece, settle_predictions
from storage.gdrive import read_category, sync_category, write_remote_parquet


def month_starts(start: date, end: date):
    d = date(start.year, start.month, 1)
    while d <= end:
        yield d
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    a = p.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)

    years = list(range(start.year - v0.COURSE_YEARS - 1, end.year + 1))
    sync_category("results", years)
    sync_category("program", years)
    results_all = read_category("results", f"{start.year - v0.COURSE_YEARS - 1}-01-01", a.end)
    program_all = read_category("program", (start - timedelta(days=92)).isoformat(), a.end)

    fit_start = (start - timedelta(days=92)).isoformat()
    # alphaを決める期間（開始前92日）の結果を含めないよう、その期間の初日時点で計算（Geminiレビューで修正）
    rates0 = v0.course_win_rates(results_all, fit_start)
    fit_prog = program_all[(program_all["date"] >= fit_start) & (program_all["date"] < a.start)]
    fit_res = results_all[(results_all["date"] >= fit_start) & (results_all["date"] < a.start)]
    alpha = v0.fit_alpha(fit_prog, fit_res, rates0)
    print(f"[BACKTEST] alpha={alpha} (fit {fit_start}..{a.start})")

    parts = []
    for ms in month_starts(start, end):
        me = min(date(ms.year + (ms.month == 12), ms.month % 12 + 1, 1) - timedelta(days=1), end)
        lo = max(ms, start).isoformat()
        rates = v0.course_win_rates(results_all, ms.isoformat())
        prog = program_all[(program_all["date"] >= lo) & (program_all["date"] <= me.isoformat())]
        if prog.empty:
            continue
        pred = v0.predict_program(prog, rates, alpha)
        res = results_all[(results_all["date"] >= lo) & (results_all["date"] <= me.isoformat())]
        s = settle_predictions(pred, res)
        parts.append(s)
        print(f"[BACKTEST] {ms:%Y-%m} races={len(s)} logloss={s['logloss'].mean():.4f} brier={s['brier'].mean():.4f}", flush=True)

    settled = pd.concat(parts, ignore_index=True)
    write_remote_parquet(settled, "backtest/v0/settled.parquet")
    cal = calibration_table(settled)
    print(f"[BACKTEST] total races={len(settled)} logloss={settled['logloss'].mean():.4f} "
          f"brier={settled['brier'].mean():.4f} ECE={ece(cal):.4f}")
    print(cal.to_string(index=False))


if __name__ == "__main__":
    main()
