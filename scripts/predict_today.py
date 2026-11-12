"""朝の予想（Claude実装）：当日の番組表を取得し、v0で各レースの1着確率を出して保存する。

保存先: data/predictions/YYYY/YYYY-MM-DD.parquet
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fetch.official_files import download_and_extract_lzh
from fetch.parse_program import parse_program_text
from model import v0
from storage.gdrive import read_category, sync_category, upsert_monthly_parquet, write_remote_parquet

JST = timezone(timedelta(hours=9))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None)
    a = p.parse_args()
    d = datetime.strptime(a.date, "%Y-%m-%d").date() if a.date else datetime.now(JST).date()
    ds = d.isoformat()

    text = download_and_extract_lzh(d, "B")
    if not text:
        print(f"[PREDICT] {ds} 番組表なし（未公開または開催なし）")
        return
    prog = pd.DataFrame([r.model_dump() for r in parse_program_text(text, d)])
    upsert_monthly_parquet(prog, "program", ds[:7], ["race_id", "boat_no"])

    years = list(range(d.year - v0.COURSE_YEARS - 1, d.year + 1))
    sync_category("results", years)
    sync_category("program", years)
    start3y = f"{d.year - v0.COURSE_YEARS}{ds[4:]}"
    results = read_category("results", start3y, ds)
    rates = v0.course_win_rates(results, ds)

    fit_start = (d - timedelta(days=92)).isoformat()
    fit_end = (d - timedelta(days=1)).isoformat()
    rates_fit = v0.course_win_rates(results, fit_start)  # alphaを決める期間の結果を含めない
    alpha = v0.fit_alpha(read_category("program", fit_start, fit_end), read_category("results", fit_start, fit_end), rates_fit)

    pred = v0.predict_program(prog, rates, alpha)
    write_remote_parquet(pred, f"predictions/{d:%Y}/{ds}.parquet")
    print(f"[PREDICT] {ds} races={len(pred)} alpha={alpha}")


if __name__ == "__main__":
    main()
