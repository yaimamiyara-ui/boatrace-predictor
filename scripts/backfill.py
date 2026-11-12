import argparse
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fetch.official_files import download_and_extract_lzh
from fetch.parse_program import parse_program_text
from fetch.parse_results import parse_results_text
from storage.gdrive import get_existing_dates, upsert_monthly_parquet


def daterange(start_date: date, end_date: date):
    curr = start_date
    while curr <= end_date:
        yield curr
        curr += timedelta(days=1)


def run_backfill(start_str: str, end_str: str) -> None:
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()
    if end_dt < start_dt:
        raise ValueError("end_date must be >= start_date")

    # 年月ごとに日付リストをグループ化
    ym_dates = defaultdict(list)
    for d in daterange(start_dt, end_dt):
        ym_dates[d.strftime("%Y-%m")].append(d)

    total_prog_rows = 0
    total_res_rows = 0
    total_pay_rows = 0
    fetched_days = 0
    skipped_days = 0

    for ym, dates in ym_dates.items():
        existing_prog_dates = get_existing_dates("program", ym)
        existing_res_dates = get_existing_dates("results", ym)

        prog_list = []
        res_list = []
        pay_list = []

        for d in dates:
            d_str = d.strftime("%Y-%m-%d")
            need_b = d_str not in existing_prog_dates
            need_k = d_str not in existing_res_dates

            if not need_b and not need_k:
                skipped_days += 1
                continue

            day_fetched = False

            if need_b:
                b_text = download_and_extract_lzh(d, "B")
                if b_text:
                    b_recs = parse_program_text(b_text, d)
                    prog_list.extend([r.model_dump() for r in b_recs])
                    day_fetched = True

            if need_k:
                k_text = download_and_extract_lzh(d, "K")
                if k_text:
                    k_boats, k_pays = parse_results_text(k_text, d)
                    res_list.extend([r.model_dump() for r in k_boats])
                    pay_list.extend([r.model_dump() for r in k_pays])
                    day_fetched = True

            if day_fetched:
                fetched_days += 1
            else:
                skipped_days += 1

        if prog_list:
            df_prog = pd.DataFrame(prog_list)
            upsert_monthly_parquet(df_prog, "program", ym, ["race_id", "boat_no"])
            total_prog_rows += len(df_prog)

        if res_list:
            df_res = pd.DataFrame(res_list)
            upsert_monthly_parquet(df_res, "results", ym, ["race_id", "boat_no"])
            total_res_rows += len(df_res)

        if pay_list:
            df_pay = pd.DataFrame(pay_list)
            upsert_monthly_parquet(df_pay, "payouts", ym, ["race_id"])
            total_pay_rows += len(df_pay)

        # 月ごとの進捗（途中で止まっても、どこまで保存できたか分かるように）
        print(
            f"[MONTH DONE] {ym} program={len(prog_list)} results={len(res_list)} payouts={len(pay_list)}",
            flush=True,
        )

    print(
        f"[BACKFILL SUMMARY] range={start_str}..{end_str} "
        f"fetched_days={fetched_days} skipped_days={skipped_days} "
        f"program_rows={total_prog_rows} result_rows={total_res_rows} payout_rows={total_pay_rows}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    run_backfill(args.start_date, args.end_date)
