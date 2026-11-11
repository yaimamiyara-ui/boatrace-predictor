import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.backfill import run_backfill

JST = timezone(timedelta(hours=9))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-date",
        default=None,
        help="YYYY-MM-DD (省略時はJST基準の前日)",
    )
    args = parser.parse_args()

    if args.target_date:
        target_str = args.target_date
    else:
        now_jst = datetime.now(JST)
        yesterday = (now_jst - timedelta(days=1)).date()
        target_str = yesterday.strftime("%Y-%m-%d")

    run_backfill(target_str, target_str)


if __name__ == "__main__":
    main()
