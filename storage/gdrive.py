import subprocess
import tempfile
from pathlib import Path
from typing import Set
import pandas as pd

GDRIVE_ROOT = "gdrive:boatrace-predictor"


def _remote_parquet_path(category: str, year_month: str) -> str:
    """
    category: 'program' | 'results' | 'payouts'
    year_month: 'YYYY-MM'
    return: 'gdrive:boatrace-predictor/data/{category}/{YYYY}/{YYYY-MM}.parquet'
    """
    year = year_month.split("-")[0]
    return f"{GDRIVE_ROOT}/data/{category}/{year}/{year_month}.parquet"


def load_monthly_parquet(category: str, year_month: str) -> pd.DataFrame:
    """Googleドライブ上の年月別Parquetを取得してDataFrameで返す。存在しない場合は空DataFrame"""
    remote_path = _remote_parquet_path(category, year_month)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / f"{year_month}.parquet"
        res = subprocess.run(
            ["rclone", "copyto", remote_path, str(local_file)],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0 and local_file.exists():
            return pd.read_parquet(local_file)
        return pd.DataFrame()


def save_monthly_parquet(df: pd.DataFrame, category: str, year_month: str) -> None:
    """DataFrameをParquetとしてGoogleドライブへアップロードする"""
    if df.empty:
        return
    remote_path = _remote_parquet_path(category, year_month)
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / f"{year_month}.parquet"
        df.to_parquet(local_file, index=False)
        subprocess.run(
            ["rclone", "copyto", str(local_file), remote_path],
            check=True,
            capture_output=True,
            text=True,
        )


def get_existing_dates(category: str, year_month: str) -> Set[str]:
    """指定年月のParquetに既に含まれている日付 ('YYYY-MM-DD') の集合を返す"""
    df = load_monthly_parquet(category, year_month)
    if df.empty or "date" not in df.columns:
        return set()
    return set(df["date"].dropna().unique().tolist())


def upsert_monthly_parquet(
    new_df: pd.DataFrame,
    category: str,
    year_month: str,
    key_cols: list[str],
) -> int:
    """既存Parquetと結合し、key_colsで重複排除・ソートして保存する。保存後の総行数を返す"""
    if new_df.empty:
        return 0
    existing_df = load_monthly_parquet(category, year_month)
    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = new_df.drop_duplicates(subset=key_cols, keep="last")

    combined = combined.sort_values(by=key_cols).reset_index(drop=True)
    save_monthly_parquet(combined, category, year_month)
    return len(combined)
