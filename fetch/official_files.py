import io
import time
from datetime import date
from typing import Optional
import requests
import lhafile

BASE_URL_B = "https://www1.mbrace.or.jp/od2/B/{yyyy}{mm}/b{yy}{mm}{dd}.lzh"
BASE_URL_K = "https://www1.mbrace.or.jp/od2/K/{yyyy}{mm}/k{yy}{mm}{dd}.lzh"


def download_and_extract_lzh(
    target_date: date,
    file_type: str,
    sleep_sec: float = 3.2,
    max_retries: int = 3,
) -> Optional[str]:
    """
    指定日のB(番組表)またはK(競走成績)のLZHをダウンロードし、解凍後のテキスト(str)を返す。
    404(開催なし・未公開)の場合はNoneを返す。
    """
    yyyy = target_date.strftime("%Y")
    yy = target_date.strftime("%y")
    mm = target_date.strftime("%m")
    dd = target_date.strftime("%d")

    if file_type.upper() == "B":
        url = BASE_URL_B.format(yyyy=yyyy, yy=yy, mm=mm, dd=dd)
    elif file_type.upper() == "K":
        url = BASE_URL_K.format(yyyy=yyyy, yy=yy, mm=mm, dd=dd)
    else:
        raise ValueError(f"Invalid file_type: {file_type}")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BoatracePredictor/1.0)"
    }

    for attempt in range(1, max_retries + 1):
        time.sleep(sleep_sec)
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()

            bio = io.BytesIO(resp.content)
            lzh = lhafile.Lhafile(bio)
            namelist = lzh.namelist()
            if not namelist:
                return None

            raw_bytes = lzh.read(namelist[0])
            return raw_bytes.decode("cp932", errors="replace")

        except requests.RequestException:
            if attempt == max_retries:
                raise
            time.sleep(sleep_sec * attempt)

    return None
