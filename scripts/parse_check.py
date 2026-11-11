"""パーサーの確認用（Claude追加）。指定日のB/Kファイルを取得し、
公式ファイルの先頭行と、パース件数・パースできなかった行の例を表示する。
Googleドライブには書き込まない。"""
import argparse
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fetch.official_files import download_and_extract_lzh
from fetch.parse_program import parse_program_text, _slice_boat_line
from fetch.parse_results import parse_results_text


def show(title: str, text: str, n: int) -> None:
    print(f"===== {title}: 先頭{n}行 =====")
    for i, line in enumerate(text.splitlines()[:n]):
        print(f"{i:03d}|{line}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--lines", type=int, default=80)
    a = p.parse_args()
    d = datetime.strptime(a.date, "%Y-%m-%d").date()

    b = download_and_extract_lzh(d, "B")
    if b:
        show("B(番組表)", b, a.lines)
        recs = parse_program_text(b, d)
        venues = sorted({r.venue_code for r in recs})
        print(f"[B] parsed_boats={len(recs)} venues={len(venues)} {venues}")
        # 艇行らしいのにマッチしなかった行（先頭が1-6の数字＋空白）
        miss = [
            l for l in b.splitlines()
            if unicodedata.normalize("NFKC", l)[:2].strip() in list("123456")
            and l[1:2] == " " and _slice_boat_line(l) is None
        ]
        print(f"[B] unmatched_boat_like_lines={len(miss)}")
        for l in miss[:10]:
            print("  B-MISS|" + l)
    else:
        print("[B] no file")

    k = download_and_extract_lzh(d, "K")
    if k:
        show("K(競走成績)", k, a.lines)
        boats, pays = parse_results_text(k, d)
        print(f"[K] parsed_boats={len(boats)} parsed_races={len(pays)}")
        print(f"[K] races_with_trifecta={sum(1 for x in pays if x.trifecta_comb)}")
        print(f"[K] races_with_weather={len({x.race_id for x in boats if x.weather})}")
        # 艇数が6でないレース・3連単のないレースの生の行を表示
        from collections import Counter
        cnt = Counter(x.race_id for x in boats)
        odd = [p.race_id for p in pays if cnt.get(p.race_id, 0) != 6 or not p.trifecta_comb]
        print(f"[K] odd_races={len(odd)} {odd[:10]}")
        if odd:
            import re
            lines = k.splitlines()
            for target in odd[:4]:
                v, r = target.split("_")[1], int(target.split("_")[2])
                venue = None
                for i, l in enumerate(lines):
                    n = unicodedata.normalize("NFKC", l).strip()
                    if re.match(r"^\d{2}KBGN", n):
                        venue = n[:2]
                    if venue == v and re.match(rf"^{r}R\s+.*H\d{{4}}m", n):
                        for l2 in lines[i:i + 20]:
                            print(f"  ODD[{target}]|{l2}")
                        break
        for x in boats[:3]:
            print("  K-SAMPLE|" + x.model_dump_json())
        for x in pays[:2]:
            print("  P-SAMPLE|" + x.model_dump_json())
    else:
        print("[K] no file")


if __name__ == "__main__":
    main()
