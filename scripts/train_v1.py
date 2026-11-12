"""v1の学習と検証（Claude実装）。

期間の分け方（時系列。未来のデータで過去を予想しない）:
  学習   : 各パターンの開始日 〜 valid開始の前日
  検証   : valid_start 〜 test開始の前日（学習の止め時と温度Tを決める）
  テスト : test_start 〜 test_end（v0の過去検証と同じ期間で比べる）
学習期間は 3年・5年・10年・全期間 の4パターンを同じテストで比べる。
保存先: data/models/v1/{パターン}/ と data/backtest/v1_{パターン}/settled.parquet
"""
import argparse
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))

from model import features, v1
from model.evaluate import calibration_table, ece, settle_predictions
from storage.gdrive import GDRIVE_ROOT, read_category, read_remote_parquet, sync_category, write_remote_parquet

FIRST_DATA = "2012-01-01"


def shift_years(d: str, years: int) -> str:
    y, rest = int(d[:4]) - years, d[4:]
    return f"{y}{rest}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--valid-start", default="2025-05-01")
    p.add_argument("--test-start", default="2025-11-01")
    p.add_argument("--test-end", default="2026-11-10")
    p.add_argument("--windows", default="3y,5y,10y,all")
    a = p.parse_args()

    starts = {"3y": shift_years(a.valid_start, 3), "5y": shift_years(a.valid_start, 5),
              "10y": shift_years(a.valid_start, 10), "all": "2013-01-01"}
    windows = [w for w in a.windows.split(",") if w in starts]
    earliest = min(starts[w] for w in windows)

    sync_category("results")
    sync_category("program")
    results = read_category("results", FIRST_DATA, a.test_end, columns=features.RESULT_COLS)
    program = read_category("program", earliest, a.test_end, columns=features.PROGRAM_COLS)
    print(f"[V1] loaded results={len(results):,} program={len(program):,}", flush=True)

    feat = features.build(program, results)
    data = features.labels(feat, results)
    del feat, program
    print(f"[V1] dataset rows={len(data):,} races={len(data)//6:,} features={len(features.FEATURES)}", flush=True)

    valid = data[(data["date"] >= a.valid_start) & (data["date"] < a.test_start)]
    test = data[(data["date"] >= a.test_start) & (data["date"] <= a.test_end)]
    res_test = results[(results["date"] >= a.test_start) & (results["date"] <= a.test_end)]

    v0s = read_remote_parquet("backtest/v0/settled.parquet")
    summary = []
    settled_by = {}
    for w in windows:
        tr = data[(data["date"] >= starts[w]) & (data["date"] < a.valid_start)]
        print(f"[V1] --- window={w} train={starts[w]}..{a.valid_start} races={len(tr)//6:,}", flush=True)
        booster = v1.train(tr, valid)
        T = v1.fit_temperature(v1.raw_scores(booster, valid), valid["y"].to_numpy())
        pred = v1.predict_frame(booster, T, test, w)
        s = settle_predictions(pred, res_test)
        settled_by[w] = s
        cal = calibration_table(s)
        row = {"window": w, "train_races": len(tr) // 6, "best_iter": booster.best_iteration, "T": T,
               "test_races": len(s), "logloss": s["logloss"].mean(), "brier": s["brier"].mean(), "ece": ece(cal)}
        summary.append(row)
        print(f"[V1] {row}", flush=True)
        print(cal.to_string(index=False), flush=True)

        imp = pd.Series(booster.feature_importance("gain"), index=features.FEATURES).sort_values(ascending=False)
        print("[V1] 重要な特徴量 上位12:", ", ".join(f"{k}({v/imp.sum():.0%})" for k, v in imp.head(12).items()), flush=True)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / w
            v1.save(booster, T, {"window": w, "train_start": starts[w], "valid_start": a.valid_start,
                                 "test_start": a.test_start, "test_end": a.test_end,
                                 "trained_on": date.today().isoformat(), **{k: float(row[k]) for k in ("logloss", "brier", "ece")}}, out)
            subprocess.run(["rclone", "copy", str(out), f"{GDRIVE_ROOT}/data/models/v1/{w}"], check=True, capture_output=True)
        write_remote_parquet(s, f"backtest/v1_{w}/settled.parquet")

    print("\n[V1] ===== まとめ（テスト期間 %s〜%s）=====" % (a.test_start, a.test_end))
    print(pd.DataFrame(summary).to_string(index=False))

    def paired(name_a, sa, name_b, sb):
        m = sa[["race_id", "logloss"]].merge(sb[["race_id", "logloss"]], on="race_id", suffixes=("_a", "_b"))
        d = m["logloss_a"] - m["logloss_b"]
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        try:
            pw = stats.wilcoxon(d).pvalue
        except ValueError:
            pw = float("nan")
        print(f"[V1] {name_a} vs {name_b}: races={len(m):,} LogLoss {m['logloss_a'].mean():.4f} vs {m['logloss_b'].mean():.4f} "
              f"差={d.mean():+.4f} t={t:+.1f} Wilcoxon p={pw:.2g}")

    if not v0s.empty:
        for w in windows:
            paired(f"v1_{w}", settled_by[w], "v0", v0s)
    base = "5y" if "5y" in settled_by else windows[0]
    for w in windows:
        if w != base:
            paired(f"v1_{w}", settled_by[w], f"v1_{base}", settled_by[base])


if __name__ == "__main__":
    main()
