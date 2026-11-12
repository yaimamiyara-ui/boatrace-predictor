"""基準モデル v0（Claude実装）。

1着確率 p_i ∝ その場のコース別1着率(コース_i) × 全国勝率_i ^ alpha
  - 朝の予想ではコース＝枠番（枠なり）とみなす
  - コース別1着率は「予想日の前日まで」の直近3年分から計算（未来の情報は使わない）
  - alpha は予想期間より前の3か月で、勝った艇の対数尤度が最大になる値を選ぶ
3連単確率は Harville 方式で計算する。
"""
from itertools import permutations
from math import log

import numpy as np
import pandas as pd

MODEL_VER = "v0"
DATA_WINDOW = "3y"
COURSE_YEARS = 3
PRIOR_STRENGTH = 200  # 場別のコース1着率を全国平均へ寄せる強さ（レース数換算）


def course_win_rates(results: pd.DataFrame, asof: str) -> dict:
    """{venue_code: [1コース〜6コースの1着率]}。asof(YYYY-MM-DD)の前日までの直近3年"""
    start = str(int(asof[:4]) - COURSE_YEARS) + asof[4:]
    r = results[(results["date"] < asof) & (results["date"] >= start)]
    r = r[r["course"].between(1, 6)]
    if r.empty:
        flat = [1 / 6] * 6
        return {"_all": flat}
    r = r.assign(win=(r["finish_order"] == 1).astype(float))
    nat = r.groupby("course")["win"].mean().reindex(range(1, 7)).fillna(1 / 6).to_numpy()
    out = {"_all": list(nat)}
    for v, g in r.groupby("venue_code"):
        cnt = g.groupby("course")["win"].count().reindex(range(1, 7)).fillna(0).to_numpy()
        wins = g.groupby("course")["win"].sum().reindex(range(1, 7)).fillna(0).to_numpy()
        out[v] = list((wins + PRIOR_STRENGTH * nat) / (cnt + PRIOR_STRENGTH))
    return out


def race_probs(boats: pd.DataFrame, rates: dict, alpha: float) -> np.ndarray:
    """boats: 1レース分（boat_no, venue_code, nat_win_rate, 任意で course）。艇番1〜6順の1着確率"""
    p = np.zeros(6)
    for b in boats.itertuples():
        course = int(getattr(b, "course", 0) or b.boat_no)
        cw = rates.get(b.venue_code, rates["_all"])[course - 1]
        strength = max(float(b.nat_win_rate), 0.5) ** alpha
        p[int(b.boat_no) - 1] = cw * strength
    s = p.sum()
    return p / s if s > 0 else p


def fit_alpha(program: pd.DataFrame, results: pd.DataFrame, rates: dict) -> float:
    """勝った艇の対数尤度が最大になる alpha を格子探索で選ぶ"""
    winners = results[results["finish_order"] == 1][["race_id", "boat_no"]]
    win_map = dict(zip(winners["race_id"], winners["boat_no"]))
    races = [(rid, g) for rid, g in program.groupby("race_id") if rid in win_map and len(g) == 6]
    if not races:
        return 1.0
    best, best_ll = 1.0, -1e18
    for alpha in np.arange(0.0, 4.01, 0.25):
        ll = 0.0
        for rid, g in races:
            p = race_probs(g, rates, alpha)
            ll += log(max(p[win_map[rid] - 1], 1e-9))
        if ll > best_ll:
            best, best_ll = float(alpha), ll
    return best


def harville(p: np.ndarray) -> dict[str, float]:
    """1着確率（艇番1〜6）から3連単120通りの確率"""
    out = {}
    for a, b, c in permutations(range(6), 3):
        pa, pb, pc = p[a], p[b], p[c]
        d1 = 1 - pa
        d2 = 1 - pa - pb
        if pa <= 0 or d1 <= 0 or d2 <= 0:
            out[f"{a+1}-{b+1}-{c+1}"] = 0.0
            continue
        out[f"{a+1}-{b+1}-{c+1}"] = pa * (pb / d1) * (pc / d2)
    return out


def predict_program(program: pd.DataFrame, rates: dict, alpha: float) -> pd.DataFrame:
    """番組表（1行=1艇）から1レース1行の予想を作る"""
    rows = []
    for rid, g in program.groupby("race_id"):
        p = race_probs(g, rates, alpha)
        first = g.iloc[0]
        row = {
            "race_id": rid, "date": first["date"], "venue_code": first["venue_code"],
            "race_no": int(first["race_no"]), "model_ver": MODEL_VER, "data_window": DATA_WINDOW,
            "alpha": alpha, "has_beforeinfo": False, "n_boats": len(g),
        }
        for i in range(6):
            row[f"p{i+1}"] = float(p[i])
        rows.append(row)
    return pd.DataFrame(rows)


# ---- 買い目の選択（設計書の基準） ----
EV_MIN = 1.15
P_MIN = 0.01
ODDS_MAX = 500.0
MAX_PICKS = 5


def select_picks(combo_probs: dict[str, float], odds: dict[str, float]) -> list[dict]:
    cands = []
    for combo, pr in combo_probs.items():
        o = odds.get(combo)
        if o is None or o <= 0 or pr < P_MIN or o > ODDS_MAX:
            continue
        ev = pr * o
        if ev >= EV_MIN:
            cands.append({"combo": combo, "prob": float(pr), "odds": float(o), "ev": float(ev)})
    cands.sort(key=lambda x: x["ev"], reverse=True)
    picks = cands[:MAX_PICKS]
    # 合成オッズ（=1/Σ(1/オッズ)）が1倍未満なら、期待値の低い目から外す
    while picks and 1 / sum(1 / x["odds"] for x in picks) < 1.0:
        picks.pop()
    return picks
