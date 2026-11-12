"""1着確率から3連単の確率を作る（Claude実装）。

Harville方式は「2着・3着も1着確率の比で決まる」と仮定するが、実際は本命艇が2・3着に残る確率はそれより低く、
人気薄の艇が2・3着に入る確率は高い。そこで、2着・3着を決めるときだけ確率をべき乗（l2, l3）してから比を取る。
l2 = l3 = 1 がHarvilleと同じ。1より小さいほど、人気薄が2・3着に入りやすい形になる。
l2, l3 は過去データで、実際の2着・3着が最も起きやすくなる値を選ぶ（fit）。
"""
from itertools import permutations

import numpy as np


def trifecta_probs(p, l2: float = 1.0, l3: float = 1.0) -> dict[str, float]:
    """艇番1〜6の1着確率から、3連単120通りの確率"""
    p = np.asarray(p, dtype=float)
    q2 = np.where(p > 0, np.power(np.clip(p, 1e-12, None), l2), 0.0)
    q3 = np.where(p > 0, np.power(np.clip(p, 1e-12, None), l3), 0.0)
    out = {}
    for a, b, c in permutations(range(6), 3):
        d2 = q2.sum() - q2[a]
        d3 = q3.sum() - q3[a] - q3[b]
        if p[a] <= 0 or d2 <= 0 or d3 <= 0:
            out[f"{a+1}-{b+1}-{c+1}"] = 0.0
        else:
            out[f"{a+1}-{b+1}-{c+1}"] = float(p[a] * q2[b] / d2 * q3[c] / d3)
    return out


def winning_combo_prob(P: np.ndarray, idx: np.ndarray, l2: float = 1.0, l3: float = 1.0) -> np.ndarray:
    """P: (レース数, 6) の1着確率、idx: (レース数, 3) の実際の1・2・3着（0始まり）。当たり目の確率を返す"""
    r = np.arange(len(P))
    Pc = np.clip(P, 1e-12, None)
    Q2, Q3 = Pc ** l2, Pc ** l3
    a, b, c = idx[:, 0], idx[:, 1], idx[:, 2]
    second = Q2[r, b] / (Q2.sum(axis=1) - Q2[r, a])
    third = Q3[r, c] / (Q3.sum(axis=1) - Q3[r, a] - Q3[r, b])
    return P[r, a] * second * third


def fit(P: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """2着・3着それぞれの対数尤度が最大になる l2, l3 を選ぶ"""
    r = np.arange(len(P))
    Pc = np.clip(P, 1e-12, None)
    a, b, c = idx[:, 0], idx[:, 1], idx[:, 2]
    grid = np.round(np.arange(0.30, 1.31, 0.02), 2)
    best2 = best3 = 1.0
    ll2 = ll3 = -np.inf
    for l in grid:
        Q = Pc ** l
        s2 = np.log(Q[r, b] / (Q.sum(axis=1) - Q[r, a])).sum()
        s3 = np.log(Q[r, c] / (Q.sum(axis=1) - Q[r, a] - Q[r, b])).sum()
        if s2 > ll2:
            best2, ll2 = float(l), s2
        if s3 > ll3:
            best3, ll3 = float(l), s3
    return best2, best3
