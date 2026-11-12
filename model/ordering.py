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


# ---- 1着の枠ごとの補正表つき（Geminiの提案をもとにClaude実装） ----
# P(2着=j | 1着=i) ∝ p_j^l2 × exp(G2[i][j])、P(3着=k | 1着=i, 2着=j) ∝ p_k^l3 × exp(G3[i][k])
# G2, G3 は6×6の表（1着の枠 i ごとに、各枠が2着・3着に入りやすいかの補正）。過去データから最尤法で決める。
from scipy.optimize import minimize
from scipy.special import logsumexp

COMBOS = [(a, b, c) for a, b, c in permutations(range(6), 3)]
COMBO_NAMES = [f"{a+1}-{b+1}-{c+1}" for a, b, c in COMBOS]


def _fit_stage(logP, a, target, exclude, reg=1e-3):
    n = len(logP)
    r = np.arange(n)
    mask = np.zeros((n, 6), dtype=bool)
    for e in exclude:
        mask[r, e] = True

    def nll(theta):
        Z = theta[0] * logP + theta[1:].reshape(6, 6)[a]
        Z = np.where(mask, -np.inf, Z)
        return -(Z[r, target] - logsumexp(Z, axis=1)).mean() + reg * np.sum(theta[1:] ** 2)

    x0 = np.concatenate([[1.0], np.zeros(36)])
    res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 300})
    return float(res.x[0]), res.x[1:].reshape(6, 6)


def fit_table(P: np.ndarray, idx: np.ndarray) -> dict:
    logP = np.log(np.clip(P, 1e-12, None))
    a, b, c = idx[:, 0], idx[:, 1], idx[:, 2]
    l2, G2 = _fit_stage(logP, a, b, [a])
    l3, G3 = _fit_stage(logP, a, c, [a, b])
    return {"l2": l2, "G2": G2.tolist(), "l3": l3, "G3": G3.tolist()}


def all_probs(P: np.ndarray, params: dict) -> np.ndarray:
    """(レース数, 6) の1着確率から、(レース数, 120) の3連単確率。列の並びは COMBO_NAMES"""
    P = np.asarray(P, dtype=float).reshape(-1, 6)
    logP = np.log(np.clip(P, 1e-12, None))
    G2, G3 = np.asarray(params["G2"]), np.asarray(params["G3"])
    n = len(P)
    out = np.zeros((n, 120))
    col = {cmb: i for i, cmb in enumerate(COMBOS)}
    dead = P <= 0  # 欠場などで確率0の艇
    for a in range(6):
        Z2 = params["l2"] * logP + G2[a]
        Z2 = np.where(dead, -np.inf, Z2)
        Z2[:, a] = -np.inf
        S2 = np.exp(Z2 - logsumexp(Z2, axis=1, keepdims=True))
        Z3base = params["l3"] * logP + G3[a]
        Z3base = np.where(dead, -np.inf, Z3base)
        Z3base[:, a] = -np.inf
        for b in range(6):
            if b == a:
                continue
            Z3 = Z3base.copy()
            Z3[:, b] = -np.inf
            S3 = np.exp(Z3 - logsumexp(Z3, axis=1, keepdims=True))
            for c in range(6):
                if c in (a, b):
                    continue
                out[:, col[(a, b, c)]] = P[:, a] * S2[:, b] * S3[:, c]
    return np.nan_to_num(out, nan=0.0)


def trifecta_probs_table(p, params: dict) -> dict[str, float]:
    row = all_probs(np.asarray(p, dtype=float).reshape(1, 6), params)[0]
    return {name: float(v) for name, v in zip(COMBO_NAMES, row)}
