from itertools import permutations
import numpy as np
import pytest
from model.market import (
    scratched_boats,
    renormalize,
    implied_win_probs,
    implied_combo_probs,
    payout_rate,
)


def _make_full_odds(payout: float = 0.75, scratched: set[int] | None = None) -> dict[str, float | None]:
    scratched = scratched or set()
    base_p = np.array([0.50, 0.20, 0.12, 0.08, 0.06, 0.04], dtype=float)
    for b in scratched:
        base_p[b - 1] = 0.0
    if base_p.sum() > 0:
        base_p = base_p / base_p.sum()

    odds: dict[str, float | None] = {}
    for a, b, c in permutations(range(1, 7), 3):
        key = f"{a}-{b}-{c}"
        if a in scratched or b in scratched or c in scratched:
            odds[key] = None
            continue
        pa, pb, pc = base_p[a - 1], base_p[b - 1], base_p[c - 1]
        pr = pa * (pb / (1.0 - pa)) * (pc / (1.0 - pa - pb))
        odds[key] = round(payout / pr, 4) if pr > 0 else None
    return odds


def test_normal_full_odds():
    odds = _make_full_odds(payout=0.75)
    assert scratched_boats(odds) == set()

    pr = payout_rate(odds)
    assert pr is not None
    assert pytest.approx(pr, rel=1e-3) == 0.75

    c_probs = implied_combo_probs(odds)
    assert len(c_probs) == 120
    assert pytest.approx(sum(c_probs.values()), abs=1e-9) == 1.0

    w_probs = implied_win_probs(odds)
    assert w_probs is not None
    assert w_probs.shape == (6,)
    assert pytest.approx(float(w_probs.sum()), abs=1e-9) == 1.0
    assert pytest.approx(w_probs[0], rel=1e-2) == 0.50


def test_with_scratched_boats():
    odds = _make_full_odds(payout=0.75, scratched={4, 6})
    assert scratched_boats(odds) == {4, 6}

    c_probs = implied_combo_probs(odds)
    assert len(c_probs) == 24
    assert all("4" not in k and "6" not in k for k in c_probs)
    assert pytest.approx(sum(c_probs.values()), abs=1e-9) == 1.0

    w_probs = implied_win_probs(odds)
    assert w_probs is not None
    assert w_probs[3] == 0.0
    assert w_probs[5] == 0.0
    assert pytest.approx(float(w_probs.sum()), abs=1e-9) == 1.0


def test_renormalize_immutability_and_zero_handling():
    orig = np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05], dtype=float)
    orig_copy = orig.copy()

    res = renormalize(orig, {1, 6})
    np.testing.assert_array_equal(orig, orig_copy)
    assert res[0] == 0.0
    assert res[5] == 0.0
    assert pytest.approx(float(res.sum()), abs=1e-9) == 1.0
    assert pytest.approx(res[1], abs=1e-9) == 0.2 / 0.55

    all_zero = renormalize(orig, {1, 2, 3, 4, 5, 6})
    np.testing.assert_array_equal(all_zero, np.zeros(6))


def test_empty_and_all_none_odds():
    assert scratched_boats({}) == set()
    assert implied_combo_probs({}) == {}
    assert implied_win_probs({}) is None
    assert payout_rate({}) is None

    all_none = {f"{a}-{b}-{c}": None for a, b, c in permutations(range(1, 7), 3)}
    assert scratched_boats(all_none) == {1, 2, 3, 4, 5, 6}
    assert implied_combo_probs(all_none) == {}
    assert implied_win_probs(all_none) is None
    assert payout_rate(all_none) is None
