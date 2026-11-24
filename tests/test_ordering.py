import numpy as np
import pytest
from model.ordering import (
    COMBO_NAMES,
    trifecta_probs,
    winning_combo_prob,
    fit_table,
    all_probs,
    trifecta_probs_table,
)


def test_trifecta_probs_sum_and_ordering():
    p = np.array([0.45, 0.22, 0.14, 0.09, 0.06, 0.04])
    probs = trifecta_probs(p, l2=0.8, l3=0.6)

    assert len(probs) == 120
    assert pytest.approx(sum(probs.values()), abs=1e-9) == 1.0
    assert all(v >= 0.0 for v in probs.values())

    # Since p[0] > p[1] > p[2] > p[3] > p[4] > p[5], 1-2-3 is strictly the most likely combo
    # and combos preserve ordering of win probabilities
    assert probs["1-2-3"] > probs["1-2-4"] > probs["1-3-4"] > probs["2-3-4"] > probs["6-5-4"]


def test_trifecta_probs_matches_harville_when_l2_l3_one():
    p = np.array([0.40, 0.25, 0.15, 0.10, 0.06, 0.04])
    probs = trifecta_probs(p, l2=1.0, l3=1.0)

    # Check against plain Harville formula p_a * p_b / (1 - p_a) * p_c / (1 - p_a - p_b)
    for (a, b, c) in [(0, 1, 2), (2, 0, 4), (5, 4, 3)]:
        expected = p[a] * (p[b] / (1.0 - p[a])) * (p[c] / (1.0 - p[a] - p[b]))
        assert probs[f"{a+1}-{b+1}-{c+1}"] == pytest.approx(expected, rel=1e-12)

    # Also verify winning_combo_prob matches trifecta_probs
    idx = np.array([[0, 1, 2], [2, 0, 4]])
    win_p = winning_combo_prob(np.vstack([p, p]), idx, l2=1.0, l3=1.0)
    assert win_p[0] == pytest.approx(probs["1-2-3"], rel=1e-12)
    assert win_p[1] == pytest.approx(probs["3-1-5"], rel=1e-12)


def test_trifecta_probs_zero_probability_boat():
    # Boat 6 scratched (probability 0)
    p = np.array([0.45, 0.25, 0.15, 0.10, 0.05, 0.0])
    probs = trifecta_probs(p, l2=0.9, l3=0.7)

    assert len(probs) == 120
    for combo, val in probs.items():
        if "6" in combo.split("-"):
            assert val == 0.0
        else:
            assert val > 0.0
    assert pytest.approx(sum(probs.values()), abs=1e-9) == 1.0


def test_fit_table_all_probs_and_trifecta_probs_table():
    rng = np.random.default_rng(42)
    n_races = 30
    raw = rng.uniform(0.05, 0.5, size=(n_races, 6))
    P = raw / raw.sum(axis=1, keepdims=True)
    idx = np.array([rng.choice(6, size=3, replace=False) for _ in range(n_races)])

    params = fit_table(P, idx)
    assert set(params.keys()) == {"l2", "G2", "l3", "G3"}
    assert np.asarray(params["G2"]).shape == (6, 6)
    assert np.asarray(params["G3"]).shape == (6, 6)

    mat = all_probs(P[:4], params)
    assert mat.shape == (4, 120)
    np.testing.assert_allclose(mat.sum(axis=1), np.ones(4), atol=1e-9)

    single = trifecta_probs_table(P[0], params)
    assert len(single) == 120
    assert list(single.keys()) == COMBO_NAMES
    assert pytest.approx(sum(single.values()), abs=1e-9) == 1.0
    assert single["1-2-3"] == pytest.approx(mat[0, COMBO_NAMES.index("1-2-3")], rel=1e-12)

    # Zero-probability boat in all_probs / trifecta_probs_table gets zero combos
    p_scratched = np.array([0.50, 0.20, 0.15, 0.10, 0.05, 0.0])
    single_scratched = trifecta_probs_table(p_scratched, params)
    assert pytest.approx(sum(single_scratched.values()), abs=1e-9) == 1.0
    for combo, val in single_scratched.items():
        if "6" in combo.split("-"):
            assert val == 0.0
