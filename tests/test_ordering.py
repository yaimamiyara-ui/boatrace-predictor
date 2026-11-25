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
    n_races = 60
    raw = rng.uniform(0.10, 0.35, size=(n_races, 6))
    P = raw / raw.sum(axis=1, keepdims=True)

    # Clear, known synthetic bias: whenever boat 1 (idx 0) wins, boat 4 (idx 3) is 2nd and boat 6 (idx 5) is 3rd
    # in the vast majority of races, despite boats 2 and 3 having comparable or higher win probabilities.
    idx = np.zeros((n_races, 3), dtype=int)
    for r in range(n_races):
        if r < 48:
            idx[r] = [0, 3, 5]  # 1-4-6 finish
        else:
            idx[r] = [1, 2, 4]  # 2-3-5 finish

    params = fit_table(P, idx)
    assert set(params.keys()) == {"l2", "G2", "l3", "G3"}
    G2 = np.asarray(params["G2"])
    G3 = np.asarray(params["G3"])
    assert G2.shape == (6, 6)
    assert G3.shape == (6, 6)

    # Initial/default parameters (no fitting: l2=1.0, l3=1.0, G2=0, G3=0)
    default_params = {
        "l2": 1.0,
        "G2": np.zeros((6, 6)).tolist(),
        "l3": 1.0,
        "G3": np.zeros((6, 6)).tolist(),
    }

    fitted_mat = all_probs(P, params)
    default_mat = all_probs(P, default_params)

    win_cols = [COMBO_NAMES.index(f"{a+1}-{b+1}-{c+1}") for a, b, c in idx]
    ll_fitted = np.sum(np.log(np.clip(fitted_mat[np.arange(n_races), win_cols], 1e-15, None)))
    ll_default = np.sum(np.log(np.clip(default_mat[np.arange(n_races), win_cols], 1e-15, None)))

    # Fitting must strictly and substantially improve log-likelihood and boost the targeted combo (1-4-6)
    assert ll_fitted > ll_default + 10.0
    col_146 = COMBO_NAMES.index("1-4-6")
    assert np.all(fitted_mat[:, col_146] > default_mat[:, col_146])
    assert G2[0, 3] > G2[0, 1]
    assert G3[0, 5] > G3[0, 1]

    mat = fitted_mat[:4]
    assert mat.shape == (4, 120)
    np.testing.assert_allclose(mat.sum(axis=1), np.ones(4), atol=1e-9)

    single = trifecta_probs_table(P[0], params)
    assert len(single) == 120
    assert list(single.keys()) == COMBO_NAMES
    assert pytest.approx(sum(single.values()), abs=1e-9) == 1.0
    assert single["1-2-3"] == pytest.approx(mat[0, COMBO_NAMES.index("1-2-3")], rel=1e-12)


def test_scratched_boats_in_all_probs_and_trifecta_probs_table():
    params = {
        "l2": 0.85,
        "G2": np.zeros((6, 6)).tolist(),
        "l3": 0.70,
        "G3": np.zeros((6, 6)).tolist(),
    }
    # Boat 6 is scratched (probability 0)
    p_scratched = np.array([0.45, 0.25, 0.15, 0.10, 0.05, 0.0])
    single = trifecta_probs_table(p_scratched, params)
    mat = all_probs(p_scratched, params)

    assert pytest.approx(sum(single.values()), abs=1e-9) == 1.0
    assert pytest.approx(float(mat.sum()), abs=1e-9) == 1.0

    # Explicitly check scratched boat 6 placed 1st, 2nd, and 3rd
    assert single["6-1-2"] == 0.0  # 1st
    assert single["1-6-2"] == 0.0  # 2nd
    assert single["1-2-6"] == 0.0  # 3rd
    assert mat[0, COMBO_NAMES.index("6-1-2")] == 0.0
    assert mat[0, COMBO_NAMES.index("1-6-2")] == 0.0
    assert mat[0, COMBO_NAMES.index("1-2-6")] == 0.0

    for combo, val in single.items():
        if "6" in combo.split("-"):
            assert val == 0.0
        else:
            assert val > 0.0


_SCRATCHED_P = np.array([0.45, 0.25, 0.15, 0.10, 0.05, 0.0])  # boat 6 (index 5) scratched


def _wcp_single(combo):
    return winning_combo_prob(_SCRATCHED_P[None, :], np.array([combo]), l2=0.9, l3=0.7)[0]


def test_winning_combo_prob_scratched_boat_1st():
    assert _wcp_single([5, 0, 1]) == 0.0  # 6-1-2


def test_winning_combo_prob_scratched_boat_2nd():
    assert _wcp_single([0, 5, 1]) == 0.0  # 1-6-2


def test_winning_combo_prob_scratched_boat_3rd():
    assert _wcp_single([0, 1, 5]) == 0.0  # 1-2-6


def test_winning_combo_prob_no_scratched_boats_unchanged():
    p = np.array([0.42, 0.23, 0.15, 0.10, 0.06, 0.04])
    l2, l3 = 0.85, 0.65
    combos = [(0, 1, 2), (1, 0, 3), (2, 4, 5), (5, 3, 1)]
    P_batch = np.tile(p, (len(combos), 1))
    idx = np.array(combos)

    actual = winning_combo_prob(P_batch, idx, l2=l2, l3=l3)

    q2 = p ** l2
    q3 = p ** l3
    for i, (a, b, c) in enumerate(combos):
        expected = p[a] * (q2[b] / (q2.sum() - q2[a])) * (q3[c] / (q3.sum() - q3[a] - q3[b]))
        assert actual[i] == pytest.approx(expected, rel=1e-15)


def test_exactly_three_runnable_boats():
    # Boats 1, 2, 3 runnable; boats 4, 5, 6 scratched (0 probability)
    p3 = np.array([0.50, 0.30, 0.20, 0.0, 0.0, 0.0])
    valid_combos = {"1-2-3", "1-3-2", "2-1-3", "2-3-1", "3-1-2", "3-2-1"}

    probs = trifecta_probs(p3, l2=0.85, l3=0.75)
    assert pytest.approx(sum(probs.values()), abs=1e-9) == 1.0
    for combo, val in probs.items():
        if combo in valid_combos:
            assert val > 0.0
        else:
            assert val == 0.0

    params = {
        "l2": 0.85,
        "G2": np.zeros((6, 6)).tolist(),
        "l3": 0.75,
        "G3": np.zeros((6, 6)).tolist(),
    }
    table_probs = trifecta_probs_table(p3, params)
    assert pytest.approx(sum(table_probs.values()), abs=1e-9) == 1.0
    for combo, val in table_probs.items():
        if combo in valid_combos:
            assert val > 0.0
        else:
            assert val == 0.0


def test_fewer_than_three_runnable_boats():
    # Only 2 runnable boats (race void for trifecta: no 3-boat permutation can finish)
    p2 = np.array([0.60, 0.40, 0.0, 0.0, 0.0, 0.0])
    # Only 1 runnable boat
    p1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for p in (p2, p1):
        probs = trifecta_probs(p, l2=0.9, l3=0.7)
        assert all(v == 0.0 for v in probs.values())
        assert sum(probs.values()) == 0.0

    params = {
        "l2": 0.9,
        "G2": np.zeros((6, 6)).tolist(),
        "l3": 0.7,
        "G3": np.zeros((6, 6)).tolist(),
    }
    mat = all_probs(np.vstack([p2, p1]), params)
    assert not np.isnan(mat).any()
    np.testing.assert_allclose(mat, np.zeros((2, 120)), atol=1e-12)

    for p in (p2, p1):
        table_probs = trifecta_probs_table(p, params)
        assert all(v == 0.0 for v in table_probs.values())
