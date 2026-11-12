"""オッズから市場の確率を逆算する関数と、欠場時の確率の割り直し（Gemini実装・Claude確認）。"""
from typing import Iterable, Mapping
import numpy as np


def _parse_combo_boats(combo: str) -> tuple[int, ...]:
    parts = combo.split("-")
    if len(parts) != 3:
        return ()
    try:
        boats = tuple(int(p) for p in parts)
    except ValueError:
        return ()
    if any(b < 1 or b > 6 for b in boats):
        return ()
    return boats


def scratched_boats(odds: Mapping[str, float | None]) -> set[int]:
    if not odds:
        return set()

    active_boats: set[int] = set()
    for combo, val in odds.items():
        if val is not None and val > 0:
            boats = _parse_combo_boats(combo)
            active_boats.update(boats)

    return set(range(1, 7)) - active_boats


def renormalize(p: np.ndarray | Iterable[float], scratched: Iterable[int]) -> np.ndarray:
    arr = np.array(p, dtype=float, copy=True)
    if arr.shape != (6,):
        raise ValueError(f"Expected length-6 array, got shape {arr.shape}")

    for b in scratched:
        if 1 <= b <= 6:
            arr[b - 1] = 0.0

    total = float(arr.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros(6, dtype=float)

    return arr / total


def implied_combo_probs(odds: Mapping[str, float | None]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for combo, val in odds.items():
        if val is not None and val > 0 and np.isfinite(val):
            if _parse_combo_boats(combo):
                weights[combo] = 1.0 / float(val)

    total_w = sum(weights.values())
    if total_w <= 0.0:
        return {}

    return {combo: w / total_w for combo, w in weights.items()}


def implied_win_probs(odds: Mapping[str, float | None]) -> np.ndarray | None:
    combo_probs = implied_combo_probs(odds)
    if not combo_probs:
        return None

    win_probs = np.zeros(6, dtype=float)
    for combo, prob in combo_probs.items():
        boats = _parse_combo_boats(combo)
        if boats:
            win_probs[boats[0] - 1] += prob

    total = float(win_probs.sum())
    if total <= 0.0:
        return None

    return win_probs / total


def payout_rate(odds: Mapping[str, float | None]) -> float | None:
    total_inv = 0.0
    valid_count = 0
    for combo, val in odds.items():
        if val is not None and val > 0 and np.isfinite(val):
            if _parse_combo_boats(combo):
                total_inv += 1.0 / float(val)
                valid_count += 1

    if valid_count == 0 or total_inv <= 0.0:
        return None

    return 1.0 / total_inv
