"""Recovery tests for the Bradley-Terry scoring in emotion_vectors.preferences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from emotion_vectors.preferences import ELO_SCALE, bradley_terry, elo, soft_wins, win_matrix

if TYPE_CHECKING:
    from jaxtyping import Float, Int


def _simulate(
    true_strengths: "Float[np.ndarray, 'items']",
) -> tuple[
    "Int[np.ndarray, 'pairs']", "Int[np.ndarray, 'pairs']", "Float[np.ndarray, 'pairs']"
]:
    """All ordered pairs with exact Bradley-Terry win probabilities as soft wins."""
    n = len(true_strengths)
    pair_i, pair_j, p_first = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                pair_i.append(i)
                pair_j.append(j)
                p_first.append(true_strengths[i] / (true_strengths[i] + true_strengths[j]))
    return np.array(pair_i), np.array(pair_j), np.array(p_first)


def test_soft_wins_is_two_way_softmax() -> None:
    logits = np.array([[2.0, 0.0], [0.0, 0.0], [-1.0, 3.0]])
    p = soft_wins(logits)
    expected = 1.0 / (1.0 + np.exp(logits[:, 1] - logits[:, 0]))
    np.testing.assert_allclose(p, expected, atol=1e-12)


def test_bradley_terry_recovers_known_strengths() -> None:
    true = np.array([4.0, 2.0, 1.0, 0.5])
    true = true / np.exp(np.mean(np.log(true)))  # geometric mean 1, the fit's normalization
    pair_i, pair_j, p_first = _simulate(true)
    wins = win_matrix(pair_i, pair_j, p_first, len(true))
    fitted = bradley_terry(wins)
    np.testing.assert_allclose(fitted, true, rtol=1e-6)


def test_elo_gap_matches_log_strength_ratio() -> None:
    strengths = np.array([10.0, 1.0])
    scores = elo(strengths)
    np.testing.assert_allclose(scores[0] - scores[1], ELO_SCALE * np.log(10.0), rtol=1e-12)
    np.testing.assert_allclose(scores.mean(), 1000.0, atol=1e-9)
