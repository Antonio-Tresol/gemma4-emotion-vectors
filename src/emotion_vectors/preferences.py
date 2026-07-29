"""Bradley-Terry scoring for the preference experiment (Q1.H3, paper Figure 4).

Laptop-safe: numpy only. Input is the collection output of
scripts/run_preferences.py: per ordered pair (i, j), the model's next-token
logits for "A" (activity i) and "B" (activity j) after the paper's preference
prompt. Soft win probabilities come from the two-way softmax; strengths from
Bradley-Terry maximum likelihood via minorization; the Elo scale is
400/ln(10) * log-strength, anchored so the mean Elo is 1000 (the paper's
anchor is unpublished, so absolute Elo values are not comparable across
papers — differences and rankings are).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from einops import rearrange

if TYPE_CHECKING:
    from jaxtyping import Float, Int

ELO_SCALE = 400.0 / np.log(10.0)
ELO_ANCHOR = 1000.0


def soft_wins(
    ab_logits: "Float[np.ndarray, 'pairs two']",
) -> "Float[np.ndarray, 'pairs']":
    """P(pick A) per ordered pair, from the (A, B) token logits."""
    z = ab_logits - ab_logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 0] / e.sum(axis=1)


def win_matrix(
    pair_i: "Int[np.ndarray, 'pairs']",
    pair_j: "Int[np.ndarray, 'pairs']",
    p_first: "Float[np.ndarray, 'pairs']",
    n_items: int,
) -> "Float[np.ndarray, 'items items']":
    """W[i, j] = soft wins of i over j, both presentation orders accumulated."""
    w = np.zeros((n_items, n_items))
    np.add.at(w, (pair_i, pair_j), p_first)
    np.add.at(w, (pair_j, pair_i), 1.0 - p_first)
    return w


def bradley_terry(
    wins: "Float[np.ndarray, 'items items']",
    n_iter: int = 500,
    tol: float = 1e-10,
) -> "Float[np.ndarray, 'items']":
    """Bradley-Terry strengths by the standard minorization iteration.

    wins[i, j] is the (possibly fractional) number of wins of i over j;
    comparisons N[i, j] = wins[i, j] + wins[j, i]. Returns strengths
    normalized to geometric mean 1.
    """
    n = wins.shape[0]
    comparisons = wins + wins.T
    total_wins = wins.sum(axis=1)
    pi = np.ones(n)
    for _ in range(n_iter):
        # pairwise sums pi[i] + pi[j]: rows are i, columns are j, matching
        # comparisons[i, j]. The sum over axis 1 then runs over j.
        strength_i = rearrange(pi, "i -> i 1")
        strength_j = rearrange(pi, "j -> 1 j")
        denom = (comparisons / (strength_i + strength_j + 1e-30)).sum(axis=1)
        new = total_wins / np.maximum(denom, 1e-30)
        new = new / np.exp(np.mean(np.log(np.maximum(new, 1e-30))))
        if np.max(np.abs(new - pi)) < tol:
            pi = new
            break
        pi = new
    return pi


def elo(strengths: "Float[np.ndarray, 'items']") -> "Float[np.ndarray, 'items']":
    """Elo-scale scores, mean anchored at ELO_ANCHOR."""
    scores = ELO_SCALE * np.log(np.maximum(strengths, 1e-30))
    return scores - scores.mean() + ELO_ANCHOR
