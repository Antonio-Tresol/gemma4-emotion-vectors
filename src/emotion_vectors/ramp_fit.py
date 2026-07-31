"""Q3 hierarchical ramp-width estimator (registered read R2, Q3.H1.E1).

Model: each transition i contributes a fixed window y_i(t), t in [-W2, W2),
fitted as  y_i = a_i + b_i * logistic((t - c_i) / s)  with per-transition
intercept a_i, amplitude b_i, center c_i, and a SHARED 10-90% width w = 4.6*s
profiled over a grid. width 0 (numerically 1e-6) is the step model.

Rationale (results/trajectory_instrument_calibration_it.json): per-transition
step-vs-ramp fits need amplitude ~4x noise sd for 50-70% power, so the width
is estimated hierarchically — pooled over the transitions of a condition,
with per-transition centers absorbing timing jitter (which otherwise smears
averaged steps into pseudo-ramps of width ~2.6x jitter sd).

Estimator must be validated on simulated steps and ramps at matched noise
BEFORE touching true-probe data: scripts/validate_ramp_estimator.py.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float

WIDTH_GRID = (1e-6, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
CENTER_GRID = tuple(range(-8, 9, 2))


def logistic_curve(
    tgrid: Float[np.ndarray, "window"], center: float, width: float
) -> Float[np.ndarray, "window"]:
    """Unit logistic with 10-90% rise width `width` (width->0 gives a step)."""
    z = np.clip(-(tgrid - center) / max(width / 4.6, 1e-6), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))


def transition_sse_matrix(
    ys: list[Float[np.ndarray, "window"]],
    tgrid: Float[np.ndarray, "window"],
    widths=WIDTH_GRID,
    centers=CENTER_GRID,
) -> Float[np.ndarray, "transitions widths"]:
    """(n_transitions, n_widths) SSE, minimized over centers per transition.

    Intercept and amplitude solved by least squares at each (center, width).
    """
    n = len(tgrid)
    sse = np.full((len(ys), len(widths)), np.inf)
    for wi, w in enumerate(widths):
        for c in centers:
            f = logistic_curve(tgrid, c, w)
            X = np.stack([np.ones(n), f], axis=1)
            # hat-matrix trick: residual = y - X (X^+ y), shared X across ys
            pinv = np.linalg.pinv(X)
            H = X @ pinv
            for i, y in enumerate(ys):
                r = y - H @ y
                s = float(r @ r)
                if s < sse[i, wi]:
                    sse[i, wi] = s
    return sse


def profile_width(
    sse: Float[np.ndarray, "transitions widths"],
    widths=WIDTH_GRID,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, object]:
    """Shared-width point estimate + bootstrap CI from a per-transition SSE
    matrix. Resamples transitions (rows) with replacement."""
    total = sse.sum(axis=0)
    w_hat = float(widths[int(np.argmin(total))])
    rng = np.random.default_rng(seed)
    n = sse.shape[0]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = widths[int(np.argmin(sse[idx].sum(axis=0)))]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return dict(
        width=w_hat,
        ci95=[float(lo), float(hi)],
        boot_median=float(np.median(boots)),
        n_transitions=int(n),
    )


def verdict(fit: dict[str, object]) -> str:
    """Registered rule: ramp if width >= 4 with CI excluding <= 2, step if the
    CI includes <= 2, else undecided (CI straddles)."""
    if fit["ci95"][0] > 2.0 and fit["width"] >= 4.0:
        return "ramp"
    if fit["ci95"][0] <= 2.0 and fit["ci95"][1] <= 2.0:
        return "step"
    if fit["ci95"][1] <= 2.0:
        return "step"
    if fit["ci95"][0] <= 2.0:
        return "undecided"
    return "undecided"
