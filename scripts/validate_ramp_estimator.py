"""Validate the Q3 hierarchical ramp-width estimator on simulation ONLY.

Registered precondition for read R2 (TREE.md Q3.H1.E1): before the estimator
touches true-probe data it must, at noise matched to the measured instrument
(sigma, AR1 from results/trajectory_instrument_calibration_it.json):
  - recover width <= 2 tokens on simulated TRUE STEPS (all amplitudes,
    timing jitter sd 4);
  - recover the true width on simulated TRUE RAMPS (widths 8/16/24);
  - stay honest on a MIXTURE where half the transitions carry no signal
    (amp 0), mimicking heterogeneous real transitions.

Writes results/ramp_estimator_validation.json with a pass flag per cell.

    uv run python scripts/validate_ramp_estimator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emotion_vectors.ramp_fit import (  # noqa: E402
    WIDTH_GRID,
    logistic_curve,
    profile_width,
    transition_sse_matrix,
    verdict,
)

WINDOW = 48
N_TRANS = 200  # transitions per simulated condition (~category scale /6)
JITTER_SD = 4.0
N_BOOT = 500


def gen_condition(
    true_width: float,
    amp_sd: float,
    sigma: float,
    phi: float,
    rng: np.random.Generator,
    frac_null: float = 0.0,
) -> list[np.ndarray]:
    tgrid = np.arange(WINDOW) - WINDOW / 2
    ys = []
    for _ in range(N_TRANS):
        e = rng.standard_normal(WINDOW) * sigma * np.sqrt(1 - phi**2)
        x = np.empty(WINDOW)
        x[0] = e[0] / np.sqrt(1 - phi**2)
        for t in range(1, WINDOW):
            x[t] = phi * x[t - 1] + e[t]
        if rng.random() >= frac_null:
            c = float(np.round(rng.standard_normal() * JITTER_SD))
            x += amp_sd * sigma * logistic_curve(tgrid, c, true_width)
        ys.append(x)
    return ys


SEG = 32  # flanking segment tokens used ONLY for the independent amplitude gate


def gen_gated_mixture(
    true_width: float,
    amp_sd: float,
    sigma: float,
    phi: float,
    rng: np.random.Generator,
    frac_null: float,
) -> tuple[list, list]:
    """Mixture condition with flanking segments for the amplitude gate.

    The gate statistic (mean of the incoming flank minus mean of the outgoing
    flank, both OUTSIDE the fit window) is width-independent, so selecting on
    it cannot bias the width estimate — it only removes no-signal transitions.
    """
    L = WINDOW + 2 * SEG
    tgrid_full = np.arange(L) - L / 2
    ys, gate = [], []
    for _ in range(N_TRANS):
        e = rng.standard_normal(L) * sigma * np.sqrt(1 - phi**2)
        x = np.empty(L)
        x[0] = e[0] / np.sqrt(1 - phi**2)
        for t in range(1, L):
            x[t] = phi * x[t - 1] + e[t]
        if rng.random() >= frac_null:
            c = float(np.round(rng.standard_normal() * JITTER_SD))
            x += amp_sd * sigma * logistic_curve(tgrid_full, c, true_width)
        gate.append(float(x[-SEG:].mean() - x[:SEG].mean()))
        ys.append(x[SEG : SEG + WINDOW])
    return ys, gate


def main() -> None:
    calib = json.load(open("results/trajectory_instrument_calibration_it.json"))
    sigma = calib["random_probe"]["33"]["phase_noise_sd"]
    phi = calib["random_probe"]["33"]["ar1"]
    tgrid = np.arange(WINDOW) - WINDOW / 2
    rng = np.random.default_rng(11)

    cells = {}
    conditions = [
        *[(0.0, a, 0.0) for a in (1, 2, 4)],  # true steps
        *[(w, a, 0.0) for w in (8, 16, 24) for a in (1, 2, 4)],  # true ramps
        (16.0, 2.0, 0.5),  # mixture: half null
    ]
    for true_w, amp, frac_null in conditions:
        ys = gen_condition(true_w, amp, sigma, phi, rng, frac_null)
        sse = transition_sse_matrix(ys, tgrid)
        fit = profile_width(sse, n_boot=N_BOOT)
        fit["verdict"] = verdict(fit)
        if true_w == 0.0:
            ok = fit["width"] <= 2.0 and fit["verdict"] != "ramp"
        else:
            ok = 0.5 * true_w <= fit["width"] <= 1.5 * true_w
        name = f"w{true_w:g}_amp{amp:g}sd" + ("_mix" if frac_null else "")
        cells[name] = dict(
            true_width=true_w, amp_sd=amp, frac_null=frac_null, **fit, passes=bool(ok)
        )
        print(name, "->", fit["width"], fit["ci95"], fit["verdict"], "PASS" if ok else "FAIL")

    # amended procedure: amplitude-gate (>= 2 sigma on the flank statistic),
    # then fit the retained subset — the mixture rescue
    for true_w, amp, frac_null, tag in [
        (16.0, 2.0, 0.5, "w16_amp2sd_mix_gated"),
        (0.0, 2.0, 0.5, "w0_amp2sd_mix_gated"),
    ]:
        ys, gate = gen_gated_mixture(true_w, amp, sigma, phi, rng, frac_null)
        keep = [y for y, g in zip(ys, gate) if g >= 2 * sigma]
        sse = transition_sse_matrix(keep, tgrid)
        fit = profile_width(sse, n_boot=N_BOOT)
        fit["verdict"] = verdict(fit)
        fit["retained_frac"] = len(keep) / len(ys)
        if true_w == 0.0:
            ok = fit["width"] <= 2.0 and fit["verdict"] != "ramp"
        else:
            ok = 0.5 * true_w <= fit["width"] <= 1.5 * true_w
        cells[tag] = dict(
            true_width=true_w, amp_sd=amp, frac_null=frac_null, **fit, passes=bool(ok)
        )
        print(
            tag,
            "->",
            fit["width"],
            fit["ci95"],
            fit["verdict"],
            f"retained {fit['retained_frac']:.2f}",
            "PASS" if ok else "FAIL",
        )

    out = dict(
        sigma=sigma,
        phi=phi,
        window=WINDOW,
        n_transitions=N_TRANS,
        jitter_sd=JITTER_SD,
        n_boot=N_BOOT,
        width_grid=list(WIDTH_GRID),
        cells=cells,
        all_pass=bool(all(c["passes"] for c in cells.values())),
    )
    json.dump(out, open("results/ramp_estimator_validation.json", "w"), indent=1)
    print("all_pass:", out["all_pass"])


if __name__ == "__main__":
    main()
