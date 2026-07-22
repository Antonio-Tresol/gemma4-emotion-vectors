"""Q1.H2.E8 — numerical-template confound diagnostic (laptop-side).

Are the intensity curves emotion-specific signal, or generic geometry riding
along with N? Reads registered in TREE.md Q1.H2.E8 before running:

  R1 common-mode share: pooled R^2 of predicting each tracked probe's
     across-N curve from the 4-probe mean curve, per template (< 0.8 on most
     panels = emotion-specific).
  R2 random-direction null: 1,000 random unit directions in the span of the
     171 centered probes; real tracked-probe |Spearman rho| vs N must exceed
     the random 95th percentile per panel.
  R3 norm confound: Spearman of activation L2 norm vs N per template.
  R4 amplitude parity: cosine excursion range per panel vs the paper's ~0.16
     full range (Figure 3 spans roughly -0.08 to +0.08).
  R5 R1-R2 repeated after the neutral-PC projection (E7).

Writes results/e8_template_diagnostic.json. Instruct model, chat format,
last-token readout, layer 33 (the Figure 3 replication's configuration).

    uv run python scripts/diagnose_templates.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr

from emotion_vectors.analysis import project_out_neutral
from emotion_vectors.scoring import battery_matrix

if TYPE_CHECKING:
    from jaxtyping import Float

LAYER = 33
TRACKED = ("afraid", "calm", "happy", "sad")
N_RANDOM = 1_000
RNG = np.random.default_rng(20260722)


def common_mode_share(curves: "Float[np.ndarray, 'probes n_x']") -> float:
    """Pooled R^2 of each centered curve regressed on the centered mean curve."""
    centered = curves - curves.mean(axis=1, keepdims=True)
    common = centered.mean(axis=0)
    denom = float(common @ common)
    if denom == 0.0:
        return 0.0
    ss_res = ss_tot = 0.0
    for row in centered:
        beta = float(row @ common) / denom
        ss_res += float(((row - beta * common) ** 2).sum())
        ss_tot += float((row**2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def panel_diagnostics(
    probes: "Float[np.ndarray, 'emotions d_model']",
    emotions: list[str],
    acts: "Float[np.ndarray, 'n_x d_model']",
    xs: list[float],
) -> dict[str, object]:
    """R1/R2/R4 for one template panel at one probe set."""
    cos = battery_matrix(probes, acts, center_pool=probes)
    tracked_idx = [emotions.index(e) for e in TRACKED]
    curves = cos[tracked_idx]
    rho_real = {e: float(spearmanr(cos[emotions.index(e)], xs).statistic) for e in TRACKED}

    centered_probes = probes - probes.mean(axis=0)
    weights = RNG.standard_normal((N_RANDOM, len(emotions)))
    rand_dirs = weights @ centered_probes
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    rand_cos = rand_dirs @ acts.T / np.linalg.norm(acts, axis=1)
    rho_rand = np.array([abs(float(spearmanr(rc, xs).statistic)) for rc in rand_cos])
    p95 = float(np.percentile(rho_rand, 95))

    return {
        "common_mode_share": round(common_mode_share(curves), 4),
        "tracked_spearman": {e: round(v, 3) for e, v in rho_real.items()},
        "random_abs_rho_p95": round(p95, 3),
        "tracked_exceeding_null": [e for e, v in rho_real.items() if abs(v) > p95],
        "per_curve_excursion": {
            e: round(float(curves[k].max() - curves[k].min()), 4) for k, e in enumerate(TRACKED)
        },
    }


def main() -> int:
    import argparse  # noqa: PLC0415  (single-use CLI shim, mirrors the module docstring)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", choices=("last", "mean_all", "mean_content"), default="last")
    args = parser.parse_args()
    prompts = [json.loads(line) for line in open("results/probe_sweep_it/prompts.jsonl")]
    sweep = np.load("results/probe_sweep_it/activations.npz", allow_pickle=True)
    layer_pos = list(map(int, sweep["layers"])).index(LAYER)
    acts_all = sweep[f"chat_{args.readout}"].astype(np.float64)[:, layer_pos, :]

    bundle = np.load("results/emotion_vectors_it_means.npz", allow_pickle=True)
    emotions = list(map(str, bundle["emotions"]))
    probes = bundle["means"][:, list(map(int, bundle["layers"])).index(LAYER), :].astype(np.float64)

    neutral = np.load("results/e7_neutral_bundle.npz", allow_pickle=True)
    neutral_vecs = neutral["vectors"][:, list(map(int, neutral["layers"])).index(LAYER), :].astype(
        np.float64
    )
    projected, n_removed = project_out_neutral(probes, neutral_vecs)

    panels: dict[str, dict[str, object]] = {}
    for name in sorted({p["name"] for p in prompts if p["kind"] == "template"}):
        rows = [
            (i, p) for i, p in enumerate(prompts) if p["kind"] == "template" and p["name"] == name
        ]
        rows.sort(key=lambda ip: ip[1]["x"])
        idx = [i for i, _ in rows]
        xs = [float(p["x"]) for _, p in rows]
        acts = acts_all[idx]
        raw = panel_diagnostics(probes, emotions, acts, xs)
        raw["norm_vs_n_spearman"] = round(
            float(spearmanr(np.linalg.norm(acts, axis=1), xs).statistic), 3
        )
        raw["neutral_projected"] = panel_diagnostics(projected, emotions, acts, xs)
        panels[name] = raw

    shares = [p["common_mode_share"] for p in panels.values()]
    out = {
        "experiment": "Q1.H2.E8",
        "config": f"gemma-4-31b-it, chat format, {args.readout} readout, layer 33",
        "neutral_pcs_removed": int(n_removed),
        "panels": panels,
        "r1_read": f"common-mode share median {float(np.median(shares)):.3f}; "
        f"{sum(s < 0.8 for s in shares)}/{len(shares)} panels below 0.8",
        "r4_paper_reference_range": 0.16,
    }
    Path(
        f"results/e8_template_diagnostic_{args.readout}.json"
        if args.readout != "last"
        else "results/e8_template_diagnostic.json"
    ).write_text(json.dumps(out, indent=2) + "\n")
    for name, p in panels.items():
        print(
            f"{name:10s} common={p['common_mode_share']:.2f} "
            f"norm-rho={p['norm_vs_n_spearman']:+.2f} "
            f"excursion={max(p['per_curve_excursion'].values()):.3f} "
            f"beat-null={p['tracked_exceeding_null']} "
            f"(null p95={p['random_abs_rho_p95']:.2f})"
        )
    print(out["r1_read"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
