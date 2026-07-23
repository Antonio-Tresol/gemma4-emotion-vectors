"""Section 7: what instruction tuning changes.

No figure — this section is a printed record: first-component variance
shares, the base model's valence direction read out on the instruct vectors
against the pre-fix record, and the neutral-subspace projection test
(prediction P1). ``s7_tuning_stats`` returns a stats dict whose ``lines``
the notebook prints.
"""

from __future__ import annotations

import glob
import json
from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from emotion_vectors.analysis import project_out_neutral
from emotion_vectors.artifacts import fetch

from ._context import N_COMPONENTS, GeometryContext


def _cross_readout_lines(ctx: GeometryContext) -> tuple[list[str], float]:
    """Base model's valence direction read out on the instruct model's vectors,
    with the pre-fix instrument's numbers as comparators."""
    layer_pos = ctx.layers.index(ctx.default_layer)
    base_plane = ctx.plane[ctx.base_label][ctx.default_layer]
    direction = base_plane["pca"].components_[base_plane["vb"]]  # base valence-best (PC1)
    it_means = ctx.models[ctx.it_label][:, layer_pos, :]
    it_centered = it_means - it_means.mean(axis=0)
    cross_abs_r = abs(float(pearsonr(it_centered[ctx.matched] @ direction, ctx.valence).statistic))
    prefix = json.loads(fetch("geometry_pc_demotion.json").read_text())  # pre-fix instrument
    lines = [
        f"base PC{base_plane['vb'] + 1} valence direction on instruct vectors:"
        f" |r|={cross_abs_r:.3f} "
        f"(pre-fix instrument: {prefix['it_on_base_pc1_abs_r_valence']:.3f})",
        f"base own PC1-valence |r| at layer {ctx.default_layer}: {base_plane['r_vb']:.3f} "
        f"(pre-fix instrument: {prefix['base']['best_r']:.3f})",
    ]
    return lines, cross_abs_r


def _neutral_projection_lines(ctx: GeometryContext) -> list[str]:
    """Does projecting out the neutral-story subspace move valence back toward
    PC1? (prediction P1) — re-fit the PCA before and after the projection."""
    layer_pos = ctx.layers.index(ctx.default_layer)
    shard_dir = fetch("neutral_vectors_it_postfix/shards")  # post-fix neutral, one per story
    neutral = np.stack(
        [np.load(f) for f in sorted(glob.glob(str(shard_dir / "neutral__*.npy")))]
    ).astype(np.float64)
    it_means = ctx.models[ctx.it_label][:, layer_pos, :]
    lines = []
    for name, probe_means in (
        ("unprojected", it_means),
        ("projected", project_out_neutral(it_means, neutral[:, layer_pos, :])[0]),
    ):
        centered = probe_means - probe_means.mean(axis=0)
        scores = PCA(n_components=N_COMPONENTS, svd_solver="full").fit_transform(centered)
        abs_r_per_pc = [
            abs(float(pearsonr(scores[ctx.matched, k], ctx.valence).statistic))
            for k in range(N_COMPONENTS)
        ]
        lines.append(
            f"instruct {name:12s}: valence best at PC{int(np.argmax(abs_r_per_pc)) + 1}"
            f" (|r|={max(abs_r_per_pc):.3f}); "
            f"per-PC |r| {[round(r, 2) for r in abs_r_per_pc[:5]]}"
        )
    return lines


def s7_tuning_stats(ctx: GeometryContext) -> dict[str, Any]:
    """Section 7 record: every number the section text quotes (no figure).

    ``stats["lines"]`` holds, in print order: PC1 variance shares (base then
    instruct), the cross-model readout against the pre-fix record, and the
    neutral-projection before/after read; ``stats["cross_model_abs_r"]`` is
    the base-direction-on-instruct-vectors |r|.
    """
    lines = [
        f"{label}: PC1 variance share at layer {ctx.default_layer} = "
        f"{ctx.plane[label][ctx.default_layer]['evr'][0]:.2f}"
        for label in (ctx.base_label, ctx.it_label)
    ]
    readout_lines, cross_abs_r = _cross_readout_lines(ctx)
    lines += readout_lines
    lines += _neutral_projection_lines(ctx)
    return {"lines": lines, "cross_model_abs_r": cross_abs_r}
