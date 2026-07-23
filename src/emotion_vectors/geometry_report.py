"""Notebook-02 sections 8-9 exhibit library: valence displacement and RSA fragmentation.

The geometry report's sections 8 (what displaced valence, TREE Q1.H1.E3) and
9 (why the instruct RSA fragments, TREE Q1.H1.E4) are load-call-show over this
module. Each section builder returns ``(figure, stats)`` where ``stats["lines"]``
holds every line the notebook prints, so the printed record and the figure come
from one computation. Sections 1-7 still live in the notebook and lean on
``emotion_vectors.interactive``; they share their PCA state with this module
through :class:`GeometryContext`.

Number-identity contract: the computations here are line-for-line ports of the
formerly inline notebook cells (commit 9dd6e30), so re-running the notebook
reproduces the committed printed numbers exactly.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float
from plotly.subplots import make_subplots
from scipy.linalg import subspace_angles
from scipy.stats import pearsonr

# One layer's PCA record, as built by the notebook's section-1 cell:
# keys pca (fitted sklearn PCA), scores [emotions pcs] (sign-oriented),
# vb / ab (valence- / arousal-best component index), r_vb / r_ab,
# rv / ra (per-component |r| lists), evr (explained variance ratios).
LayerPlane = dict[str, Any]

# RSA band edges shared by sections 6 and 9: the mid block and the late block
# that the instruct model's fragmentation separates.
MID_BAND: tuple[int, int] = (12, 24)
LATE_BAND: tuple[int, int] = (30, 57)


@dataclass(frozen=True)
class GeometryContext:
    """Everything sections 8-9 need from the notebook's section-1 state.

    ``models[label]`` holds the per-emotion mean activations
    [emotions layers d_model]; ``plane[label][layer]`` the fitted PCA record
    (:data:`LayerPlane`). ``matched`` indexes the emotions found in the NRC
    lexicon; ``default_layer`` is where sliders and single-layer reads sit.
    """

    it_label: str
    base_label: str
    models: dict[str, Float[np.ndarray, "emotions layers d_model"]]
    plane: dict[str, dict[int, LayerPlane]]
    emotions: list[str]
    layers: list[int]
    matched: list[int]
    default_layer: int

    @property
    def labels(self) -> tuple[str, str]:
        """(instruct, base), the panel order used by every paired figure."""
        return (self.it_label, self.base_label)


def load_dominance(lexicon_path: Path) -> dict[str, float]:
    """Parse the NRC VAD dominance column (term -> dominance in [0, 1]).

    ``emotion_vectors.analysis.load_nrc_vad`` returns only (valence, arousal);
    section 8 additionally checks dominance, the one reference axis no other
    section uses, so it is parsed here.
    """
    dominance_by_term: dict[str, float] = {}
    with open(lexicon_path) as handle:
        next(handle)  # header: term, valence, arousal, dominance
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                dominance_by_term[parts[0].lower()] = float(parts[3])
    return dominance_by_term


def corpus_mean_story_lengths(
    dataset_id: str, emotions: list[str]
) -> tuple[Float[np.ndarray, " emotions"] | None, str | None]:
    """Per-emotion mean story length (characters) from the extraction corpus.

    Returns ``(mean_lengths, None)`` on success, ``(None, degradation_note)``
    when the Hugging Face dataset cannot be read; the caller prints the note so
    the degradation is explicit, never a silent skip.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415 — heavy import, only needed here

        lengths_by_emotion = defaultdict(list)
        for row in load_dataset(dataset_id, split="train"):
            lengths_by_emotion[row["emotion"]].extend(len(story) for story in row["stories"])
        return np.array([float(np.mean(lengths_by_emotion[e])) for e in emotions]), None
    except Exception as err:  # noqa: BLE001 — read degrades to a note, never crashes the report
        return None, (
            f"story-length read unavailable ({type(err).__name__}); see results/it_pc_structure.json"
        )


def _s8_profile_lines(
    ctx: GeometryContext,
    dominance: Float[np.ndarray, " matched"],
    mean_length: Float[np.ndarray, " emotions"] | None,
    n_pcs: int,
) -> list[str]:
    """Per-PC profile table at the default layer: variance share and |r| against
    each NRC axis plus the story-length nuisance, base first then instruct."""
    lines = []
    for label in (ctx.base_label, ctx.it_label):
        layer_plane = ctx.plane[label][ctx.default_layer]
        lines.append(f"{label} at layer {ctx.default_layer}:")
        for k in range(n_pcs):
            r_dom = abs(float(pearsonr(layer_plane["scores"][ctx.matched, k], dominance).statistic))
            r_len = (
                abs(float(pearsonr(layer_plane["scores"][:, k], mean_length).statistic))
                if mean_length is not None
                else float("nan")
            )
            lines.append(
                f"  PC{k + 1}: evr {layer_plane['evr'][k]:.3f} | "
                f"|r| valence {layer_plane['rv'][k]:.2f} arousal {layer_plane['ra'][k]:.2f} "
                f"dominance {r_dom:.2f} story-length {r_len:.2f}"
            )
    return lines


def _s8_alignment(
    ctx: GeometryContext, n_pcs: int
) -> tuple[Float[np.ndarray, "it_pcs base_pcs"], Float[np.ndarray, " angles"], list[str]]:
    """Cross-model alignment at the default layer: |score correlation| between
    instruct and base components, principal angles between top-3 subspaces, and
    the extremes of the instruct mystery axis (it-PC1)."""
    layer_pos = ctx.layers.index(ctx.default_layer)
    # raw-signed scores: pca.transform centers with the fitted mean and is
    # untouched by the circumplex sign flips applied to plane[...]["scores"]
    scores_raw = {
        label: ctx.plane[label][ctx.default_layer]["pca"].transform(
            ctx.models[label][:, layer_pos, :]
        )
        for label in ctx.labels
    }
    it, base = ctx.labels
    corr = np.array(
        [
            [
                float(pearsonr(scores_raw[it][:, i], scores_raw[base][:, j]).statistic)
                for j in range(n_pcs)
            ]
            for i in range(n_pcs)
        ]
    )
    angles = np.degrees(
        subspace_angles(
            ctx.plane[base][ctx.default_layer]["pca"].components_[:3].T,
            ctx.plane[it][ctx.default_layer]["pca"].components_[:3].T,
        )
    )
    lines = [
        "",
        f"|score correlation|, instruct PCs (rows) x base PCs (columns), layer {ctx.default_layer}:",
    ]
    for i in range(n_pcs):
        lines.append(
            f"  it-PC{i + 1}: " + "  ".join(f"{abs(corr[i, j]):.2f}" for j in range(n_pcs))
        )
    lines.append(
        f"principal angles between top-3 subspaces: {[f'{a:.1f}' for a in angles]} degrees"
    )
    lines.append(
        f"headline: it-PC1 best base correlate {float(np.abs(corr[0]).max()):.2f} (new structure); "
        f"it-PC3 x base-PC1 {abs(corr[2, 0]):.2f} (valence bundle survives); "
        f"it-PC2 x base-PC2 {abs(corr[1, 1]):.2f} (arousal survivor)"
    )
    order = np.argsort(scores_raw[it][:, 0])
    lines.append("")
    lines.append(f"it-PC1 extremes: low = {[ctx.emotions[i] for i in order[:8]]}")
    lines.append(f"                 high = {[ctx.emotions[i] for i in order[-8:][::-1]]}")
    return corr, angles, lines


def _s8_title(ctx: GeometryContext, layer: int) -> str:
    """Per-layer question title: the slider retitles so the verdict never
    freezes over moving data."""
    d_it = ctx.plane[ctx.it_label][layer]
    d_base = ctx.plane[ctx.base_label][layer]
    return (
        f"What sits above valence in the instruct reader at layer {layer}?<br>"
        f"valence-best: instruct PC{d_it['vb'] + 1} (|r|={d_it['r_vb']:.2f}) vs "
        f"base PC{d_base['vb'] + 1} (|r|={d_base['r_vb']:.2f}); "
        f"instruct PC1 (the inserted axis) holds {d_it['evr'][0]:.0%} of variance"
        f"<br><sup>one bar = |r| between one component's per-emotion scores and one NRC axis "
        f"({len(ctx.matched)} matched emotions); dotted gray line = variance share; "
        "dashed line = strong anchor: base PC1-valence |r| at this layer</sup>"
        "<br><sup>vectors read by each panel's model from gemma-4-4B-written stories | "
        "failure anchor: bars at 0 = component carries no affective signal</sup>"
    )


def _s8_add_layer_traces(
    fig: go.Figure,
    ctx: GeometryContext,
    dominance: Float[np.ndarray, " matched"],
    layer: int,
    n_pcs: int,
) -> None:
    """Add one layer's five traces per panel (three |r| bars, the variance-share
    line, the strong-anchor line), visible only at the default layer."""
    axis_colors = {"valence": "#1f77b4", "arousal": "#ff7f0e", "dominance": "#2ca02c"}
    pcs_x = [f"PC{k + 1}" for k in range(n_pcs)]
    # the dashed strong anchor: matching it means "as strong as the base
    # model's replicated PC1-valence correlation at this same layer"
    anchor_level = float(ctx.plane[ctx.base_label][layer]["rv"][0])
    for col, label in enumerate(ctx.labels, start=1):
        layer_plane = ctx.plane[label][layer]
        r_dom_layer = [
            abs(float(pearsonr(layer_plane["scores"][ctx.matched, k], dominance).statistic))
            for k in range(n_pcs)
        ]
        profiles = {
            "valence": layer_plane["rv"][:n_pcs],
            "arousal": layer_plane["ra"][:n_pcs],
            "dominance": r_dom_layer,
        }
        for axis_name, values in profiles.items():
            fig.add_bar(
                x=pcs_x,
                y=list(values),
                name=axis_name,
                marker_color=axis_colors[axis_name],
                visible=(layer == ctx.default_layer),
                showlegend=(col == 1),
                legendgroup=axis_name,
                row=1,
                col=col,
            )
        fig.add_scatter(
            x=pcs_x,
            y=list(layer_plane["evr"][:n_pcs]),
            name="variance share",
            mode="lines+markers",
            line=dict(color="#7f7f7f", dash="dot"),
            visible=(layer == ctx.default_layer),
            showlegend=(col == 1),
            legendgroup="evr",
            row=1,
            col=col,
        )
        fig.add_scatter(
            x=pcs_x,
            y=[anchor_level] * n_pcs,
            name="strong anchor: base PC1-valence |r|",
            mode="lines",
            line=dict(color="#d62728", dash="dash", width=1.5),
            visible=(layer == ctx.default_layer),
            showlegend=(col == 1),
            legendgroup="anchor",
            row=1,
            col=col,
        )


def _s8_figure(
    ctx: GeometryContext, dominance: Float[np.ndarray, " matched"], n_pcs: int
) -> go.Figure:
    """Per-PC |r| profile per layer (slider), instruct vs base; variance share
    as a dotted line, the base PC1-valence level as the dashed strong anchor."""
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, subplot_titles=list(ctx.labels), horizontal_spacing=0.06
    )
    for layer in ctx.layers:
        _s8_add_layer_traces(fig, ctx, dominance, layer, n_pcs)
    traces_per_layer = 5 * 2  # (3 bars + evr line + anchor line) x 2 panels
    steps = [
        dict(
            method="update",
            label=str(layer),
            args=[
                {
                    "visible": [
                        pos // traces_per_layer == li
                        for pos in range(len(ctx.layers) * traces_per_layer)
                    ]
                },
                {"title.text": _s8_title(ctx, layer)},
            ],
        )
        for li, layer in enumerate(ctx.layers)
    ]
    fig.update_layout(
        title=_s8_title(ctx, ctx.default_layer),
        title_font_size=14,
        barmode="group",
        height=530,
        width=1150,
        margin=dict(t=150),
        yaxis_title="|Pearson r| (bars), variance share (line)",
        sliders=[
            dict(
                active=ctx.layers.index(ctx.default_layer),
                currentvalue={"prefix": "layer "},
                pad={"t": 40},
                steps=steps,
            )
        ],
    )
    return fig


def s8_displacement_figure(
    ctx: GeometryContext,
    dominance_by_term: dict[str, float],
    mean_story_length: Float[np.ndarray, " emotions"] | None,
    n_pcs: int = 5,
) -> tuple[go.Figure, dict[str, Any]]:
    """Section 8 exhibit: what displaced valence in the instruct model (Q1.H1.E3).

    Returns ``(figure, stats)``. ``stats["lines"]`` is the printed record:
    per-PC profile tables at the default layer, the instruct-vs-base score
    correlation matrix, principal angles between top-3 subspaces, and the
    extremes of the instruct mystery axis. ``stats`` also carries the raw
    ``corr`` [it_pcs base_pcs] and ``angles`` arrays.
    """
    dominance = np.array([dominance_by_term[ctx.emotions[i].lower()] for i in ctx.matched])
    lines = _s8_profile_lines(ctx, dominance, mean_story_length, n_pcs)
    corr, angles, alignment_lines = _s8_alignment(ctx, n_pcs)
    stats = {"lines": lines + alignment_lines, "corr": corr, "angles": angles}
    return _s8_figure(ctx, dominance, n_pcs), stats


def _s9_ablated_similarities(
    ctx: GeometryContext,
) -> dict[tuple[str, int, int], Float[np.ndarray, " pairs"]]:
    """Upper-triangle cosine similarities per (model label, layer, top-k
    components removed), k in 0..2; raw-signed scores via ``pca.transform``."""
    upper = np.triu_indices(len(ctx.emotions), k=1)
    flat_by_key: dict[tuple[str, int, int], Float[np.ndarray, " pairs"]] = {}
    for label in ctx.labels:
        for layer_pos, layer in enumerate(ctx.layers):
            centered = ctx.models[label][:, layer_pos, :] - ctx.models[label][:, layer_pos, :].mean(
                axis=0
            )
            pca = ctx.plane[label][layer]["pca"]
            raw_scores = pca.transform(ctx.models[label][:, layer_pos, :])
            for k in (0, 1, 2):
                ablated = centered - raw_scores[:, :k] @ pca.components_[:k]
                normed = ablated / np.linalg.norm(ablated, axis=1, keepdims=True)
                flat_by_key[label, layer, k] = (normed @ normed.T)[upper]
    return flat_by_key


def _s9_band_stats(
    ctx: GeometryContext, flat_by_key: dict[tuple[str, int, int], Float[np.ndarray, " pairs"]]
) -> tuple[dict[str, Any], list[str]]:
    """Within-late and mid-to-late RSA means per (model, top-k removed), plus
    cross-model late-band agreement unablated and with instruct top-k removed."""
    mid_pos = [
        ctx.layers.index(layer) for layer in ctx.layers if MID_BAND[0] <= layer <= MID_BAND[1]
    ]
    late_pos = [
        ctx.layers.index(layer) for layer in ctx.layers if LATE_BAND[0] <= layer <= LATE_BAND[1]
    ]
    stats: dict[str, Any] = {"bands": {}, "cross_late": {}}
    lines = []
    for label in ctx.labels:
        parts = []
        for k in (0, 1, 2):
            rsa = np.corrcoef(np.stack([flat_by_key[label, layer, k] for layer in ctx.layers]))
            within = rsa[np.ix_(late_pos, late_pos)][np.triu_indices(len(late_pos), k=1)].mean()
            mid_to_late = rsa[np.ix_(mid_pos, late_pos)].mean()
            stats["bands"][label, k] = (float(within), float(mid_to_late))
            parts.append(
                f"top-{k} removed: within-late {within:.3f}, mid-to-late {mid_to_late:.3f}"
            )
        lines.append(f"{label:22s} " + " | ".join(parts))
    it, base = ctx.labels
    for k in (0, 1, 2):
        cross = np.array(
            [
                [
                    float(np.corrcoef(flat_by_key[it, li, k], flat_by_key[base, lj, 0])[0, 1])
                    for lj in ctx.layers
                ]
                for li in ctx.layers
            ]
        )
        stats["cross_late"][k] = float(cross[np.ix_(late_pos, late_pos)].mean())
        if k == 0:
            stats["cross_matrix_unablated"] = cross
        tag = "unablated" if k == 0 else f"instruct top-{k} removed (post-hoc)"
        lines.append(f"cross-model late x late, {tag}: {stats['cross_late'][k]:.3f}")
    return stats, lines


def _s9_figure(ctx: GeometryContext, stats: dict[str, Any]) -> go.Figure:
    """Four-panel mechanism figure: instruct RSA unablated and top-1-ablated,
    cross-model RSA, and top-component continuity with the handoff band."""
    it, base = ctx.labels
    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
        subplot_titles=(
            "instruct RSA, unablated (section 6's left panel)",
            "instruct RSA, per-layer top component removed",
            "cross-model RSA: instruct (rows) x base (cols)",
            "top-component continuity between adjacent layers",
        ),
    )
    heat_kw = dict(colorscale="Viridis", zmin=0, zmax=1, showscale=False)
    tick_kw = dict(
        tickvals=list(range(len(ctx.layers))),
        ticktext=[str(layer) for layer in ctx.layers],
        tickfont=dict(size=8),
    )
    fig.add_trace(go.Heatmap(z=stats["rsa_it_unablated"], **heat_kw), row=1, col=1)
    fig.add_trace(go.Heatmap(z=stats["rsa_it_top1_removed"], **heat_kw), row=1, col=2)
    fig.add_trace(
        go.Heatmap(
            z=stats["cross_matrix_unablated"],
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(
                title="agreement r<br>(1 = same similarity<br>structure, 0 = unrelated)",
                len=0.4,
                y=0.8,
            ),
        ),
        row=2,
        col=1,
    )
    for row, col in ((1, 1), (1, 2), (2, 1)):
        fig.update_xaxes(title_text="layer", row=row, col=col, **tick_kw)
        fig.update_yaxes(title_text="layer", autorange="reversed", row=row, col=col, **tick_kw)
    _s9_continuity_panel(fig, ctx, stats)
    fig.update_layout(
        title=_s9_title(ctx, stats),
        title_font_size=14,
        height=820,
        width=1150,
        # horizontal legend below the panels: keeps the continuity curves clear
        legend=dict(orientation="h", y=-0.06, x=0.5, xanchor="center"),
        margin=dict(t=140),
    )
    return fig


def _s9_continuity_panel(fig: go.Figure, ctx: GeometryContext, stats: dict[str, Any]) -> None:
    """Bottom-right panel: adjacent-layer |cos| of each model's top component,
    the 24-30 handoff band, and in-plot labels grading high vs low cosine."""
    boundary_mid = [(a + b) / 2 for a, b in zip(ctx.layers[:-1], ctx.layers[1:])]
    for label, color in ((ctx.it_label, "#d62728"), (ctx.base_label, "#1f77b4")):
        fig.add_scatter(
            x=boundary_mid,
            y=stats["continuity"][label],
            mode="lines+markers",
            name=label,
            line=dict(color=color),
            row=2,
            col=2,
        )
    fig.add_vrect(x0=24, x1=30, fillcolor="#d62728", opacity=0.12, line_width=0, row=2, col=2)
    # grade the continuity panel in place: what high and low cosine mean
    fig.add_annotation(
        x=boundary_mid[len(boundary_mid) // 2],
        y=1.04,
        showarrow=False,
        text="|cos| = 1: same top axis at adjacent layers",
        font=dict(size=9),
        row=2,
        col=2,
    )
    fig.add_annotation(
        x=boundary_mid[len(boundary_mid) // 2],
        y=0.05,
        showarrow=False,
        text="|cos| = 0: top axis replaced (regime change)",
        font=dict(size=9),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="layer boundary", row=2, col=2)
    fig.update_yaxes(title_text="|cos| adjacent PC1", range=[0, 1.1], row=2, col=2)


def _s9_title(ctx: GeometryContext, stats: dict[str, Any]) -> str:
    """Question-form title whose headline values come from the computed stats."""
    it, base = ctx.labels
    return (
        "Why does the instruct RSA fragment? Two inserted axis regimes over a preserved "
        "emotion geometry"
        f"<br><sup>removing each layer's top component lifts instruct mid-to-late agreement "
        f"{stats['bands'][it, 0][1]:.2f} to {stats['bands'][it, 1][1]:.2f} but lowers base "
        f"{stats['bands'][base, 0][1]:.2f} to {stats['bands'][base, 1][1]:.2f}; "
        f"instruct-vs-base late-band agreement {stats['cross_late'][0]:.2f} unablated, "
        f"{stats['cross_late'][2]:.2f} with instruct top-2 removed</sup>"
        "<br><sup>shaded band: the 24-30 handoff where the instruct top component changes "
        "identity | vectors read by each model from gemma-4-4B-written stories</sup>"
    )


def s9_fragmentation_figure(ctx: GeometryContext) -> tuple[go.Figure, dict[str, Any]]:
    """Section 9 exhibit: why the instruct RSA fragments (Q1.H1.E4).

    Returns ``(figure, stats)``. ``stats["lines"]`` is the printed record:
    within-late / mid-to-late RSA means per model under top-0/1/2 component
    ablation, then cross-model late-band agreement unablated and with the
    instruct top components removed. ``stats["continuity"][label]`` holds the
    adjacent-layer |cos| of each model's top component direction.
    """
    flat_by_key = _s9_ablated_similarities(ctx)
    stats, lines = _s9_band_stats(ctx, flat_by_key)
    stats["lines"] = lines
    it = ctx.it_label
    stats["rsa_it_unablated"] = np.corrcoef(
        np.stack([flat_by_key[it, layer, 0] for layer in ctx.layers])
    )
    stats["rsa_it_top1_removed"] = np.corrcoef(
        np.stack([flat_by_key[it, layer, 1] for layer in ctx.layers])
    )
    stats["continuity"] = {
        label: [
            abs(
                float(
                    ctx.plane[label][a]["pca"].components_[0]
                    @ ctx.plane[label][b]["pca"].components_[0]
                )
            )
            for a, b in zip(ctx.layers[:-1], ctx.layers[1:])
        ]
        for label in ctx.labels
    }
    return _s9_figure(ctx, stats), stats
