"""Sections 4 and 5: the scale test and probe-direction convergence.

Section 4 asks whether more probe data fixes detection (16 to 256 stories per
emotion); section 5 asks whether the probe directions are still moving at
those corpus sizes, which is what "more data would help" would require. Both
read the same frozen bundle, ``results/e6_scale_means.npz``.

Scores here center on the 12 probes themselves: the scale corpus extracted
the 12 battery emotions only (see :mod:`._data`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float
from plotly.subplots import make_subplots

from emotion_vectors.artifacts import fetch

from ._data import (
    CHANCE_CORRECT,
    IT_LABEL,
    PASS_BAR,
    READOUT_LABELS,
    READOUTS,
    REGISTERED_LAYER,
    ModelSweep,
    battery_counts,
    figure_title,
)

SCALE_WIDTH_PX = 900
# One scored configuration: (prompt format, layer, readout, paper count, held-out count).
Configuration = tuple[str, int, str, int, int]


@dataclass(frozen=True)
class ScaleCorpus:
    """The scale bundle: probe means at four corpus sizes, plus neutral text.

    ``means`` [corpus_sizes battery_emotions layers d_model] holds the probe
    means built from n stories per emotion, ``n_buckets`` the sizes in
    ascending order, ``emotions`` the row order, ``layers`` the layer grid,
    and ``neutral`` [neutral_texts layers d_model] the neutral-transcript
    activations section 6 projects out.
    """

    means: Float[np.ndarray, "corpus_sizes battery_emotions layers d_model"]
    n_buckets: list[int]
    emotions: list[str]
    layers: list[int]
    neutral: Float[np.ndarray, "neutral_texts layers d_model"]

    @property
    def reference_size(self) -> int:
        """The largest corpus size, which every smaller size is compared to."""
        return self.n_buckets[-1]


def load_scale_corpus(layers: list[int]) -> ScaleCorpus:
    """Load the scale bundle and the neutral-transcript bundle.

    ``layers`` is the shared layer grid; a bundle that disagrees with it
    raises, because sections 4 to 6 score against the same swept activations.
    """
    bundle = np.load(fetch("e6_scale_means.npz"), allow_pickle=True)
    bundle_layers = [int(layer) for layer in bundle["layers"]]
    if bundle_layers != layers:
        raise ValueError(f"scale bundle layers {bundle_layers} differ from the sweep's {layers}")
    neutral = np.load(fetch("e7_neutral_bundle.npz"))["vectors"].astype(np.float32)
    return ScaleCorpus(
        means=bundle["means"].astype(np.float32),
        n_buckets=[int(n) for n in bundle["n_buckets"]],
        emotions=[str(emotion) for emotion in bundle["emotions"]],
        layers=bundle_layers,
        neutral=neutral,
    )


def best_scores(
    model: ModelSweep, probe_means: Float[np.ndarray, "battery_emotions layers d_model"]
) -> tuple[dict[str, int], Configuration]:
    """Sweep one probe set over every configuration and keep the maxima.

    Returns ``(best, configuration)``. ``best`` holds the highest count on
    each battery taken independently (so the two can come from different
    configurations), while ``configuration`` is the single combination whose
    WORSE battery is highest: that is the one the registered dual-battery
    rule grades. Ties keep the first configuration in sweep order.
    """
    best = {"scenario": 0, "heldout": 0}
    best_configuration: Configuration | None = None
    best_worse_battery = -1
    for fmt in model.formats:
        for layer_pos, layer in enumerate(model.layers):
            for readout in READOUTS:
                paper, held_out = battery_counts(
                    model, probe_means[:, layer_pos, :], fmt, readout, layer_pos
                )
                best["scenario"] = max(best["scenario"], paper)
                best["heldout"] = max(best["heldout"], held_out)
                if min(paper, held_out) > best_worse_battery:
                    best_worse_battery = min(paper, held_out)
                    best_configuration = (fmt, int(layer), readout, paper, held_out)
    if best_configuration is None:
        raise ValueError(f"{model.label}: no configurations to score")
    return best, best_configuration


def describe_configuration(configuration: Configuration) -> str:
    """One configuration in words, for a printed line or a figure label."""
    fmt, layer, readout, paper, held_out = configuration
    return (
        f"{fmt} format, layer {layer}, {READOUT_LABELS[readout]} readout"
        f" ({paper}/12 paper, {held_out}/12 held-out)"
    )


def scale_rows(model: ModelSweep, scale: ScaleCorpus) -> list[tuple[int, int, int, Configuration]]:
    """Per corpus size: (n, best paper, best held-out, best joint configuration)."""
    rows = []
    for size_pos, n in enumerate(scale.n_buckets):
        best, configuration = best_scores(model, scale.means[size_pos])
        rows.append((n, best["scenario"], best["heldout"], configuration))
    return rows


def _add_scale_anchors(fig: go.Figure, n_buckets: list[int]) -> None:
    """Draw the grading scale: the registered pass bar and the chance level."""
    fig.add_hline(
        y=PASS_BAR,
        line_dash="dot",
        line_color="#2ca02c",
        annotation_text=f"registered pass bar: {PASS_BAR} of 12 on BOTH batteries",
        annotation_position="top left",
    )
    fig.add_hline(
        y=CHANCE_CORRECT,
        line_dash="dot",
        line_color="#888888",
        annotation_text=f"failure anchor: guessing lands near {CHANCE_CORRECT:.0f} of 12",
        annotation_position="bottom left",
    )
    fig.update_xaxes(tickvals=n_buckets)


def scale_figure(
    rows: list[tuple[int, int, int, Configuration]],
) -> tuple[go.Figure, dict[str, Any]]:
    """Section 4 exhibit: best battery score against probe-corpus size.

    Input: ``scale_rows`` output. Returns ``(figure, stats)``;
    ``stats["lines"]`` prints each corpus size's bests and the configuration
    that achieved the best joint score, and ``stats["peak_joint"]`` is the
    highest score any single combination reached on both batteries.
    """
    sizes = [row[0] for row in rows]
    joint = [min(row[3][3], row[3][4]) for row in rows]
    peak_joint = max(joint)
    verdict = "No" if peak_joint < PASS_BAR else "Yes"
    fig = go.Figure()
    for values, name, dash, color in (
        (
            [row[1] for row in rows],
            "best on the paper battery (any combination)",
            "solid",
            "#1f77b4",
        ),
        (
            [row[2] for row in rows],
            "best on the held-out battery (any combination)",
            "dash",
            "#ff7f0e",
        ),
        (
            joint,
            "best ONE combination scored on both batteries (the graded read)",
            "solid",
            "#d62728",
        ),
    ):
        fig.add_scatter(
            x=sizes,
            y=values,
            mode="lines+markers",
            name=name,
            line=dict(dash=dash, color=color, width=3 if color == "#d62728" else 2),
            marker=dict(size=8),
        )
    _add_scale_anchors(fig, sizes)
    fig.update_layout(
        title=figure_title(
            f"Does 16x more probe data fix detection? {verdict}: the best single combination"
            f" measured on BOTH batteries peaks at {peak_joint} of 12 across the whole range,"
            f" against a registered bar of {PASS_BAR}",
            [
                "one marker = one probe-corpus size; the blue and orange lines take the best"
                " score on each battery separately, so they can come from DIFFERENT"
                " combinations; the red line is the one combination graded by the registered"
                " rule",
                f"failure anchor: guessing lands near {CHANCE_CORRECT:.0f} of 12; strength"
                f" anchor: the registered {PASS_BAR}-of-12 bar. Scale test on {IT_LABEL}"
                " (TREE Q1.H2.E6, results/e6_scale_means.npz); our extension, no source figure",
            ],
            SCALE_WIDTH_PX,
        ),
        title_font_size=13,
        # grow the wrapped title downward from the top of the margin
        title_y=0.985,
        title_yanchor="top",
        xaxis_title="stories per emotion the probes were built from (log scale)",
        xaxis_type="log",
        yaxis_title="scenarios correct (of 12)",
        yaxis_range=[0, 12],
        width=SCALE_WIDTH_PX,
        height=620,
        # bottom margin holds the x-axis title above a three-row legend
        margin=dict(t=170, b=190),
        legend=dict(orientation="h", yanchor="top", y=-0.20, font=dict(size=10)),
    )
    lines = [
        f"n={n:3d}: best paper {paper}/12, best heldout {held_out}/12,"
        f" best joint config {describe_configuration(configuration)}"
        for n, paper, held_out, configuration in rows
    ]
    lines.append(f"peak joint score over all corpus sizes: {peak_joint}/12, bar {PASS_BAR}/12")
    return fig, {"lines": lines, "peak_joint": peak_joint}


def convergence_cosines(
    scale: ScaleCorpus,
) -> tuple[Float[np.ndarray, "layers smaller_sizes"], Float[np.ndarray, "layers smaller_sizes"]]:
    """(mean, minimum) cosine to the largest-corpus probe directions.

    For every layer and every corpus size below the reference, each emotion's
    contrast direction (its probe minus the set mean) is compared with the
    same emotion's direction built from the full corpus. Returns the
    per-emotion mean and minimum of those cosines.
    """
    n_smaller = len(scale.n_buckets) - 1
    means = np.zeros((len(scale.layers), n_smaller))
    minima = np.zeros((len(scale.layers), n_smaller))
    for layer_pos in range(len(scale.layers)):
        reference = scale.means[-1, :, layer_pos, :]
        reference = reference - reference.mean(axis=0)
        for size_pos in range(n_smaller):
            current = scale.means[size_pos, :, layer_pos, :]
            current = current - current.mean(axis=0)
            cosines = (current * reference).sum(1) / (
                np.linalg.norm(current, axis=1) * np.linalg.norm(reference, axis=1)
            )
            means[layer_pos, size_pos] = float(cosines.mean())
            minima[layer_pos, size_pos] = float(cosines.min())
    return means, minima


def _convergence_title(
    mean_cosines: Float[np.ndarray, " smaller_sizes"], scale: ScaleCorpus, layer: int
) -> str:
    """Per-slider-step title carrying THIS layer's convergence verdict."""
    largest_smaller = scale.n_buckets[-2]
    reached = float(mean_cosines[-1])
    verdict = "Yes" if reached >= 0.99 else "Not fully"
    return figure_title(
        f"Do the probe directions stabilize as the corpus grows? {verdict}: at layer {layer},"
        f" {largest_smaller} stories per emotion already reproduce the"
        f" {scale.reference_size}-story directions at cosine {reached:.3f}",
        [
            "one marker = one corpus size, scored as the cosine between its probe directions and"
            f" the full {scale.reference_size}-story directions, averaged over the 12 emotions"
            " (each set's own mean removed first, so this compares what separates the emotions)",
            "failure anchor: 0 = unrelated directions; comparator: the dashed line is how far a"
            " genuinely DIFFERENT probe set sits (self-generated stories against the gemma-4-4B"
            " corpus probes, section 3, at this same layer); strength anchor: 1 = identical",
            f"slider picks the layer; layer {REGISTERED_LAYER} (*) is where the printed numbers"
            f" and the campaign record report (TREE Q1.H2.E6, {IT_LABEL})",
        ],
        SCALE_WIDTH_PX,
    )


def _add_convergence_traces(
    fig: go.Figure,
    scale: ScaleCorpus,
    mean_cosines: Float[np.ndarray, "layers smaller_sizes"],
    comparator: float,
) -> None:
    """Add the convergence curve, its comparator line, and the two anchors.

    The comparator is a trace rather than a reference line because it is a
    per-layer measurement and has to move when the slider does.
    """
    sizes = scale.n_buckets[:-1]
    fig.add_scatter(
        x=sizes,
        y=mean_cosines,
        mode="lines+markers",
        name=f"probes from n stories vs the full {scale.reference_size}-story probes",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=9),
    )
    fig.add_scatter(
        x=sizes,
        y=[comparator] * len(sizes),
        mode="lines",
        name="comparator: a different probe set (self-generated vs 4B corpus, section 3)",
        line=dict(color="#7f7f7f", dash="dash", width=2),
    )
    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="#2ca02c",
        annotation_text="1.0 = identical to the full-corpus directions",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=0.0, line_dash="dot", line_color="#888888", annotation_text="0 = unrelated directions"
    )


def _convergence_slider(
    scale: ScaleCorpus,
    mean_cosines: Float[np.ndarray, "layers smaller_sizes"],
    comparator_by_layer: dict[int, float],
) -> list[dict[str, Any]]:
    """One slider step per layer: move both traces, then RETITLE.

    The verdict (how close the smaller corpora come to the full-corpus
    directions) changes with the layer, so a frozen title over moving data
    would be a defect; every step carries its own layer's verdict.
    """
    n_sizes = len(scale.n_buckets) - 1
    return [
        dict(
            method="update",
            args=[
                {
                    "y": [
                        mean_cosines[layer_pos].tolist(),
                        [comparator_by_layer[int(layer)]] * n_sizes,
                    ]
                },
                {"title.text": _convergence_title(mean_cosines[layer_pos], scale, int(layer))},
            ],
            label=f"{layer}*" if layer == REGISTERED_LAYER else str(layer),
        )
        for layer_pos, layer in enumerate(scale.layers)
    ]


def convergence_figure(
    scale: ScaleCorpus, comparator_by_layer: dict[int, float]
) -> tuple[go.Figure, dict[str, Any]]:
    """Section 5 exhibit: probe-direction convergence, with a layer slider.

    Inputs: the :class:`ScaleCorpus` and ``comparator_by_layer``, section 3's
    mean cosine between the self-generated and corpus probe sets at each
    layer, which is drawn as the "a different probe set looks like this"
    grading anchor. Returns ``(figure, stats)``; ``stats["lines"]`` prints the
    registered layer's mean and minimum per corpus size.
    """
    mean_cosines, min_cosines = convergence_cosines(scale)
    sizes = scale.n_buckets[:-1]
    registered_pos = scale.layers.index(REGISTERED_LAYER)
    fig = go.Figure()
    _add_convergence_traces(
        fig, scale, mean_cosines[registered_pos], comparator_by_layer[REGISTERED_LAYER]
    )
    fig.update_layout(
        title=_convergence_title(mean_cosines[registered_pos], scale, REGISTERED_LAYER),
        title_font_size=13,
        # grow the wrapped title downward from the top of the margin
        title_y=0.985,
        title_yanchor="top",
        xaxis_title="stories per emotion the probes were built from (log scale)",
        xaxis_type="log",
        xaxis_tickvals=sizes,
        yaxis_title="mean cosine (1 = identical)",
        yaxis_range=[-0.05, 1.08],
        width=SCALE_WIDTH_PX,
        height=640,
        # bottom margin holds the x-axis title, the legend, and the slider
        margin=dict(t=180, b=200),
        legend=dict(orientation="h", yanchor="top", y=-0.18, font=dict(size=10)),
        sliders=[
            dict(
                active=registered_pos,
                currentvalue=dict(prefix="layer: "),
                pad=dict(t=110),
                steps=_convergence_slider(scale, mean_cosines, comparator_by_layer),
            )
        ],
    )
    lines = [
        f"n={n:3d} vs n={scale.reference_size}: mean contrast-direction cosine"
        f" {mean_cosines[registered_pos, size_pos]:.3f}"
        f" (min {min_cosines[registered_pos, size_pos]:.3f})"
        for size_pos, n in enumerate(sizes)
    ]
    return fig, {"lines": lines, "mean_cosines": mean_cosines, "min_cosines": min_cosines}
