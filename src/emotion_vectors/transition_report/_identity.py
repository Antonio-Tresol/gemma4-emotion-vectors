"""Section 1 exhibit: the identity read, story set by story set.

One grouped-bar panel per story set, one bar per (probe set, layer), showing
the tagged emotion's median rank as a fraction of how many probes the set
holds, so a 12-probe set and the 171-probe set share one axis. The builder
returns ``(figure, stats)`` where ``stats["lines"]`` holds every number the
notebook prints, so no number appears in markdown without being computed
beside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._data import (
    CHANCE_FRACTION,
    IDENTITY_BAR_FRACTION,
    PROBE_SET_COLORS,
    PROBE_SET_LABEL,
    PROBE_SET_SIZE,
    PROBED_MODEL,
    StorySet,
    TransitionEvidence,
    figure_title,
    title_top,
)

IDENTITY_WIDTH_PX = 1400
IDENTITY_HEIGHT_PX = 520
ANCHOR_COLOR = "#2a3f5f"
EVIDENCE_NOTE = (
    "evidence: q3_gate_r1_it.json, q3_gate_r1_deepseek.json, q3_gate_r1_it_v2.json"
    " (experiment-artifacts dataset)"
)


@dataclass(frozen=True)
class IdentityRow:
    """One (story set, probe set) row of the identity read.

    Fields: ``story_set`` and ``probe_set`` (which cell this is), ``panel`` (the
    1-based subplot column its story set occupies), ``ranks`` (median rank per
    layer, in layer order), ``fractions`` (the same divided by how many probes
    the set holds, which is what the bars plot), ``worst_layer`` and
    ``worst_p`` (the layer whose wrong-emotion shuffle p value is largest, and
    that p value).
    """

    story_set: StorySet
    probe_set: str
    panel: int
    ranks: list[float]
    fractions: list[float]
    worst_layer: int
    worst_p: float


def identity_rows(evidence: TransitionEvidence) -> list[IdentityRow]:
    """Every scored (story set, probe set) cell of the identity read.

    Input: the loaded :class:`~._data.TransitionEvidence`. Returns one
    :class:`IdentityRow` per cell, in figure order. The panel index comes from
    the story set's position in the evidence, so the subplot grid can be sized
    from this list rather than from a typed column count.
    """
    rows = []
    for panel, story_set in enumerate(evidence.story_sets.values(), start=1):
        # "gate_G" is the identity read's key in the evidence files, kept as-is
        for probe_set in story_set.banks("gate_G"):
            cells = [story_set.identity_cell(probe_set, layer) for layer in evidence.layers]
            if any(cell is None for cell in cells):
                raise ValueError(
                    f"{story_set.label}/{probe_set}: identity read has a missing layer"
                )
            ranks = [float(cell["median_rank"]) for cell in cells]  # type: ignore[index]
            worst_pos = max(
                range(len(cells)),
                key=lambda pos: float(cells[pos]["n2_p"]),  # type: ignore[index]
            )
            rows.append(
                IdentityRow(
                    story_set=story_set,
                    probe_set=probe_set,
                    panel=panel,
                    ranks=ranks,
                    fractions=[rank / PROBE_SET_SIZE[probe_set] for rank in ranks],
                    worst_layer=evidence.layers[worst_pos],
                    worst_p=float(cells[worst_pos]["n2_p"]),  # type: ignore[index]
                )
            )
    return rows


def _identity_traces(fig: go.Figure, rows: list[IdentityRow], layers: list[int]) -> None:
    """Draw one grouped bar per (story set, probe set) into its panel.

    Each probe set gets exactly one legend entry, claimed by whichever panel
    draws it first, so the legend names three probe sets rather than repeating
    them once per story set.
    """
    probe_sets_in_legend: set[str] = set()
    for row in rows:
        fig.add_bar(
            x=[str(layer) for layer in layers],
            y=row.fractions,
            name=PROBE_SET_LABEL[row.probe_set],
            marker_color=PROBE_SET_COLORS[row.probe_set],
            legendgroup=row.probe_set,
            showlegend=row.probe_set not in probe_sets_in_legend,
            hovertemplate=(
                f"{row.story_set.label}<br>{PROBE_SET_LABEL[row.probe_set]}"
                "<br>layer %{x}<br>rank fraction %{y:.3f}<extra></extra>"
            ),
            row=1,
            col=row.panel,
        )
        probe_sets_in_legend.add(row.probe_set)


def _identity_anchors(fig: go.Figure) -> None:
    """Draw the two grading anchors and name them in the legend.

    Failure anchor: chance, where the wrong-emotion shuffle (N2) puts a
    meaningless assignment. Strength anchor: the registered top-decile bar.
    Both are drawn across every panel; their labels live in the legend because
    per-panel annotation text collided with the bars.
    """
    fig.add_hline(y=CHANCE_FRACTION, line_dash="dot", line_color=ANCHOR_COLOR)
    fig.add_hline(y=IDENTITY_BAR_FRACTION, line_dash="dash", line_color=ANCHOR_COLOR)
    for dash, name in (
        ("dot", f"FAILURE anchor: chance = {CHANCE_FRACTION} (measured by the N2 shuffle)"),
        (
            "dash",
            f"STRENGTH anchor: registered bar = {IDENTITY_BAR_FRACTION}"
            " (top decile: rank 17 of 171, rank 1 of 12)",
        ),
    ):
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(dash=dash, color=ANCHOR_COLOR),
            name=name,
        )


def _identity_title(rows: list[IdentityRow]) -> str:
    """Question-form title whose verdict is computed from the plotted rows."""
    small_ranks = [
        rank for row in rows if PROBE_SET_SIZE[row.probe_set] == 12 for rank in row.ranks
    ]
    corpus_ranks = [
        rank for row in rows if PROBE_SET_SIZE[row.probe_set] == 171 for rank in row.ranks
    ]
    n_bars = sum(len(row.fractions) for row in rows)
    n_at_bar = sum(fraction <= IDENTITY_BAR_FRACTION for row in rows for fraction in row.fractions)
    return figure_title(
        "Does the model know which emotion the current phase is? Yes with the sets of twelve"
        f" probes, no with the 171-probe corpus set: median rank {min(small_ranks):.0f} to"
        f" {max(small_ranks):.0f} of 12 for the two sets of twelve, versus"
        f" {min(corpus_ranks):.0f} to {max(corpus_ranks):.0f} of 171 for the corpus set",
        [
            "one bar = one (story set, probe set, layer): the tagged emotion's median rank"
            " over that cell's phases, as a fraction of how many probes the set holds, so sets"
            " of different sizes share one axis; lower is better",
            f"FAILURE anchor: chance = {CHANCE_FRACTION} of the set, where the wrong-emotion"
            f" shuffle (N2) lands. STRENGTH anchor: the registered top-decile bar at"
            f" {IDENTITY_BAR_FRACTION}, which {n_at_bar} of {n_bars} bars reach",
            "panels = story sets (who wrote what the model was reading), colours = which stories"
            f" the probes were built from | {PROBED_MODEL} | {EVIDENCE_NOTE}",
        ],
        IDENTITY_WIDTH_PX,
    )


def identity_figure(evidence: TransitionEvidence) -> tuple[go.Figure, dict[str, Any]]:
    """Section 1 exhibit: the identity read per story set, probe set and layer.

    Input: the loaded :class:`~._data.TransitionEvidence`. Returns
    ``(figure, stats)``. ``stats["lines"]`` prints each cell's median-rank
    range and its worst wrong-emotion-shuffle p value; ``stats["n_panels"]``
    and ``stats["n_cells"]`` are the layout shape DERIVED from the evidence,
    which the notebook asserts against the evidence it loaded.
    """
    rows = identity_rows(evidence)
    n_panels = len(evidence.story_sets)
    fig = make_subplots(
        rows=1,
        cols=n_panels,
        subplot_titles=[each.tick_label for each in evidence.story_sets.values()],
        shared_yaxes=True,
        horizontal_spacing=0.03,
    )
    _identity_traces(fig, rows, evidence.layers)
    _identity_anchors(fig)
    for annotation in fig.layout.annotations:  # only the panel titles exist at this point
        annotation.font = dict(size=11)
    fig.update_xaxes(title_text="layer (residual stream depth)", title_font_size=11)
    fig.update_yaxes(
        title_text="median rank as a fraction of the probe set (lower is better)", row=1, col=1
    )
    fig.update_layout(
        title=_identity_title(rows),
        title_font_size=13,
        # anchor the wrapped title to the very top so it grows downward into
        # the margin instead of centring over the first row of panel titles
        title_y=title_top(IDENTITY_HEIGHT_PX),
        title_yanchor="top",
        barmode="group",
        width=IDENTITY_WIDTH_PX,
        height=IDENTITY_HEIGHT_PX,
        margin=dict(t=155, r=430, b=70),
        legend=dict(x=1.005, y=1.0, xanchor="left", yanchor="top", font=dict(size=10)),
    )
    lines = [
        f"{row.story_set.label:44s} {PROBE_SET_LABEL[row.probe_set]:52s}"
        f" median rank {min(row.ranks):.0f} to {max(row.ranks):.0f}"
        f" of {PROBE_SET_SIZE[row.probe_set]};"
        f" worst shuffle p = {row.worst_p:.4f} (layer {row.worst_layer})"
        for row in rows
    ]
    return fig, {
        "lines": lines,
        "rows": rows,
        "n_panels": n_panels,
        "n_cells": len(rows),
    }
