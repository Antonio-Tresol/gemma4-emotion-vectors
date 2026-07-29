"""Section 3 exhibit: above chance, useful, or meaningless?

Every (story set, probe set) identity result becomes one dot on a
rank-fraction axis divided into three named zones, at the layer the slider
selects. The dot's POSITION answers "is it useful?"; the p value printed
beside it answers the different question "is it distinguishable from chance?".
Every slider step rewrites the title with its own layer's counts, because the
number of dots past the bar changes with depth. The builder returns
``(figure, stats)`` where ``stats["lines"]`` holds every number the notebook
prints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go

from ._data import (
    CHANCE_FRACTION,
    CHANCE_LIKE_FRACTION,
    IDENTITY_BAR_FRACTION,
    PRIMARY_LAYER,
    PROBE_SET_COLORS,
    PROBE_SET_LABEL,
    PROBE_SET_SHORT,
    PROBE_SET_SIZE,
    PROBED_MODEL,
    STORY_SET_SYMBOLS,
    TransitionEvidence,
    figure_title,
    title_top,
)

ZONE_WIDTH_PX = 1200
ZONE_HEIGHT_PX = 900  # 11 two-line y ticks collide at anything shorter
ZONE_X_MAX = 0.72  # leaves room to the right of the widest dot for its label
# One blank category's worth of headroom above the top dot, so the zone labels
# printed across the top of the plot never land on a dot's text.
ZONE_Y_HEADROOM = 1.4
ANCHOR_COLOR = "#2a3f5f"
EVIDENCE_NOTE = (
    "evidence: q3_gate_r1_it.json, q3_gate_r1_deepseek.json, q3_gate_r1_it_v2.json"
    " (experiment-artifacts dataset)"
)


@dataclass(frozen=True)
class ZonePoint:
    """One identity result placed on the usefulness axis at one layer.

    Fields: ``category`` (the y tick, naming the probe set and the story set
    it was read on), ``probe_set`` and ``symbol`` (the colour and shape
    encodings), ``fraction`` (median rank as a fraction of how many probes the
    set holds, the x position), ``rank`` (the same in absolute ranks) and
    ``n2_p`` (the wrong-emotion shuffle p value).
    """

    category: str
    probe_set: str
    symbol: str
    fraction: float
    rank: float
    n2_p: float


def zone_points(evidence: TransitionEvidence, layer: int) -> list[ZonePoint]:
    """Every scored identity cell at one layer, as plottable dots.

    Input: the evidence and the layer to slice. Returns the dots in figure
    order (story set outer, probe set inner), which is also the order the y
    axis stacks its categories in.
    """
    symbols = {
        label: STORY_SET_SYMBOLS[pos % len(STORY_SET_SYMBOLS)]
        for pos, label in enumerate(evidence.story_sets)
    }
    points = []
    for story_set in evidence.story_sets.values():
        # "gate_G" is the identity read's key in the evidence files, kept as-is
        for probe_set in story_set.banks("gate_G"):
            cell = story_set.identity_cell(probe_set, layer)
            if cell is None:
                raise ValueError(
                    f"{story_set.label}/{probe_set}: no identity cell at layer {layer}"
                )
            points.append(
                ZonePoint(
                    category=f"{PROBE_SET_SHORT[probe_set]}<br><i>reading {story_set.label}</i>",
                    probe_set=probe_set,
                    symbol=symbols[story_set.label],
                    fraction=float(cell["median_rank"]) / PROBE_SET_SIZE[probe_set],
                    rank=float(cell["median_rank"]),
                    n2_p=float(cell["n2_p"]),
                )
            )
    return points


def _zone_title(evidence: TransitionEvidence, layer: int) -> str:
    """The title for one slider step: that layer's three-way count."""
    points = zone_points(evidence, layer)
    n_chance_like = sum(point.fraction >= CHANCE_LIKE_FRACTION for point in points)
    n_past_bar = sum(point.fraction <= IDENTITY_BAR_FRACTION for point in points)
    n_between = len(points) - n_chance_like - n_past_bar
    return figure_title(
        "Is any identity reading meaningless, and is any useful by the registered bar?"
        f" At layer {layer}: {n_chance_like} of {len(points)} readings are chance-like,"
        f" {n_past_bar} clear the registered bar, and {n_between} sit above chance but short"
        " of it",
        [
            "one dot = one (probe set, story set) at the selected layer; x = the tagged"
            " emotion's median rank as a fraction of how many probes the set holds, lower is"
            " better. The text beside each dot prints the absolute rank and its"
            " wrong-emotion-shuffle p value, which answers the DIFFERENT question of whether"
            " the reading is distinguishable from chance",
            f"FAILURE anchor: chance = {CHANCE_FRACTION} of the set, measured by 10,000 N2"
            f" shuffles (red zone starts at {CHANCE_LIKE_FRACTION}). STRENGTH anchor: the"
            f" registered top-decile bar at {IDENTITY_BAR_FRACTION} (green zone)",
            "dot colour = which stories the probes were built from, dot shape ="
            f" story set (who wrote what the model was reading) | {PROBED_MODEL}"
            f" | {EVIDENCE_NOTE}",
        ],
        ZONE_WIDTH_PX,
    )


def _zone_traces(fig: go.Figure, evidence: TransitionEvidence) -> int:
    """Add one dot trace per (layer, story set, probe set); return the per-layer count.

    Every layer's dots are drawn once and hidden; the slider reveals exactly
    one layer's block, so the returned count is what the visibility masks are
    built from rather than a typed number.
    """
    per_layer = 0
    for layer in evidence.layers:
        points = zone_points(evidence, layer)
        if layer == evidence.layers[0]:
            per_layer = len(points)
        elif len(points) != per_layer:
            raise ValueError(
                f"layer {layer} scored {len(points)} identity cells, layer"
                f" {evidence.layers[0]} scored {per_layer}: the slider needs one block per layer"
            )
        for point in points:
            fig.add_scatter(
                x=[point.fraction],
                y=[point.category],
                mode="markers+text",
                visible=layer == PRIMARY_LAYER,
                marker=dict(size=14, color=PROBE_SET_COLORS[point.probe_set], symbol=point.symbol),
                text=[
                    f"  rank {point.rank:.0f}/{PROBE_SET_SIZE[point.probe_set]}, p={point.n2_p:.4f}"
                ],
                textposition="middle right",
                textfont=dict(size=11),
                hovertemplate="%{y}<br>rank fraction %{x:.3f}<extra></extra>",
                showlegend=False,
            )
    return per_layer


def _zone_anchors(fig: go.Figure, evidence: TransitionEvidence) -> None:
    """Draw the graded zones and give every colour and shape a legend entry."""
    fig.add_vline(x=CHANCE_FRACTION, line_dash="dot", line_color=ANCHOR_COLOR)
    fig.add_vline(x=IDENTITY_BAR_FRACTION, line_dash="dash", line_color=ANCHOR_COLOR)
    zones = [
        (0.0, IDENTITY_BAR_FRACTION, "green", "good work"),
        (IDENTITY_BAR_FRACTION, CHANCE_LIKE_FRACTION, "orange", "above chance, below the bar"),
        (CHANCE_LIKE_FRACTION, ZONE_X_MAX, "red", "chance-like"),
    ]
    for start, end, color, label in zones:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=0.07,
            line_width=0,
            annotation_text=label,
            annotation_position="top",
            annotation_font_size=10,
        )
    # legend entries carry the two encodings: colour names which stories the
    # probes came from, shape names the story set they were read on
    for probe_set, label in PROBE_SET_LABEL.items():
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=12, color=PROBE_SET_COLORS[probe_set], symbol="circle"),
            name=f"colour: {label}",
        )
    for pos, label in enumerate(evidence.story_sets):
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=12,
                color="#444444",
                symbol=STORY_SET_SYMBOLS[pos % len(STORY_SET_SYMBOLS)],
            ),
            name=f"shape: read on {label}",
        )


def zone_figure(evidence: TransitionEvidence) -> tuple[go.Figure, dict[str, Any]]:
    """Section 3 exhibit: the three-way usefulness diagnosis, with a layer slider.

    Input: the loaded :class:`~._data.TransitionEvidence`. Returns
    ``(figure, stats)``. ``stats["lines"]`` prints each layer's three-way
    count; ``stats["n_dots_per_layer"]`` is the dot count DERIVED from the
    evidence, which the notebook asserts against the cells it loaded.
    """
    fig = go.Figure()
    per_layer = _zone_traces(fig, evidence)
    _zone_anchors(fig, evidence)
    n_legend = len(PROBE_SET_LABEL) + len(evidence.story_sets)
    steps = [
        dict(
            method="update",
            label=f"layer {layer}",
            args=[
                {
                    "visible": [
                        pos == layer_pos
                        for pos in range(len(evidence.layers))
                        for _ in range(per_layer)
                    ]
                    + [True] * n_legend
                },
                {"title.text": _zone_title(evidence, layer)},
            ],
        )
        for layer_pos, layer in enumerate(evidence.layers)
    ]
    fig.update_layout(
        sliders=[
            dict(
                active=evidence.layers.index(PRIMARY_LAYER),
                steps=steps,
                y=-0.10,
                currentvalue=dict(prefix="showing: ", font=dict(size=12)),
            )
        ],
        title=_zone_title(evidence, PRIMARY_LAYER),
        title_font_size=13,
        title_y=title_top(ZONE_HEIGHT_PX),
        title_yanchor="top",
        xaxis_title="median rank as a fraction of the probe set (lower is better)",
        xaxis_range=[0, ZONE_X_MAX],
        # explicit numeric range over the category indices: it adds a blank
        # slot at the top for the zone labels without inventing a category
        yaxis=dict(tickfont=dict(size=10), range=[-0.6, per_layer - 1 + ZONE_Y_HEADROOM]),
        width=ZONE_WIDTH_PX,
        height=ZONE_HEIGHT_PX,
        # the slider sits directly under the x-axis title and the legend
        # below the slider's own layer ticks: at any closer spacing the two
        # collide, which is what the rendered PNGs showed
        margin=dict(t=155, b=270, l=270, r=30),
        legend=dict(orientation="h", y=-0.34, x=0, font=dict(size=9)),
    )
    lines = []
    for layer in evidence.layers:
        points = zone_points(evidence, layer)
        n_chance_like = sum(point.fraction >= CHANCE_LIKE_FRACTION for point in points)
        n_past_bar = sum(point.fraction <= IDENTITY_BAR_FRACTION for point in points)
        worst = max(points, key=lambda point: point.n2_p)
        lines.append(
            f"layer {layer:2d}: {n_chance_like}/{len(points)} chance-like,"
            f" {n_past_bar}/{len(points)} past the registered bar;"
            f" largest shuffle p = {worst.n2_p:.4f} ({PROBE_SET_SHORT[worst.probe_set]},"
            f" rank {worst.rank:.0f}/{PROBE_SET_SIZE[worst.probe_set]})"
        )
    return fig, {"lines": lines, "n_dots_per_layer": per_layer}
