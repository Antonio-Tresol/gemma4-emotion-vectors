"""Section 1 exhibit: the scenario sweep over layer x pooling x prompt format.

One builder, used twice: once for the instruction-tuned model (two prompt
formats) and once for the base model (plain format only). Every score here
centers on the full 171-word extraction, the sweep convention documented in
:mod:`._data`. The builder returns ``(figure, stats)`` where
``stats["lines"]`` holds every number the notebook prints, so no number
appears in markdown without being computed beside it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Int
from plotly.subplots import make_subplots

from ._data import (
    CHANCE_CORRECT,
    N_SCENARIOS,
    PASS_BAR,
    READOUT_LABELS,
    READOUTS,
    SCENARIO_SET_ROLES,
    SCENARIO_SET_SHORT,
    ModelSweep,
    battery_counts,
    figure_title,
)

SWEEP_WIDTH_PX = 1000
PANEL_HEIGHT_PX = 520
SweepGrids = dict[tuple[str, str], Int[np.ndarray, "layers poolings"]]


def sweep_grids(model: ModelSweep) -> SweepGrids:
    """Score every (format, scenario set, layer, pooling) cell of one sweep.

    Input: one :class:`~._data.ModelSweep`. Returns {(format, scenario set):
    grid}, each grid [layers poolings] holding scenarios of 12 whose target
    emotion ranked in the top 3. Centering is the sweep convention (the full
    extraction), which is what sections 1 and 2 registered.
    """
    grids: SweepGrids = {}
    for fmt in model.formats:
        # "scenario" and "heldout" are the stored prompt ``kind`` values
        columns = {"scenario": [], "heldout": []}  # type: dict[str, list[list[int]]]
        for layer_pos in range(len(model.layers)):
            counts = [
                battery_counts(
                    model,
                    model.probe_means(layer_pos),
                    fmt,
                    pooling,
                    layer_pos,
                    center_pool=model.center_pool(layer_pos),
                )
                for pooling in READOUTS
            ]
            columns["scenario"].append([paper for paper, _ in counts])
            columns["heldout"].append([held_out for _, held_out in counts])
        for kind, rows in columns.items():
            grids[fmt, kind] = np.array(rows)
    return grids


def combination_rows(grids: SweepGrids, layers: list[int]) -> list[tuple[int, int, str, int, str]]:
    """Every swept combination as (paper, held-out, format, layer, pooling).

    Sorted best first, so the notebook cell can print the head of the ranking
    and then apply the registered pass rule to the whole list in plain sight.
    """
    formats = sorted({fmt for fmt, _ in grids})
    rows = [
        (
            int(grids[fmt, "scenario"][layer_pos, pooling_pos]),
            int(grids[fmt, "heldout"][layer_pos, pooling_pos]),
            fmt,
            int(layers[layer_pos]),
            pooling,
        )
        for fmt in formats
        for layer_pos in range(len(layers))
        for pooling_pos, pooling in enumerate(READOUTS)
    ]
    rows.sort(reverse=True)
    return rows


def _sweep_title(model_label: str, grids: SweepGrids, layers: list[int], source_note: str) -> str:
    """Question-form title whose verdict is computed from the grids.

    Reports two numbers a reader could otherwise confuse: the single
    brightest cell anywhere (which is a selection-inflated best) and the best
    combination measured on BOTH scenario sets, which is what the registered
    rule actually grades.
    """
    rows = combination_rows(grids, layers)
    best_cell = max(int(grid.max()) for grid in grids.values())
    best_joint = max(min(paper, held_out) for paper, held_out, _, _, _ in rows)
    n_cells = sum(int(grid.size) for grid in grids.values())
    n_at_bar = sum(int((grid >= PASS_BAR).sum()) for grid in grids.values())
    verdict = "No" if best_joint < PASS_BAR else "Yes"
    return figure_title(
        f"Can any combination of layer, pooling and prompt format detect the implied emotion?"
        f" {verdict}: the brightest single cell reaches {best_cell} of 12, the best combination"
        f" measured on BOTH sets of scenarios reaches {best_joint} of 12, and the registered bar"
        f" is {PASS_BAR}",
        [
            f"one cell = one (layer, pooling) combination, scored as the number of scenarios"
            f" (of {N_SCENARIOS}) whose true emotion ranked in the top 3 of the 12 probes;"
            " left panel = the scenarios used to pick the best cell, right panel = fresh"
            " held-out scenarios",
            f"failure anchor: guessing puts the target in the top 3 about"
            f" {CHANCE_CORRECT:.0f} of 12 times; strength anchor: the registered"
            f" {PASS_BAR}-of-12 bar, printed in bold where a cell reaches it"
            f" ({n_at_bar} of {n_cells} cells do)",
            f"probes and scenario activations from {model_label}; {source_note}",
        ],
        SWEEP_WIDTH_PX,
    )


def _add_sweep_panel(
    fig: go.Figure,
    grid: Int[np.ndarray, "layers poolings"],
    layers: list[int],
    row: int,
    col: int,
    show_scale: bool,
) -> None:
    """Draw one panel: the heatmap, its per-cell counts, and its axes.

    The cell count is written into every cell because the color scale alone
    cannot be read to one unit, and cells at or above the pass bar are bolded
    so a passing cell would be visible without reading the colorbar.
    """
    fig.add_trace(
        go.Heatmap(
            z=grid,
            colorscale="Viridis",
            zmin=0,
            zmax=N_SCENARIOS,
            showscale=show_scale,
            colorbar=dict(
                # short horizontal title above a half-height bar: a rotated
                # full sentence would run the width of the right margin
                title=dict(text="scenarios<br>correct<br>(of 12)", side="top", font=dict(size=11)),
                x=1.02,
                len=0.45,
                thickness=14,
                tickfont=dict(size=10),
                tickvals=[0, CHANCE_CORRECT, PASS_BAR, N_SCENARIOS],
                ticktext=[
                    "0 = never",
                    f"{CHANCE_CORRECT:.0f} = chance",
                    f"{PASS_BAR} = PASS BAR",
                    "12 = all",
                ],
            ),
            hovertemplate="layer %{y}<br>pooling %{x}<br>%{z} of 12 correct<extra></extra>",
        ),
        row=row,
        col=col,
    )
    for layer_pos in range(grid.shape[0]):
        for pooling_pos in range(grid.shape[1]):
            value = int(grid[layer_pos, pooling_pos])
            fig.add_annotation(
                x=pooling_pos,
                y=layer_pos,
                text=f"<b>{value}</b>" if value >= PASS_BAR else str(value),
                showarrow=False,
                font=dict(size=9, color="white" if value < 7 else "black"),
                row=row,
                col=col,
            )
    fig.update_xaxes(
        tickvals=list(range(len(READOUTS))),
        ticktext=[READOUT_LABELS[pooling] for pooling in READOUTS],
        tickfont=dict(size=9),
        row=row,
        col=col,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickvals=list(range(len(layers))),
        ticktext=[str(layer) for layer in layers],
        row=row,
        col=col,
    )


def sweep_figure(
    model: ModelSweep, grids: SweepGrids, source_note: str
) -> tuple[go.Figure, dict[str, Any]]:
    """Section 1 exhibit: the sweep grids, one panel pair per prompt format.

    Inputs: the model's :class:`~._data.ModelSweep`, its ``sweep_grids``
    output, and ``source_note`` (the sentence naming the source figure this
    extends). Returns ``(figure, stats)``; ``stats["lines"]`` prints the per
    (format, scenario set) maxima and ``stats["best_joint"]`` is the best score
    a single combination achieved on both sets.
    """
    formats = model.formats
    # two short lines per panel title: plotly does not wrap them, and one long
    # line collides with the neighbouring panel's title
    titles = [
        f"{fmt} format<br><sub>{SCENARIO_SET_ROLES[kind]}</sub>"
        for fmt in formats
        for kind in SCENARIO_SET_ROLES
    ]
    fig = make_subplots(
        rows=len(formats),
        cols=2,
        subplot_titles=titles,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.10,
    )
    for format_pos, fmt in enumerate(formats):
        for kind_pos, kind in enumerate(SCENARIO_SET_ROLES):
            _add_sweep_panel(
                fig,
                grids[fmt, kind],
                model.layers,
                row=format_pos + 1,
                col=kind_pos + 1,
                show_scale=(format_pos == 0 and kind_pos == 0),
            )
        fig.update_yaxes(title_text="layer", row=format_pos + 1, col=1)
    for col in (1, 2):  # x-axis title on the bottom row only, so it cannot collide
        fig.update_xaxes(
            title_text="how the prompt is pooled into one vector", row=len(formats), col=col
        )
    for annotation in fig.layout.annotations[: 2 * len(formats)]:  # the subplot titles
        annotation.font = dict(size=11)
    fig.update_layout(
        title=_sweep_title(model.label, grids, model.layers, source_note),
        title_font_size=13,
        # anchor the wrapped title to the very top so it grows downward into
        # the margin instead of centring over the first row of panel titles
        title_y=0.985,
        title_yanchor="top",
        height=PANEL_HEIGHT_PX * len(formats) + 140,
        width=SWEEP_WIDTH_PX,
        margin=dict(t=175, r=170),
    )
    rows = combination_rows(grids, model.layers)
    best_joint = max(min(paper, held_out) for paper, held_out, _, _, _ in rows)
    lines = [
        f"{model.label}, {fmt} format, {SCENARIO_SET_SHORT[kind]} scenarios:"
        f" best cell {int(grid.max())} of 12 (chance {CHANCE_CORRECT:.0f}, bar {PASS_BAR})"
        for (fmt, kind), grid in grids.items()
    ]
    lines.append(
        f"best single combination across BOTH scenario sets: {best_joint} of 12, bar {PASS_BAR}"
    )
    return fig, {"lines": lines, "best_joint": best_joint, "rows": rows}
