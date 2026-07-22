"""Reusable Plotly layer-scrubber widgets for the layer-exploration notebook.

Every figure here is driven by a self-contained Plotly slider (no ipywidgets, no
live kernel), so the interactive control survives in a statically-viewed
notebook and in exported HTML. Two builders:

- ``layer_scatter_scrubber`` — one slider drives an N-panel scatter (e.g. the
  circumplex, instruct beside base). Per layer it swaps each point's x/y and the
  axis titles; colour and size stay fixed (they encode external NRC ratings, not
  the model's per-layer components), which is exactly what keeps the left-right
  valence gradient reading the same direction across every layer.
- ``layer_table_scrubber`` — one slider swaps a table's cell values per layer
  (e.g. the logit-lens top/bottom tokens).

The notebook computes the per-layer data (PCA scores, lens tables); these helpers
only own the fiddly slider machinery so the notebook cells stay skimmable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _as_list(a: Any) -> list[Any]:
    """Slider ``args`` are serialised raw, so numpy arrays must become lists."""
    return np.asarray(a).tolist()


def _axis_key(kind: str, col: int) -> str:
    """Plotly names the first subplot axis ``xaxis``/``yaxis`` and the rest
    ``xaxis2``/``yaxis2``/... — return the ``.title.text`` relayout key."""
    suffix = "" if col == 1 else str(col)
    return f"{kind}axis{suffix}.title.text"


def layer_scatter_scrubber(
    layers: list[int],
    panels: list[dict],
    *,
    default_layer: int,
    title: str = "",
    colorbar_title: str = "",
    colorscale: str = "RdYlGn",
    figsize: tuple[int, int] = (1150, 620),
) -> go.Figure:
    """Build an N-panel scatter with a shared layer slider.

    ``panels`` is one dict per subplot column, left to right, with keys:
      ``name``     panel title (fixed across layers)
      ``x``,``y``  {layer: 1-D array} — the per-layer point coordinates
      ``xtitle``,``ytitle``  {layer: str} — per-layer axis titles
      ``color``,``size``,``text``  per-point arrays/lists, FIXED across layers
      ``showscale``  bool — show the colour bar on this panel only
    """
    ncol = len(panels)
    fig = make_subplots(
        rows=1,
        cols=ncol,
        subplot_titles=[p["name"] for p in panels],
        horizontal_spacing=0.10,
    )
    for c, p in enumerate(panels, start=1):
        fig.add_trace(
            go.Scatter(
                x=_as_list(p["x"][default_layer]),
                y=_as_list(p["y"][default_layer]),
                mode="markers",
                marker=dict(
                    color=_as_list(p["color"]),
                    colorscale=colorscale,
                    size=_as_list(p["size"]),
                    opacity=0.85,
                    line=dict(color="black", width=0.3),
                    colorbar=dict(title=colorbar_title) if p.get("showscale") else None,
                    showscale=bool(p.get("showscale")),
                ),
                text=list(p["text"]),
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=c,
        )
        fig.update_xaxes(title_text=p["xtitle"][default_layer], row=1, col=c)
        fig.update_yaxes(title_text=p["ytitle"][default_layer], row=1, col=c)

    steps = []
    for layer in layers:
        restyle = {
            "x": [_as_list(p["x"][layer]) for p in panels],
            "y": [_as_list(p["y"][layer]) for p in panels],
        }
        relayout: dict[str, str] = {}
        for c, p in enumerate(panels, start=1):
            relayout[_axis_key("x", c)] = p["xtitle"][layer]
            relayout[_axis_key("y", c)] = p["ytitle"][layer]
        steps.append(dict(method="update", label=str(layer), args=[restyle, relayout]))

    fig.update_layout(
        title=title,
        width=figsize[0],
        height=figsize[1],
        sliders=[
            dict(
                active=layers.index(default_layer),
                currentvalue={"prefix": "layer "},
                pad={"t": 50},
                steps=steps,
            )
        ],
    )
    return fig


def layer_table_scrubber(
    layers: list[int],
    columns_by_layer: dict[int, list[list[str]]],
    *,
    header: list[str],
    default_layer: int,
    title: str = "",
    height: int = 470,
) -> go.Figure:
    """Build a single Plotly table whose cell values a slider swaps per layer.

    ``columns_by_layer`` maps a layer to that layer's table as a list of columns
    (one list per header column), matching ``go.Table``'s ``cells.values``.
    """
    fig = go.Figure(
        go.Table(
            header=dict(values=header, align="left"),
            cells=dict(values=columns_by_layer[default_layer], align="left", height=26),
        )
    )
    steps = [
        dict(method="restyle", label=str(layer), args=[{"cells.values": [columns_by_layer[layer]]}])
        for layer in layers
    ]
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(t=90, b=10),
        sliders=[
            dict(
                active=layers.index(default_layer),
                currentvalue={"prefix": "layer "},
                pad={"t": 30},
                steps=steps,
            )
        ],
    )
    return fig
