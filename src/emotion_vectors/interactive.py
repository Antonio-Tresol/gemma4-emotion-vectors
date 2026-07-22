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


def _hover(tokens: list[str] | None, n: int) -> list[str]:
    """Per-position hover label: the decoded token text if given, else just
    the index. Token identity doesn't depend on which layer is scrubbed to,
    so this is computed once and reused across every slider step."""
    if tokens is None:
        return [f"t={t}" for t in range(n)]
    return [f"t={t} {tokens[t]!r}" for t in range(n)]


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


def trajectory_lines_scrubber(
    layers: list[int],
    cosines_by_layer: dict[int, Any],
    emotions: list[str],
    phase_starts: list[int],
    *,
    default_layer: int,
    title: str = "",
    colors: tuple[str, str, str] = ("#dc2626", "#6366f1", "#d97706"),
    tokens: list[str] | None = None,
) -> go.Figure:
    """Per-token cosine lines with a layer slider.

    ``cosines_by_layer`` maps a layer to a [tokens, 3] array (column j = phase
    j's emotion). The y-range is fixed to the global extremes so amplitude
    differences between layers stay visible while scrubbing. ``tokens``, if
    given, is the decoded token string at each position, shown on hover so a
    point on the curve reads back to the word that produced it — token
    identity is layer-independent, so this doesn't need updating per step.
    """
    n = len(cosines_by_layer[default_layer])
    lo = min(float(np.min(c)) for c in cosines_by_layer.values())
    hi = max(float(np.max(c)) for c in cosines_by_layer.values())
    pad = 0.1 * (hi - lo)
    hover = _hover(tokens, n)
    fig = go.Figure()
    for k, name in enumerate(emotions):
        fig.add_trace(
            go.Scatter(
                x=list(range(n)),
                y=_as_list(cosines_by_layer[default_layer][:, k]),
                name=name,
                line=dict(color=colors[k % 3], width=2),
                text=hover,
                hoverinfo="text+y+name",
            )
        )
    for start in phase_starts[1:]:
        fig.add_vline(x=start, line_dash="dash", line_color="gray")
    steps = [
        dict(
            method="restyle",
            label=str(layer),
            args=[{"y": [_as_list(cosines_by_layer[layer][:, k]) for k in range(3)]}],
        )
        for layer in layers
    ]
    fig.update_layout(
        title=title,
        xaxis_title="token",
        yaxis_title="centered cosine",
        yaxis_range=[lo - pad, hi + pad],
        width=940,
        height=540,
        sliders=[
            dict(
                active=layers.index(default_layer),
                currentvalue={"prefix": "layer "},
                pad={"t": 40},
                steps=steps,
            )
        ],
    )
    return fig


def trajectory_ternary_scrubber(
    layers: list[int],
    barycentric_by_layer: dict[int, Any],
    emotions: list[str],
    phase_starts: list[int],
    *,
    default_layer: int,
    title: str = "",
    tokens: list[str] | None = None,
) -> go.Figure:
    """Emotion-triangle trajectory with a layer slider.

    ``barycentric_by_layer`` maps a layer to a [tokens, 3] barycentric array
    (rows sum to 1 — the caller owns the softmax display transform).
    ``tokens``, if given, is the decoded token string at each position,
    shown on hover instead of the bare index.
    """
    bary0 = np.asarray(barycentric_by_layer[default_layer])
    n = len(bary0)
    fig = go.Figure(
        go.Scatterternary(
            a=_as_list(bary0[:, 0]),
            b=_as_list(bary0[:, 1]),
            c=_as_list(bary0[:, 2]),
            mode="lines+markers",
            marker=dict(
                size=4,
                color=list(range(n)),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="token"),
            ),
            line=dict(width=1, color="rgba(120,120,120,0.4)"),
            text=_hover(tokens, n),
            hoverinfo="text",
            showlegend=False,
        )
    )
    starts_pts = {layer: np.asarray(barycentric_by_layer[layer])[phase_starts] for layer in layers}
    fig.add_trace(
        go.Scatterternary(
            a=_as_list(starts_pts[default_layer][:, 0]),
            b=_as_list(starts_pts[default_layer][:, 1]),
            c=_as_list(starts_pts[default_layer][:, 2]),
            mode="markers",
            marker=dict(size=12, symbol="diamond", color="black"),
            text=[f"phase start t={s}" for s in phase_starts],
            hoverinfo="text",
            showlegend=False,
        )
    )
    steps = []
    for layer in layers:
        bary = np.asarray(barycentric_by_layer[layer])
        steps.append(
            dict(
                method="restyle",
                label=str(layer),
                args=[
                    {
                        "a": [_as_list(bary[:, 0]), _as_list(starts_pts[layer][:, 0])],
                        "b": [_as_list(bary[:, 1]), _as_list(starts_pts[layer][:, 1])],
                        "c": [_as_list(bary[:, 2]), _as_list(starts_pts[layer][:, 2])],
                    }
                ],
            )
        )
    fig.update_layout(
        title=title,
        ternary=dict(
            aaxis=dict(title=emotions[0]),
            baxis=dict(title=emotions[1]),
            caxis=dict(title=emotions[2]),
        ),
        width=760,
        height=700,
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


def trajectory_heatmap_scrubber(
    layers: list[int],
    heat_by_layer: dict[int, Any],
    probe_names: list[str],
    phase_starts: list[int],
    *,
    default_layer: int,
    title: str = "",
    tokens: list[str] | None = None,
) -> go.Figure:
    """All-probe heatmap ([tokens, probes] per layer) with a layer slider.

    The colour range is fixed to the global |max| so intensity is comparable
    across layers while scrubbing. ``tokens``, if given, labels the x-axis at
    each phase start with its actual word and adds per-cell hover text —
    token identity is layer-independent, so neither needs a slider step.
    """
    zmax = max(float(np.abs(h).max()) for h in heat_by_layer.values())
    z0 = np.asarray(heat_by_layer[default_layer]).T
    heat_kwargs: dict[str, Any] = {}
    if tokens is not None:
        heat_kwargs["text"] = [tokens] * len(probe_names)
        heat_kwargs["hovertemplate"] = "%{y}<br>t=%{x} %{text}<br>cosine=%{z:.3f}<extra></extra>"
    fig = go.Figure(
        go.Heatmap(
            z=_as_list(z0),
            y=probe_names,
            colorscale="RdBu",
            zmin=-zmax,
            zmax=zmax,
            colorbar=dict(title="centered cosine"),
            **heat_kwargs,
        )
    )
    for start in phase_starts[1:]:
        fig.add_vline(x=start, line_dash="dash", line_color="black")
    if tokens is not None:
        fig.update_xaxes(tickvals=phase_starts, ticktext=[tokens[s] for s in phase_starts])
    steps = [
        dict(
            method="restyle",
            label=str(layer),
            args=[{"z": [_as_list(np.asarray(heat_by_layer[layer]).T)]}],
        )
        for layer in layers
    ]
    fig.update_layout(
        title=title,
        xaxis_title="token",
        width=960,
        height=620,
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


def trajectory_ternary_animation(
    barycentric: Any,
    emotions: list[str],
    phase_starts: list[int],
    *,
    stride: int = 2,
    frame_ms: int = 60,
    title: str = "",
) -> go.Figure:
    """A playable animation of the token state moving through the triangle.

    The full path sits faint underneath; Play sweeps a dot along it with the
    trail brightening behind. Frames are strided (default every 2nd token) to
    keep the saved notebook light. Self-contained: the play/pause buttons are
    Plotly updatemenus, no kernel needed.
    """
    bary = np.asarray(barycentric)
    n = len(bary)
    ticks = list(range(0, n, stride)) + ([n - 1] if (n - 1) % stride else [])
    path = dict(
        a=_as_list(bary[:, 0]),
        b=_as_list(bary[:, 1]),
        c=_as_list(bary[:, 2]),
        mode="lines",
        line=dict(width=1, color="rgba(120,120,120,0.25)"),
        hoverinfo="skip",
        showlegend=False,
    )

    def dot(t: int) -> dict:
        return dict(
            a=[float(bary[t, 0])],
            b=[float(bary[t, 1])],
            c=[float(bary[t, 2])],
            mode="markers+text",
            marker=dict(size=13, color="#6366f1"),
            text=[f"t={t}"],
            textposition="top center",
            hoverinfo="text",
            showlegend=False,
        )

    def trail(t: int) -> dict:
        return dict(
            a=_as_list(bary[: t + 1, 0]),
            b=_as_list(bary[: t + 1, 1]),
            c=_as_list(bary[: t + 1, 2]),
            mode="lines",
            line=dict(width=2.5, color="#6366f1"),
            hoverinfo="skip",
            showlegend=False,
        )

    fig = go.Figure(
        data=[go.Scatterternary(path), go.Scatterternary(trail(0)), go.Scatterternary(dot(0))],
        frames=[
            go.Frame(
                data=[go.Scatterternary(trail(t)), go.Scatterternary(dot(t))],
                traces=[1, 2],
                name=str(t),
            )
            for t in ticks
        ],
    )
    for start in phase_starts:
        fig.add_trace(
            go.Scatterternary(
                a=[float(bary[start, 0])],
                b=[float(bary[start, 1])],
                c=[float(bary[start, 2])],
                mode="markers",
                marker=dict(size=11, symbol="diamond", color="black"),
                text=[f"phase start t={start}"],
                hoverinfo="text",
                showlegend=False,
            )
        )
    fig.update_layout(
        title=title,
        ternary=dict(
            aaxis=dict(title=emotions[0]),
            baxis=dict(title=emotions[1]),
            caxis=dict(title=emotions[2]),
        ),
        width=760,
        height=740,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.05,
                y=-0.06,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_ms, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0), mode="immediate")],
                    ),
                ],
            )
        ],
        # token slider bound to the same frames: drag to see the path up to t
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="t = "),
                pad=dict(t=30),
                steps=[
                    dict(
                        method="animate",
                        label=str(t),
                        args=[
                            [str(t)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for t in ticks
                ],
            )
        ],
    )
    return fig


def trajectory_story_dropdown(
    entries: list[dict],
    *,
    title: str = "",
    colors: tuple[str, str, str] = ("#dc2626", "#6366f1", "#d97706"),
) -> go.Figure:
    """Cosine-lines view with a dropdown over stories.

    Each entry: ``label``, ``emotions`` (3), ``cosines`` [tokens, 3] (column j
    = phase j's emotion), ``starts`` (phase token starts). Phase markers are
    per-story shapes swapped through the dropdown's relayout args.
    """
    first = entries[0]
    fig = go.Figure()
    for k in range(3):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(first["cosines"]))),
                y=_as_list(np.asarray(first["cosines"])[:, k]),
                name=first["emotions"][k],
                line=dict(color=colors[k], width=2),
            )
        )

    def shapes(entry: dict) -> list[dict]:
        return [
            dict(
                type="line",
                x0=s,
                x1=s,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(dash="dash", color="gray"),
            )
            for s in entry["starts"][1:]
        ]

    buttons = []
    for entry in entries:
        cos = np.asarray(entry["cosines"])
        buttons.append(
            dict(
                label=entry["label"],
                method="update",
                args=[
                    {
                        "x": [list(range(len(cos)))] * 3,
                        "y": [_as_list(cos[:, k]) for k in range(3)],
                        "name": list(entry["emotions"]),
                    },
                    {"shapes": shapes(entry)},
                ],
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="token",
        yaxis_title="centered cosine",
        shapes=shapes(first),
        width=940,
        height=540,
        updatemenus=[
            dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.18, yanchor="top")
        ],
    )
    return fig
