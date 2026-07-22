"""Q3 trajectory figures: phase-space ternary view + cosine-vs-token lines.

Two linked views of the same story, by convention shown together:

- ``lines_figure`` — the measurement view: per-token (centered) cosine to each
  of the triple's three emotion probes, with phase-transition markers. All
  quantitative reads come from this data.
- ``ternary_figure`` — the intuition view: the token state as a point inside
  a triangle whose corners are the three emotions. Barycentric coordinates
  come from a softmax over the three cosines; the softmax temperature is a
  DISPLAY transform (documented on the figure), not a measurement, and probe
  directions are not orthogonal, so positions and traversal order are
  meaningful while metric distances are not.
- ``layer_ternaries`` — small multiples of the ternary across layer bands:
  the paper's token -> local-context -> planned-emotion story predicts corner
  snaps at lexical cues early and anticipatory arcs mid-late.

Cosines are computed from the stored shard substrate (raw dots onto unit
probes + raw/centered norms): centered dots are the stored dots minus their
per-story token mean (dot products are linear), divided by the stored
centered norms.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float
from plotly.subplots import make_subplots

DEFAULT_TEMPERATURE = 28.0
SERIES_COLORS = ["#6366f1", "#dc2626", "#d97706"]


def probe_index(labels: list[str], emotion: str, lineage: str) -> int:
    return labels.index(f"{lineage}:{emotion}")


def story_cosines(
    shard: dict,
    labels: list[str],
    emotions: list[str],
    layer_pos: int,
    lineage: str = "selfgen",
    centered: bool = True,
) -> Float[np.ndarray, "tokens three"]:
    """Per-token cosine to the triple's three probes at one captured layer.

    ``layer_pos`` indexes the run's captured-layer list (run_config layers),
    not the model layer number.
    """
    dots = shard["dots"].astype(np.float32)[:, layer_pos]  # [tokens, probes]
    idx = [probe_index(labels, e, lineage) for e in emotions]
    picked = dots[:, idx]
    if centered:
        picked = picked - picked.mean(axis=0, keepdims=True)
        norms = shard["norms_centered"].astype(np.float32)[:, layer_pos]
    else:
        norms = shard["norms"].astype(np.float32)[:, layer_pos]
    return picked / np.clip(norms[:, None], 1e-6, None)


def smooth(x: Float[np.ndarray, "tokens three"], window: int = 8) -> np.ndarray:
    """Centered moving average per column; window <= 1 is a no-op."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.stack(
        [
            np.convolve(
                np.pad(c, (window // 2, window - 1 - window // 2), mode="edge"), kernel, "valid"
            )
            for c in x.T
        ],
        axis=1,
    )


def barycentric(
    cosines: Float[np.ndarray, "tokens three"], temperature: float = DEFAULT_TEMPERATURE
) -> np.ndarray:
    """Softmax over the three cosines — the ternary display transform."""
    z = cosines * temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _hover(tokens: list[str] | None, n: int) -> list[str]:
    if tokens is None:
        return [f"t={t}" for t in range(n)]
    return [f"t={t} {tokens[t]!r}" for t in range(n)]


def ternary_figure(
    cosines: Float[np.ndarray, "tokens three"],
    emotions: list[str],
    phase_token_starts: list[int],
    tokens: list[str] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> go.Figure:
    """Phase-space view: token trajectory inside the emotion triangle."""
    bary = barycentric(cosines, temperature)
    n = len(bary)
    fig = go.Figure(
        go.Scatterternary(
            a=bary[:, 0],
            b=bary[:, 1],
            c=bary[:, 2],
            mode="lines+markers",
            marker={
                "size": 4,
                "color": np.arange(n),
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "token"},
            },
            line={"width": 1, "color": "rgba(120,120,120,0.4)"},
            text=_hover(tokens, n),
            hoverinfo="text",
        )
    )
    for start in phase_token_starts:
        fig.add_trace(
            go.Scatterternary(
                a=[bary[start, 0]],
                b=[bary[start, 1]],
                c=[bary[start, 2]],
                mode="markers",
                marker={"size": 12, "symbol": "diamond", "color": "black"},
                text=[f"phase start t={start}"],
                hoverinfo="text",
                showlegend=False,
            )
        )
    fig.update_layout(
        ternary={
            "aaxis": {"title": emotions[0]},
            "baxis": {"title": emotions[1]},
            "caxis": {"title": emotions[2]},
        },
        title=f"Emotion phase space (softmax display transform, T={temperature:g})",
        showlegend=False,
        width=560,
        height=520,
    )
    return fig


def lines_figure(
    cosines: Float[np.ndarray, "tokens three"],
    emotions: list[str],
    phase_token_starts: list[int],
    tokens: list[str] | None = None,
) -> go.Figure:
    """Measurement view: centered cosine per token, transitions marked."""
    n = len(cosines)
    hover = _hover(tokens, n)
    fig = go.Figure()
    for k, name in enumerate(emotions):
        fig.add_trace(
            go.Scatter(
                x=np.arange(n),
                y=cosines[:, k],
                name=name,
                line={"color": SERIES_COLORS[k % 3], "width": 2},
                text=hover,
                hoverinfo="text+y+name",
            )
        )
    for start in phase_token_starts[1:]:
        fig.add_vline(x=start, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="token",
        yaxis_title="centered cosine",
        title="Per-token probe trajectories",
        width=700,
        height=380,
    )
    return fig


def layer_ternaries(
    per_layer_cosines: list[Float[np.ndarray, "tokens three"]],
    layer_names: list[str],
    emotions: list[str],
    temperature: float = DEFAULT_TEMPERATURE,
) -> go.Figure:
    """Small multiples across layer bands — the layer-contrast picture."""
    fig = make_subplots(
        rows=1,
        cols=len(per_layer_cosines),
        specs=[[{"type": "ternary"}] * len(per_layer_cosines)],
        subplot_titles=layer_names,
    )
    for col, cosines in enumerate(per_layer_cosines, start=1):
        bary = barycentric(cosines, temperature)
        fig.add_trace(
            go.Scatterternary(
                a=bary[:, 0],
                b=bary[:, 1],
                c=bary[:, 2],
                mode="lines+markers",
                marker={"size": 3, "color": np.arange(len(bary)), "colorscale": "Viridis"},
                line={"width": 1, "color": "rgba(120,120,120,0.35)"},
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    axis_titles = {
        "aaxis": {"title": emotions[0]},
        "baxis": {"title": emotions[1]},
        "caxis": {"title": emotions[2]},
    }
    for col in range(1, len(per_layer_cosines) + 1):
        fig.update_ternaries(axis_titles, row=1, col=col)
    fig.update_layout(
        width=320 * len(per_layer_cosines), height=380, title="Same story across layer bands"
    )
    return fig
