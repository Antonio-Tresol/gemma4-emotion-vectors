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

Cosines are computed from the arrays each stored shard holds (raw dots onto
unit probes, plus raw and centered norms): centered dots are the stored dots
minus their per-story token mean (dot products are linear), divided by the
stored centered norms.
"""

from __future__ import annotations

import html
import textwrap
from collections.abc import Mapping

import numpy as np
import plotly.graph_objects as go
from einops import rearrange
from jaxtyping import Float
from plotly.subplots import make_subplots

DEFAULT_TEMPERATURE = 28.0
SERIES_COLORS = ["#6366f1", "#dc2626", "#d97706"]


# ``lineage`` names the story source a probe was built from ("corpus",
# "selfgen"). The parameter keeps that name because notebooks 05, 06 and 09
# pass it by keyword, and because the value is the literal prefix of each
# stored probe label, "<story source>:<emotion>".
def probe_index(labels: list[str], emotion: str, lineage: str) -> int:
    """Row of ``labels`` holding one emotion's probe from one story source."""
    return labels.index(f"{lineage}:{emotion}")


def story_cosines(
    shard: Mapping[str, Float[np.ndarray, "..."]],
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
    denominator = rearrange(np.clip(norms, 1e-6, None), "tokens -> tokens 1")
    return picked / denominator


def smooth(
    x: Float[np.ndarray, "tokens three"], window: int = 8
) -> Float[np.ndarray, "tokens three"]:
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
) -> Float[np.ndarray, "tokens three"]:
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
        width=760,
        height=680,
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
        width=940,
        height=500,
    )
    return fig


def layer_ternaries(
    per_layer_cosines: list[Float[np.ndarray, "tokens three"]],
    layer_names: list[str],
    emotions: list[str],
    temperature: float = DEFAULT_TEMPERATURE,
    tokens: list[str] | None = None,
) -> go.Figure:
    """Small multiples across layer bands — the layer-contrast picture."""
    fig = make_subplots(
        rows=1,
        cols=len(per_layer_cosines),
        specs=[[{"type": "ternary"}] * len(per_layer_cosines)],
        subplot_titles=layer_names,
        horizontal_spacing=0.14,
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
                text=_hover(tokens, len(bary)),
                hoverinfo="text",
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
    _caption(
        fig,
        "How to read: same story, one triangle per layer band. Corner snaps at cue words "
        "early vs curved anticipatory arcs mid-late would support the paper's layer story.",
    )
    fig.update_layout(
        width=400 * len(per_layer_cosines), height=500, title="Same story across layer bands"
    )
    return fig


def _caption(fig: go.Figure, text: str) -> None:
    """A 'how to read' note pinned under the plot area, wrapped to fit."""
    wrapped = "<br>".join(textwrap.wrap(text, width=110))
    fig.add_annotation(
        text=wrapped,
        xref="paper",
        yref="paper",
        x=0,
        y=0,
        yanchor="top",
        yshift=-78,  # pixels below the plot area: clears tick labels and the axis title
        showarrow=False,
        align="left",
        font={"size": 11, "color": "gray"},
    )
    fig.update_layout(margin={"b": 150})


def transition_locked_figure(
    incoming: Float[np.ndarray, "trans width"],
    outgoing: Float[np.ndarray, "trans width"],
    window: int,
    label: str = "",
    n_boot: int = 1000,
    seed: int = 0,
) -> go.Figure:
    """The population evidence figure: mean incoming/outgoing cosine around
    all transitions aligned at t=0, with bootstrap 95% bands."""
    x = np.arange(-window, window + 1)
    rng = np.random.default_rng(seed)
    fig = go.Figure()
    for name, rows, color in (("incoming", incoming, "#0e7490"), ("outgoing", outgoing, "#b45309")):
        mean = np.nanmean(rows, axis=0)
        picks = rng.integers(0, len(rows), size=(n_boot, len(rows)))
        boot = np.nanmean(rows[picks], axis=1)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(go.Scatter(x=x, y=mean, name=name, line={"color": color, "width": 2.5}))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Transition-locked average ({len(incoming)} transitions){': ' + label if label else ''}",
        xaxis_title="tokens relative to phase start",
        yaxis_title="centered cosine",
        width=940,
        height=540,
    )
    _caption(
        fig,
        "How to read: every phase transition aligned at t=0. A ramp = sloped crossover spanning "
        "many tokens; a step = jump at t=0. Incoming rising above its band BEFORE t=0 = "
        "anticipation ('planned emotion'). Bands are bootstrap 95% over transitions.",
    )
    return fig


def probe_heatmap_figure(
    cosines_all: Float[np.ndarray, "tokens probes"],
    probe_names: list[str],
    phase_starts: list[int],
    tokens: list[str] | None = None,
) -> go.Figure:
    """The confound check: every probe (not just the triple's three) over tokens."""
    fig = go.Figure(
        go.Heatmap(
            z=cosines_all.T,
            x=np.arange(len(cosines_all)),
            y=probe_names,
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "centered cosine"},
        )
    )
    for start in phase_starts[1:]:
        fig.add_vline(x=start, line_dash="dash", line_color="black")
    if tokens is not None:
        fig.update_xaxes(tickvals=phase_starts, ticktext=[tokens[s] for s in phase_starts])
    fig.update_layout(title="All probes over tokens", xaxis_title="token", width=960, height=600)
    _caption(
        fig,
        "How to read: rows are ALL probes, dashed lines are phase starts. The assigned emotion's "
        "row should dominate its own phase; an off-triple row lighting up everywhere means the "
        "probes are reading shared valence or style, not emotion identity.",
    )
    return fig


def circumplex_figure(
    projection: Float[np.ndarray, "tokens two"],
    phase_starts: list[int],
    phase_emotions: list[str],
    tokens: list[str] | None = None,
) -> go.Figure:
    """The bridge to Q1: the token state wandering the valence/arousal plane."""
    n = len(projection)
    fig = go.Figure(
        go.Scatter(
            x=projection[:, 0],
            y=projection[:, 1],
            mode="lines+markers",
            marker={
                "size": 5,
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
    for start, emotion in zip(phase_starts, phase_emotions):
        fig.add_annotation(
            x=projection[start, 0], y=projection[start, 1], text=emotion, showarrow=True
        )
    fig.update_layout(
        title="Trajectory on the Q1 circumplex plane",
        xaxis_title="PC1 of emotion means (valence-aligned, see H1)",
        yaxis_title="PC2 (arousal-aligned)",
        width=760,
        height=660,
    )
    _caption(
        fig,
        "How to read: axes are the PCA plane of the 171 emotion means (Q1.H1's circumplex). "
        "The story should drift toward each phase emotion's known position; same-valence "
        "transitions move little here by design; this view buys interpretable axes, not "
        "per-emotion resolution.",
    )
    return fig


def cosine_3d_figure(
    cosines: Float[np.ndarray, "tokens three"],
    emotions: list[str],
    phase_starts: list[int],
    tokens: list[str] | None = None,
    up_to: int | None = None,
) -> go.Figure:
    """The untransformed view: raw centered cosines as 3D coordinates.

    ``up_to`` renders the path only through token t (the rest stays faint) and
    puts a cone arrow at t pointing in the direction of motion."""
    n = len(cosines)
    t_end = n - 1 if up_to is None else min(up_to, n - 1)
    fig = go.Figure(
        go.Scatter3d(
            x=cosines[:, 0],
            y=cosines[:, 1],
            z=cosines[:, 2],
            mode="lines",
            line={"width": 1, "color": "rgba(120,120,120,0.25)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=cosines[: t_end + 1, 0],
            y=cosines[: t_end + 1, 1],
            z=cosines[: t_end + 1, 2],
            mode="lines+markers",
            marker={
                "size": 4,
                "color": np.arange(t_end + 1),
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": n - 1,
                "showscale": True,
                "colorbar": {"title": "token"},
            },
            line={
                "width": 5,
                "color": np.arange(t_end + 1),
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": n - 1,
            },
            text=_hover(tokens, n)[: t_end + 1],
            hoverinfo="text",
            showlegend=False,
        )
    )
    if t_end >= 1:
        direction = cosines[t_end] - cosines[t_end - 1]
        fig.add_trace(
            go.Cone(
                x=[float(cosines[t_end, 0])],
                y=[float(cosines[t_end, 1])],
                z=[float(cosines[t_end, 2])],
                u=[float(direction[0])],
                v=[float(direction[1])],
                w=[float(direction[2])],
                sizemode="absolute",
                sizeref=float(np.abs(cosines).max()) * 0.15,
                showscale=False,
                colorscale=[[0, "#111"], [1, "#111"]],
                hoverinfo="skip",
            )
        )
    for start in phase_starts:
        fig.add_trace(
            go.Scatter3d(
                x=[cosines[start, 0]],
                y=[cosines[start, 1]],
                z=[cosines[start, 2]],
                mode="markers",
                marker={"size": 7, "symbol": "diamond", "color": "black"},
                text=[f"phase start t={start}"],
                hoverinfo="text",
                showlegend=False,
            )
        )
    fig.update_layout(
        scene={
            "xaxis_title": emotions[0],
            "yaxis_title": emotions[1],
            "zaxis_title": emotions[2],
        },
        title="Raw cosine coordinates (no display transform)",
        showlegend=False,
        width=800,
        height=700,
    )
    _caption(
        fig,
        "How to read: actual measured coordinates, magnitude preserved, no softmax. Color runs "
        "dark (start) to yellow; the cone arrow points in the direction of motion at t. Rotate "
        "it; axes are oblique in activation space, so read positions and order, not distances.",
    )
    return fig


def speed_figure(
    speed: Float[np.ndarray, "tokens"],
    phase_starts: list[int],
    tokens: list[str] | None = None,
) -> go.Figure:
    """Step-vs-ramp diagnostic independent of which emotions are involved."""
    n = len(speed)
    fig = go.Figure(
        go.Scatter(
            x=np.arange(1, n + 1),
            y=speed,
            line={"color": "#6366f1", "width": 2},
            text=_hover(tokens, n + 1)[1:],
            hoverinfo="text+y",
        )
    )
    for start in phase_starts[1:]:
        fig.add_vline(x=start, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Trajectory speed ||a(t) - a(t-1)||",
        xaxis_title="token",
        yaxis_title="residual-stream speed",
        width=940,
        height=420,
    )
    _caption(
        fig,
        "How to read: a sharp spike exactly at the dashed phase start = the state jumps at the "
        "cue word (step); elevated speed spread over the window = gradual movement (ramp). "
        "Works without choosing probes at all.",
    )
    return fig


def text_panel_html(
    clean_text: str,
    phase_char_starts: list[int],
    phase_emotions: list[str],
    colors: list[str] = SERIES_COLORS,
) -> str:
    """The story's own text with each phase colored to match its trajectory
    line/point color above — the "which words gave which curve" view a
    per-token hover can't give at a glance. Meant to be displayed directly
    under a lines/ternary/heatmap figure for the same story via
    ``IPython.display.HTML``.

    SEQUENTIAL stories (``len(phase_char_starts) > 1``) get one colored span
    per phase, split at the same char offsets ``parse_story`` recorded.
    SIMULTANEOUS stories (a single phase covering the whole text, per
    ``ParsedStory``'s convention) get the whole text plain plus a 3-color
    legend, since coloring by character position would misrepresent all
    three emotions as co-active throughout rather than pick one span.
    """
    legend = " &nbsp; ".join(
        f'<span style="border-bottom:3px solid {colors[i % len(colors)]};padding-bottom:1px">'
        f"{html.escape(e)}</span>"
        for i, e in enumerate(phase_emotions)
    )
    if len(phase_char_starts) == 1:
        body = html.escape(clean_text)
    else:
        bounds = list(phase_char_starts) + [len(clean_text)]
        spans = []
        for i, emotion in enumerate(phase_emotions):
            chunk = html.escape(clean_text[bounds[i] : bounds[i + 1]])
            color = colors[i % len(colors)]
            spans.append(
                f'<span style="background-color:{color}22;border-bottom:2px solid {color}" '
                f'title="{html.escape(emotion)}">{chunk}</span>'
            )
        body = "".join(spans)
    return (
        f'<div style="font-family:ui-monospace,monospace;line-height:1.9;'
        f'max-width:940px;white-space:pre-wrap;padding:8px 0">{body}</div>'
        f'<div style="font-size:12px;color:#666;margin-top:2px">{legend}</div>'
    )
