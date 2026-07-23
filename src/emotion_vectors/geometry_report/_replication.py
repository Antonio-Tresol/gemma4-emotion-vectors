"""Sections 1, 2, and 5: the replication exhibits.

|r|-vs-layer against the published values (section 1), the paired circumplex
scrubber (section 2), and the ordered loadings bars, paper Figure 7
(section 5). Each builder returns ``(figure, stats)`` with ``stats["lines"]``
holding every line the notebook prints.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from emotion_vectors.artifacts import fetch
from emotion_vectors.interactive import layer_scatter_scrubber

from ._context import GeometryContext

# valence = blue family, arousal = orange family; instruct saturated, base lighter
S1_LINE_COLORS = {
    ("instruct", "val"): "#1f77b4",
    ("base", "val"): "#9ecae1",
    ("instruct", "aro"): "#ff7f0e",
    ("base", "aro"): "#fdbe85",
}


def _s1_reference_lines(fig: go.Figure) -> None:
    """Grading anchors: published values = replication targets, |r| = 0 = failure floor."""
    fig.add_hline(
        y=0.81,
        line_dash="dash",
        line_color="#1f77b4",
        opacity=0.6,
        annotation_text="replication target: Anthropic valence 0.81 (Russell)",
        annotation_position="bottom left",
    )
    fig.add_hline(
        y=0.83,
        line_dash="dot",
        line_color="#1f77b4",
        opacity=0.6,
        annotation_text="replication target: open replication 0.83 (NRC)",
        annotation_position="top left",
    )
    fig.add_hline(
        y=0.66,
        line_dash="dash",
        line_color="#ff7f0e",
        opacity=0.6,
        annotation_text="replication target: Anthropic arousal 0.66 (Russell)",
        annotation_position="bottom left",
    )
    fig.add_hline(
        y=0.0,
        line_color="#888888",
        line_width=1,
        annotation_text="failure floor: |r| = 0, component carries no signal about the rating",
        annotation_position="top right",
    )


def s1_replication_figure(ctx: GeometryContext) -> tuple[go.Figure, dict[str, Any]]:
    """Section 1 exhibit: does the published valence/arousal tracking replicate?

    Overlays |r| vs layer for both models, with the published values as
    replication-target anchors and |r| = 0 as the failure floor. Returns
    ``(figure, stats)``; ``stats["lines"]`` is the printed record: the
    default-layer post-fix values and the pre-fix comparators from
    ``results/e4b_extraction_impact.json``.
    """
    fig = go.Figure()
    for label, short, dash, width in (
        (ctx.it_label, "instruct", None, 2.5),
        (ctx.base_label, "base", "dash", 1.5),
    ):
        abs_r_valence = [ctx.plane[label][layer]["rv"][0] for layer in ctx.layers]  # PC1-valence
        abs_r_arousal = [ctx.plane[label][layer]["ra"][1] for layer in ctx.layers]  # PC2-arousal
        fig.add_scatter(
            x=ctx.layers,
            y=abs_r_valence,
            mode="lines+markers",
            name=f"{short} reader, |r| PC1-valence",
            line=dict(color=S1_LINE_COLORS[(short, "val")], dash=dash, width=width),
        )
        fig.add_scatter(
            x=ctx.layers,
            y=abs_r_arousal,
            mode="lines+markers",
            name=f"{short} reader, |r| PC2-arousal",
            line=dict(color=S1_LINE_COLORS[(short, "aro")], dash=dash, width=width),
            marker_symbol="square",
        )
    _s1_reference_lines(fig)
    fig.add_annotation(
        x=9,
        y=ctx.plane[ctx.it_label][9]["rv"][0],
        text="instruct valence collapses after layer 6;<br>base climbs to the published band",
        showarrow=True,
        arrowhead=2,
        ax=90,
        ay=-60,
        font=dict(size=11),
    )
    # headline values, in the title and printed again below the figure
    it_at_default = ctx.plane[ctx.it_label][ctx.default_layer]["rv"][0]
    base_at_default = ctx.plane[ctx.base_label][ctx.default_layer]["rv"][0]
    base_peak_layer = max(ctx.layers, key=lambda layer: ctx.plane[ctx.base_label][layer]["rv"][0])
    base_peak = ctx.plane[ctx.base_label][base_peak_layer]["rv"][0]
    fig.update_layout(
        title=(
            "Does the base reader replicate the published valence geometry, and does tuning keep"
            " it?<br>"
            f"Base yes: peak |r| = {base_peak:.2f} (layer {base_peak_layer}) in the published "
            f"0.81-0.83 band; instruct collapses to {it_at_default:.2f}"
            "<br><sup>one point = one layer's |Pearson r| between that reader's own PC scores and"
            f" NRC human ratings ({len(ctx.matched)} matched emotions); stories written by"
            " gemma-4-4B; post-fix bundles</sup>"
            "<br><sup>maps to: open replication fig_valence/arousal_trajectory; the"
            " base-vs-instruct comparison is our extension</sup>"
        ),
        title_font_size=14,
        xaxis_title="layer",
        yaxis_title="|Pearson r|",
        yaxis_range=[0, 1],
        height=520,
        width=1000,
        margin=dict(t=150),
    )
    audit = json.loads(fetch("e4b_extraction_impact.json").read_text())
    per_layer_key = str(ctx.default_layer)
    audited_base = audit["lineages"]["corpus_base"]["geometry"]["per_layer"][per_layer_key]
    audited_it = audit["lineages"]["corpus_it"]["geometry"]["per_layer"][per_layer_key]
    lines = [
        f"|r| PC1-valence at layer {ctx.default_layer}, post-fix instrument: "
        f"base {base_at_default:.3f}, instruct {it_at_default:.3f}",
        "pre-fix comparators (results/e4b_extraction_impact.json): "
        f"base {audited_base['pc1_valence_abs_r_before']:.3f}, "
        f"instruct {audited_it['pc1_valence_abs_r_before']:.3f}",
    ]
    return fig, {"lines": lines, "base_peak": base_peak, "base_peak_layer": base_peak_layer}


def _selected_component_caption(layer_plane: dict[str, Any], which: str) -> str:
    """'component N (valence |r|=0.83, 15% variance)' for one selected component."""
    component, abs_r = (
        (layer_plane["vb"], layer_plane["r_vb"])
        if which == "valence"
        else (layer_plane["ab"], layer_plane["r_ab"])
    )
    return (
        f"component {component + 1} "
        f"({which} |r|={abs_r:.2f}, {layer_plane['evr'][component]:.0%} variance)"
    )


def _s2_panels(ctx: GeometryContext) -> list[dict]:
    """One panel dict per reader for ``layer_scatter_scrubber``: per-layer x/y
    are the valence-best x arousal-best component scores of the NRC-matched
    emotions; color, size, and text encode the fixed NRC ratings."""
    marker_sizes = 8 + 22 * (ctx.arousal - ctx.arousal.min()) / np.ptp(ctx.arousal)
    panels = []
    for label in ctx.labels:
        per_layer = ctx.plane[label]
        panels.append(
            dict(
                name=label,
                x={
                    layer: per_layer[layer]["scores"][ctx.matched, per_layer[layer]["vb"]]
                    for layer in ctx.layers
                },
                y={
                    layer: per_layer[layer]["scores"][ctx.matched, per_layer[layer]["ab"]]
                    for layer in ctx.layers
                },
                xtitle={
                    layer: _selected_component_caption(per_layer[layer], "valence")
                    for layer in ctx.layers
                },
                ytitle={
                    layer: _selected_component_caption(per_layer[layer], "arousal")
                    for layer in ctx.layers
                },
                color=ctx.valence,
                size=marker_sizes,
                text=[ctx.emotions[pos] for pos in ctx.matched],
                showscale=(label == ctx.base_label),
            )
        )
    return panels


def _s2_title(ctx: GeometryContext, layer: int) -> str:
    """Per-layer verdict: which components each reader selects, and how well
    they read valence."""
    plane_it = ctx.plane[ctx.it_label][layer]
    plane_base = ctx.plane[ctx.base_label][layer]
    return (
        f"Where does each reader keep its circumplex at layer {layer}? "
        f"base: components {plane_base['vb'] + 1} x {plane_base['ab'] + 1}, "
        f"instruct: components {plane_it['vb'] + 1} x {plane_it['ab'] + 1} "
        f"(valence |r| {plane_base['r_vb']:.2f} vs {plane_it['r_vb']:.2f})"
        "<br><sup>one dot = one emotion; x = that panel's valence-best component, y = its"
        " arousal-best; color = NRC valence, size = NRC arousal; activations read by the panel's"
        " model from gemma-4-4B-written stories</sup>"
        "<br><sup>success: smooth red-to-green left-to-right gradient (the circumplex); failure:"
        " color-shuffled cloud | maps to: Anthropic Figure 8 + open replication fig2_pca</sup>"
    )


def s2_circumplex_figure(ctx: GeometryContext) -> tuple[go.Figure, dict[str, Any]]:
    """Section 2 exhibit: where each reader keeps its circumplex plane.

    Paired scatter with a layer slider: each layer's valence-best x
    arousal-best plane, the slider re-titling the figure with its own layer's
    verdict. Returns ``(figure, stats)``; ``stats["lines"]`` is the printed
    record (each reader's selected components at the default layer).
    """
    fig = layer_scatter_scrubber(
        ctx.layers,
        _s2_panels(ctx),
        default_layer=ctx.default_layer,
        title=_s2_title(ctx, ctx.default_layer),
        colorbar_title="NRC valence",
    )
    # the helper's slider steps already swap the per-layer axis titles; add the
    # per-layer figure title so the headline verdict never freezes over moving data
    retitled_steps = []
    for step, layer in zip(fig.layout.sliders[0].steps, ctx.layers):
        restyle, relayout = step.args
        relayout = dict(relayout)
        relayout["title.text"] = _s2_title(ctx, layer)
        retitled_steps.append(dict(method=step.method, label=step.label, args=[restyle, relayout]))
    fig.layout.sliders[0].steps = retitled_steps
    fig.update_layout(title_font_size=14, margin=dict(t=130))
    lines = []
    for label in ctx.labels:
        layer_plane = ctx.plane[label][ctx.default_layer]
        lines.append(
            f"{label} at layer {ctx.default_layer}: valence-best component"
            f" {layer_plane['vb'] + 1} "
            f"(|r|={layer_plane['r_vb']:.3f}, {layer_plane['evr'][layer_plane['vb']]:.1%}"
            " variance); "
            f"arousal-best component {layer_plane['ab'] + 1} "
            f"(|r|={layer_plane['r_ab']:.3f}, {layer_plane['evr'][layer_plane['ab']]:.1%}"
            " variance)"
        )
    return fig, {"lines": lines}


def _loading_view(
    ctx: GeometryContext, label: str, layer: int, which: str
) -> tuple[list[float], list[str], str]:
    """(bar heights, every-10th tick labels, subplot title) for one model's
    selected component at one layer, emotions sorted by their score."""
    layer_plane = ctx.plane[label][layer]
    component, abs_r = (
        (layer_plane["vb"], layer_plane["r_vb"])
        if which == "valence"
        else (layer_plane["ab"], layer_plane["r_ab"])
    )
    sorted_order = np.argsort(layer_plane["scores"][:, component])
    bar_heights = np.round(layer_plane["scores"][sorted_order, component], 3).tolist()
    tick_positions = list(range(0, len(ctx.emotions), 10))
    tick_labels = [ctx.emotions[pos] for pos in sorted_order[tick_positions]]
    title = (
        f"{label} component {component + 1} "
        f"({layer_plane['evr'][component]:.0%} var, |r|={abs_r:.2f} vs {which})"
    )
    return bar_heights, tick_labels, title


def _s5_title(ctx: GeometryContext, layer: int) -> str:
    """Per-layer verdict naming each reader's selected components."""
    plane_it = ctx.plane[ctx.it_label][layer]
    plane_base = ctx.plane[ctx.base_label][layer]
    return (
        f"Does sorting emotions by component score still walk despair-to-joy at layer {layer}? "
        f"valence lives on base component {plane_base['vb'] + 1} vs instruct component"
        f" {plane_it['vb'] + 1}; "
        f"arousal on {plane_base['ab'] + 1} vs {plane_it['ab'] + 1}"
        "<br><sup>one bar = one emotion's score on that panel's component, sorted ascending; top"
        " row = valence-best component, bottom row = arousal-best; left column read by the"
        " instruct model, right by base</sup>"
        "<br><sup>success: a smooth red-to-green ramp walking despair to joy (top) or serene to"
        " agitated (bottom); failure: an unordered jumble hugging the 0 line | maps to: Anthropic"
        " Figure 7</sup>"
    )


def _s5_slider_steps(ctx: GeometryContext, views: list[tuple[str, str]]) -> list[dict]:
    """Each step swaps bar heights/colors, tick labels, subplot titles, and the
    figure title for its layer."""
    steps = []
    for layer in ctx.layers:
        per_trace = [_loading_view(ctx, label, layer, which) for which, label in views]
        restyle = {
            "y": [view[0] for view in per_trace],
            "marker.color": [view[0] for view in per_trace],
        }
        relayout = {f"annotations[{pos}].text": view[2] for pos, view in enumerate(per_trace)}
        relayout["title.text"] = _s5_title(ctx, layer)
        for pos, view in enumerate(per_trace):
            axis_suffix = "" if pos == 0 else str(pos + 1)
            relayout[f"xaxis{axis_suffix}.ticktext"] = view[1]
        steps.append(dict(method="update", label=str(layer), args=[restyle, relayout]))
    return steps


def s5_loadings_figure(ctx: GeometryContext) -> tuple[go.Figure, dict[str, Any]]:
    """Section 5 exhibit: the despair-to-joy walk on the selected components
    (paper Figure 7), ordered loadings bars with a layer slider.

    Returns ``(figure, stats)``. Section 5 prints nothing, so ``stats``
    carries only the per-panel captions at the default layer.
    """
    views = [(which, label) for which in ("valence", "arousal") for label in ctx.labels]
    default_views = [_loading_view(ctx, label, ctx.default_layer, which) for which, label in views]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[view[2] for view in default_views],
        horizontal_spacing=0.06,
        vertical_spacing=0.16,
    )
    for pos, (bar_heights, tick_labels, _) in enumerate(default_views):
        row, col = divmod(pos, 2)
        fig.add_bar(
            x=list(range(len(ctx.emotions))),
            y=bar_heights,
            row=row + 1,
            col=col + 1,
            marker_color=bar_heights,
            marker_colorscale="RdYlGn",
            showlegend=False,
        )
        fig.update_xaxes(
            tickvals=list(range(0, len(ctx.emotions), 10)),
            ticktext=tick_labels,
            tickangle=60,
            tickfont=dict(size=6),
            row=row + 1,
            col=col + 1,
        )
    fig.add_hline(y=0, line_color="#999999", line_width=1, row="all", col="all")  # 0 = neutral
    fig.update_layout(
        title=_s5_title(ctx, ctx.default_layer),
        title_font_size=14,
        height=860,
        width=1200,
        margin=dict(t=150),
        sliders=[
            dict(
                active=ctx.layers.index(ctx.default_layer),
                currentvalue={"prefix": "layer "},
                pad={"t": 40},
                steps=_s5_slider_steps(ctx, views),
            )
        ],
    )
    return fig, {"panel_titles": [view[2] for view in default_views]}
