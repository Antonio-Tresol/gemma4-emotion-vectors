"""Section 7's caveat exhibit: does the extraction format change the vectors?

Every number in notebook 07 comes from RAW-TEXT extraction (the stories fed
to the model as plain text). The E4b audit re-extracted the same
self-generated corpus through the model's chat template and compared the two
sets of directions layer by layer. This module turns that audit's per-layer
record into a figure, so the caveat is gradeable rather than a wall of
printed numbers.

``extraction_format_figure`` returns ``(figure, stats)`` with
``stats["lines"]`` holding the printed record, matching the house contract
of the rest of the package. It only slices
``results/raw_vs_chat_extraction_cosine.json``; nothing here re-extracts.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from ._data import GEOMETRY_LAYER, PROBED_MODEL, StorySourceEvidence, figure_title

# Agreement a reader should demand before treating two extraction formats as
# interchangeable: cosine 1.0 is the same direction, and we draw 0.9 as the
# "close enough to swap" band edge. These are reading aids for the figure,
# not registered bars: E4b registered no pass bar on this read.
IDENTICAL_COSINE = 1.0
INTERCHANGEABLE_BAND = 0.9


def _per_layer_records(evidence: StorySourceEvidence) -> dict[int, dict[str, Any]]:
    """The audit's per-layer block with integer layer keys (the JSON stores
    them as strings); raises if the audit is empty."""
    records = {int(layer): stats for layer, stats in evidence.raw_vs_chat["per_layer"].items()}
    if not records:
        raise ValueError("raw_vs_chat_extraction_cosine.json has an empty per_layer block")
    return records


def _add_agreement_series(
    fig: go.Figure,
    layers: list[int],
    records: dict[int, dict[str, Any]],
    raw_means: list[float],
    contrast_means: list[float],
    contrast_minima: list[float],
) -> None:
    """Draw the three agreement curves: uncentered means, the contrast
    directions this notebook actually uses, and the worst single emotion."""
    fig.add_scatter(
        x=layers,
        y=raw_means,
        mode="lines+markers",
        name="uncentered mean vectors (before 12-pool centering)",
        line=dict(color="#7f7f7f", dash="dash"),
    )
    fig.add_scatter(
        x=layers,
        y=contrast_means,
        mode="lines+markers",
        name="contrast directions, averaged over the 12 emotions (what this notebook uses)",
        line=dict(color="#1f77b4", width=3),
    )
    fig.add_scatter(
        x=layers,
        y=contrast_minima,
        mode="lines+markers",
        name="contrast direction of the single worst emotion at that layer",
        line=dict(color="#d62728", dash="dot"),
        text=[records[layer]["contrast_cos_argmin"] for layer in layers],
        hovertemplate="layer %{x}: cosine %{y:.2f} (%{text})<extra></extra>",
    )


def _add_agreement_anchors(
    fig: go.Figure, worst_layer: int, worst_value: float, worst_emotion: str
) -> None:
    """Add the grading scale: identical directions above, the
    "interchangeable" band edge, unrelated directions at the bottom, and a
    callout naming the worst emotion so the low point is not anonymous."""
    fig.add_hline(
        y=IDENTICAL_COSINE,
        line_dash="dot",
        line_color="#2ca02c",
        annotation_text="identical directions (cosine 1.0): the format would not matter",
        annotation_position="bottom left",
    )
    fig.add_hline(
        y=INTERCHANGEABLE_BAND,
        line_dash="dash",
        line_color="#2ca02c",
        opacity=0.5,
        annotation_text="cosine 0.9: close enough to treat the two formats as one instrument",
        annotation_position="bottom right",
    )
    # the failure anchor sits just inside the plot floor, so it cannot collide
    # with the layer tick labels underneath the axis
    fig.add_annotation(
        xref="paper",
        yref="y",
        x=0.01,
        y=0.03,
        text="failure anchor: cosine 0, the two formats share no direction at all",
        showarrow=False,
        font=dict(size=11, color="#666666"),
        xanchor="left",
    )
    fig.add_annotation(
        x=worst_layer,
        y=worst_value,
        text=f"worst emotion overall:<br>{worst_emotion}, layer {worst_layer}, cosine"
        f" {worst_value:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=-70,
        ay=-60,
        font=dict(size=11),
    )


def _agreement_lines(
    layers: list[int],
    records: dict[int, dict[str, Any]],
    contrast_means: list[float],
    raw_means: list[float],
    worst: tuple[float, int, str],
) -> list[str]:
    """The printed record: the per-layer table, then the summary ranges."""
    worst_value, worst_layer, worst_emotion = worst
    lines = [
        "raw-vs-chat extraction agreement per layer (contrast mean / worst emotion / raw mean):"
    ]
    for layer in layers:
        stats = records[layer]
        lines.append(
            f"  layer {layer}: {stats['contrast_cos_mean']:.2f} /"
            f" {stats['contrast_cos_min']:.2f} ({stats['contrast_cos_argmin']}) /"
            f" {stats['raw_cos_mean']:.2f}"
        )
    lines.append(
        f"contrast means {min(contrast_means):.2f} to {max(contrast_means):.2f}; worst"
        f" per-emotion agreement {worst_value:.2f} ({worst_emotion}, layer {worst_layer});"
        f" uncentered means {min(raw_means):.2f} to {max(raw_means):.2f}"
    )
    return lines


def extraction_format_figure(evidence: StorySourceEvidence) -> tuple[go.Figure, dict[str, Any]]:
    """Section 7 caveat exhibit: raw-text vs chat-template extraction agreement.

    Input: ``load_evidence()`` output. Three series against layer, all of
    them cosines between the raw-text and chat-template versions of the SAME
    probe: the uncentered mean vectors (which agree almost perfectly), the
    12-pool contrast directions averaged over emotions (the objects this
    notebook actually uses), and the single worst emotion's contrast cosine
    at each layer. Grading anchors: 1.0 = identical directions, a 0.9 band
    edge for "interchangeable", and 0 = unrelated directions.

    Returns ``(figure, stats)``; ``stats["lines"]`` is the printed record
    (the per-layer table plus a range summary) and ``stats["summary"]`` holds
    the contrast-mean range, the worst per-emotion agreement with its layer
    and emotion, and the uncentered-mean range.
    """
    records = _per_layer_records(evidence)
    layers = sorted(records)
    raw_means = [records[layer]["raw_cos_mean"] for layer in layers]
    contrast_means = [records[layer]["contrast_cos_mean"] for layer in layers]
    contrast_minima = [records[layer]["contrast_cos_min"] for layer in layers]
    worst_layer = min(layers, key=lambda layer: records[layer]["contrast_cos_min"])
    worst_value = records[worst_layer]["contrast_cos_min"]
    worst_emotion = records[worst_layer]["contrast_cos_argmin"]

    fig = go.Figure()
    _add_agreement_series(fig, layers, records, raw_means, contrast_means, contrast_minima)
    _add_agreement_anchors(fig, worst_layer, worst_value, worst_emotion)
    width_px = 1050
    fig.update_layout(
        title=figure_title(
            "Would the same corpus give the same probe directions under the other extraction"
            " format? No: the contrast directions this notebook uses agree at only"
            f" {min(contrast_means):.2f} to {max(contrast_means):.2f} across layers, worst"
            f" single emotion {worst_value:.2f}, while the uncentered means agree at"
            f" {min(raw_means):.2f} to {max(raw_means):.2f}",
            [
                "one point = one layer's cosine between the raw-text and the chat-template"
                " version of the same probe, both built from the same self-generated corpus;"
                " higher is better and 1.0 would mean the format changed nothing",
                f"probes read from {PROBED_MODEL}; the E4b audit registered no pass bar on this"
                " read, so the 0.9 line is a reading aid rather than a registered bar;"
                " evidence: raw_vs_chat_extraction_cosine.json",
            ],
            width_px,
        ),
        title_font_size=13,
        xaxis=dict(
            title=f"layer of {PROBED_MODEL} (only the layers the audit covered)",
            tickmode="array",
            tickvals=layers,
        ),
        yaxis_title="cosine between the two extraction formats<br>(1 = identical, 0 = unrelated)",
        yaxis_range=[0, 1.08],
        width=width_px,
        height=620,
        margin=dict(t=210, b=140),
        legend=dict(orientation="h", yanchor="top", y=-0.18),
    )
    lines = _agreement_lines(
        layers, records, contrast_means, raw_means, (worst_value, worst_layer, worst_emotion)
    )
    summary = {
        "contrast_mean_range": (min(contrast_means), max(contrast_means)),
        "worst_value": worst_value,
        "worst_layer": worst_layer,
        "worst_emotion": worst_emotion,
        "raw_mean_range": (min(raw_means), max(raw_means)),
        "geometry_layer_contrast_mean": records[GEOMETRY_LAYER]["contrast_cos_mean"],
    }
    return fig, {"lines": lines, "summary": summary}
