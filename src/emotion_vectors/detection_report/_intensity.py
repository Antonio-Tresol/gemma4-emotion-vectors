"""Section 2 exhibit: numerical intensity, the replication of paper Figure 3.

Six templates repeat one sentence with only a number changed. The registered
read (TREE.md Q1.H2.E1, prediction P2) is the SIGN of the rank correlation
between the number and each tracked probe's cosine, for 11 named (template,
probe) pairs. The figure carries a layer slider; because the verdict changes
with the layer, every slider step retitles the figure and every panel's
grading line with that layer's own count. The printed scoring stays pinned to
the registered layer 33, which is the read the campaign registered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from emotion_vectors.probe_prompts import TRACKED_PROBES
from emotion_vectors.scoring import battery_matrix

from ._data import BASE_LABEL, IT_LABEL, MODEL_SOURCES, REGISTERED_LAYER, SweepContext, figure_title

# TREE.md Q1.H2.E1 (prediction P2): the 11 named (template, probe, expected
# sign) directions, registered before any scoring. +1 means the probe's cosine
# should RISE with the number in the sentence, -1 that it should fall.
REGISTERED_DIRECTIONS: list[tuple[str, str, int]] = [
    ("tylenol", "afraid", +1),
    ("tylenol", "calm", -1),
    ("fasting", "afraid", +1),
    ("sister", "sad", -1),
    ("sister", "calm", +1),
    ("sister", "happy", +1),
    ("dog_missing", "sad", +1),
    ("runway", "afraid", -1),
    ("runway", "calm", +1),
    ("exam", "happy", +1),
    ("exam", "afraid", -1),
]
# Under the generic-drift null each registered direction is a coin flip, so
# half of them land correct by luck: that is the figure's failure anchor.
CHANCE_DIRECTIONS = len(REGISTERED_DIRECTIONS) / 2
TEMPLATE_ORDER = ["tylenol", "fasting", "sister", "dog_missing", "runway", "exam"]
PROBE_COLORS = {"afraid": "#d62728", "calm": "#1f77b4", "happy": "#2ca02c", "sad": "#ff7f0e"}
# (model label, line dash, line width, the role the line plays)
MODEL_STYLES = (
    (IT_LABEL, None, 2.5, "instruct, chat fmt"),
    (BASE_LABEL, "dash", 1.5, "base, plain fmt"),
)
INTENSITY_WIDTH_PX = 1180


@dataclass(frozen=True)
class TemplateCurve:
    """One template's cosine curves at every collected layer.

    ``x_values`` are the numbers the sentence takes, ``cosines``
    [layers probes x_values] their cosine with each tracked probe,
    ``axis_label`` names the number's unit and ``sentence`` is the template
    text with the number replaced by {X}.
    """

    name: str
    x_values: list[int]
    axis_label: str
    sentence: str
    cosines: Float[np.ndarray, "layers probes x_values"]


def intensity_curves(ctx: SweepContext) -> dict[str, dict[str, TemplateCurve]]:
    """Cosine curves per model and template, at every collected layer.

    Input: the loaded :class:`~._data.SweepContext`. Returns {model label:
    {template name: :class:`TemplateCurve`}}. Each model is read in the
    prompt format it was swept in (chat for the instruct model, plain for the
    base model, which has no chat template) at the last token, the readout
    the registered E1 read specifies. Cosines are centered on the full
    extraction, the sweep convention.
    """
    curves: dict[str, dict[str, TemplateCurve]] = {}
    for label, (_means_path, _sweep_dir, fmt) in MODEL_SOURCES.items():
        model = ctx.model(label)
        activations = model.activations(fmt, "last")
        model_curves: dict[str, TemplateCurve] = {}
        for name, points in model.template_rows().items():
            x_values = [x for x, _row in points]
            series = np.stack([activations[row] for _x, row in points])
            # cosines at EVERY collected layer, so the slider can move freely
            cosines = np.stack(
                [
                    battery_matrix(
                        model.emotion_means(TRACKED_PROBES, layer_pos),
                        series[:, layer_pos, :],
                        center_pool=model.center_pool(layer_pos),
                    )
                    for layer_pos in range(len(model.layers))
                ]
            )
            text = model.prompts[points[0][1]]["text"]
            model_curves[name] = TemplateCurve(
                name=name,
                x_values=x_values,
                axis_label=model.prompts[points[0][1]]["axis"],
                sentence=text.replace(str(x_values[0]), "{X}", 1),
                cosines=cosines,
            )
        curves[label] = model_curves
    return curves


def direction_scores(
    curves: dict[str, dict[str, TemplateCurve]], layer_pos: int
) -> dict[str, dict[str, Any]]:
    """The registered sign test at one layer, per model.

    Returns {model label: {"correct": int, "rows": [(template, probe,
    registered sign, observed rho, correct?)]}}. A direction counts as
    correct when the sign of the Spearman rank correlation between the number
    and the probe's cosine matches the registered sign.
    """
    scores: dict[str, dict[str, Any]] = {}
    for label, model_curves in curves.items():
        rows = []
        for name, probe, sign in REGISTERED_DIRECTIONS:
            curve = model_curves[name]
            rho = spearmanr(
                curve.x_values, curve.cosines[layer_pos, TRACKED_PROBES.index(probe)]
            ).statistic
            rows.append((name, probe, sign, float(rho), bool(np.sign(rho) == sign)))
        scores[label] = {"correct": sum(row[4] for row in rows), "rows": rows}
    return scores


def _panel_grading(scores: dict[str, dict[str, Any]], template: str, layer: int) -> str:
    """One panel's grading line: what was registered, and what this layer did.

    Named directions are spelled out in words ("afraid rises") so the panel
    can be graded without consulting the registered list elsewhere.
    """
    registered = [
        f"{probe} {'rises' if sign > 0 else 'falls'}"
        for name, probe, sign in REGISTERED_DIRECTIONS
        if name == template
    ]
    correct = {
        label: sum(row[4] for row in model["rows"] if row[0] == template)
        for label, model in scores.items()
    }
    total = len(registered)
    return (
        f"<sub>registered: {', '.join(registered)} with the number"
        f" | at layer {layer}: instruct {correct[IT_LABEL]} of {total},"
        f" base {correct[BASE_LABEL]} of {total} correct</sub>"
    )


def _intensity_title(scores: dict[str, dict[str, Any]], layer: int) -> str:
    """Question-form title carrying THIS layer's verdict, for a slider step."""
    n_directions = len(REGISTERED_DIRECTIONS)
    instruct, base = scores[IT_LABEL]["correct"], scores[BASE_LABEL]["correct"]
    return figure_title(
        "Does each probe reading move the right way when only the number in a sentence changes?"
        f" At layer {layer}: the instruct model gets {instruct} of {n_directions} registered"
        f" directions right and the base model {base} of {n_directions}",
        [
            "one line = one probe's cosine with the sentence activation at the last token; solid"
            " = instruct model (chat format), dashed = base model (plain format); one panel = one"
            " sentence whose only change is a number",
            f"failure anchor: under the generic-drift null each direction is a coin flip, so"
            f" {CHANCE_DIRECTIONS:.1f} of {n_directions} come out right by luck; strength anchor:"
            f" all {n_directions}. Each panel states its own registered directions and this"
            " layer's score",
            f"replicates Anthropic Figure 3 (the Tylenol plot). The slider picks the layer; layer"
            f" {REGISTERED_LAYER} (*) is the registered read, and the sign scoring printed below"
            " stays pinned there whatever the slider shows",
        ],
        INTENSITY_WIDTH_PX,
    )


def _add_intensity_traces(
    fig: go.Figure, curves: dict[str, dict[str, TemplateCurve]], layer_pos: int
) -> None:
    """Add every model x template x probe curve, in the slider's trace order."""
    for template_pos, name in enumerate(TEMPLATE_ORDER):
        row, col = template_pos // 2 + 1, template_pos % 2 + 1
        for label, dash, width, role in MODEL_STYLES:
            curve = curves[label][name]
            for probe_pos, probe in enumerate(TRACKED_PROBES):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(curve.x_values))),
                        y=curve.cosines[layer_pos, probe_pos],
                        mode="lines+markers",
                        name=f"{probe} probe | {role}",
                        legendgroup=f"{probe}-{label}",
                        showlegend=(template_pos == 0),
                        line=dict(color=PROBE_COLORS[probe], dash=dash, width=width),
                        marker=dict(size=5),
                    ),
                    row=row,
                    col=col,
                )
        fig.add_hline(y=0, line_color="#bbbbbb", line_width=1, row=row, col=col)
        axis_curve = curves[IT_LABEL][name]  # both models walk the same numbers
        fig.update_xaxes(
            tickvals=list(range(len(axis_curve.x_values))),
            ticktext=[str(x) for x in axis_curve.x_values],
            title_text=axis_curve.axis_label,
            title_font=dict(size=10),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="cosine with the probe" if col == 1 else None,
            title_font=dict(size=10),
            row=row,
            col=col,
        )


def _intensity_slider(
    curves: dict[str, dict[str, TemplateCurve]], layers: list[int]
) -> list[dict[str, Any]]:
    """One slider step per layer: restyle the curves, then RETITLE.

    A frozen title over moving data would be a defect here, because the
    verdict (how many registered directions come out right) changes with the
    layer. Each step therefore rewrites the headline and all six panel
    grading lines with that layer's own counts.
    """
    steps = []
    for layer_pos, layer in enumerate(layers):
        scores = direction_scores(curves, layer_pos)
        ys = [
            curves[label][name].cosines[layer_pos, probe_pos].tolist()
            for name in TEMPLATE_ORDER
            for label, _dash, _width, _role in MODEL_STYLES
            for probe_pos in range(len(TRACKED_PROBES))
        ]
        relayout: dict[str, Any] = {"title.text": _intensity_title(scores, layer)}
        for template_pos, name in enumerate(TEMPLATE_ORDER):
            relayout[f"annotations[{template_pos}].text"] = (
                f"{curves[IT_LABEL][name].sentence}<br>{_panel_grading(scores, name, layer)}"
            )
        steps.append(
            dict(
                method="update",
                args=[{"y": ys}, relayout],
                label=f"{layer}*" if layer == REGISTERED_LAYER else str(layer),
            )
        )
    return steps


def intensity_figure(
    curves: dict[str, dict[str, TemplateCurve]], layers: list[int]
) -> tuple[go.Figure, dict[str, Any]]:
    """Section 2 exhibit: six numerical-intensity panels with a layer slider.

    Inputs: ``intensity_curves`` output and the shared layer grid. Returns
    ``(figure, stats)``; ``stats["lines"]`` prints the registered sign test at
    layer 33 for both models and ``stats["correct"]`` maps model label ->
    directions correct at that registered layer.
    """
    if REGISTERED_LAYER not in layers:
        raise ValueError(f"registered layer {REGISTERED_LAYER} missing from the layer grid")
    registered_pos = layers.index(REGISTERED_LAYER)
    registered_scores = direction_scores(curves, registered_pos)
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            f"{curves[IT_LABEL][name].sentence}<br>"
            f"{_panel_grading(registered_scores, name, REGISTERED_LAYER)}"
            for name in TEMPLATE_ORDER
        ],
        vertical_spacing=0.13,
        horizontal_spacing=0.09,
    )
    for annotation in fig.layout.annotations:  # only the six panel titles exist yet
        annotation.font = dict(size=10)
    _add_intensity_traces(fig, curves, registered_pos)
    fig.update_layout(
        title=_intensity_title(registered_scores, REGISTERED_LAYER),
        title_font_size=13,
        # grow the wrapped title downward from the top of the margin
        title_y=0.99,
        title_yanchor="top",
        height=1180,
        width=INTENSITY_WIDTH_PX,
        # right margin holds the vertical legend; bottom margin holds the
        # x-axis titles and the layer slider without either overlapping
        margin=dict(t=200, b=120, r=230),
        legend=dict(
            orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01, font=dict(size=10)
        ),
        sliders=[
            dict(
                active=registered_pos,
                currentvalue=dict(prefix="layer: "),
                pad=dict(t=55),
                steps=_intensity_slider(curves, layers),
            )
        ],
    )
    lines = []
    for label, score in registered_scores.items():
        lines.append(f"{label} - Spearman sign per registered (template, probe) direction:")
        for name, probe, sign, rho, correct in score["rows"]:
            lines.append(
                f"  {name:12s} {probe:7s} registered {'+' if sign > 0 else '-'}  "
                f"rho {rho:+.3f}  {'correct' if correct else 'WRONG'}"
            )
        lines.append(
            f"{label}: {score['correct']}/{len(REGISTERED_DIRECTIONS)} registered directions"
            f" correct, read at the registered layer {REGISTERED_LAYER}\n"
        )
    correct = {label: score["correct"] for label, score in registered_scores.items()}
    return fig, {"lines": lines, "correct": correct}
