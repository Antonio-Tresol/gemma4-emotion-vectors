"""Section 10: reading the emotion vectors through the output vocabulary.

The source paper's Table 1 shows each emotion vector raising the probability
of words that belong to that emotion (sad toward grief, tears). This module
rebuilds that read for both models, folded in from the paper-parity notebook
(``notebooks/04_paper_plot_parity.ipynb``, its section 2), so the result sits
next to notebook 02's other evidence about what instruction tuning does to the
representation.

There is no registered numeric bar for this exhibit: the paper's table is
judged by eye, and so is ours. The judgment recorded when the lens was first
read lives in :data:`JUDGED_AFFECTIVE`, one entry per (model, layer), so the
figure can show it per row and state it in its own title instead of leaving a
reader to squint at token lists.

``s10_logit_lens_figure`` returns ``(figure, stats)`` with ``stats["lines"]``
holding the printed record, like every other builder in this package.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

import plotly.graph_objects as go

from emotion_vectors.artifacts import fetch  # local results/ first, HF otherwise

# Only two layers were ever computed per model (33 and 57, final-RMSNorm
# scaled); the slider covers every available results/logit_lens_*.json, no
# other layers exist, so this is the whole view rather than a selection.
LENS_FILES: dict[str, dict[int, str]] = {
    "gemma-4-31b-it": {33: "logit_lens_it_L33_normed.json", 57: "logit_lens_it_L57.json"},
    "gemma-4-31b (base)": {33: "logit_lens_base_L33.json", 57: "logit_lens_base_L57.json"},
}
# The layer each model opens on: the instruct negative was reported at its
# deepest computed layer, the base partial positive at layer 33.
LENS_DEFAULT_LAYER = {"gemma-4-31b-it": 57, "gemma-4-31b (base)": 33}

# The by-eye judgment recorded when the lens was read: which emotions show an
# affective token neighborhood. An empty tuple means "none did"; ``None``
# means no per-emotion judgment was recorded at that layer (base layer 57 was
# read as the same picture as layer 33 but noisier, and never itemized).
JUDGED_AFFECTIVE: dict[tuple[str, int], tuple[str, ...] | None] = {
    ("gemma-4-31b-it", 33): (),
    ("gemma-4-31b-it", 57): (),
    ("gemma-4-31b (base)", 33): ("happy", "proud", "desperate", "angry", "guilty"),
    ("gemma-4-31b (base)", 57): None,
}

LENS_WIDTH_PX = 1000
# Plotly never wraps a title; these are characters that fit per line at this
# width, measured from rendered PNGs (the <sup> subtitle lines render smaller).
HEADLINE_WRAP_CHARS = 120
SUBTITLE_WRAP_CHARS = 165
# What the judgment column says, per recorded judgment.
JUDGMENT_LABELS = {True: "yes", False: "no", None: "not recorded"}


def _lens_title(model_label: str, layer: int, headline_verdict: str, note: str) -> str:
    """Wrap this layer's question-form title, its answer, and the subtitles.

    Inputs: the model, the layer shown, the verdict clause computed for that
    layer, and the lens file's own provenance note. Returns the ``<br>``-joined
    title string.
    """
    headline = (
        "Do the emotion vectors raise the probability of emotion words?"
        f" {headline_verdict} ({model_label}, layer {layer})"
    )
    subtitles = [
        "one row = one of the twelve emotions; its cells are the tokens that emotion's vector most"
        " raises and most lowers when its direction is read through the model's output"
        " vocabulary (the logit lens)",
        "grading, by eye and by eye only (no registered numeric bar): a REPRODUCTION looks like"
        " the source paper's Table 1, an emotion's own word family (sad toward grief, tears);"
        " a FAILURE looks like unrelated word fragments and punctuation, which is what chance"
        " token neighborhoods look like",
        f"maps to: Anthropic Table 1. {note}. Slider covers both computed layers"
        f" (files results/{LENS_FILES[model_label][33]} and"
        f" results/{LENS_FILES[model_label][57]}); no other layer was ever computed",
    ]
    lines = textwrap.wrap(headline, width=HEADLINE_WRAP_CHARS)
    for subtitle in subtitles:
        lines += [f"<sup>{line}</sup>" for line in textwrap.wrap(subtitle, SUBTITLE_WRAP_CHARS)]
    return "<br>".join(lines)


def _layer_verdict(model_label: str, layer: int, n_emotions: int) -> str:
    """This layer's answer to the figure's question, in words.

    Inputs: the model, the layer, and how many emotions the table holds.
    Returns the clause the title states after the question.
    """
    judged = JUDGED_AFFECTIVE[(model_label, layer)]
    if judged is None:
        return (
            "Not itemized: this layer was read as the same picture as layer 33 but noisier,"
            " and no per-emotion judgment was recorded"
        )
    if not judged:
        return f"No: 0 of {n_emotions} emotions show an affective token neighborhood"
    return (
        f"Partly: {len(judged)} of {n_emotions} emotions show an affective token neighborhood"
        f" ({', '.join(judged)})"
    )


def _lens_columns(model_label: str, layer: int) -> tuple[list[list[str]], str]:
    """One layer's table columns plus that file's provenance note.

    Input: the model and layer. Returns ``(columns, note)`` where ``columns``
    is [emotions, up-weighted tokens, down-weighted tokens, judgment] and the
    note is the scaling convention recorded in the evidence file. Raises if
    the file's own layer field disagrees with the layer requested.
    """
    lens = json.loads(fetch(LENS_FILES[model_label][layer]).read_text())
    if int(lens["layer"]) != layer:
        raise ValueError(
            f"{LENS_FILES[model_label][layer]} says layer {lens['layer']}, expected {layer}"
        )
    judged = JUDGED_AFFECTIVE[(model_label, layer)]
    emotions = list(lens["table"])
    judgment = [
        JUDGMENT_LABELS[None] if judged is None else JUDGMENT_LABELS[emotion in judged]
        for emotion in emotions
    ]
    columns = [
        emotions,
        [", ".join(tokens["up"]) for tokens in lens["table"].values()],
        [", ".join(tokens["down"]) for tokens in lens["table"].values()],
        judgment,
    ]
    return columns, str(lens["note"])


def _verdict_line(model_label: str) -> str:
    """The verdict this model earned across both computed layers, as one line.

    Input: the model label. Returns the printed record line (the same sentence
    the parity notebook printed, so the campaign record reads identically).
    """
    judged = JUDGED_AFFECTIVE[(model_label, 33)]
    if judged is None:
        raise ValueError(f"no layer-33 judgment recorded for {model_label}")
    if not judged:
        return (
            f"verdict, {model_label}: no emotion-word neighborhoods at layer 33 or 57;"
            " Table 1 does not reproduce"
        )
    return (
        f"verdict, {model_label}: affective neighborhoods for {len(judged)}/12 at layer 33"
        f" ({', '.join(judged)}); valence-consistent down-lists for most others"
    )


def s10_logit_lens_figure(model_label: str) -> tuple[go.Figure, dict[str, Any]]:
    """Section 10 exhibit: one model's logit-lens token table, with a layer slider.

    Input: the model label, a key of :data:`LENS_FILES`. Returns
    ``(figure, stats)``; ``stats["lines"]`` is the printed record (this model's
    verdict and the evidence files behind it) and ``stats["judged_affective"]``
    maps each computed layer to the emotions judged to show an affective
    neighborhood there.
    """
    if model_label not in LENS_FILES:
        raise KeyError(f"no logit lens computed for {model_label}; have {list(LENS_FILES)}")
    layers = sorted(LENS_FILES[model_label])
    default_layer = LENS_DEFAULT_LAYER[model_label]
    fig = go.Figure()
    titles: dict[int, str] = {}
    for layer in layers:
        columns, note = _lens_columns(model_label, layer)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[
                        "emotion",
                        "tokens this vector raises most",
                        "tokens this vector lowers most",
                        "affective neighborhood? (by eye)",
                    ],
                    align="left",
                    font=dict(size=11),
                ),
                columnwidth=[0.10, 0.36, 0.36, 0.18],
                cells=dict(values=columns, align="left", height=26, font=dict(size=11)),
                visible=layer == default_layer,
            )
        )
        titles[layer] = _lens_title(
            model_label, layer, _layer_verdict(model_label, layer, len(columns[0])), note
        )
    fig.update_layout(
        title=titles[default_layer],
        title_font_size=13,
        # the wrapped title grows downward from the top of the margin
        title_y=0.985,
        title_yanchor="top",
        width=LENS_WIDTH_PX,
        height=620,
        # top margin holds the five wrapped title lines, bottom margin the slider
        margin=dict(t=150, b=80, l=10, r=10),
        sliders=[
            dict(
                active=layers.index(default_layer),
                currentvalue=dict(prefix="layer: "),
                pad=dict(t=20),
                steps=[
                    dict(
                        method="update",
                        args=[
                            {"visible": [other == layer for other in layers]},
                            {"title.text": titles[layer]},
                        ],
                        label=str(layer),
                    )
                    for layer in layers
                ],
            )
        ],
    )
    lines = [
        _verdict_line(model_label),
        f"  evidence, {model_label}: "
        + ", ".join(f"layer {layer} results/{LENS_FILES[model_label][layer]}" for layer in layers),
    ]
    return fig, {
        "lines": lines,
        "judged_affective": {layer: JUDGED_AFFECTIVE[(model_label, layer)] for layer in layers},
    }
