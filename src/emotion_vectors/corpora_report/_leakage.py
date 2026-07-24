"""Section 2: the emotion-word leakage quality control.

Leakage = a generated text naming its target emotion despite the generation
instruction "do not name the emotion". Counted with the same stem lists the
scoring pipeline uses (``emotion_vectors.scoring.EMOTION_STEMS``), so this
audit and the probe work agree on what "naming the emotion" means.
``s2_leakage_figure`` returns ``(figure, stats)`` with ``stats["lines"]``
holding the printed record.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.colors import qualitative

from emotion_vectors.scoring import EMOTION_STEMS, leakage

from ._data import CorporaContext

# Legend entries name the ROLE of each color: who wrote the texts, and
# whether that writer was following the "do not name the emotion" instruction.
GENERATOR_LEGEND = {
    "gemma-4-4B": "written by gemma-4-4B (reference corpus authors)",
    "gemma-4-31b base": "written by gemma-4-31b base (ours; base models follow instructions poorly)",
    "gemma-4-31b-it": "written by gemma-4-31b-it (ours, instruction-tuned)",
    "deepseek-v4-pro": "written by deepseek-v4-pro (ours, external API generator)",
}
GENERATOR_COLORS = dict(zip(GENERATOR_LEGEND, qualitative.Plotly))

# (corpus key in the catalog, generator short name); "published" is counted
# from the HF rows, every other corpus from its grouped jsonl.
LEAKAGE_BARS = [
    ("published", "gemma-4-4B"),
    ("dialogues_base", "gemma-4-31b base"),
    ("dialogues_it", "gemma-4-31b-it"),
    ("self_stories", "gemma-4-31b-it"),
    ("deepseek_fixed", "deepseek-v4-pro"),
    ("deepseek_diverse", "deepseek-v4-pro"),
]


def _published_leakage(published: dict[str, list[str]]) -> tuple[int, int]:
    """(leaked, total) over the emotions covered by EMOTION_STEMS, mirroring
    ``scoring.leakage`` for a corpus already loaded as {emotion: [story, ...]}."""
    leaked = total = 0
    for emotion, stories in published.items():
        stems = EMOTION_STEMS.get(emotion)
        if not stems:
            continue
        for story in stories:
            total += 1
            leaked += any(stem in story.lower() for stem in stems)
    return leaked, total


def _axis_label(corpus_label: str) -> str:
    """Two-line tick label for a corpus, so six bar labels fit unrotated.

    Breaks at the comma (or the opening parenthesis) that separates a corpus
    family from its variant, e.g. "dialogues, base model" -> two lines.
    """
    if ", " in corpus_label:
        head, _, tail = corpus_label.partition(", ")
        return f"{head},<br>{tail}"
    if " (" in corpus_label:
        head, _, tail = corpus_label.partition(" (")
        return f"{head}<br>({tail}"
    return corpus_label


def _s2_title(base_percent: float, worst_following_percent: float) -> str:
    """Verdict title with both headline percentages computed from the counts.

    Every line is hand-wrapped: plotly titles do not wrap, so a long line is
    silently truncated at the figure edge.
    """
    return (
        "Do generated texts avoid naming their target emotion?"
        f"<br>Yes, except the base model: every instruction-following corpus at or below"
        f" {worst_following_percent:.1f}%, the base model at {base_percent:.1f}%"
        "<br><sup>one bar = one corpus; height = % of its audited texts containing a word-stem"
        " of their target emotion,"
        f"<br>counted over the {len(EMOTION_STEMS)} battery emotions that have stem lists"
        " (emotion_vectors.scoring.EMOTION_STEMS); 0% = perfect compliance</sup>"
        "<br><sup>grading: inside the green band (at or below the reference corpus the source"
        " papers worked with) = clean enough to reuse;"
        "<br>near the base-model bar = lexical confound (vectors could encode the word, not the"
        " concept)</sup>"
    )


def _leakage_counts(ctx: CorporaContext) -> list[tuple[str, str, int, int]]:
    """Per audited corpus: (label, generator short name, leaked, audited texts).

    The reference corpus is counted from its already-loaded Hugging Face rows,
    every other corpus from its grouped jsonl with ``scoring.leakage``. The
    diverse DeepSeek arm is skipped when the context could not fetch it (the
    context already printed that degradation).
    """
    counts: list[tuple[str, str, int, int]] = []
    for key, generator in LEAKAGE_BARS:
        if key == "deepseek_diverse" and not ctx.diverse_available:
            continue
        entry = ctx.entry(key)
        if key == "published":
            leaked, total = _published_leakage(ctx.published)
        else:
            leaked, total = leakage(entry.grouped_path)
        counts.append((entry.label, generator, leaked, total))
    return counts


def _add_leakage_bars(
    fig: go.Figure, counts: list[tuple[str, str, int, int]], percent: dict[str, float]
) -> None:
    """One bar trace per generator, so the legend names each color's role once.

    Hover carries the raw fraction behind each bar, so a reader can check the
    arithmetic without leaving the figure.
    """
    for generator, legend_name in GENERATOR_LEGEND.items():
        rows = [row for row in counts if row[1] == generator]
        if not rows:
            continue
        fig.add_trace(
            go.Bar(
                x=[_axis_label(label) for label, _, _, _ in rows],
                y=[percent[label] for label, _, _, _ in rows],
                name=legend_name,
                marker_color=GENERATOR_COLORS[generator],
                text=[f"{percent[label]:.1f}%" for label, _, _, _ in rows],
                textposition="outside",
                customdata=[[label, leaked, total] for label, _, leaked, total in rows],
                hovertemplate=(
                    "%{customdata[0]}<br>%{customdata[1]} of %{customdata[2]} audited texts"
                    " name their target emotion (%{y:.2f}%)<extra></extra>"
                ),
            )
        )


def _add_grading_anchors(
    fig: go.Figure, reference_percent: float, base_percent: float, base_label: str
) -> None:
    """Draw the scale a reader grades a bar against, without reading the prose.

    A green band runs from the perfect-compliance floor (0%) up to the
    reference corpus level, which is the CEILING of the acceptable set (the
    cleanliness the source papers worked with), and a red marker labels the
    base-model bar as the failure anchor.
    """
    fig.add_hrect(
        y0=0,
        y1=reference_percent,
        fillcolor="seagreen",
        opacity=0.10,
        line_width=0,
        layer="below",
    )
    fig.add_hline(
        y=reference_percent,
        line_dash="dash",
        line_color="seagreen",
        annotation_text=f"top of the clean band: reference corpus ({reference_percent:.1f}%)",
        annotation_position="top right",
        annotation_font=dict(size=11, color="seagreen"),
    )
    # The 0% floor gets a leader-line annotation planted in empty plot space
    # (between two short bars) rather than an hline label, which would sit on
    # top of the leftmost bar.
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_annotation(
        xref="paper",
        x=0.5,
        yref="y",
        y=0,
        ax=0,
        ay=-58,
        showarrow=True,
        arrowhead=2,
        arrowwidth=1,
        arrowcolor="gray",
        text="floor of the scale: 0% = perfect compliance",
        font=dict(size=11, color="gray"),
    )
    fig.add_annotation(
        x=_axis_label(base_label),
        y=base_percent,
        yshift=34,
        showarrow=False,
        text="failure anchor: instruction ignored,<br>lexical confound in base-arm probes",
        font=dict(size=10, color="crimson"),
    )


def s2_leakage_figure(ctx: CorporaContext) -> tuple[go.Figure, dict[str, Any]]:
    """Section 2 exhibit: emotion-word leakage per corpus.

    Input: the shared :class:`CorporaContext`. Output: ``(figure, stats)``:
    a bar chart (one bar per corpus, colored by generator, with the
    reference-corpus level and the perfect-compliance floor at 0% drawn as
    grading anchors bounding a green "clean enough to reuse" band) and
    ``stats["lines"]``, the printed per-corpus counts. ``stats["percent"]``
    maps corpus label -> leakage percentage, ``stats["audited"]`` maps
    corpus label -> (leaked, audited texts), and
    ``stats["reference_percent"]`` / ``stats["base_percent"]`` carry the two
    grading anchors so the notebook's prose can quote them without retyping.
    """
    # Count every bar first, so the title can state the verdict it draws.
    counts = _leakage_counts(ctx)
    percent = {label: 100 * leaked / total for label, _, leaked, total in counts}
    base_label = ctx.entry("dialogues_base").label
    reference_label = ctx.entry("published").label
    worst_following = max(pct for label, pct in percent.items() if label != base_label)

    fig = go.Figure()
    _add_leakage_bars(fig, counts, percent)
    _add_grading_anchors(fig, percent[reference_label], percent[base_label], base_label)
    fig.update_layout(
        title=dict(text=_s2_title(percent[base_label], worst_following), font=dict(size=13)),
        xaxis_title="corpus (rows of the catalog table in section 1)",
        xaxis_tickangle=0,
        yaxis_title="audited texts naming their target emotion (%)",
        yaxis_range=[0, percent[base_label] * 1.32],
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.995,
            font=dict(size=10),
            title=dict(text="who WROTE the texts", font=dict(size=11)),
            bgcolor="rgba(255,255,255,0.80)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
        height=650,
        width=1120,
        margin=dict(t=190, b=70),
    )
    lines = [
        f"{label}: {leaked}/{total} audited texts name their emotion ({leaked / total:.1%})"
        for label, _, leaked, total in counts
    ]
    lines.append(
        f"audited = texts whose target emotion is one of the {len(EMOTION_STEMS)} battery"
        " emotions with stem lists, so a corpus covering more emotions (the reference corpus"
        " covers 171) is audited on that battery subset only"
    )
    return fig, {
        "lines": lines,
        "percent": percent,
        "audited": {label: (leaked, total) for label, _, leaked, total in counts},
        "reference_percent": percent[reference_label],
        "base_percent": percent[base_label],
        "reference_label": reference_label,
        "base_label": base_label,
        "n_battery_emotions": len(EMOTION_STEMS),
    }
