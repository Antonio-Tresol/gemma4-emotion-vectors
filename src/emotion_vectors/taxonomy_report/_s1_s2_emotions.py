"""S1 and S2 exhibits: per-emotion tracking quality and designed-family
difficulty (the tagged probe's rank and the R1 lead per family, with
cluster-bootstrap CIs)."""

from __future__ import annotations

from functools import partial

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Float, Num
from plotly.subplots import make_subplots

from ._data import (
    FAMILIES,
    LAYERS,
    Arms,
    LayerStats,
    cluster_boot_ci,
    layer_slider,
    tagged_ranks,
    true_leads,
)

_S2_ARM_COLORS = {"it_v2": "#4c78a8", "deepseek": "#f58518"}
# The two twelve-probe sets S1 compares, named as the record dumps name them.
PROBE_SETS = ["selfgen", "deepseek"]
# Plain reading of those names, for panel titles and legends.
PROBE_SET_DISPLAY = {
    "selfgen": "probes from the model's own stories",
    "deepseek": "probes from DeepSeek-written stories",
}
# plain names for the designed story types; the registry code stays in parentheses
FAMILY_DISPLAY = {
    "A_superposition": "superposition<br>(A)",
    "B_conflict": "conflict<br>(B)",
    "D_timescale": "timescale<br>(D)",
    "E_arousal_mismatch": "arousal<br>mismatch (E)",
    "F_valence_spread": "valence<br>spread (F)",
}


def _s1_stats(arms: Arms) -> dict[str, LayerStats]:
    """Per probe set and layer: emotions sorted by top-1 rate, with median rank and n.

    ``stats[probe_set][layer] = {"emotions", "top1", "median_rank", "n"}``, all
    lists in best-first order at that layer.
    """
    # "selfgen" and "deepseek" are the probe-set names written into the record dumps
    reads = {probe_set: tagged_ranks(arms, "it_v2", probe_set) for probe_set in PROBE_SETS}
    stats: dict[str, LayerStats] = {probe_set: {} for probe_set in PROBE_SETS}
    for layer_pos, layer in enumerate(LAYERS):
        for probe_set in PROBE_SETS:
            ranks, _, phase_rows = reads[probe_set]
            emotions = sorted({row["emotion"] for row in phase_rows})
            # per-emotion summary at this layer: top-1 rate, median rank, n
            top1, median_rank, n_phases = [], [], []
            for emotion in emotions:
                phase_idx = [pos for pos, row in enumerate(phase_rows) if row["emotion"] == emotion]
                emotion_ranks = ranks[phase_idx, layer_pos]
                top1.append(float((emotion_ranks == 1).mean()))
                median_rank.append(float(np.median(emotion_ranks)))
                n_phases.append(len(phase_idx))
            best_first = np.argsort(top1)[::-1]
            stats[probe_set][layer] = {
                "emotions": [emotions[pos] for pos in best_first],
                "top1": [top1[pos] for pos in best_first],
                "median_rank": [median_rank[pos] for pos in best_first],
                "n": [n_phases[pos] for pos in best_first],
            }
    return stats


def _s1_title(stats: dict[str, LayerStats], layer: int) -> str:
    """S1's title for ONE layer: the question, then that layer's own answer.

    Every number is read from ``stats[probe_set][layer]``, so the slider cannot
    leave another layer's verdict on screen. Plotly never wraps a title, so
    the text is hand-wrapped with a fixed six-line shape at every layer.
    """
    layer_summary = stats["selfgen"][layer]
    above_half = sum(1 for rate in layer_summary["top1"] if rate >= 0.5)
    best_emotion = layer_summary["emotions"][0]
    best_rate = layer_summary["top1"][0]
    return (
        "Which emotions does the tracker actually recognise?"
        f" At layer {layer}, {above_half} of {len(layer_summary['emotions'])} tagged"
        "<br>emotions win outright among the twelve more than half the time,"
        f" best is {best_emotion} at {best_rate:.0%}"
        "<br><sup>one bar = one tagged emotion; height = the fraction of that emotion's story"
        " phases where its own probe ranks first of the twelve;</sup>"
        "<br><sup>bar label = median rank and number of phases."
        " Failure anchor: the dotted line at 1/12 is where guessing lands;</sup>"
        "<br><sup>strength anchor: 1.0 would mean the right probe always wins. Stories"
        " written by Gemma, read with v2 probes;</sup>"
        "<br><sup>the slider picks the layer, and the bars, the sort order and this"
        " headline all follow it</sup>"
    )


def s1_top1_figure(arms: Arms) -> tuple[go.Figure, dict[str, LayerStats]]:
    """S1: per-emotion top-1 rate and median rank of the tagged probe.

    One bar per tagged emotion, for each of the two twelve-probe sets (the
    model's own stories left, the DeepSeek-written stories right) on the
    primary arm, sorted by top-1 rate, with median rank and n as the bar
    label. Returns the figure (layer slider) and the ``_s1_stats`` dict.
    """
    stats = _s1_stats(arms)
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=[PROBE_SET_DISPLAY[name] for name in PROBE_SETS]
    )
    # traces added layer-major (both probe sets per layer) so the slider can toggle blocks
    for layer in LAYERS:
        for panel_pos, probe_set in enumerate(PROBE_SETS):
            layer_summary = stats[probe_set][layer]
            fig.add_trace(
                go.Bar(
                    x=layer_summary["emotions"],
                    y=layer_summary["top1"],
                    text=[
                        f"med {med:.0f}<br>n={n}"
                        for med, n in zip(layer_summary["median_rank"], layer_summary["n"])
                    ],
                    textposition="outside",
                    marker_color="#4c78a8" if probe_set == "selfgen" else "#f58518",
                    showlegend=False,
                ),
                row=1,
                col=panel_pos + 1,
            )
    for panel in (1, 2):
        fig.update_yaxes(title="top-1 rate (fraction of phases)", range=[0, 1.05], row=1, col=panel)
        fig.update_xaxes(title="tagged emotion (sorted by top-1 rate)", row=1, col=panel)
        fig.add_hline(
            y=1 / 12,
            line_dash="dot",
            line_color="#888",
            row=1,
            col=panel,
            # anchored bottom left: the bars are sorted descending, so the right
            # side of the panel is where the low bars crowd the chance line
            annotation_text="chance 1/12",
            annotation_position="bottom left",
            annotation_font_size=11,
            # the label lands on top of the tall left-hand bars, so it carries
            # its own background rather than sitting dark-on-dark
            annotation_bgcolor="rgba(255,255,255,0.8)",
        )
    fig.update_layout(
        width=1150,
        height=680,
        title_font_size=15,
        # top margin clears the six wrapped title lines (the shape is the same
        # at every layer), bottom margin holds the rotated emotion labels, the
        # axis titles and the layer slider
        margin=dict(t=250, b=190),
    )
    # the headline verdict is computed per layer and written by the slider
    layer_slider(fig, traces_per_layer=2, title_for_layer=partial(_s1_title, stats))
    return fig, stats


def _s2_family_cell(
    rank_part: tuple[Num[np.ndarray, " family_phases"], list[int]],
    lead_part: tuple[Float[np.ndarray, " family_trans"], list[int]],
    rng: np.random.Generator,
) -> dict[str, object]:
    """Median rank + CI and lead mean + CI for one family at one layer.

    ``rank_part`` / ``lead_part`` carry (values, triple_ids) for the family's
    phases and transitions. RNG order: rank CI first, then lead CI, matching
    the original cell's evaluation order; both are None for an empty family.
    """
    family_ranks, phase_triples = rank_part
    family_leads, trans_triples = lead_part
    median_rank = float(np.median(family_ranks)) if len(family_ranks) else None
    median_rank_ci = (
        cluster_boot_ci(family_ranks, phase_triples, rng, np.median) if len(family_ranks) else None
    )
    lead_mean = float(family_leads.mean()) if len(family_leads) else None
    lead_ci = cluster_boot_ci(family_leads, trans_triples, rng) if len(family_leads) else None
    return {
        "median_rank": median_rank,
        "median_rank_ci": median_rank_ci,
        "n_phases": len(family_ranks),
        "lead_mean": lead_mean,
        "lead_ci": lead_ci,
        "n_transitions": len(family_leads),
    }


def _s2_family_stats(arms: Arms, rng: np.random.Generator) -> dict[str, LayerStats]:
    """Median rank of the tagged probe and mean R1 lead per designed family,
    for both story arms.

    Returns ``stats[arm][layer][family] = {"median_rank", "median_rank_ci",
    "n_phases", "lead_mean", "lead_ci", "n_transitions"}``, read with the
    probes the model built from its own stories. RNG order: arm-major, then
    layer, then family.
    """
    stats: dict[str, LayerStats] = {}
    for arm in ["it_v2", "deepseek"]:
        ranks, _, phase_rows = tagged_ranks(arms, arm, "selfgen")
        leads, transition_rows = true_leads(arms, arm, "selfgen")
        stats[arm] = {}
        for layer_pos, layer in enumerate(LAYERS):
            family_stats = {}
            for family in FAMILIES:
                phase_idx = [pos for pos, row in enumerate(phase_rows) if row["category"] == family]
                trans_idx = [
                    pos for pos, row in enumerate(transition_rows) if row["category"] == family
                ]
                family_stats[family] = _s2_family_cell(
                    (
                        ranks[phase_idx, layer_pos],
                        [phase_rows[pos]["triple_id"] for pos in phase_idx],
                    ),
                    (
                        leads[trans_idx, layer_pos],
                        [transition_rows[pos]["triple_id"] for pos in trans_idx],
                    ),
                    rng,
                )
            stats[arm][layer] = family_stats
    return stats


def _s2_add_arm_traces(fig: go.Figure, family_stats: dict[str, object], arm: str) -> None:
    """Add one arm's median-rank bars (left panel) and lead bars (right panel)
    for one layer, with the cluster-CI error bars."""
    median_ranks = [family_stats[family]["median_rank"] for family in FAMILIES]
    median_rank_cis = [family_stats[family]["median_rank_ci"] for family in FAMILIES]
    fig.add_trace(
        go.Bar(
            x=[FAMILY_DISPLAY[family] for family in FAMILIES],
            y=median_ranks,
            name="reading Gemma-written stories"
            if arm == "it_v2"
            else "reading DeepSeek-written stories",
            legendgroup=arm,
            marker_color=_S2_ARM_COLORS[arm],
            error_y=dict(
                array=[ci[1] - value for value, ci in zip(median_ranks, median_rank_cis)],
                arrayminus=[value - ci[0] for value, ci in zip(median_ranks, median_rank_cis)],
            ),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    lead_means = [family_stats[family]["lead_mean"] for family in FAMILIES]
    lead_cis = [family_stats[family]["lead_ci"] for family in FAMILIES]
    fig.add_trace(
        go.Bar(
            x=[FAMILY_DISPLAY[family] for family in FAMILIES],
            y=lead_means,
            legendgroup=arm,
            marker_color=_S2_ARM_COLORS[arm],
            showlegend=False,
            error_y=dict(
                array=[ci[1] - value for value, ci in zip(lead_means, lead_cis)],
                arrayminus=[value - ci[0] for value, ci in zip(lead_means, lead_cis)],
            ),
        ),
        row=1,
        col=2,
    )


def _s2_rank_span(medians: list[float]) -> str:
    """How far apart the five story types sit on the identity read, in words.

    Ties are common (several types share a median rank), so an all-equal
    layer says so rather than pretending a best and a worst type exist.
    """
    if min(medians) == max(medians):
        return f"all five sit at median rank {min(medians):.0f} of 12"
    return f"the five span median rank {min(medians):.0f} to {max(medians):.0f} of 12"


def _s2_title(stats: dict[str, LayerStats], layer: int) -> str:
    """S2's title for ONE layer: the question, then that layer's own answer.

    Reads the primary arm's per-family cells at ``layer``: the identity read
    is summarised by the median-rank span, the anticipation read by how many
    of the five types have a bootstrap CI entirely above zero. Hand-wrapped
    to the same five-line shape at every layer.
    """
    family_stats = stats["it_v2"][layer]
    medians = [family_stats[family]["median_rank"] for family in FAMILIES]
    rises = sum(1 for family in FAMILIES if family_stats[family]["lead_ci"][0] > 0)
    return (
        f"Which designed story types does the tracker handle best? At layer {layer},"
        f"<br>on Gemma stories {_s2_rank_span(medians)}, and {rises} of {len(FAMILIES)}"
        " anticipate the next emotion with a CI clear of zero"
        "<br><sup>one bar = one designed story type read on one story set; left height ="
        " median rank of the tagged probe among 12;</sup>"
        "<br><sup>right height = mean rise of the incoming emotion before the boundary."
        " Failure anchors: the dotted line at rank 6.5 is where guessing lands, and"
        " 0 rise = no anticipation;</sup>"
        "<br><sup>strength anchor: the dashed line at rank 1 would mean the right probe"
        " always wins. Error bars = 95% cluster-bootstrap CI over triples;"
        " self-generated probes</sup>"
    )


def s2_family_figure(
    arms: Arms, rng: np.random.Generator
) -> tuple[go.Figure, dict[str, LayerStats]]:
    """S2: tagged-probe rank and anticipation lead per designed family, with CIs.

    Read with the probes the model built from its own stories, primary arm vs
    DeepSeek-story arm. Returns the grouped-bar figure (layer slider) and the
    ``_s2_family_stats`` dict.
    """
    stats = _s2_family_stats(arms, rng)
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.09,
        subplot_titles=[
            "identity: is the right emotion recognized?",
            "anticipation: does it rise before the boundary?",
        ],
    )
    # traces added layer-major (both arms per layer) so the slider can toggle blocks
    for layer in LAYERS:
        for arm in ["it_v2", "deepseek"]:
            _s2_add_arm_traces(fig, stats[arm][layer], arm)
    fig.update_yaxes(
        title="median rank of tagged probe<br>(1 = best of 12)", row=1, col=1, range=[0, 8.5]
    )
    fig.update_yaxes(title="rise of the incoming emotion<br>(centered-cosine units)", row=1, col=2)
    for panel in (1, 2):
        # labels wrap to two lines instead of tilting, so the axis title stays
        # clear of the layer slider underneath it
        fig.update_xaxes(title="designed story type", tickangle=0, row=1, col=panel)
    # grading anchors: what failure and success look like, named in the plot
    fig.add_hline(
        y=6.5,
        line_dash="dot",
        line_color="#888",
        row=1,
        col=1,
        annotation_text="chance rank = 6.5",
        annotation_position="top right",
    )
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="#54a24b",
        row=1,
        col=1,
        annotation_text="perfect = 1",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=0,
        line_color="#888",
        line_width=1,
        row=1,
        col=2,
        annotation_text="0 = no anticipation",
        annotation_position="top left",
        annotation_font_size=11,
        # bars sit on both sides of the line at some layers, so the label
        # carries its own background and stays readable at every slider step
        annotation_bgcolor="rgba(255,255,255,0.8)",
    )
    fig.update_layout(
        width=1150,
        height=680,
        barmode="group",
        # top margin clears the five wrapped title lines plus the legend row;
        # the bottom one holds the tilted type labels, axis titles and slider
        margin=dict(t=210, b=150),
        legend=dict(orientation="h", y=1.19, x=0.0),
        title=dict(y=0.98, yanchor="top", font_size=15),
    )
    layer_slider(fig, traces_per_layer=4, title_for_layer=partial(_s2_title, stats))
    return fig, stats
