"""S1 and S2 exhibits: per-emotion tracking quality and designed-family
difficulty (gate rank and R1 lead per family, with cluster-bootstrap CIs)."""

from __future__ import annotations

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
    gate_ranks,
    layer_slider,
    true_leads,
)

_S2_ARM_COLORS = {"it_v2": "#4c78a8", "deepseek": "#f58518"}
# plain names for the designed story types; the registry code stays in parentheses
FAMILY_DISPLAY = {
    "A_superposition": "superposition (A)",
    "B_conflict": "conflict (B)",
    "D_timescale": "timescale (D)",
    "E_arousal_mismatch": "arousal mismatch (E)",
    "F_valence_spread": "valence spread (F)",
}


def _s1_stats(arms: Arms) -> dict[str, LayerStats]:
    """Per bank and layer: emotions sorted by top-1 rate, with median rank and n.

    ``stats[bank][layer] = {"emotions", "top1", "median_rank", "n"}``, all
    lists in best-first order at that layer.
    """
    bank_reads = {bank: gate_ranks(arms, "it_v2", bank) for bank in ["selfgen", "deepseek"]}
    stats: dict[str, LayerStats] = {"selfgen": {}, "deepseek": {}}
    for layer_pos, layer in enumerate(LAYERS):
        for bank in ["selfgen", "deepseek"]:
            ranks, _, phase_rows = bank_reads[bank]
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
            stats[bank][layer] = {
                "emotions": [emotions[pos] for pos in best_first],
                "top1": [top1[pos] for pos in best_first],
                "median_rank": [median_rank[pos] for pos in best_first],
                "n": [n_phases[pos] for pos in best_first],
            }
    return stats


def s1_top1_figure(arms: Arms) -> tuple[go.Figure, dict[str, LayerStats]]:
    """S1: per-emotion top-1 rate and median gate rank, both battery-12 banks.

    One bar per tagged emotion (selfgen panel left, deepseek panel right) on
    the primary arm, sorted by top-1 rate, with median rank and n as the bar
    label. Returns the figure (layer slider) and the ``_s1_stats`` dict.
    """
    stats = _s1_stats(arms)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["selfgen bank", "deepseek bank"])
    # traces added layer-major (both banks per layer) so the slider can toggle blocks
    for layer in LAYERS:
        for panel_pos, bank in enumerate(["selfgen", "deepseek"]):
            layer_summary = stats[bank][layer]
            fig.add_trace(
                go.Bar(
                    x=layer_summary["emotions"],
                    y=layer_summary["top1"],
                    text=[
                        f"med {med:.0f}<br>n={n}"
                        for med, n in zip(layer_summary["median_rank"], layer_summary["n"])
                    ],
                    textposition="outside",
                    marker_color="#4c78a8" if bank == "selfgen" else "#f58518",
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
            annotation_text="chance 1/12",
            annotation_position="top right",
        )
    fig.update_layout(
        height=440,
        title="S1: how often the tagged emotion's probe wins its bank outright "
        "(Gemma stories, v2 probes; bar label = median rank and n)",
    )
    layer_slider(fig, traces_per_layer=2)
    return fig, stats


def _s2_family_cell(
    gate_part: tuple[Num[np.ndarray, " family_phases"], list[int]],
    lead_part: tuple[Float[np.ndarray, " family_trans"], list[int]],
    rng: np.random.Generator,
) -> dict[str, object]:
    """Gate median + CI and lead mean + CI for one family at one layer.

    ``gate_part`` / ``lead_part`` carry (values, triple_ids) for the family's
    phases and transitions. RNG order: gate CI first, then lead CI, matching
    the original cell's evaluation order; both are None for an empty family.
    """
    family_ranks, phase_triples = gate_part
    family_leads, trans_triples = lead_part
    gate_median = float(np.median(family_ranks)) if len(family_ranks) else None
    gate_ci = (
        cluster_boot_ci(family_ranks, phase_triples, rng, np.median) if len(family_ranks) else None
    )
    lead_mean = float(family_leads.mean()) if len(family_leads) else None
    lead_ci = cluster_boot_ci(family_leads, trans_triples, rng) if len(family_leads) else None
    return {
        "gate_median": gate_median,
        "gate_ci": gate_ci,
        "n_phases": len(family_ranks),
        "lead_mean": lead_mean,
        "lead_ci": lead_ci,
        "n_transitions": len(family_leads),
    }


def _s2_family_stats(arms: Arms, rng: np.random.Generator) -> dict[str, LayerStats]:
    """Gate median rank and mean R1 lead per designed family, both story arms.

    Returns ``stats[arm][layer][family] = {"gate_median", "gate_ci",
    "n_phases", "lead_mean", "lead_ci", "n_transitions"}``, selfgen bank.
    RNG order: arm-major, then layer, then family.
    """
    stats: dict[str, LayerStats] = {}
    for arm in ["it_v2", "deepseek"]:
        ranks, _, phase_rows = gate_ranks(arms, arm, "selfgen")
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
    """Add one arm's gate-rank bars (left panel) and lead bars (right panel)
    for one layer, with the cluster-CI error bars."""
    gate_medians = [family_stats[family]["gate_median"] for family in FAMILIES]
    gate_cis = [family_stats[family]["gate_ci"] for family in FAMILIES]
    fig.add_trace(
        go.Bar(
            x=[FAMILY_DISPLAY[family] for family in FAMILIES],
            y=gate_medians,
            name="reading Gemma-written stories"
            if arm == "it_v2"
            else "reading DeepSeek-written stories",
            legendgroup=arm,
            marker_color=_S2_ARM_COLORS[arm],
            error_y=dict(
                array=[ci[1] - value for value, ci in zip(gate_medians, gate_cis)],
                arrayminus=[value - ci[0] for value, ci in zip(gate_medians, gate_cis)],
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


def s2_family_figure(
    arms: Arms, rng: np.random.Generator
) -> tuple[go.Figure, dict[str, LayerStats]]:
    """S2: gate rank and anticipation lead per designed family, with CIs.

    Selfgen bank, primary arm vs DeepSeek-story arm. Returns the grouped-bar
    figure (layer slider) and the ``_s2_family_stats`` dict.
    """
    bank = "selfgen"
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
        fig.update_xaxes(
            title="designed story type", tickangle=18, title_standoff=28, row=1, col=panel
        )
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
        annotation_text="no anticipation",
        annotation_position="bottom right",
    )
    fig.update_layout(
        height=560,
        barmode="group",
        margin=dict(t=150, b=120),
        legend=dict(orientation="h", y=1.22, x=0.0),
        title=dict(
            text="Which designed story types does the tracker handle best?"
            "<br><sup>self-generated probes read both story sets; bars pair the two story "
            "authors per type; error bars = 95% cluster-bootstrap CI over triples</sup>",
            y=0.97,
        ),
    )
    layer_slider(fig, traces_per_layer=4)
    return fig, stats
