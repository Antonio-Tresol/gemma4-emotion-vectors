"""S5 exhibit (position and non-affect cuts) and the cross-arm summary block
that closes Part 1 of the notebook."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._data import (
    LAYERS,
    PRIMARY_LAYER,
    Arms,
    LayerStats,
    bank_columns,
    cluster_boot_ci,
    gate_ranks,
    layer_slider,
    true_leads,
)


def _s5_stats(arms: Arms, has_nonaffect: dict[int, bool], rng: np.random.Generator) -> LayerStats:
    """The two S5 cuts per layer, selfgen bank, primary arm.

    ``stats[layer] = {"position": {...}, "nonaffect": {...}}`` where each cut
    maps its label to (point estimate, CI, n). RNG order per layer:
    transition 1, transition 2, then flag False, flag True.
    """
    ranks, _, phase_rows = gate_ranks(arms, "it_v2", "selfgen")
    leads, transition_rows = true_leads(arms, "it_v2", "selfgen")
    stats: LayerStats = {}
    for layer_pos, layer in enumerate(LAYERS):
        # left panel cut: mean lead at the story's first vs second boundary
        position_cut = {}
        for position in [1, 2]:
            trans_idx = [
                pos
                for pos, row in enumerate(transition_rows)
                if row["transition_index"] == position
            ]
            ci = cluster_boot_ci(
                leads[trans_idx, layer_pos],
                [transition_rows[pos]["triple_id"] for pos in trans_idx],
                rng,
            )
            position_cut[f"transition {position}"] = (
                float(leads[trans_idx, layer_pos].mean()),
                ci,
                len(trans_idx),
            )
        # right panel cut: gate median rank with vs without a non-affect phase
        nonaffect_cut = {}
        for flag in [False, True]:
            phase_idx = [
                pos for pos, row in enumerate(phase_rows) if has_nonaffect[row["triple_id"]] == flag
            ]
            ci = cluster_boot_ci(
                ranks[phase_idx, layer_pos],
                [phase_rows[pos]["triple_id"] for pos in phase_idx],
                rng,
                np.median,
            )
            nonaffect_cut[f"has_nonaffect={flag}"] = (
                float(np.median(ranks[phase_idx, layer_pos])),
                ci,
                len(phase_idx),
            )
        stats[layer] = {"position": position_cut, "nonaffect": nonaffect_cut}
    return stats


def _s5_add_cut_bars(fig: go.Figure, cut: dict[str, object], panel: int, color: str) -> None:
    """One bar per cut label with the cluster-CI error bar and n on the bar."""
    fig.add_trace(
        go.Bar(
            x=list(cut),
            y=[cut[label][0] for label in cut],
            marker_color=color,
            error_y=dict(
                array=[cut[label][1][1] - cut[label][0] for label in cut],
                arrayminus=[cut[label][0] - cut[label][1][0] for label in cut],
            ),
            text=[f"n={cut[label][2]}" for label in cut],
            showlegend=False,
        ),
        row=1,
        col=panel,
    )


def s5_position_nonaffect_figure(
    arms: Arms, has_nonaffect: dict[int, bool], rng: np.random.Generator
) -> tuple[go.Figure, LayerStats]:
    """S5: R1 lead by transition position; gate rank by the nonaffect flag.

    Selfgen bank, primary arm. Returns the two-panel bar figure (layer
    slider) and the ``_s5_stats`` dict.
    """
    stats = _s5_stats(arms, has_nonaffect, rng)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["R1 lead by transition position", "gate median rank by has_nonaffect"],
    )
    for layer in LAYERS:
        _s5_add_cut_bars(fig, stats[layer]["position"], panel=1, color="#4c78a8")
        _s5_add_cut_bars(fig, stats[layer]["nonaffect"], panel=2, color="#b279a2")
    fig.update_yaxes(title="mean R1 lead (centered-cosine units)", row=1, col=1)
    fig.update_yaxes(title="median rank of tagged probe (1 = best of 12)", row=1, col=2)
    fig.add_hline(y=0, line_color="#888", line_width=1, row=1, col=1)
    fig.update_layout(
        height=440,
        title="S5: does difficulty depend on which transition, or on a non-affect "
        "distractor phase? (selfgen bank, Gemma stories; error bars = 95% cluster CI)",
    )
    layer_slider(fig, traces_per_layer=2)
    return fig, stats


def cross_arm_lines(
    arms: Arms,
    s1_stats: dict[str, LayerStats],
    s3_stats: LayerStats,
    s4_stats: LayerStats,
) -> list[str]:
    """The cross-arm table at the primary layer plus Part 1 headline numbers.

    One line per arm (gate median rank and mean R1 lead, selfgen bank), then
    the S1 best/worst emotions, the S3 valence contrast, and the S4
    winner-vs-shuffle distances, pulled from the section stats. Returns the
    lines exactly as the notebook prints them.
    """
    primary_pos = LAYERS.index(PRIMARY_LAYER)
    lines = [f"=== primary layer {PRIMARY_LAYER}, selfgen bank ==="]
    for arm in ["it_v2", "it", "deepseek", "deepseek_constant", "base"]:
        if not bank_columns(arms, arm, "selfgen")[0]:
            lines.append(
                f"{arm:22s} no selfgen bank in this arm (base lineage); its corpus-bank "
                "read lives in S6 and in notebook 10's factorial"
            )
            continue
        ranks, _, phase_rows = gate_ranks(arms, arm, "selfgen")
        leads, transition_rows = true_leads(arms, arm, "selfgen")
        lines.append(
            f"{arm:22s} gate median rank {np.median(ranks[:, primary_pos]):4.1f} "
            f"(n={len(phase_rows)})   R1 mean lead {leads[:, primary_pos].mean():+.4f} "
            f"(n={len(transition_rows)})"
        )
    s1_primary = s1_stats["selfgen"][PRIMARY_LAYER]
    lines.append("")
    lines.append(
        "S1 best-tracked emotions: "
        + ", ".join(s1_primary["emotions"][:3])
        + " | worst: "
        + ", ".join(s1_primary["emotions"][-3:])
    )
    cuts = s3_stats[PRIMARY_LAYER]["cuts"]
    lines.append(
        f"S3 cross-valence lead {cuts['cross-valence'][0]:+.4f} "
        f"vs same-valence {cuts['same-valence'][0]:+.4f}"
    )
    s4_primary = s4_stats[PRIMARY_LAYER]
    lines.append(
        f"S4 wrong-winner VAD distance {s4_primary['wrong_mean']:.2f} "
        f"vs shuffle {s4_primary['shuffle_mean']:.2f}"
    )
    return lines
