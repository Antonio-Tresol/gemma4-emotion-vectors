"""S8 exhibit: the model's own probe geometry as a difficulty predictor,
with NRC VAD affective distance as the competing (partialled-out) predictor."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from einops import einsum
from jaxtyping import Float, Num
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from emotion_vectors.artifacts import fetch
from emotion_vectors.trajectories import unit_contrast_probes

from ._data import (
    LAYERS,
    PRIMARY_LAYER,
    Arms,
    GateRead,
    LayerStats,
    LeadRead,
    Vad,
    bank_columns,
    gate_ranks,
    layer_slider,
    true_leads,
)


def rank_partial(
    x: Num[np.ndarray, " pairs"],
    y: Num[np.ndarray, " pairs"],
    z: Num[np.ndarray, " pairs"],
) -> float:
    """Spearman partial corr r(x, y | z): correlate rank residuals."""
    x_ranks, y_ranks, z_ranks = (
        np.argsort(np.argsort(values)).astype(float) for values in (x, y, z)
    )
    # residualize both rank vectors on z's ranks, then correlate what is left
    x_residual = x_ranks - np.polyval(np.polyfit(z_ranks, x_ranks, 1), z_ranks)
    y_residual = y_ranks - np.polyval(np.polyfit(z_ranks, y_ranks, 1), z_ranks)
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def selfgen_probe_cosines(
    arms: Arms,
) -> tuple[Float[np.ndarray, "layers bank_probes bank_probes"], list[str]]:
    """Reconstruct the exact selfgen probe bank and its within-bank cosines.

    Rebuilds the unit contrast probes from the stored extraction means (the
    n=256 bucket the extraction used), reorders them to the recorded label
    order, and returns (probe-probe cosines [layers, bank_probes,
    bank_probes] on the LAYERS grid, bank emotion names).
    """
    corpus_bundle = np.load(
        fetch("emotion_vectors_it_postfix/emotion_means.npz"), allow_pickle=True
    )
    # the extraction bundles share one layer grid; map LAYERS onto it
    bundle_layer_pos = [list(corpus_bundle["layers"]).index(layer) for layer in LAYERS]
    selfgen_bundle = np.load(fetch("e6_scale_means.npz"), allow_pickle=True)
    bundle_names = list(map(str, selfgen_bundle["emotions"]))
    # bucket -1 = n=256, the bank the extraction used; layer grid shared with corpus
    selfgen_dirs = unit_contrast_probes(selfgen_bundle["means"][-1].astype(np.float32))[
        :, bundle_layer_pos
    ]
    _, bank_names = bank_columns(arms, "it_v2", "selfgen")
    bundle_order = [bundle_names.index(emotion) for emotion in bank_names]
    ordered_dirs = selfgen_dirs[bundle_order]
    probe_cos = einsum(
        ordered_dirs,
        ordered_dirs,
        "probe_a layers d_model, probe_b layers d_model -> layers probe_a probe_b",
    )
    return probe_cos, bank_names


def _vad_distance_matrix(
    vad: Vad, names: list[str]
) -> Float[np.ndarray, "bank_probes bank_probes"]:
    """Pairwise Euclidean distance in the valence-arousal plane."""
    return np.array(
        [[np.hypot(vad[a][0] - vad[b][0], vad[a][1] - vad[b][1]) for b in names] for a in names]
    )


def _s8_crowding(
    gate: GateRead,
    probe_cos: Float[np.ndarray, "layers bank_probes bank_probes"],
    names: list[str],
) -> tuple[LayerStats, list[str]]:
    """Sub-read (a): per layer, Spearman rho between an emotion's crowding
    (mean cosine to the other 11 probes) and its top-1 rate. Returns
    (``stats[layer] = {"crowding_rho", "crowding_p"}``, printed lines)."""
    _, winner, phase_rows = gate
    not_self = ~np.eye(len(names), dtype=bool)
    lines = ["", "(a) crowding vs tracking quality, per layer (spearman, n=12 emotions):"]
    stats: LayerStats = {}
    for layer_pos, layer in enumerate(LAYERS):
        crowding, top1 = [], []
        for emotion_pos, emotion in enumerate(names):
            phase_idx = [pos for pos, row in enumerate(phase_rows) if row["emotion"] == emotion]
            crowding.append(probe_cos[layer_pos, emotion_pos, not_self[emotion_pos]].mean())
            top1.append((winner[phase_idx, layer_pos] == emotion_pos).mean())
        rho = spearmanr(crowding, top1)
        stats[layer] = {"crowding_rho": float(rho.statistic), "crowding_p": float(rho.pvalue)}
        lines.append(
            f"  L{layer}: rho = {rho.statistic:+.2f} (p={rho.pvalue:.3f})"
            + ("  <- crowded probes track worse" if rho.statistic < -0.3 else "")
        )
    return stats, lines


def _s8_pair_confusion(
    gate: GateRead,
    probe_cos: Float[np.ndarray, "layers bank_probes bank_probes"],
    vad_dist: Float[np.ndarray, "bank_probes bank_probes"],
    names: list[str],
) -> tuple[dict[str, float], list[str]]:
    """Sub-read (b): over the 132 ordered pairs at the primary layer, does
    probe cosine predict WHICH wrong answer wins, beyond VAD closeness?

    Returns (stats with the raw Spearmans and both rank partials, printed
    lines). The probe-cos partial controlling VAD is the model-idiosyncratic
    read.
    """
    _, winner, phase_rows = gate
    primary_pos = LAYERS.index(PRIMARY_LAYER)
    confusion_rate, pair_cos, vad_closeness = [], [], []
    for emotion_pos, emotion in enumerate(names):
        phase_idx = [pos for pos, row in enumerate(phase_rows) if row["emotion"] == emotion]
        report_counts = np.bincount(winner[phase_idx, primary_pos], minlength=len(names))
        for other_pos in range(len(names)):
            if other_pos == emotion_pos:
                continue
            confusion_rate.append(report_counts[other_pos] / len(phase_idx))
            pair_cos.append(probe_cos[primary_pos, emotion_pos, other_pos])
            vad_closeness.append(-vad_dist[emotion_pos, other_pos])
    confusion_rate, pair_cos, vad_closeness = map(
        np.array, (confusion_rate, pair_cos, vad_closeness)
    )
    stats = {
        "spearman_cos": float(spearmanr(confusion_rate, pair_cos).statistic),
        "spearman_vad": float(spearmanr(confusion_rate, vad_closeness).statistic),
        "partial_cos_given_vad": rank_partial(confusion_rate, pair_cos, vad_closeness),
        "partial_vad_given_cos": rank_partial(confusion_rate, vad_closeness, pair_cos),
    }
    lines = [
        "",
        f"(b) which wrong answer wins (132 pairs, layer {PRIMARY_LAYER}):",
        f"  confusion ~ probe cos          spearman {stats['spearman_cos']:+.2f}",
        f"  confusion ~ VAD closeness      spearman {stats['spearman_vad']:+.2f}",
        f"  confusion ~ probe cos | VAD    partial  {stats['partial_cos_given_vad']:+.2f}"
        "   <- the model-idiosyncratic read",
        f"  confusion ~ VAD | probe cos    partial  {stats['partial_vad_given_cos']:+.2f}",
    ]
    return stats, lines


def _s8_transitions(
    arms: Arms,
    gate: GateRead,
    probe_cos: Float[np.ndarray, "layers bank_probes bank_probes"],
    names: list[str],
) -> tuple[dict[str, float], list[str]]:
    """Sub-read (c): are transitions between similar probes harder?

    At the primary layer, correlates each transition's R1 lead and its
    post-boundary gate rank with cos(from-probe, to-probe), for transitions
    whose outgoing emotion is also in the bank. Returns (stats, printed
    lines).
    """
    ranks, _, phase_rows = gate
    primary_pos = LAYERS.index(PRIMARY_LAYER)
    leads, transition_rows = true_leads(arms, "it_v2", "selfgen")
    # the phase AFTER transition k carries phase_index k, so its gate-rank row
    # is looked up by (story_id, transition_index)
    rank_by_key = {
        (row["story_id"], row["phase_index"]): ranks[pos] for pos, row in enumerate(phase_rows)
    }
    trans_cos, trans_lead, trans_rank = [], [], []
    for trans_pos, transition in enumerate(transition_rows):
        if transition["from_emotion"] not in names:
            continue
        key = (transition["story_id"], transition["transition_index"])
        if key not in rank_by_key:
            continue
        trans_cos.append(
            probe_cos[
                primary_pos,
                names.index(transition["from_emotion"]),
                names.index(transition["to_emotion"]),
            ]
        )
        trans_lead.append(leads[trans_pos, primary_pos])
        trans_rank.append(rank_by_key[key][primary_pos])
    trans_cos, trans_lead, trans_rank = map(np.array, (trans_cos, trans_lead, trans_rank))
    stats = {
        "n": int(len(trans_cos)),
        "lead_vs_cos": float(spearmanr(trans_lead, trans_cos).statistic),
        "rank_vs_cos": float(spearmanr(trans_rank, trans_cos).statistic),
    }
    lines = [
        "",
        f"(c) transitions with both emotions in the bank (n={len(trans_cos)}, "
        f"layer {PRIMARY_LAYER}):",
        f"  R1 lead ~ cos(from, to)             spearman {stats['lead_vs_cos']:+.2f}",
        f"  post-boundary gate rank ~ cos       spearman {stats['rank_vs_cos']:+.2f}"
        "   (positive = similar probes -> worse rank)",
    ]
    return stats, lines


def _s8_pair_points(
    gate: GateRead,
    probe_cos: Float[np.ndarray, "layers bank_probes bank_probes"],
    names: list[str],
    layer_pos: int,
) -> tuple[list[float], list[float]]:
    """(probe cosine, confusion rate) per ordered emotion pair at one layer,
    the left S8 panel's point cloud."""
    _, winner, phase_rows = gate
    cosines, rates = [], []
    for emotion_pos, emotion in enumerate(names):
        phase_idx = [pos for pos, row in enumerate(phase_rows) if row["emotion"] == emotion]
        report_counts = np.bincount(winner[phase_idx, layer_pos], minlength=len(names))
        for other_pos in range(len(names)):
            if other_pos != emotion_pos:
                rates.append(report_counts[other_pos] / len(phase_idx))
                cosines.append(probe_cos[layer_pos, emotion_pos, other_pos])
    return cosines, rates


def _s8_lead_points(
    lead_read: LeadRead,
    probe_cos: Float[np.ndarray, "layers bank_probes bank_probes"],
    names: list[str],
    layer_pos: int,
) -> tuple[list[float], list[float]]:
    """(cos(from, to), R1 lead) per transition whose outgoing emotion is in
    the bank at one layer, the right S8 panel's point cloud."""
    leads, transition_rows = lead_read
    lead_cosines, lead_values = [], []
    for trans_pos, transition in enumerate(transition_rows):
        if transition["from_emotion"] in names:
            lead_cosines.append(
                probe_cos[
                    layer_pos,
                    names.index(transition["from_emotion"]),
                    names.index(transition["to_emotion"]),
                ]
            )
            lead_values.append(leads[trans_pos, layer_pos])
    return lead_cosines, lead_values


def _s8_scatter_figure(
    arms: Arms,
    gate: GateRead,
    probe_cos: Float[np.ndarray, "layers bank_probes bank_probes"],
    names: list[str],
) -> go.Figure:
    """The two S8 pair-level scatter panels with a layer slider: confusion
    rate vs probe cosine (ordered pairs), and R1 lead vs cos(from, to)."""
    lead_read = true_leads(arms, "it_v2", "selfgen")
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "confusion rate vs probe cosine (pairs)",
            "anticipation lead vs cos(from, to)",
        ],
    )
    for layer_pos in range(len(LAYERS)):
        cosines, rates = _s8_pair_points(gate, probe_cos, names, layer_pos)
        fig.add_trace(
            go.Scatter(
                x=cosines,
                y=rates,
                mode="markers",
                showlegend=False,
                marker=dict(size=6, color="#4c78a8", opacity=0.7),
            ),
            row=1,
            col=1,
        )
        lead_cosines, lead_values = _s8_lead_points(lead_read, probe_cos, names, layer_pos)
        fig.add_trace(
            go.Scatter(
                x=lead_cosines,
                y=lead_values,
                mode="markers",
                showlegend=False,
                marker=dict(size=5, color="#f58518", opacity=0.5),
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(
        title_text="cosine between the two probes (model-side similarity)", row=1, col=1
    )
    fig.update_yaxes(title_text="P(model reports this wrong emotion | target)", row=1, col=1)
    fig.update_xaxes(title_text="cos(from-probe, to-probe) of the transition", row=1, col=2)
    fig.update_yaxes(title_text="R1 lead of the incoming emotion (cosine units)", row=1, col=2)
    fig.add_hline(y=0, line_color="#888", line_width=1, row=1, col=2)
    fig.update_layout(
        height=450,
        title="S8: does the model's own probe geometry predict its difficulties? "
        "(it_v2, selfgen bank; each left point = one ordered emotion pair, "
        "each right point = one transition)",
    )
    layer_slider(fig, traces_per_layer=2)
    return fig


def s8_geometry_figure(arms: Arms, vad: Vad) -> tuple[go.Figure, dict[str, object]]:
    """S8: probe geometry as a difficulty predictor, vs the VAD competitor.

    Reconstructs the exact selfgen probe bank, then runs the three registered
    sub-reads: (a) crowding vs top-1 rate per layer, (b) pairwise probe
    cosine vs confusion rate with VAD rank partials, (c) transition
    difficulty vs cos(from, to). Returns the two-panel scatter figure (layer
    slider) and ``stats = {"crowding", "pairs", "transitions", "lines"}``
    where ``lines`` is the exact printed block.
    """
    probe_cos, names = selfgen_probe_cosines(arms)
    primary_pos = LAYERS.index(PRIMARY_LAYER)
    vad_dist = _vad_distance_matrix(vad, names)
    not_self = ~np.eye(len(names), dtype=bool)
    # the two candidate predictors are themselves correlated, which is why the
    # sub-reads below lean on partials rather than raw correlations
    predictor_corr = spearmanr(probe_cos[primary_pos][not_self], -vad_dist[not_self]).statistic
    lines = [
        f"probe cos vs VAD distance over the 132 ordered pairs, layer {PRIMARY_LAYER}: "
        f"spearman {predictor_corr:+.2f} "
        "(correlated, hence the partials below)"
    ]
    gate = gate_ranks(arms, "it_v2", "selfgen")
    crowding_stats, crowding_lines = _s8_crowding(gate, probe_cos, names)
    lines += crowding_lines
    pair_stats, pair_lines = _s8_pair_confusion(gate, probe_cos, vad_dist, names)
    lines += pair_lines
    transition_stats, transition_lines = _s8_transitions(arms, gate, probe_cos, names)
    lines += transition_lines
    fig = _s8_scatter_figure(arms, gate, probe_cos, names)
    stats = {
        "crowding": crowding_stats,
        "pairs": pair_stats,
        "transitions": transition_stats,
        "lines": lines,
    }
    return fig, stats
