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
) -> tuple[LayerStats, list[str]]:
    """Sub-read (b), per layer: does probe cosine predict WHICH wrong answer
    wins, beyond human VAD closeness - and by how much, concretely?

    For every layer: the raw Spearmans, both rank partials, and a concrete
    contrast (mean confusion rate among the most-similar quartile of pairs
    vs the least-similar quartile). Returns (stats_by_layer, printed lines
    for the primary layer)."""
    _, winner, phase_rows = gate
    stats: LayerStats = {}
    for layer_pos, layer in enumerate(LAYERS):
        confusion_rate, pair_cos, vad_closeness = [], [], []
        for emotion_pos, emotion in enumerate(names):
            phase_idx = [pos for pos, row in enumerate(phase_rows) if row["emotion"] == emotion]
            report_counts = np.bincount(winner[phase_idx, layer_pos], minlength=len(names))
            for other_pos in range(len(names)):
                if other_pos == emotion_pos:
                    continue
                confusion_rate.append(report_counts[other_pos] / len(phase_idx))
                pair_cos.append(probe_cos[layer_pos, emotion_pos, other_pos])
                vad_closeness.append(-vad_dist[emotion_pos, other_pos])
        confusion_rate, pair_cos, vad_closeness = map(
            np.array, (confusion_rate, pair_cos, vad_closeness)
        )
        # concrete contrast: are the model's "similar" pairs confused more?
        quartile = len(pair_cos) // 4
        by_similarity = np.argsort(pair_cos)
        stats[layer] = {
            "spearman_cos": float(spearmanr(confusion_rate, pair_cos).statistic),
            "spearman_vad": float(spearmanr(confusion_rate, vad_closeness).statistic),
            "partial_cos_given_vad": rank_partial(confusion_rate, pair_cos, vad_closeness),
            "partial_vad_given_cos": rank_partial(confusion_rate, vad_closeness, pair_cos),
            "similar_quartile_rate": float(confusion_rate[by_similarity[-quartile:]].mean()),
            "dissimilar_quartile_rate": float(confusion_rate[by_similarity[:quartile]].mean()),
            "overall_rate": float(confusion_rate.mean()),
            "n_quartile": int(quartile),
        }
    primary = stats[PRIMARY_LAYER]
    ratio = primary["similar_quartile_rate"] / max(primary["dissimilar_quartile_rate"], 1e-9)
    lines = [
        "",
        f"(b) which wrong answer wins (132 pairs, layer {PRIMARY_LAYER}):",
        f"  confusion ~ probe cos          spearman {primary['spearman_cos']:+.2f}",
        f"  confusion ~ VAD closeness      spearman {primary['spearman_vad']:+.2f}",
        f"  confusion ~ probe cos | VAD    partial  {primary['partial_cos_given_vad']:+.2f}"
        "   <- the model-idiosyncratic read",
        f"  confusion ~ VAD | probe cos    partial  {primary['partial_vad_given_cos']:+.2f}",
        f"  concrete contrast: the model's most-similar pair quartile is confused "
        f"{primary['similar_quartile_rate']:.1%} of the time vs "
        f"{primary['dissimilar_quartile_rate']:.1%} for the least-similar quartile "
        f"({ratio:.1f}x)",
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


def _s8_add_verdict_traces(fig: go.Figure, layer_stats: dict[str, float]) -> None:
    """One layer's two verdict bars: the predictor face-off (left panel) and
    the similar-vs-dissimilar quartile contrast (right panel)."""
    predictor_labels = [
        "model similarity<br>(raw)",
        "human similarity<br>(raw)",
        "model, beyond<br>human (partial)",
        "human, beyond<br>model (partial)",
    ]
    # blue = the model's ruler, green = the human ruler, in both raw and partial form
    predictor_colors = ["#4c78a8", "#54a24b", "#4c78a8", "#54a24b"]
    fig.add_trace(
        go.Bar(
            x=predictor_labels,
            y=[
                layer_stats["spearman_cos"],
                layer_stats["spearman_vad"],
                layer_stats["partial_cos_given_vad"],
                layer_stats["partial_vad_given_cos"],
            ],
            marker_color=predictor_colors,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=["most-similar quartile<br>(by probe cosine)", "least-similar quartile"],
            y=[layer_stats["similar_quartile_rate"], layer_stats["dissimilar_quartile_rate"]],
            marker_color=["#4c78a8", "#9d9d9d"],
            text=[
                f"{layer_stats['similar_quartile_rate']:.1%}",
                f"{layer_stats['dissimilar_quartile_rate']:.1%}",
            ],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )


def _s8_verdict_figure(pair_stats: LayerStats) -> go.Figure:
    """The two S8 verdict panels with a layer slider.

    Left: how well each candidate ruler predicts which wrong emotion wins
    (raw and unique-contribution rank correlations, zero = no power).
    Right: the concrete contrast - mean confusion rate among the model's
    most-similar pair quartile vs the least-similar quartile, against the
    all-pairs average."""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.11,
        subplot_titles=[
            "which ruler predicts the confusions?",
            "how much more are 'similar' pairs confused?",
        ],
    )
    for layer in LAYERS:
        _s8_add_verdict_traces(fig, pair_stats[layer])
    fig.update_yaxes(title="rank correlation with confusion rate", range=[-0.3, 0.5], row=1, col=1)
    fig.update_yaxes(title="mean P(model reports that wrong emotion)", row=1, col=2)
    fig.add_hline(
        y=0,
        line_color="#888",
        line_width=1,
        row=1,
        col=1,
        annotation_text="0 = no predictive power",
        annotation_position="bottom right",
    )
    overall = pair_stats[PRIMARY_LAYER]["overall_rate"]
    fig.add_hline(
        y=overall,
        line_dash="dot",
        line_color="#888",
        row=1,
        col=2,
        annotation_text=f"all-pairs average {overall:.1%}",
        annotation_position="top right",
    )
    fig.update_layout(
        height=520,
        margin=dict(t=130, b=90),
        title=dict(
            text="Whose notion of 'similar emotions' predicts the tracker's mistakes?"
            "<br><sup>blue = the model's ruler (probe cosine), green = the human ruler "
            "(NRC valence-arousal closeness); 'partial' = what each predicts after "
            "removing the other | 132 ordered emotion pairs, selfgen probes, "
            "Gemma-written stories</sup>",
            y=0.97,
        ),
    )
    layer_slider(fig, traces_per_layer=2)
    return fig


def s8_geometry_figure(arms: Arms, vad: Vad) -> tuple[go.Figure, dict[str, object]]:
    """S8: probe geometry as a difficulty predictor, vs the VAD competitor.

    Reconstructs the exact selfgen probe bank, then runs the three registered
    sub-reads: (a) crowding vs top-1 rate per layer, (b) pairwise probe
    cosine vs confusion rate with VAD rank partials, (c) transition
    difficulty vs cos(from, to). Returns the two-panel verdict figure (layer
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
    fig = _s8_verdict_figure(pair_stats)
    stats = {
        "crowding": crowding_stats,
        "pairs": pair_stats,
        "transitions": transition_stats,
        "lines": lines,
    }
    return fig, stats
