"""S3 and S4 exhibits: does tracking follow affective (valence-arousal)
distance, and does the confusion structure land on affective neighbors?"""

from __future__ import annotations

from functools import partial

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Bool, Float
from plotly.subplots import make_subplots

from ._data import (
    LAYERS,
    S3_VALENCE_GAP_EDGES,
    Arms,
    LayerStats,
    Vad,
    bank_columns,
    cluster_boot_ci,
    gate_ranks,
    layer_slider,
    true_leads,
)


def _s3_transition_geometry(
    arms: Arms, vad: Vad
) -> tuple[
    Float[np.ndarray, "vad_trans layers"],
    Float[np.ndarray, " vad_trans"],
    Float[np.ndarray, " vad_trans"],
    Bool[np.ndarray, " vad_trans"],
    list[int],
]:
    """Per-transition VAD geometry, for transitions with both emotions in the
    lexicon: (leads [vad_transitions, layers], |valence gap|, |arousal gap|,
    cross-valence flag, triple_ids)."""
    leads, transition_rows = true_leads(arms, "it_v2", "selfgen")
    covered = [
        pos
        for pos, row in enumerate(transition_rows)
        if row["from_emotion"] in vad and row["to_emotion"] in vad
    ]
    valence_gap = np.array(
        [
            abs(
                vad[transition_rows[pos]["to_emotion"]][0]
                - vad[transition_rows[pos]["from_emotion"]][0]
            )
            for pos in covered
        ]
    )
    arousal_gap = np.array(
        [
            abs(
                vad[transition_rows[pos]["to_emotion"]][1]
                - vad[transition_rows[pos]["from_emotion"]][1]
            )
            for pos in covered
        ]
    )
    # cross-valence: the from/to pair straddles the pleasant-unpleasant boundary
    crosses_valence = np.array(
        [
            (vad[transition_rows[pos]["from_emotion"]][0] > 0)
            != (vad[transition_rows[pos]["to_emotion"]][0] > 0)
            for pos in covered
        ]
    )
    triple_ids = [transition_rows[pos]["triple_id"] for pos in covered]
    return leads[covered], valence_gap, arousal_gap, crosses_valence, triple_ids


def _s3_stats(arms: Arms, vad: Vad, rng: np.random.Generator) -> LayerStats:
    """Per layer: the binned |delta valence| read, the lead-vs-gap Pearson
    correlations, and the cross-valence vs same-valence cuts.

    RNG order per layer: the four bins (CI skipped when n <= 5), then the
    cross-valence cut, then the same-valence cut.
    """
    leads, valence_gap, arousal_gap, crosses_valence, triple_ids = _s3_transition_geometry(
        arms, vad
    )
    stats: LayerStats = {}
    for layer_pos, layer in enumerate(LAYERS):
        layer_leads = leads[:, layer_pos]
        # mean lead per |delta valence| bin, cluster CI when the bin is big enough
        binned = []
        for bin_lo, bin_hi in zip(S3_VALENCE_GAP_EDGES[:-1], S3_VALENCE_GAP_EDGES[1:]):
            in_bin = (valence_gap >= bin_lo) & (valence_gap < bin_hi)
            ci = (
                cluster_boot_ci(
                    layer_leads[in_bin],
                    [t for t, member in zip(triple_ids, in_bin) if member],
                    rng,
                )
                if in_bin.sum() > 5
                else (np.nan, np.nan)
            )
            binned.append(
                (
                    float(layer_leads[in_bin].mean()) if in_bin.any() else np.nan,
                    ci,
                    int(in_bin.sum()),
                )
            )
        r_valence = float(np.corrcoef(valence_gap, layer_leads)[0, 1])
        r_arousal = float(np.corrcoef(arousal_gap, layer_leads)[0, 1])
        cuts = {}
        for cut_name, mask in [
            ("cross-valence", crosses_valence),
            ("same-valence", ~crosses_valence),
        ]:
            ci = cluster_boot_ci(
                layer_leads[mask], [t for t, member in zip(triple_ids, mask) if member], rng
            )
            cuts[cut_name] = (float(layer_leads[mask].mean()), ci, int(mask.sum()))
        stats[layer] = {"binned": binned, "r_dval": r_valence, "r_darr": r_arousal, "cuts": cuts}
    return stats


def _s3_add_layer_traces(
    fig: go.Figure, layer_stats: dict[str, object], bin_labels: list[str]
) -> None:
    """Add one layer's binned-lead bars (left) and valence-cut bars (right)."""
    binned = layer_stats["binned"]
    cuts = layer_stats["cuts"]
    fig.add_trace(
        go.Bar(
            x=bin_labels,
            y=[mean for mean, _, _ in binned],
            marker_color="#4c78a8",
            error_y=dict(
                array=[ci[1] - mean for mean, ci, _ in binned],
                arrayminus=[mean - ci[0] for mean, ci, _ in binned],
            ),
            text=[f"n={n}" for _, _, n in binned],
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=list(cuts),
            y=[cuts[name][0] for name in cuts],
            marker_color=["#54a24b", "#b79a20"],
            error_y=dict(
                array=[cuts[name][1][1] - cuts[name][0] for name in cuts],
                arrayminus=[cuts[name][0] - cuts[name][1][0] for name in cuts],
            ),
            text=[f"n={cuts[name][2]}" for name in cuts],
            showlegend=False,
        ),
        row=1,
        col=2,
    )


def _s3_title(stats: LayerStats, layer: int) -> str:
    """S3's title for ONE layer: the plain question, then that layer's answer.

    Reads that layer's own Pearson correlation of lead with the valence gap
    and its two valence-cut means, so the headline moves with the slider.
    Hand-wrapped to the same five-line shape at every layer.
    """
    layer_stats = stats[layer]
    cross_mean = layer_stats["cuts"]["cross-valence"][0]
    same_mean = layer_stats["cuts"]["same-valence"][0]
    return (
        "Does the tracker see a bigger emotional jump coming? At layer"
        f" {layer} the rise tracks the valence gap at r = {layer_stats['r_dval']:+.2f},"
        f"<br>and cross-valence jumps rise {cross_mean:+.3f} against {same_mean:+.3f}"
        " for same-valence ones (registry code S3)"
        "<br><sup>one left bar = every transition whose two emotions differ by that much"
        " NRC valence; one right bar = every transition of that type;</sup>"
        "<br><sup>height = mean rise of the incoming emotion before the boundary, in"
        " centered-cosine units; failure anchor: the marked line at 0 = no"
        " anticipation;</sup>"
        "<br><sup>a real distance effect would climb left to right. Error bars = 95%"
        " cluster-bootstrap CI over triples; Gemma-written stories, self-generated"
        " probes</sup>"
    )


def s3_affective_distance_figure(
    arms: Arms, vad: Vad, rng: np.random.Generator
) -> tuple[go.Figure, LayerStats]:
    """S3: does anticipation follow affective distance? (selfgen bank, it_v2)

    Left panel: mean R1 lead binned by |NRC valence gap|; right panel:
    cross-valence vs same-valence mean lead. Returns the figure (layer
    slider) and the ``_s3_stats`` dict (which includes the per-layer Pearson
    correlations the notebook prints).
    """
    stats = _s3_stats(arms, vad, rng)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["lead vs |delta valence| (binned)", "cross-valence vs same-valence lead"],
    )
    bin_labels = [
        f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(S3_VALENCE_GAP_EDGES[:-1], S3_VALENCE_GAP_EDGES[1:])
    ]
    for layer in LAYERS:
        _s3_add_layer_traces(fig, stats[layer], bin_labels)
    fig.update_yaxes(title="mean R1 lead (centered-cosine units)", row=1, col=1)
    fig.update_yaxes(title="mean R1 lead (centered-cosine units)", row=1, col=2)
    fig.update_xaxes(title="|NRC valence difference| of the from-to pair", row=1, col=1)
    fig.update_xaxes(title="transition type (bar label = n transitions)", row=1, col=2)
    # the failure anchor, named in the plot: zero rise = no anticipation at all
    for panel in (1, 2):
        fig.add_hline(
            y=0,
            line_color="#888",
            line_width=1,
            row=1,
            col=panel,
            annotation_text="0 = no anticipation",
            annotation_position="top left",
            annotation_font_size=11,
            # the bars cross the line at some layers, so the label carries its
            # own background and stays readable at every slider step
            annotation_bgcolor="rgba(255,255,255,0.8)",
        )
    fig.update_layout(
        width=1150,
        height=560,
        # top margin clears the five wrapped title lines at every slider step
        margin=dict(t=175, b=95),
        title=dict(y=0.98, yanchor="top", font_size=15),
    )
    layer_slider(fig, traces_per_layer=2, title_for_layer=partial(_s3_title, stats))
    return fig, stats


def _s4_stats(
    arms: Arms, vad: Vad, rng: np.random.Generator
) -> tuple[list[str], dict[int, Float[np.ndarray, "expected reported"]], LayerStats]:
    """The tagged-by-winner confusion structure per layer, selfgen bank, it_v2.

    Per layer: the row-normalized confusion matrix, plus the mean VAD
    distance from wrong winners to their target vs the same mean under a
    uniformly random wrong probe (one rng draw per lexicon-covered wrong
    phase, in phase order). Returns (bank emotion names, {layer: matrix
    [expected, reported]}, ``stats[layer] = {"top1_rate", "wrong_mean",
    "shuffle_mean", "n_wrong"}``).
    """
    _, winner, phase_rows = gate_ranks(arms, "it_v2", "selfgen")
    _, names = bank_columns(arms, "it_v2", "selfgen")
    heatmaps: dict[int, Float[np.ndarray, "expected reported"]] = {}
    stats: LayerStats = {}
    for layer_pos, layer in enumerate(LAYERS):
        confusion_counts = np.zeros((len(names), len(names)))
        wrong_dist, shuffle_dist = [], []
        for phase_pos, row in enumerate(phase_rows):
            target_pos = names.index(row["emotion"])
            winner_pos = int(winner[phase_pos, layer_pos])
            confusion_counts[target_pos, winner_pos] += 1
            # for wrong winners covered by the lexicon: distance to target,
            # plus the same distance for a uniformly random wrong probe
            if winner_pos != target_pos and names[winner_pos] in vad and names[target_pos] in vad:
                wrong_dist.append(
                    float(
                        np.hypot(
                            vad[names[winner_pos]][0] - vad[names[target_pos]][0],
                            vad[names[winner_pos]][1] - vad[names[target_pos]][1],
                        )
                    )
                )
                other_names = [name for name in names if name != row["emotion"] and name in vad]
                random_wrong = other_names[rng.integers(0, len(other_names))]
                shuffle_dist.append(
                    float(
                        np.hypot(
                            vad[random_wrong][0] - vad[names[target_pos]][0],
                            vad[random_wrong][1] - vad[names[target_pos]][1],
                        )
                    )
                )
        heatmaps[layer] = confusion_counts / np.clip(
            confusion_counts.sum(axis=1, keepdims=True), 1, None
        )
        stats[layer] = {
            "top1_rate": float(np.trace(confusion_counts) / confusion_counts.sum()),
            "wrong_mean": float(np.mean(wrong_dist)),
            "shuffle_mean": float(np.mean(shuffle_dist)),
            "n_wrong": len(wrong_dist),
        }
    return names, heatmaps, stats


def _s4_title(stats: LayerStats, layer: int) -> str:
    """S4's title for ONE layer: the plain question, then that layer's answer.

    Reads that layer's own top-1 rate and the two mean valence-arousal
    distances (actual wrong winners vs the matched random draw), and names
    which way the comparison went. Same seven-line shape at every layer.
    """
    layer_stats = stats[layer]
    wrong_mean, shuffle_mean = layer_stats["wrong_mean"], layer_stats["shuffle_mean"]
    verdict = (
        "yes, wrong answers are affective neighbours"
        if wrong_mean < shuffle_mean
        else "no, wrong answers are no closer than chance"
    )
    return (
        f"When the tracker names the wrong emotion, is it a near miss? At layer {layer}"
        f" the tagged probe wins {layer_stats['top1_rate']:.0%} of phases,"
        f"<br>and its wrong winners sit {wrong_mean:.2f} from the target in"
        f" valence-arousal space against {shuffle_mean:.2f} for a random wrong probe:"
        f"<br>{verdict} (registry code S4)"
        "<br><sup>left: one cell = the fraction of that row emotion's phases where the"
        " column emotion's probe won; rows sum to 1; the diagonal is the top-1 rate;</sup>"
        "<br><sup>right: one bar = mean valence-arousal distance from a wrong winner to"
        f" the tagged emotion, over that layer's {layer_stats['n_wrong']} wrong phases;</sup>"
        "<br><sup>the grey bar is the failure anchor: the same distance when each wrong"
        " winner is replaced by a uniformly random wrong probe, so equal bars would"
        " mean unstructured confusions;</sup>"
        "<br><sup>Gemma-written stories, self-generated probes</sup>"
    )


def s4_confusion_figure(
    arms: Arms, vad: Vad, rng: np.random.Generator
) -> tuple[go.Figure, LayerStats]:
    """S4: the confusion heatmap and the affective-neighbor test.

    Left panel: row-normalized 12x12 tagged-by-winner heatmap. Right panel:
    mean VAD distance of wrong winners to the target vs the random-wrong
    reference. Returns the figure (layer slider) and the ``_s4_stats`` per
    layer dict (the lines the notebook prints).
    """
    names, heatmaps, stats = _s4_stats(arms, vad, rng)
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        # wide gutter: the heatmap's colorbar and the right panel's axis title
        # both live between the two panels
        horizontal_spacing=0.19,
        subplot_titles=["confusion (row-normalized)", "winner->target VAD distance vs shuffle"],
    )
    for layer in LAYERS:
        fig.add_trace(
            go.Heatmap(
                z=heatmaps[layer],
                x=names,
                y=names,
                colorscale="Blues",
                zmin=0,
                zmax=1,
                # every layer carries its own colorbar: only one layer's traces
                # are visible at a time, so putting it on layer 6 alone would
                # leave the default view (layer 33) with no scale at all
                colorbar=dict(
                    title="fraction of<br>the row's<br>phases",
                    x=0.455,
                    len=0.8,
                    thickness=14,
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=["actual winners", "random wrong probe"],
                y=[stats[layer]["wrong_mean"], stats[layer]["shuffle_mean"]],
                marker_color=["#4c78a8", "#9d9d9d"],
                text=[f"n={stats[layer]['n_wrong']}"] * 2,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(autorange="reversed", title="expected (tagged) emotion", row=1, col=1)
    fig.update_xaxes(title="emotion the model reports (winning probe)", row=1, col=1)
    fig.update_yaxes(title="mean valence-arousal distance to target", row=1, col=2)
    fig.update_xaxes(title="wrong winners vs matched random draw", row=1, col=2)
    fig.update_layout(
        width=1150,
        height=760,
        # top margin clears the seven wrapped title lines at every slider step
        margin=dict(t=225, b=130),
        title=dict(y=0.98, yanchor="top", font_size=15),
    )
    layer_slider(fig, traces_per_layer=2, title_for_layer=partial(_s4_title, stats))
    return fig, stats
