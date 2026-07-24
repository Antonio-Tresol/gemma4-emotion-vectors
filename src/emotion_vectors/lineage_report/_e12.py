"""E12 exhibits: the dose-response curves and the preference read.

Each builder returns ``(figure, stats)`` with ``stats["lines"]`` holding the
printed record, sliced from the frozen evidence files (nothing re-scores).
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._data import (
    GEOMETRY_LAYER,
    LINEAGE_COLORS,
    LINEAGE_LABELS,
    LINEAGE_ORDER,
    LINEAGE_TICK_LABELS,
    PROBED_MODEL,
    LineageEvidence,
    figure_title,
    seed_mean_by_n,
)

# dose-response arm -> (canonical lineage key, role label for the legend)
ARM_ROLES = {
    "fixed": ("fixed_deepseek", "fixed-prompt DeepSeek stories"),
    "diverse": ("diverse_deepseek", "diverse-prompt DeepSeek stories"),
}
N_TOTAL_LAYERS = 20  # the R1 grid scores 20 layers (0..57, every third)


def _matched_n_detection(evidence: LineageEvidence) -> dict[str, float]:
    """The matched-n detection comparison quoted in the verdict: diverse arm
    at n=256 vs the fixed arm at its full corpus (n=255, the closest size the
    fixed corpus reaches)."""
    arms = evidence.scale_curves["arms"]
    fixed_means = seed_mean_by_n(arms["fixed"]["points"], "n_passing_layers")
    diverse_means = seed_mean_by_n(arms["diverse"]["points"], "n_passing_layers")
    if 256 not in diverse_means:
        raise ValueError("diverse arm has no n=256 point")
    fixed_full_n = max(fixed_means)
    return {
        "diverse_n256": diverse_means[256],
        "fixed_full": fixed_means[fixed_full_n],
        "fixed_full_n": fixed_full_n,
        "diverse_full": diverse_means[max(diverse_means)],
        "diverse_full_n": max(diverse_means),
    }


def _add_dose_response_traces(fig: go.Figure, arms: dict[str, Any]) -> None:
    """Draw both arms into the two panels: faint per-seed markers and the
    seed-mean line, detection on the left, geometry on the right."""
    for arm, (lineage_key, role) in ARM_ROLES.items():
        points = arms[arm]["points"]
        color = LINEAGE_COLORS[lineage_key]
        for col, metric in ((1, "n_passing_layers"), (2, "mean_contrast_cos_to_selfgen_L33")):
            seed_means = seed_mean_by_n(points, metric)
            # faint markers: individual seeded subsamples; solid line: seed mean
            fig.add_scatter(
                x=[point["n"] for point in points],
                y=[point[metric] for point in points],
                mode="markers",
                marker=dict(color=color, opacity=0.35),
                name=f"{role}, one seeded subsample",
                legendgroup=arm,
                showlegend=(col == 1),
                row=1,
                col=col,
            )
            fig.add_scatter(
                x=list(seed_means),
                y=list(seed_means.values()),
                mode="lines+markers",
                line=dict(color=color),
                name=f"{role}, seed mean",
                legendgroup=arm,
                showlegend=(col == 1),
                row=1,
                col=col,
            )


def _add_dose_response_anchors(
    fig: go.Figure, fixed_ceiling: int, selfgen_n256: int, fixed_to_selfgen_cos: float
) -> None:
    """Add the grading scale to both panels: strength comparators above and
    the failure floor (nothing passes / unrelated directions) at zero."""
    # grading anchors, left panel: ceiling and comparator above, failure floor at 0
    fig.add_hline(
        y=fixed_ceiling,
        line_dash="dot",
        annotation_text=f"fixed full-corpus ceiling ({fixed_ceiling} layers)",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=selfgen_n256,
        line_dash="dot",
        annotation_text=f"self-generated n=256 ({selfgen_n256} layers)",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="failure: no layer passes",
        annotation_position="bottom right",
        row=1,
        col=1,
    )
    # grading anchors, right panel: full-fixed-corpus cosine and the unrelated floor
    fig.add_hline(
        y=fixed_to_selfgen_cos,
        line_dash="dot",
        annotation_text=f"full fixed corpus ({fixed_to_selfgen_cos:.2f})",
        row=1,
        col=2,
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="failure: unrelated directions",
        annotation_position="bottom right",
        row=1,
        col=2,
    )


def _dose_response_lines(
    arms: dict[str, Any], matched: dict[str, float], fixed_ceiling: int
) -> list[str]:
    """The printed record: seed-mean passing layers per arm, then the
    matched-n detection comparison the verdict asserts on."""
    lines = []
    for arm, (_lineage_key, role) in ARM_ROLES.items():
        seed_means = seed_mean_by_n(arms[arm]["points"], "n_passing_layers")
        lines.append(
            f"{role}, seed-mean passing layers by n: "
            + ", ".join(f"n={size}: {mean:.1f}" for size, mean in seed_means.items())
        )
    lines.append(
        f"matched-n detection: diverse at n=256 {matched['diverse_n256']:.1f} layers vs fixed at"
        f" its full corpus n={matched['fixed_full_n']:.0f} {matched['fixed_full']:.1f} layers;"
        f" diverse at full n={matched['diverse_full_n']:.0f} reaches"
        f" {matched['diverse_full']:.1f}, ceiling {fixed_ceiling}"
    )
    return lines


def dose_response_figure(evidence: LineageEvidence) -> tuple[go.Figure, dict[str, Any]]:
    """Section 4 exhibit: how many stories per emotion do probes need?

    Left panel: layers passing the dual-battery bar vs stories per emotion
    (log x), one faint marker per seeded subsample, line = seed mean, with
    reference lines for the fixed-corpus ceiling, the self-generated n=256
    result, and the zero-layers failure floor. Right panel: mean contrast
    cosine of each subsample's probes to the self-generated probes at layer
    33 (the registered geometry read exists only there), anchored by the
    full-fixed-corpus cosine and the zero (unrelated) floor.

    Returns ``(figure, stats)``: ``stats["lines"]`` prints the seed-mean
    tables and the matched-n comparison; ``stats["matched"]`` holds the
    matched-n numbers the verdict asserts on.
    """
    arms = evidence.scale_curves["arms"]
    fixed_ceiling = len(evidence.e11["r1_dual_battery"]["strong_external"]["passing_layers"])
    selfgen_n256 = len(evidence.e11["r1_dual_battery"]["selfgen"]["passing_layers"])
    fixed_to_selfgen_cos = evidence.e11["r2_cross_lineage_layer33"]["selfgen_vs_strong_external"][
        "mean_contrast_cos"
    ]
    matched = _matched_n_detection(evidence)
    fixed_at_64 = seed_mean_by_n(arms["fixed"]["points"], "n_passing_layers")[64]
    width_px = 1150

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "detection: layers passing the dual-battery bar",
            f"geometry: cosine to self-generated probes (layer {GEOMETRY_LAYER})",
        ),
    )
    _add_dose_response_traces(fig, arms)
    _add_dose_response_anchors(fig, fixed_ceiling, selfgen_n256, fixed_to_selfgen_cos)
    fig.update_xaxes(type="log", title="stories per emotion (log scale)")
    fig.update_yaxes(
        title=f"layers passing the dual-battery bar (of {N_TOTAL_LAYERS})", row=1, col=1
    )
    fig.update_yaxes(title="mean contrast cosine to self-generated probes", row=1, col=2)
    fig.update_layout(
        title=figure_title(
            "How many stories per emotion do detection probes need? About 64: the fixed-prompt"
            f" arm reaches {fixed_at_64:.1f} of its {fixed_ceiling}-layer ceiling by n=64, and"
            " prompt diversity is a detection tax at matched size"
            f" ({matched['diverse_n256']:.1f} layers for diverse at n=256 against"
            f" {matched['fixed_full']:.1f} for fixed at n={matched['fixed_full_n']:.0f})",
            [
                "one faint marker = one seeded subsample (5 seeds per corpus size); the solid"
                " line joins the seed means; the largest n per arm is that arm's full corpus"
                f" ({arms['fixed']['n_available_min']} fixed,"
                f" {arms['diverse']['n_available_min']} diverse, each the smallest per-emotion"
                " count), where subsampling is a no-op",
                "left panel anchors: the fixed full-corpus ceiling and the self-generated"
                " n=256 result above, no layer passing at 0 below; right panel anchors: the"
                " full fixed corpus above, unrelated directions at 0 below; probes read from"
                f" {PROBED_MODEL}; evidence: e12_scale_curve.json, reference lines from"
                " e11_lineage.json",
            ],
            width_px,
        ),
        title_font_size=13,
        width=width_px,
        height=660,
        # the wrapped title runs to five lines above two subplot titles
        margin=dict(t=250, b=120),
        legend=dict(orientation="h", yanchor="top", y=-0.20),
    )
    lines = _dose_response_lines(arms, matched, fixed_ceiling)
    return fig, {"lines": lines, "matched": matched, "fixed_ceiling": fixed_ceiling}


def _add_preference_anchors(fig: go.Figure, abs_r: dict[str, float]) -> None:
    """Add the grading scale: the weakest generator's level as the comparator a
    new lineage has to beat, and |r| = 0 as the failure floor.

    Both labels live in the right margin because every bar reaches past the
    comparator line, so an in-plot label would sit on top of a bar.
    """
    fig.add_hline(y=abs_r["weak_external"], line_dash="dot")
    for anchor_y, anchor_text in (
        (
            abs_r["weak_external"],
            f"comparator: the weakest<br>generator's level ({abs_r['weak_external']:.3f})",
        ),
        (0.0, "failure anchor: |r| = 0,<br>the probe says nothing<br>about preference"),
    ):
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=1.01,
            y=anchor_y,
            text=anchor_text,
            showarrow=False,
            align="left",
            xanchor="left",
            font=dict(size=11, color="#444444"),
        )


def _preference_bars(
    abs_r: dict[str, float], r3_sources: dict[str, Any], bar_labels: list[str]
) -> go.Bar:
    """One bar per lineage, colored by the shared lineage palette and labeled
    with the layer and emotion that won that maximum."""
    return go.Bar(
        x=bar_labels,
        y=[abs_r[key] for key in LINEAGE_ORDER],
        marker_color=[LINEAGE_COLORS[key] for key in LINEAGE_ORDER],
        text=[
            f"{abs_r[key]:.3f}<br>layer {r3_sources[key]['layer']}, {r3_sources[key]['emotion']}"
            for key in LINEAGE_ORDER
        ],
        textposition="outside",
        # no legend entry: each bar is named by its own x tick, and one
        # swatch cannot stand for four different lineage colors. The
        # subtitle names the color encoding instead.
        showlegend=False,
    )


def preference_figure(evidence: LineageEvidence) -> tuple[go.Figure, dict[str, Any]]:
    """Section 6 exhibit: the R3 preference read (probe cosine vs Elo).

    One bar per lineage: the best absolute Pearson correlation between any
    probe's cosine and the preference Elo over 64 activities (a max over 12
    emotions x 4 layers, stated on the figure). A diamond with a seed range
    overlays the diverse bar: the exploratory matched-n check at n=256.

    Returns ``(figure, stats)``; ``stats["lines"]`` prints every bar and the
    matched-n check, ``stats["abs_r"]`` maps lineage -> max |r| and
    ``stats["matched_mean"]`` is the matched-n seed mean.
    """
    r3_sources = {
        "selfgen": evidence.e11["r3_preference_probe_elo"]["selfgen"],
        "weak_external": evidence.e11["r3_preference_probe_elo"]["weak_external"],
        "fixed_deepseek": evidence.full_grid["r3_preference_probe_elo"]["fixed_deepseek_n256"],
        "diverse_deepseek": evidence.full_grid["r3_preference_probe_elo"]["diverse_deepseek_n1024"],
    }
    abs_r = {key: source["abs_r"] for key, source in r3_sources.items()}
    matched = evidence.matched_n
    matched_seeds = matched["diverse_n256_seeds"]
    matched_mean = matched["diverse_n256_mean"]

    bar_labels = [LINEAGE_TICK_LABELS[key] for key in LINEAGE_ORDER]
    width_px = 1000
    fig = go.Figure(_preference_bars(abs_r, r3_sources, bar_labels))
    # the exploratory matched-n check: diverse subsampled to n=256, 5 seeds.
    # The seed range is narrow, so the error bar is a thin whisker, not a gap.
    diverse_label = bar_labels[LINEAGE_ORDER.index("diverse_deepseek")]
    fig.add_scatter(
        x=[diverse_label],
        y=[matched_mean],
        mode="markers",
        marker=dict(symbol="diamond", size=11, color="black"),
        error_y=dict(
            type="data",
            symmetric=False,
            array=[max(matched_seeds) - matched_mean],
            arrayminus=[matched_mean - min(matched_seeds)],
        ),
        name=f"diverse subsampled to n=256, {len(matched_seeds)}-seed mean and range (exploratory)",
    )
    _add_preference_anchors(fig, abs_r)
    fig.update_layout(
        title=figure_title(
            "Does corpus quality and diversity pay on the fine-grained preference read? Yes: the"
            f" best |r| climbs {abs_r['selfgen']:.2f} (self-generated) to"
            f" {abs_r['fixed_deepseek']:.2f} (fixed DeepSeek) to"
            f" {abs_r['diverse_deepseek']:.2f} (diverse DeepSeek), and diverse at matched n=256"
            f" holds {matched_mean:.2f}",
            [
                "one bar = the best absolute Pearson correlation between any probe's centered"
                " cosine and the preference Elo rating over 64 activities; it is a maximum over"
                " 12 emotions x 4 layers, so compare levels, not small gaps",
                "the label on each bar names the layer and emotion that won that maximum;"
                " bar color = probe lineage, the same palette as the figures above"
                " (self-generated green, weak external gray, fixed DeepSeek blue,"
                " diverse DeepSeek red), so color repeats the x axis and adds nothing new",
                f"probes read from {PROBED_MODEL}; evidence: e11_lineage.json,"
                " e12_diverse_fullcorpus_grid.json, e12_pref_matched_n_exploratory.json",
            ],
            width_px,
        ),
        title_font_size=13,
        yaxis_title="max |Pearson r| (0 = no relation, 1 = perfect)",
        yaxis_range=[0, 0.95],
        xaxis_title="probe lineage (who wrote the stories)",
        width=width_px,
        # height grows with the top margin so the plot area itself keeps its
        # size and the legend stays clear of the x-axis title
        height=700,
        # top margin clears the five wrapped subtitle lines; the right margin
        # holds the two grading-anchor labels
        margin=dict(t=250, b=150, r=230),
        legend=dict(orientation="h", yanchor="top", y=-0.28),
    )
    lines = [
        f"{LINEAGE_LABELS[key]}: max |r| = {abs_r[key]:.3f} (layer {r3_sources[key]['layer']},"
        f" {r3_sources[key]['emotion']})"
        for key in LINEAGE_ORDER
    ]
    lines.append(
        f"matched-n check, diverse subsampled to n=256: mean |r| {matched_mean:.3f}"
        f" ({len(matched_seeds)} seeds, {min(matched_seeds):.3f} to {max(matched_seeds):.3f})"
    )
    return fig, {"lines": lines, "abs_r": abs_r, "matched_mean": matched_mean}
