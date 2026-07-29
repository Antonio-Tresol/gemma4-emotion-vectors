"""Act IV and Act V exhibits: the behavior bars and the story-source panels."""

from __future__ import annotations

import math
from typing import Any, Mapping

import plotly.graph_objects as go
from plotly.subplots import make_subplots

Stats = dict[str, Any]

# Act V ladder: display label -> arm key inside e11_lineage.json (a committed
# evidence filename, kept as-is). The roles matter: "external" names who wrote
# the probe stories, never the probed model.
_ACT5_LADDER = [
    ("self-generated (probed model's own stories, n=256)", "selfgen"),
    ("weak external (gemma-4-4B corpus)", "weak_external"),
    ("strong external (DeepSeek-v4-pro, n=256)", "strong_external"),
]
_ACT5_LADDER_BAR_NAMES = [
    "self-generated<br>(model's own, n=256)",
    "weak external<br>(gemma-4-4B)",
    "strong external<br>(DeepSeek-v4-pro, n=256)",
]


# --------------------------------------------------------------------------- Act IV


def _act4_lines(
    preferences: Mapping, preferences_prefix: Mapping, steering_runs: Mapping[str, Mapping]
) -> list[str]:
    """The Act IV printout: preference reads (post- and pre-fix) and sign tests."""
    lines = [
        f"preference probe-Elo max |r|: post-fix probes {preferences['p2_max_abs_r']:.4f} "
        f"at layer {preferences['p2_best_layer']} "
        f"(registered bar {preferences['p2_registered_bar']}); "
        f"pre-fix probes {preferences_prefix['p2_max_abs_r']:.4f} (superseded, Act III)",
        f"preference Elo split: positive-category mean {preferences['p1_positive_mean']:.0f} "
        f"vs negative {preferences['p1_negative_mean']:.0f} (P1 pass={preferences['p1_pass']})",
    ]
    for label, scores in steering_runs.items():
        lines.append(
            f"steering {label}: valence sign test {scores['p3_sign_agreements']}/12 "
            f"(bar {scores['p3_registered_bar']}), pass={scores['p3_pass']}"
        )
    return lines


def _act4_sign_panel(fig: go.Figure, steering_runs: Mapping[str, Mapping], sign_bar: int) -> None:
    """Left panel: the sign-test bars with the registered bar and chance anchors."""
    sign_counts = [scores["p3_sign_agreements"] for scores in steering_runs.values()]
    fig.add_bar(
        x=["alpha=2 (gentler push)", "alpha=8 (stronger push)"],
        y=sign_counts,
        text=sign_counts,
        textposition="outside",
        showlegend=False,
        row=1,
        col=1,
    )
    fig.add_hline(
        y=sign_bar,
        line_dash="dash",
        row=1,
        col=1,
        annotation_text=f"registered bar = {sign_bar} of 12 (pass at or above)",
        annotation_position="bottom left",
    )
    fig.add_hline(
        y=6,
        line_dash="dot",
        row=1,
        col=1,
        annotation_text="chance = 6 of 12 (coin-flip signs)",
        annotation_position="bottom right",
    )
    fig.update_yaxes(
        range=[0, 13.4], title="emotions shifting the steered way (of 12)", row=1, col=1
    )


def _act4_preference_panel(
    fig: go.Figure, preferences: Mapping, preferences_prefix: Mapping
) -> None:
    """Right panel: probe-Elo |r| bars with the registered bar, zero, and paper band."""
    fig.add_bar(
        x=["post-fix probes", "pre-fix probes<br>(superseded, Act III)"],
        y=[preferences["p2_max_abs_r"], preferences_prefix["p2_max_abs_r"]],
        text=[
            f"{preferences['p2_max_abs_r']:.3f}",
            f"{preferences_prefix['p2_max_abs_r']:.3f}",
        ],
        textposition="outside",
        marker_color=["#d62728", "#9a9a9a"],
        showlegend=False,
        row=1,
        col=2,
    )
    fig.add_hline(
        y=preferences["p2_registered_bar"],
        line_dash="dash",
        row=1,
        col=2,
        annotation_text=f"registered bar = {preferences['p2_registered_bar']} (pass at or above)",
        annotation_position="bottom left",
    )
    fig.add_hrect(y0=0.71, y1=0.74, fillcolor="#888888", opacity=0.18, line_width=0, row=1, col=2)
    # the band label sits centered between the two bars, where nothing collides
    fig.add_annotation(
        xref="x2 domain",
        x=0.5,
        yref="y2",
        y=0.725,
        showarrow=False,
        text="paper's band 0.71-0.74",
        font=dict(size=11),
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        row=1,
        col=2,
        annotation_text="0 = no relationship",
        annotation_position="bottom right",
    )
    fig.update_yaxes(
        range=[-0.07, 0.88], title="max |Pearson r|, probe cosine vs Elo", row=1, col=2
    )


def act4_behavior_figure(
    preferences: Mapping,
    preferences_prefix: Mapping,
    steering_runs: Mapping[str, Mapping],
    model_it: str,
) -> tuple[go.Figure, Stats]:
    """Act IV: the steering sign test and the preference correlation, with bars.

    ``preferences`` / ``preferences_prefix`` are the post-fix and (superseded)
    pre-fix probe reads of ``preferences_it_chat_fixed``; ``steering_runs``
    maps a display label (``"alpha=2"``) to that run's ``scores.json``.
    Returns the two-panel figure (registered bars, chance and zero anchors,
    the paper's band) and stats with the printed ``lines``.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "steering: does the writing shift the steered way?",
            "preferences: probe reading vs revealed choice",
        ),
    )
    sign_bar = next(iter(steering_runs.values()))["p3_registered_bar"]
    _act4_sign_panel(fig, steering_runs, sign_bar)
    _act4_preference_panel(fig, preferences, preferences_prefix)
    for annotation in fig.layout.annotations[:2]:  # the two subplot titles
        annotation.font = dict(size=13)
    sign_counts = [scores["p3_sign_agreements"] for scores in steering_runs.values()]
    fig.update_layout(
        title=(
            "Do the directions cause and predict behavior? Yes: both registered bars clear<br>"
            f"<sup>steering shifts {sign_counts[0]} and {sign_counts[1]} of 12 emotions the "
            f"right way (bar {sign_bar}, chance 6); probe-Elo |r| = "
            f"{preferences['p2_max_abs_r']:.2f} clears the "
            f"{preferences['p2_registered_bar']} bar, below the paper's 0.71-0.74 band</sup><br>"
            "<sup>left: one bar = one steering strength (alpha = vector scale added at layer 33, "
            "the only steered layer, while the model writes about its day)</sup><br>"
            "<sup>right: one bar = one probe vintage, same 64-activity Elo tournament | "
            f"model: {model_it} | evidence: steering_it(_a8)_fixed, "
            "preferences_it_chat_fixed</sup>"
        ),
        title_font_size=15,
        width=1050,
        height=510,
        margin=dict(t=150),
    )
    stats = {
        "lines": _act4_lines(preferences, preferences_prefix, steering_runs),
        "sign_counts": sign_counts,
        "pref_post_fix": preferences["p2_max_abs_r"],
        "pref_pre_fix": preferences_prefix["p2_max_abs_r"],
    }
    return fig, stats


# --------------------------------------------------------------------------- Act V


def _seed_means(points: list[Mapping]) -> dict[int, float]:
    """Corpus size -> mean passing-layer count over the subsampling seeds."""
    sizes = sorted({point["n"] for point in points})
    return {
        n: sum(pt["n_passing_layers"] for pt in points if pt["n"] == n)
        / len([pt for pt in points if pt["n"] == n])
        for n in sizes
    }


def _act5_lines(
    e11: Mapping,
    e12_pref: Mapping,
    fixed_means: dict[int, float],
    diverse_means: dict[int, float],
    fixed_full: int,
    diverse_full: int,
) -> list[str]:
    """The Act V printout: ladder, seed means, corpus sizes, matched-n reads."""
    lines = []
    for label, arm in _ACT5_LADDER:
        # "r1_dual_battery" is this block's key in the evidence file, kept as-is
        dual = e11["r1_dual_battery"][arm]
        lines.append(
            f"{label:52s} passing layers: {len(dual['passing_layers']):2d}  "
            f"{dual['passing_layers']}"
        )
    for arm_label, means in (("fixed prompt", fixed_means), ("diversified prompts", diverse_means)):
        lines.append(
            f"E12 {arm_label:20s} seed-mean passing layers by corpus size: "
            + ", ".join(f"{n}: {mean:.1f}" for n, mean in means.items())
        )
    lines += [
        f"corpus sizes: fixed arm caps at {fixed_full} stories/emotion, diverse arm at "
        f"{diverse_full} ({diverse_full / fixed_full:.1f}x larger)",
        f"E12 at matched n=256 (fixed arm caps at {fixed_full} kept stories): "
        f"fixed {fixed_means[fixed_full]:.1f} vs diverse {diverse_means[256]:.1f} passing layers",
        f"E12 preference at matched n=256: diverse {e12_pref['diverse_n256_mean']:.3f} "
        f"vs fixed {e12_pref['references']['fixed_n256']:.3f} max |r| "
        "(the dissociation, exploratory)",
    ]
    return lines


def _act5_anchors(fig: go.Figure) -> None:
    """Grading anchors: the 20-layer ceiling and the random-probe floor on both
    panels; the near-saturation size on the dose-response panel."""
    for col in (1, 2):
        fig.add_hline(
            y=20,
            line_dash="dash",
            row=1,
            col=col,
            annotation_text="ceiling: all 20 measured layers",
            annotation_position="top left",
        )
        fig.add_hline(
            y=0,
            line_dash="dot",
            row=1,
            col=col,
            annotation_text="random probe sets pass 0 layers (Act II)",
            annotation_position="bottom right",
        )
    # plotly quirk on log axes: the vline SHAPE takes raw data coordinates but
    # its built-in annotation would land at 10^x, so the label is added by hand
    # (manual annotations on log axes take log10 coordinates)
    fig.add_vline(x=64, line_dash="dot", row=1, col=2)
    fig.add_annotation(
        xref="x2",
        x=math.log10(64),
        yref="y2",
        y=12,
        xanchor="left",
        showarrow=False,
        text="near saturation (about 64)",
        font=dict(size=11),
    )


def act5_story_source_figure(
    e11: Mapping,
    e12_curve: Mapping,
    e12_pref: Mapping,
    model_it: str,
) -> tuple[go.Figure, Stats]:
    """Act V: the passing-layer ladder (E10/E11) and the dose-response pair (E12).

    ``e11`` is ``e11_lineage.json`` (per-corpus passing layers, scored on both
    sets of twelve scenarios),
    ``e12_curve`` is ``e12_scale_curve.json`` (fixed vs diverse subsampling
    points), ``e12_pref`` is ``e12_pref_matched_n_exploratory.json``. Returns
    the two-panel figure with the ceiling/floor/saturation anchors and stats
    with the printed ``lines`` plus the per-arm seed means.
    """
    fixed_means = _seed_means(e12_curve["arms"]["fixed"]["points"])
    diverse_means = _seed_means(e12_curve["arms"]["diverse"]["points"])
    fixed_full = e12_curve["arms"]["fixed"]["n_available_min"]
    diverse_full = e12_curve["arms"]["diverse"]["n_available_min"]
    # "r1_dual_battery" is this block's key in the evidence file, kept as-is
    ladder_counts = [len(e11["r1_dual_battery"][arm]["passing_layers"]) for _, arm in _ACT5_LADDER]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.09,
        subplot_titles=(
            "who wrote the probe stories (E10/E11)",
            "how many stories, fixed vs diversified prompts (E12)",
        ),
    )
    fig.add_bar(
        x=_ACT5_LADDER_BAR_NAMES,
        y=ladder_counts,
        text=ladder_counts,
        textposition="outside",
        showlegend=False,
        row=1,
        col=1,
    )
    for means, legend, color in (
        (fixed_means, "fixed prompt (DeepSeek-v4-pro author)", "#1f77b4"),
        (diverse_means, "diversified prompts (same author)", "#d62728"),
    ):
        fig.add_scatter(
            x=list(means),
            y=list(means.values()),
            mode="lines+markers",
            name=legend,
            line=dict(color=color),
            row=1,
            col=2,
        )
    _act5_anchors(fig)
    fig.update_xaxes(type="log", title="stories per emotion (log scale)", row=1, col=2)
    fig.update_yaxes(title="passing layers (of 20 measured)", range=[-1.9, 21.9])
    for annotation in fig.layout.annotations[:2]:  # the two subplot titles
        annotation.font = dict(size=13)
    fig.update_layout(
        title=(
            "Whose stories make the best probes? The strongest generator's: "
            f"{ladder_counts[2]} passing layers vs {ladder_counts[0]} (self-generated) "
            f"and {ladder_counts[1]} (weak external)<br>"
            f"<sup>dose-response: the fixed-prompt corpus is near saturation by 64 stories per "
            f"emotion ({fixed_means[64]:.1f} of its full-size {fixed_means[fixed_full]:.1f} "
            "passing layers) and beats the diversified corpus at every matched size</sup><br>"
            "<sup>one bar / dot = passing layers of the 20 measured; a layer passes when the "
            "target emotion ranks top-3 in at least 8 of the 12 scenarios in each of the "
            "two scenario sets</sup><br>"
            f"<sup>probes read from {model_it} | evidence: e11_lineage.json, "
            "e12_scale_curve.json</sup>"
        ),
        title_font_size=15,
        width=1100,
        height=520,
        margin=dict(t=150),
    )
    stats = {
        "lines": _act5_lines(e11, e12_pref, fixed_means, diverse_means, fixed_full, diverse_full),
        "ladder_counts": dict(zip([arm for _, arm in _ACT5_LADDER], ladder_counts)),
        "fixed_means": fixed_means,
        "diverse_means": diverse_means,
    }
    return fig, stats
