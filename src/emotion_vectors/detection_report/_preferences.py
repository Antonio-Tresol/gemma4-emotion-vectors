"""Sections 8 and 9: the behavioural legs of the source paper's Figure 4.

Sections 1 to 7 ask whether the emotion vectors detect an implied emotion in a
scenario. These two sections ask a different question with the same vectors,
the one the paper's Figure 4 asks: do the probes predict, and does adding a
vector cause, what the model says it would rather do?

* Section 8 (Figure 4, left half): the model chooses between activities, a
  Bradley-Terry fit turns the choices into one Elo rating per activity, and
  each emotion probe's activation while the model reads an activity is
  correlated with that rating.
* Section 9 (Figure 4, right half): the same choices are re-collected while an
  emotion vector is added to the residual stream, and the Elo redistribution
  is compared with the correlational profile from section 8.

Both were folded in from the paper-parity notebook
(``notebooks/04_paper_plot_parity.ipynb``, its sections 3 and 4). Every
collection here is the padding-FIXED one (TREE Q1.H3.E4); the compromised
originals stay in the results history. Each builder returns
``(figure, stats)`` with ``stats["lines"]`` holding the printed record, and
the registered verdicts (P1, P2, P3) are read from the scorer's own output
rather than recomputed here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from emotion_vectors.artifacts import fetch  # local results/ first, HF otherwise

from ._data import IT_LABEL, figure_title

PREFERENCE_WIDTH_PX = 1050
STEERING_WIDTH_PX = 1150
# The steering scatter is wide so its point labels do not collide, but a title
# wrapped to the full width runs into the right edge; wrap it narrower.
STEERING_TITLE_WRAP_PX = 1020

# (role label, results/ subtree). The role label names what the arm IS, since
# the two arms differ only in how the choice prompt was formatted.
PREFERENCE_SOURCES = (
    ("plain (paper's exact format)", "preferences_it_fixed"),
    ("chat template", "preferences_it_chat_fixed"),
)
STEERING_SOURCES = (
    ("alpha=2", "steering_it_fixed"),
    ("alpha=8", "steering_it_a8_fixed"),
)
# The chat arm rescored against the post-fix probe set: the instrument-
# sensitivity check for section 8's headline correlation.
POSTFIX_PROBE_SCORES = "scores_postfix_probes.json"

# The paper's own values, quoted as replication targets. They are also
# recorded in each steering scores file's ``paper_reference`` string, which
# :func:`_check_paper_reference` verifies so the two cannot drift apart.
PAPER_PROBE_ELO_RANGE = (0.71, 0.74)  # hostile -0.74 to blissful +0.71
PAPER_COUPLING_R = 0.85  # the paper's prediction-vs-effect scatter slope
PAPER_MEAN_SHIFTS = {"blissful": 212, "hostile": -303}  # mean Elo shift under steering

# One fixed color per activity category, warm for the categories the model
# should like and red for the ones it should not.
CATEGORY_COLORS = {
    "helpful": "#2ca02c",
    "engaging": "#17becf",
    "social": "#1f77b4",
    "self_curiosity": "#9467bd",
    "neutral": "#7f7f7f",
    "aversive": "#8c564b",
    "misaligned": "#ff7f0e",
    "unsafe": "#d62728",
}
ELO_ANCHOR = 1000  # our Bradley-Terry fit is anchored at mean 1000 by construction


@dataclass(frozen=True)
class ScoredArm:
    """One collected arm and the scorer's verdicts for it.

    Fields: ``label`` (the role the arm plays, as every figure names it),
    ``directory`` (its ``results/`` subtree, quoted as provenance), and
    ``scores`` (the parsed ``scores.json``, including the registered P1/P2/P3
    verdicts the scorer wrote before this notebook existed).
    """

    label: str
    directory: str
    scores: dict[str, Any]


def _load_arms(sources: tuple[tuple[str, str], ...], scores_name: str) -> list[ScoredArm]:
    """Load one family of arms, failing loudly on a missing collection.

    Inputs: ``sources`` ((role label, results subtree) pairs) and the score
    file to read inside each subtree. Returns one :class:`ScoredArm` per
    source, in the order given.
    """
    arms = []
    for label, directory in sources:
        scores_file = fetch(directory) / scores_name
        if not scores_file.exists():
            raise FileNotFoundError(
                f"{label}: no {scores_name} under results/{directory};"
                " the arm has not been collected and scored"
            )
        arms.append(
            ScoredArm(label=label, directory=directory, scores=json.loads(scores_file.read_text()))
        )
    return arms


def load_preference_arms() -> list[ScoredArm]:
    """Both preference arms (plain format, then chat template)."""
    return _load_arms(PREFERENCE_SOURCES, "scores.json")


def load_steering_arms() -> list[ScoredArm]:
    """Both steering doses (alpha=2, then alpha=8)."""
    return _load_arms(STEERING_SOURCES, "scores.json")


def _preference_title(arm: ScoredArm, by_layer: dict[str, Any], is_default: bool) -> str:
    """This layer's question-form title, carrying this layer's own verdict.

    Inputs: the arm, one entry of its ``probe_elo_by_layer``, and whether that
    layer is the registered best one. Returns the wrapped title string.
    """
    scores = arm.scores
    bar = scores["p2_registered_bar"]
    best_probe = scores["probe_emotions_matched"][by_layer["argmax_probe_index"]]
    verdict = "Yes" if by_layer["max_abs_r"] >= bar else "No"
    paper_low, paper_high = PAPER_PROBE_ELO_RANGE
    headline = (
        "Do the emotion probes predict which activities the model prefers?"
        f" {verdict} at layer {by_layer['layer']}"
        + (" (the registered best layer)" if is_default else "")
        + f": the strongest probe, '{best_probe}', reaches |r| = {by_layer['max_abs_r']}"
        f" against a registered bar of {bar}"
    )
    return figure_title(
        headline,
        [
            f"left: one bar = one of the {len(scores['probe_emotions_matched'])} emotion probes"
            " matched to the NRC lexicon, its Pearson correlation between that probe's activation"
            " while the model reads an activity and the activity's Elo rating; right: one dot ="
            f" one of the {scores['n_activities']} activities, at the strongest probe",
            f"failure anchor: r = 0, the probe says nothing about preference; registered bar:"
            f" |r| = {bar} (dashed); replication target: the paper's {paper_low} to {paper_high}"
            f" band (shaded). Elo is anchored at mean {ELO_ANCHOR} by construction, so gaps and"
            " rankings compare but absolute levels do not",
            f"maps to: Anthropic Figure 4, left half. Arm: {arm.label}, model {IT_LABEL},"
            f" padding-fixed collection. Evidence results/{arm.directory}/scores.json;"
            " the activity list is ours (the paper's is unpublished)",
        ],
        PREFERENCE_WIDTH_PX,
    )


def _add_preference_anchors(fig: go.Figure, bar: float) -> None:
    """Draw section 8's grading scale on the correlation panel.

    The registered bar at both signs (the statistic is |r|), the zero line as
    the failure anchor, and the paper's reported band as the strength anchor.
    """
    paper_low, paper_high = PAPER_PROBE_ELO_RANGE
    # the paper's reported band, at both signs: it reports hostile near -0.74
    # and blissful near +0.71, and our statistic is the absolute value
    for sign in (1, -1):
        fig.add_hrect(
            y0=sign * paper_low,
            y1=sign * paper_high,
            fillcolor="#2ca02c",
            opacity=0.16,
            line_width=0,
            row=1,
            col=1,
        )
        fig.add_hline(y=sign * bar, line_dash="dash", line_color="#2ca02c", row=1, col=1)
    fig.add_annotation(
        text=f"strength anchor: the paper's {paper_low} to {paper_high} band",
        x=0.02,
        y=paper_high,
        xref="x domain",
        yref="y",
        showarrow=False,
        yanchor="bottom",
        font=dict(size=10, color="#1a6b1a"),
        row=1,
        col=1,
    )
    fig.add_annotation(
        text=f"registered bar |r| = {bar}",
        x=0.02,
        y=bar,
        xref="x domain",
        yref="y",
        showarrow=False,
        yanchor="bottom",
        font=dict(size=10, color="#2ca02c"),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=0,
        line_color="#888888",
        line_width=1,
        row=1,
        col=1,
        annotation_text="failure anchor: r = 0",
        annotation_position="bottom right",
        annotation_font_size=10,
    )
    fig.add_hline(
        y=ELO_ANCHOR,
        line_dash="dot",
        line_color="#888888",
        row=1,
        col=2,
        annotation_text=f"anchor: mean {ELO_ANCHOR}",
        annotation_position="bottom right",
        annotation_font_size=9,
    )


def _add_preference_traces(fig: go.Figure, arm: ScoredArm, by_layer: dict[str, Any]) -> list[str]:
    """Add one layer's bar panel and scatter panel; return the sorted probe names.

    The bars are every probe's correlation with Elo, ascending; the scatter is
    the strongest probe's activation against Elo, one dot per activity colored
    by activity category.
    """
    scores = arm.scores
    elo_values = np.array([activity["elo"] for activity in scores["elo_per_activity"]])
    categories = [activity["category"] for activity in scores["elo_per_activity"]]
    per_probe_r = np.array(by_layer["per_probe_r"])
    order = np.argsort(per_probe_r)
    activation = np.array(by_layer["best_probe_activation_per_activity"])
    fig.add_bar(
        x=list(range(len(per_probe_r))),
        y=per_probe_r[order],
        marker_color="#4878a8",
        row=1,
        col=1,
        showlegend=False,
        visible=False,
    )
    for category, color in CATEGORY_COLORS.items():
        rows = [pos for pos, name in enumerate(categories) if name == category]
        fig.add_scatter(
            x=activation[rows],
            y=elo_values[rows],
            mode="markers",
            name=category.replace("_", " "),
            marker=dict(color=color, size=9),
            row=1,
            col=2,
            visible=False,
        )
    return [scores["probe_emotions_matched"][pos] for pos in order]


def _preference_slider_steps(
    arm: ScoredArm, names_by_layer: list[list[str]], traces_per_layer: int, default_pos: int
) -> list[dict[str, Any]]:
    """One slider step per scored probe layer: swap traces, relabel, RETITLE.

    The verdict changes with the layer, so a frozen title over moving data
    would be a defect; each step carries its own layer's verdict and its own
    layer's probe ordering on the x axis.
    """
    by_layers = arm.scores["probe_elo_by_layer"]
    steps = []
    for layer_pos, by_layer in enumerate(by_layers):
        names = names_by_layer[layer_pos]
        tickvals = list(range(0, len(names), 8))
        steps.append(
            dict(
                method="update",
                args=[
                    {
                        "visible": [
                            other == layer_pos
                            for other in range(len(by_layers))
                            for _ in range(traces_per_layer)
                        ]
                    },
                    {
                        "title.text": _preference_title(arm, by_layer, layer_pos == default_pos),
                        "xaxis.tickvals": tickvals,
                        "xaxis.ticktext": [names[pos] for pos in tickvals],
                    },
                ],
                label=f"L{by_layer['layer']}",
            )
        )
    return steps


def preference_figure(arm: ScoredArm) -> tuple[go.Figure, dict[str, Any]]:
    """Section 8 exhibit: probe activations against revealed preference.

    Input: one :class:`ScoredArm` from :func:`load_preference_arms`. Returns
    ``(figure, stats)``; ``stats["lines"]`` is the printed record (the
    registered P1 and P2 verdicts for this arm) and ``stats["max_abs_r"]`` the
    strongest correlation at the registered best layer.
    """
    scores = arm.scores
    by_layers = scores["probe_elo_by_layer"]
    layers = [by_layer["layer"] for by_layer in by_layers]
    default_pos = layers.index(scores["p2_best_layer"])
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.09,
        subplot_titles=(
            "every probe's correlation with Elo, sorted",
            "the strongest probe at this layer, against Elo",
        ),
    )
    names_by_layer = [_add_preference_traces(fig, arm, by_layer) for by_layer in by_layers]
    traces_per_layer = 1 + len(CATEGORY_COLORS)  # one bar panel plus one scatter per category
    for trace_pos in range(default_pos * traces_per_layer, (default_pos + 1) * traces_per_layer):
        fig.data[trace_pos].visible = True
    _add_preference_anchors(fig, scores["p2_registered_bar"])
    default_names = names_by_layer[default_pos]
    default_ticks = list(range(0, len(default_names), 8))
    fig.update_xaxes(
        tickvals=default_ticks,
        ticktext=[default_names[pos] for pos in default_ticks],
        tickangle=45,
        tickfont=dict(size=9),
        row=1,
        col=1,
    )
    # leave headroom for the paper's band, which sits outside the observed range
    fig.update_yaxes(
        title_text="Pearson r between probe activation and Elo",
        range=[-0.9, 0.9],
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="probe activation (cosine)", row=1, col=2)
    fig.update_yaxes(title_text="Elo rating (mean-anchored)", row=1, col=2)
    fig.update_layout(
        title=_preference_title(arm, by_layers[default_pos], True),
        title_font_size=13,
        # the wrapped title grows downward from the top of the margin
        title_y=0.985,
        title_yanchor="top",
        width=PREFERENCE_WIDTH_PX,
        height=760,
        # bottom margin holds the rotated probe names, the legend, and the slider
        margin=dict(t=160, b=250),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.30,
            font=dict(size=10),
            title=dict(text="activity category (what the activity is)", font=dict(size=10)),
        ),
        sliders=[
            dict(
                active=default_pos,
                currentvalue=dict(prefix="probe layer: "),
                pad=dict(t=140),
                len=0.5,
                x=0.25,
                steps=_preference_slider_steps(arm, names_by_layer, traces_per_layer, default_pos),
            )
        ],
    )
    return fig, {"lines": _preference_lines(arm), "max_abs_r": scores["p2_max_abs_r"]}


def _preference_lines(arm: ScoredArm) -> list[str]:
    """This arm's registered verdicts, as the campaign record prints them."""
    scores = arm.scores
    return [
        f"[{arm.label}] P1 {'PASS' if scores['p1_pass'] else 'FAIL'} "
        f"(positive {scores['p1_positive_mean']:.0f} vs negative {scores['p1_negative_mean']:.0f}); "
        f"P2 {'PASS' if scores['p2_pass'] else 'FAIL'} "
        f"(max |r|={scores['p2_max_abs_r']} bar {scores['p2_registered_bar']},"
        f" organization r={scores['p2_valence_organization_r']},"
        f" perm p={scores['p2_valence_organization_perm_p']})",
        f"  best layer {scores['p2_best_layer']}, best probe"
        f" '{scores['p2_best_probe_emotion']}', evidence results/{arm.directory}/scores.json",
    ]


def instrument_sensitivity_stats() -> dict[str, Any]:
    """Section 8's instrument check: the same chat arm under two probe sets.

    No figure. Reads the chat arm's ``scores.json`` (pre-fix probes) and
    ``scores_postfix_probes.json`` (post-fix probes) and returns ``stats``
    whose ``lines`` print the per-layer maximum |r| under each instrument, so
    the headline correlation is stated as a range over instruments rather than
    a single number. Raises if the two files disagree on the layer grid.
    """
    chat_directory = PREFERENCE_SOURCES[1][1]
    pre_fix = json.loads((fetch(chat_directory) / "scores.json").read_text())
    post_fix = json.loads((fetch(chat_directory) / POSTFIX_PROBE_SCORES).read_text())
    bar = pre_fix["p2_registered_bar"]
    lines = [
        f"probe-Elo max |r| per layer, chat arm, by probe instrument (registered bar {bar}):",
        f"{'layer':>6} {'pre-fix probes':>15} {'post-fix probes':>16}",
    ]
    for pre_layer, post_layer in zip(pre_fix["probe_elo_by_layer"], post_fix["probe_elo_by_layer"]):
        if pre_layer["layer"] != post_layer["layer"]:
            raise ValueError(
                f"instrument files disagree on the layer grid:"
                f" {pre_layer['layer']} vs {post_layer['layer']}"
            )
        lines.append(
            f"{pre_layer['layer']:>6} {pre_layer['max_abs_r']:>15.4f}"
            f" {post_layer['max_abs_r']:>16.4f}"
        )
    lines += [
        f"sources: results/{chat_directory}/scores.json (pre-fix probes) and"
        f" {POSTFIX_PROBE_SCORES} (post-fix probes)",
        f"read: every layer clears the {bar} bar under both instruments; the fix costs |r|"
        f" {pre_fix['p2_max_abs_r']} to {post_fix['p2_max_abs_r']} at the best layer"
        f" ({post_fix['p2_best_layer']}) but does not change any pass/fail verdict",
    ]
    return {"lines": lines, "pre_fix_max_abs_r": pre_fix["p2_max_abs_r"]}


def _check_paper_reference(arms: list[ScoredArm]) -> None:
    """Fail loudly if an arm's recorded paper reference lost our quoted values.

    The figure states the paper's coupling and mean shifts as replication
    targets; each scores file also records them in a free-text field. If they
    ever disagree, the figure is quoting something the evidence no longer says.
    """
    for arm in arms:
        reference = str(arm.scores["paper_reference"])
        expected = [str(PAPER_COUPLING_R)] + [str(abs(v)) for v in PAPER_MEAN_SHIFTS.values()]
        missing = [value for value in expected if value not in reference]
        if missing:
            raise ValueError(
                f"{arm.label}: paper_reference {reference!r} no longer records {missing}"
            )


def _steering_title(arms: list[ScoredArm]) -> str:
    """The question-form title, with both doses' verdicts computed at runtime."""
    sign_clause = ", ".join(
        f"{arm.scores['p3_sign_agreements']} of {len(arm.scores['per_emotion'])} at {arm.label}"
        for arm in arms
    )
    coupling_clause = ", ".join(f"r = {arm.scores['p2_pearson_r']:+.3f}" for arm in arms)
    bar = arms[0].scores["p3_registered_bar"]
    coupling_bar = arms[0].scores["p2_registered_bar"]
    passes_sign = all(arm.scores["p3_pass"] for arm in arms)
    passes_coupling = any(arm.scores["p2_pass"] for arm in arms)
    verdict = "Partly" if passes_sign and not passes_coupling else "Yes" if passes_sign else "No"
    return figure_title(
        f"Does steering with an emotion vector move preferences the way its probe predicted?"
        f" {verdict}: the direction is right ({sign_clause} emotions move the valence-correct way,"
        f" bar {bar}) but the fine-grained coupling is not ({coupling_clause},"
        f" bar {coupling_bar}, paper {PAPER_COUPLING_R})",
        [
            "one dot = one steered emotion at one dose; x = how well that emotion's probe predicted"
            " preference WITHOUT steering (section 8's per-probe r); y = how far steering with that"
            " vector actually moved preference, as the mean Elo change of the positive-category"
            " activities",
            "failure anchor: y = 0, steering redistributes nothing (a uniform shift is invisible to"
            " pairwise choices by construction); strength anchors: the paper's reported mean shifts"
            f" of {PAPER_MEAN_SHIFTS['blissful']:+d} (blissful) and"
            f" {PAPER_MEAN_SHIFTS['hostile']:+d} (hostile), drawn as dashed lines",
            "maps to: Anthropic Figure 4, right half. Model"
            f" {IT_LABEL}, steering at layer {arms[0].scores['layer']}, all positions, padding-fixed"
            " collections. Evidence results/"
            + "/scores.json, results/".join(arm.directory for arm in arms)
            + "/scores.json",
        ],
        STEERING_TITLE_WRAP_PX,
    )


def _add_steering_anchors(fig: go.Figure) -> None:
    """Draw section 9's grading scale: no-effect line and the paper's shifts."""
    fig.add_hline(
        y=0,
        line_color="#888888",
        line_width=1,
        annotation_text="failure anchor: y = 0",
        annotation_position="bottom right",
        annotation_font_size=10,
    )
    for emotion, shift in PAPER_MEAN_SHIFTS.items():
        fig.add_hline(
            y=shift,
            line_dash="dash",
            line_color="#2ca02c",
            opacity=0.7,
            annotation_text=f"replication target: the paper's {shift:+d} Elo for {emotion} steering",
            annotation_position="top left" if shift > 0 else "bottom left",
            annotation_font_size=10,
        )
    fig.add_vline(x=0, line_color="#cccccc", line_width=1)


def steering_figure(arms: list[ScoredArm]) -> tuple[go.Figure, dict[str, Any]]:
    """Section 9 exhibit: steering redistribution against the probe prediction.

    Input: both doses from :func:`load_steering_arms`. Returns
    ``(figure, stats)``; ``stats["lines"]`` is the printed record (the
    registered P1, P2 and P3 verdicts per dose) and ``stats["sign_agreements"]``
    maps each dose label to its valence-sign count.
    """
    _check_paper_reference(arms)
    fig = go.Figure()
    # the two doses land close together, so their point labels are pushed to
    # opposite sides of the marker instead of colliding on top of each other
    label_sides = ("top left", "bottom right")
    for arm, label_side in zip(arms, label_sides):
        per_emotion = arm.scores["per_emotion"]
        fig.add_scatter(
            x=[row["probe_elo_r"] for row in per_emotion],
            y=[row["mean_delta_elo_positive_categories"] for row in per_emotion],
            mode="markers+text",
            text=[row["emotion"] for row in per_emotion],
            textposition=label_side,
            textfont=dict(size=8),
            marker=dict(size=9),
            name=(
                f"steering dose {arm.label} x residual RMS"
                f" (coupling r = {arm.scores['p2_pearson_r']:+.2f})"
            ),
        )
    _add_steering_anchors(fig)
    fig.update_layout(
        title=_steering_title(arms),
        title_font_size=13,
        # the wrapped title grows downward from the top of the margin
        title_y=0.985,
        title_yanchor="top",
        xaxis_title="probe-Elo correlation r without steering (section 8, chat arm)",
        yaxis_title="mean Elo change of positive-category activities (steered minus baseline)",
        width=STEERING_WIDTH_PX,
        height=660,
        # bottom margin holds the axis title above a two-row legend
        margin=dict(t=165, b=130),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            font=dict(size=10),
            title=dict(text="steering arm (the dose it was collected at)", font=dict(size=10)),
        ),
    )
    lines = [
        f"[{arm.label}] P1 {'PASS' if arm.scores['p1_pass'] else 'FAIL'}; "
        f"P2 {'PASS' if arm.scores['p2_pass'] else 'FAIL'}"
        f" (r={arm.scores['p2_pearson_r']:+.3f}, p={arm.scores['p2_p_value']:.3f}); "
        f"P3 {'PASS' if arm.scores['p3_pass'] else 'FAIL'}"
        f" ({arm.scores['p3_sign_agreements']}/{len(arm.scores['per_emotion'])})"
        for arm in arms
    ]
    lines.append("evidence: " + ", ".join(f"results/{arm.directory}/scores.json" for arm in arms))
    return fig, {
        "lines": lines,
        "sign_agreements": {arm.label: arm.scores["p3_sign_agreements"] for arm in arms},
    }
