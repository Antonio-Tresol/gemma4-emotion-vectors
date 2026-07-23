"""S6 and S7 exhibits: named confusions (what the model reports when it
disagrees with the tag) and the boundary-lag vs stable-relabeling timing read."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from jaxtyping import Bool, Int
from plotly.subplots import make_subplots

from ._data import (
    FAMILIES,
    LAYERS,
    PRIMARY_LAYER,
    S7_CATEGORIES,
    Arms,
    Row,
    bank_columns,
    layer_slider,
)


def confusion(
    arms: Arms,
    arm: str,
    bank: str,
    layer_pos: int,
    restrict: list[str] | None = None,
) -> tuple[Int[np.ndarray, " kept_phases"], list[Row], list[str]]:
    """Winning probe per phase for one (arm, bank) at one layer position.

    ``restrict`` optionally cuts the bank to a subset of emotion names (used
    to put the corpus bank on the battery-12 footing). Returns (winner
    position within the kept bank per kept phase, kept phase metadata rows,
    bank emotion names after the cut).
    """
    columns, names = bank_columns(arms, arm, bank)
    if restrict is not None:
        columns = [column for column, name in zip(columns, names) if name in restrict]
        names = [name for name in names if name in restrict]
    phase_records = arms[arm]["phases"]
    kept = [pos for pos, row in enumerate(phase_records) if row["emotion"] in names]
    winners = arms[arm]["phase_scores"][kept][:, layer_pos, :][:, columns].argmax(axis=1)
    return winners, [phase_records[pos] for pos in kept], names


def _s6_table_lines(arms: Arms, battery: list[str]) -> tuple[dict[str, object], list[str]]:
    """The three named-confusion tables at the primary layer.

    Per table (reader/bank combination) and per expected emotion: the top-3
    reported emotions with their win rates. Returns (``tables[label][emotion]
    = {"n", "top1", "reported"}``, the printed lines).
    """
    primary_pos = LAYERS.index(PRIMARY_LAYER)
    tables: dict[str, object] = {}
    lines: list[str] = []
    for label, (arm, bank, restrict) in [
        ("it_v2 / selfgen bank", ("it_v2", "selfgen", None)),
        ("it_v2 / deepseek bank", ("it_v2", "deepseek", None)),
        ("base reader / corpus bank, battery-12 subset", ("base", "corpus", battery)),
    ]:
        winners, phase_rows, names = confusion(arms, arm, bank, primary_pos, restrict)
        table = {}
        lines.append(f"=== {label}, layer {PRIMARY_LAYER}: expected -> model reports (rate) ===")
        for expected_pos, expected in enumerate(names):
            phase_idx = [pos for pos, row in enumerate(phase_rows) if row["emotion"] == expected]
            report_counts = np.bincount(winners[phase_idx], minlength=len(names))
            most_reported = np.argsort(-report_counts)
            reported = [
                (names[reported_pos], report_counts[reported_pos] / len(phase_idx))
                for reported_pos in most_reported[:3]
                if report_counts[reported_pos]
            ]
            table[expected] = {
                "n": len(phase_idx),
                "top1": report_counts[expected_pos] / len(phase_idx),
                "reported": reported,
            }
            lines.append(
                f"  {expected:10s} (n={len(phase_idx):3d}) -> "
                + ", ".join(f"{name} {rate:.0%}" for name, rate in reported)
            )
        tables[label] = table
    return tables, lines


def _s6_family_lines(arms: Arms) -> list[str]:
    """Per designed family: wrong-answer rate and the emotions the model
    substitutes (it_v2, selfgen bank, primary layer). Returns printed lines."""
    primary_pos = LAYERS.index(PRIMARY_LAYER)
    winners, phase_rows, names = confusion(arms, "it_v2", "selfgen", primary_pos)
    lines = ["", "=== wrong-answer profile per designed family (it_v2, selfgen bank) ==="]
    for family in FAMILIES:
        family_idx = [pos for pos, row in enumerate(phase_rows) if row["category"] == family]
        wrong_idx = [pos for pos in family_idx if names[winners[pos]] != phase_rows[pos]["emotion"]]
        # count which wrong emotion the model reported instead, most common first
        wrong_counts: dict[str, int] = {}
        for pos in wrong_idx:
            wrong_counts[names[winners[pos]]] = wrong_counts.get(names[winners[pos]], 0) + 1
        top_wrong = sorted(wrong_counts.items(), key=lambda name_count: -name_count[1])[:4]
        lines.append(
            f"  {family:20s} wrong {len(wrong_idx) / len(family_idx):4.0%}; model says: "
            + ", ".join(f"{name} x{count}" for name, count in top_wrong)
        )
    return lines


def _s6_base_heatmap(arms: Arms, battery: list[str]) -> go.Figure:
    """Row-normalized confusion heatmap for the BASE reader on the corpus
    bank cut to the battery 12, one trace per layer with a slider."""
    fig = go.Figure()
    for layer_pos, layer in enumerate(LAYERS):
        winners, phase_rows, names = confusion(arms, "base", "corpus", layer_pos, battery)
        confusion_counts = np.zeros((len(names), len(names)))
        for phase_pos, row in enumerate(phase_rows):
            confusion_counts[names.index(row["emotion"]), winners[phase_pos]] += 1
        row_normalized = confusion_counts / np.clip(
            confusion_counts.sum(axis=1, keepdims=True), 1, None
        )
        fig.add_trace(
            go.Heatmap(
                z=row_normalized,
                x=names,
                y=names,
                colorscale="Blues",
                zmin=0,
                zmax=1,
                showscale=(layer_pos == 0),
                colorbar=dict(title="fraction of the<br>row's phases"),
            )
        )
    fig.update_layout(
        height=540,
        width=680,
        yaxis_autorange="reversed",
        xaxis_title="emotion the base reader reports (winning corpus probe)",
        yaxis_title="expected (tagged) emotion",
        title="S6: what the BASE reader says the emotion is "
        "(corpus bank, battery-12 subset; rows sum to 1)",
    )
    layer_slider(fig, traces_per_layer=1)
    return fig


def s6_named_confusions(arms: Arms) -> tuple[go.Figure, dict[str, object]]:
    """S6: what the model says the emotion is, named rather than as distance.

    Returns the base-reader confusion heatmap (layer slider) and ``stats =
    {"tables": ..., "lines": ...}`` where ``lines`` is the exact printed
    block: the three named-confusion tables followed by the per-family
    wrong-answer profile.
    """
    battery = bank_columns(arms, "it_v2", "selfgen")[1]
    tables, lines = _s6_table_lines(arms, battery)
    lines += _s6_family_lines(arms)
    fig = _s6_base_heatmap(arms, battery)
    return fig, {"tables": tables, "lines": lines}


def third_match(
    arms: Arms, arm: str, bank: str
) -> tuple[Bool[np.ndarray, "kept_phases thirds layers"], list[Row]]:
    """Whether the tagged probe wins each token-count third of each phase.

    Returns (match [kept_phases, 3 thirds, layers], kept phase metadata
    rows); a phase is kept when its tagged emotion has a probe in the bank.
    """
    columns, names = bank_columns(arms, arm, bank)
    phase_records = arms[arm]["phases"]
    kept = [pos for pos, row in enumerate(phase_records) if row["emotion"] in names]
    # thirds: [kept_phases, 3 thirds, layers, bank_probes] centered cosines
    thirds = arms[arm]["phase_thirds"][kept][:, :, :, columns]
    target = np.array([names.index(phase_records[pos]["emotion"]) for pos in kept])
    match = thirds.argmax(axis=3) == target[:, None, None]
    return match, [phase_records[pos] for pos in kept]


def _s7_family_lines(arms: Arms) -> list[str]:
    """Converges vs never-right share per designed family at the primary
    layer (it_v2, selfgen bank). Returns printed lines."""
    primary_pos = LAYERS.index(PRIMARY_LAYER)
    match, phase_rows = third_match(arms, "it_v2", "selfgen")
    lines = [f"layer {PRIMARY_LAYER} per family (it_v2, selfgen): converges vs never-right"]
    for family in FAMILIES:
        family_idx = [pos for pos, row in enumerate(phase_rows) if row["category"] == family]
        first_third = match[family_idx, 0, primary_pos]
        last_third = match[family_idx, 2, primary_pos]
        converges = (~first_third & last_third).mean()
        never_right = (~first_third & ~last_third).mean()
        lines.append(
            f"  {family:20s} converges {converges:4.0%}   never right {never_right:4.0%}   "
            f"(n={len(family_idx)})"
        )
    return lines


def _s7_verdict_lines(
    arms: Arms, shares_primary: dict[str, dict[str, float]]
) -> tuple[str, list[str]]:
    """Name the dominant failure mode on the primary arm and assemble the
    printed block: the per-family split, then the verdict line."""
    verdict = (
        "boundary lag"
        if shares_primary["it_v2"]["converges"] > shares_primary["it_v2"]["never right"]
        else "stable relabeling"
    )
    lines = _s7_family_lines(arms) + [
        "",
        f"dominant failure mode at layer {PRIMARY_LAYER} (it_v2): {verdict}",
    ]
    return verdict, lines


def s7_thirds_figure(arms: Arms) -> tuple[go.Figure, dict[str, object]]:
    """S7: boundary lag vs stable relabeling, from phase-third correctness.

    Classifies every phase by whether the tagged probe wins its first and
    last token-count third, per story arm (selfgen bank). Returns the
    two-panel share figure (layer slider) and ``stats = {"shares":
    {arm: {category: share}} at the primary layer, "verdict": the dominant
    failure mode, "lines": the printed per-family block}``.
    """
    shares_primary: dict[str, dict[str, float]] = {}
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=["Gemma stories (it_v2)", "DeepSeek stories"]
    )
    for layer_pos, layer in enumerate(LAYERS):
        for panel, arm in enumerate(["it_v2", "deepseek"], start=1):
            match, _ = third_match(arms, arm, "selfgen")
            first_third, last_third = match[:, 0, layer_pos], match[:, 2, layer_pos]
            # shares in S7_CATEGORIES order: right-right, wrong-right,
            # right-wrong, wrong-wrong; they sum to 1 per arm and layer
            shares = [
                float((first_third & last_third).mean()),
                float((~first_third & last_third).mean()),
                float((first_third & ~last_third).mean()),
                float((~first_third & ~last_third).mean()),
            ]
            if layer == PRIMARY_LAYER:
                shares_primary[arm] = dict(zip(S7_CATEGORIES, shares))
            fig.add_trace(
                go.Bar(
                    x=S7_CATEGORIES,
                    y=shares,
                    showlegend=False,
                    marker_color="#4c78a8" if panel == 1 else "#f58518",
                    text=[f"{share:.0%}" for share in shares],
                    textposition="outside",
                ),
                row=1,
                col=panel,
            )
    for panel in (1, 2):
        fig.update_yaxes(title="share of phases", range=[0, 1], row=1, col=panel)
        fig.update_xaxes(title="phase class (first third vs last third correct)", row=1, col=panel)
    fig.update_layout(
        height=450,
        title="S7: is the model wrong early then right late (boundary lag), or wrong "
        "throughout (stable relabeling)? (selfgen bank; shares sum to 1 per panel)",
    )
    layer_slider(fig, traces_per_layer=2)
    verdict, lines = _s7_verdict_lines(arms, shares_primary)
    return fig, {"shares": shares_primary, "verdict": verdict, "lines": lines}
