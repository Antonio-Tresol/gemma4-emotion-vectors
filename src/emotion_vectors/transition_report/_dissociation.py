"""Section 2 exhibit: identity against anticipation, story sets x probe sets.

Two heatmaps sharing one grid, with a layer slider. The left panel is the
identity read, the right the anticipation read (R1). Every slider step
redraws BOTH panels and rewrites the title with that layer's own verdict: the
pass pattern moves with depth, so a frozen title over moving data would be
wrong at most layers. The builder returns ``(figure, stats)`` where
``stats["lines"]`` holds every number the notebook prints.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._data import (
    CHANCE_FRACTION,
    IDENTITY_BAR_FRACTION,
    PRIMARY_LAYER,
    PROBE_SET_SIZE,
    PROBE_SET_TICK_LABEL,
    PROBED_MODEL,
    R1_BAR_IN_NOISE_SD,
    TransitionEvidence,
    figure_title,
    title_top,
)

DISSOCIATION_WIDTH_PX = 1250
DISSOCIATION_HEIGHT_PX = 560
GEMMA_STORY_SET = "Gemma-written stories (v1 probes)"
DEEPSEEK_STORY_SET = "DeepSeek-written stories"
EVIDENCE_NOTE = (
    "evidence: q3_gate_r1_it.json, q3_gate_r1_deepseek.json, q3_gate_r1_it_v2.json"
    " (experiment-artifacts dataset)"
)


def dissociation_panels(
    evidence: TransitionEvidence, layer: int, probe_sets: list[str]
) -> dict[str, list[list[Any]]]:
    """Both panels' cell values and printed labels at one layer.

    Inputs: the evidence, the layer to slice, and the column order
    ``probe_sets``. Returns ``{"identity": z, "identity_text": t,
    "anticipation": z, "anticipation_text": t}``, each a
    [story sets x probe sets] nested list. A cell the arm never scored is
    ``None`` in ``z`` and ``"not carried"`` in the text, so an absent probe set
    reads as absent rather than as a zero.
    """
    panels: dict[str, list[list[Any]]] = {key: [] for key in ("identity", "anticipation")}
    panels["identity_text"], panels["anticipation_text"] = [], []
    for story_set in evidence.story_sets.values():
        identity_row, identity_text, lead_row, lead_text = [], [], [], []
        for probe_set in probe_sets:
            identity_cell = story_set.identity_cell(probe_set, layer)
            if identity_cell is None:
                identity_row.append(None)
                identity_text.append("not carried")
            else:
                star = " *" if identity_cell["passes"] else ""
                identity_row.append(identity_cell["median_rank"] / PROBE_SET_SIZE[probe_set])
                identity_text.append(
                    f"rank {identity_cell['median_rank']:.0f}/{PROBE_SET_SIZE[probe_set]}{star}"
                )
            lead = story_set.lead_in_noise_sd(probe_set, layer)
            r1_cell = story_set.r1_cell(probe_set, layer)
            if lead is None or r1_cell is None:
                lead_row.append(None)
                lead_text.append("not carried")
            else:
                lead_row.append(lead)
                lead_text.append(f"{lead:+.1f} sd{' *' if r1_cell['passes'] else ''}")
        panels["identity"].append(identity_row)
        panels["identity_text"].append(identity_text)
        panels["anticipation"].append(lead_row)
        panels["anticipation_text"].append(lead_text)
    return panels


def _pass_counts(evidence: TransitionEvidence, layer: int) -> dict[str, Any]:
    """How many cells clear each read's bar at one layer, and where.

    Returns the two pass counts, the two cell totals, and the set of story
    sets holding at least one passing anticipation cell: the per-layer
    facts the title states as its verdict.
    """
    # "gate_G" is the identity read's key in the evidence files, kept as-is
    identity_cells = [
        story_set.identity_cell(probe_set, layer)
        for story_set, probe_set in evidence.scored_cells("gate_G")
    ]
    r1_pairs = [
        (story_set, story_set.r1_cell(probe_set, layer))
        for story_set, probe_set in evidence.scored_cells("r1_anticipation")
    ]
    passing = {each.label for each, cell in r1_pairs if cell is not None and cell["passes"]}
    return {
        "identity_pass": sum(1 for cell in identity_cells if cell is not None and cell["passes"]),
        "identity_total": len(identity_cells),
        "r1_pass": sum(1 for _, cell in r1_pairs if cell is not None and cell["passes"]),
        "r1_total": len(r1_pairs),
        "r1_passing_story_sets": sorted(passing),
    }


def _dissociation_title(evidence: TransitionEvidence, layer: int, probe_sets: list[str]) -> str:
    """The title for one slider step, with that layer's verdict computed in."""
    counts = _pass_counts(evidence, layer)
    gemma_lead = evidence.story_set(GEMMA_STORY_SET).lead_in_noise_sd("selfgen", layer)
    deepseek_leads = [
        lead
        for probe_set in probe_sets
        if (lead := evidence.story_set(DEEPSEEK_STORY_SET).lead_in_noise_sd(probe_set, layer))
        is not None
    ]
    if gemma_lead is None or not deepseek_leads:
        raise ValueError(f"layer {layer}: the two headline anticipation cells are not both scored")
    if not counts["r1_passing_story_sets"]:
        where = "no story set clears it here"
    elif all("Gemma-written" in label for label in counts["r1_passing_story_sets"]):
        where = "every one of them on Gemma-written stories"
    else:
        where = "on " + ", ".join(counts["r1_passing_story_sets"])
    return figure_title(
        "Does identity follow the probe set while anticipation follows who wrote the stories?"
        f" At layer {layer}: {counts['r1_pass']} of {counts['r1_total']} anticipation cells clear"
        f" their bar, {where}; the self-gen probes' pre-boundary lead is {gemma_lead:+.1f}x noise"
        f" on Gemma-written stories against {max(deepseek_leads):+.1f}x on DeepSeek-written ones"
        f" ({counts['identity_pass']} of {counts['identity_total']} identity cells clear their"
        " bar here)",
        [
            "one cell = one (story set, probe set) at the selected layer. Left panel: the tagged"
            " emotion's median rank as a fraction of how many probes the set holds, low is good."
            " Right panel: the incoming emotion's mean rise over the 16 tokens before the"
            " boundary versus the 16 before those, in units of that layer's calibrated per-token"
            " noise",
            f"FAILURE anchors: identity chance = {CHANCE_FRACTION} of the set (N2 shuffle),"
            " anticipation zero lead. STRENGTH anchors: the registered top-decile identity bar"
            f" at {IDENTITY_BAR_FRACTION}, and the anticipation bar at {R1_BAR_IN_NOISE_SD}x noise"
            " with N2 p < 0.001; * marks a cell that clears its own bar",
            "rows = story sets (who wrote what the model was reading), columns = which stories"
            f" the probes were built from; layer slider below | {PROBED_MODEL}"
            f" | {EVIDENCE_NOTE}",
        ],
        DISSOCIATION_WIDTH_PX,
    )


def _identity_colorbar() -> dict[str, Any]:
    """Colorbar for the identity panel, ticked at both grading anchors."""
    return dict(
        title=dict(
            text="median rank as a<br>fraction of the probe set", side="top", font=dict(size=10)
        ),
        x=0.415,
        len=0.72,
        thickness=12,
        tickfont=dict(size=9),
        tickvals=[0, IDENTITY_BAR_FRACTION, CHANCE_FRACTION],
        ticktext=[
            "0 = top of the set",
            f"{IDENTITY_BAR_FRACTION} = registered bar",
            "0.5 = chance",
        ],
    )


def _anticipation_colorbar(limit: float) -> dict[str, Any]:
    """Colorbar for the anticipation panel, ticked at zero and both extremes.

    The pass bar sits at 0.5x noise, which is under 3% of this bar's length: a
    tick there collides with the zero tick, so the bar is stated in the
    colorbar title and marked on the cells themselves with a star instead.
    """
    return dict(
        title=dict(
            text="pre-boundary lead<br>(x per-token noise sd)"
            f"<br>* = clears the +{R1_BAR_IN_NOISE_SD} bar",
            side="top",
            font=dict(size=10),
        ),
        x=1.0,
        len=0.72,
        thickness=12,
        tickfont=dict(size=9),
        tickvals=[-limit, 0, limit],
        ticktext=[f"{-limit:.0f} = falling", "0 = no lead", f"+{limit:.0f} = rising"],
    )


def _dissociation_traces(
    fig: go.Figure, evidence: TransitionEvidence, probe_sets: list[str], limit: float
) -> None:
    """Add one heatmap pair per layer; only the primary layer starts visible.

    Both color scales are pinned across every slider step (identity 0 to 0.6,
    anticipation symmetric about zero at the largest lead anywhere in the
    grid), so one colour means the same value at every layer.
    """
    columns = [PROBE_SET_TICK_LABEL[probe_set] for probe_set in probe_sets]
    rows = [story_set.tick_label for story_set in evidence.story_sets.values()]
    for layer in evidence.layers:
        panels = dissociation_panels(evidence, layer, probe_sets)
        visible = layer == PRIMARY_LAYER
        fig.add_trace(
            go.Heatmap(
                z=panels["identity"],
                x=columns,
                y=rows,
                text=panels["identity_text"],
                texttemplate="%{text}",
                colorscale="Blues_r",
                zmin=0,
                zmax=0.6,
                colorbar=_identity_colorbar(),
                hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>",
                visible=visible,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=panels["anticipation"],
                x=columns,
                y=rows,
                text=panels["anticipation_text"],
                texttemplate="%{text}",
                colorscale="RdBu",
                zmin=-limit,
                zmax=limit,
                colorbar=_anticipation_colorbar(limit),
                hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>",
                visible=visible,
            ),
            row=1,
            col=2,
        )


def _lead_limit(evidence: TransitionEvidence) -> float:
    """The symmetric colour limit: the largest lead magnitude anywhere."""
    leads = [
        abs(lead)
        for story_set, probe_set in evidence.scored_cells("r1_anticipation")
        for layer in evidence.layers
        if (lead := story_set.lead_in_noise_sd(probe_set, layer)) is not None
    ]
    if not leads:
        raise ValueError("no anticipation cell scored: nothing to set a colour scale from")
    return float(int(max(leads)) + 1)


def _anticipation_by_story_set(evidence: TransitionEvidence) -> list[str]:
    """Per story set, the anticipation record pooled over every layer and probe set.

    This is what the section-2 verdict quotes about the constant-emotion
    control: how many of that set's cells clear the bar anywhere, and the
    smallest shuffle p it ever reached, so "the control never certifies" is a
    printed number rather than an impression from the heatmap.
    """
    lines = []
    for story_set in evidence.story_sets.values():
        cells = [
            cell
            for probe_set in story_set.banks("r1_anticipation")
            for layer in evidence.layers
            if (cell := story_set.r1_cell(probe_set, layer)) is not None
        ]
        n_pass = sum(1 for cell in cells if cell["passes"])
        smallest_p = min(float(cell["n2_p"]) for cell in cells)
        lines.append(
            f"{story_set.label:44s} anticipation across all layers and probe sets:"
            f" {n_pass} of {len(cells)} cells clear the bar; smallest shuffle p {smallest_p:.4f}"
        )
    return lines


def _layer_steps(evidence: TransitionEvidence, probe_sets: list[str]) -> list[dict[str, Any]]:
    """One slider step per layer: what to show, and the title to show with it.

    Each step reveals that layer's heatmap pair (two traces per layer, added in
    the same order by ``_dissociation_traces``) AND rewrites the title, so the
    stated verdict always describes the data on screen.
    """
    return [
        dict(
            method="update",
            label=f"layer {layer}",
            args=[
                {
                    "visible": [
                        pos == layer_pos for pos in range(len(evidence.layers)) for _ in (0, 1)
                    ]
                },
                {"title.text": _dissociation_title(evidence, layer, probe_sets)},
            ],
        )
        for layer_pos, layer in enumerate(evidence.layers)
    ]


def dissociation_figure(evidence: TransitionEvidence) -> tuple[go.Figure, dict[str, Any]]:
    """Section 2 exhibit: the identity / anticipation dissociation, with a slider.

    Input: the loaded :class:`~._data.TransitionEvidence`. Returns
    ``(figure, stats)``. ``stats["lines"]`` prints, per layer, how many
    identity and anticipation cells clear their bars and which story sets the
    anticipation passes sit on; ``stats["n_rows"]`` and ``stats["n_columns"]``
    are the grid shape DERIVED from the evidence, which the notebook asserts.
    """
    # "gate_G" is the identity read's key in the evidence files, kept as-is
    probe_sets = sorted(
        {probe_set for _, probe_set in evidence.scored_cells("gate_G")},
        key=lambda probe_set: list(PROBE_SET_TICK_LABEL).index(probe_set),
    )
    limit = _lead_limit(evidence)
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.17,
        subplot_titles=(
            "identity read:<br>is the current emotion readable?",
            "anticipation read (R1):<br>does the next emotion rise early?",
        ),
    )
    _dissociation_traces(fig, evidence, probe_sets, limit)
    steps = _layer_steps(evidence, probe_sets)
    for annotation in fig.layout.annotations:  # only the panel titles exist at this point
        annotation.font = dict(size=12)
    fig.update_xaxes(title_text="probe set", title_font_size=11, tickfont=dict(size=10))
    # reversed so the rows read top-to-bottom in the same order as section 1's
    # panels read left-to-right; plotly stacks heatmap categories upward
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    fig.update_yaxes(title_text="story set", title_font_size=11, row=1, col=1)
    fig.update_layout(
        sliders=[
            dict(
                active=evidence.layers.index(PRIMARY_LAYER),
                steps=steps,
                y=-0.30,
                currentvalue=dict(prefix="showing: ", font=dict(size=12)),
            )
        ],
        title=_dissociation_title(evidence, PRIMARY_LAYER, probe_sets),
        title_font_size=13,
        title_y=title_top(DISSOCIATION_HEIGHT_PX),
        title_yanchor="top",
        width=DISSOCIATION_WIDTH_PX,
        height=DISSOCIATION_HEIGHT_PX,
        margin=dict(t=185, b=150, l=180, r=190),
    )
    lines = []
    for layer in evidence.layers:
        counts = _pass_counts(evidence, layer)
        where = ", ".join(counts["r1_passing_story_sets"]) or "none"
        lines.append(
            f"layer {layer:2d}: identity {counts['identity_pass']}/{counts['identity_total']}"
            f" cells clear the bar; anticipation {counts['r1_pass']}/{counts['r1_total']}"
            f" clear theirs (story sets: {where})"
        )
    lines += _anticipation_by_story_set(evidence)
    return fig, {
        "lines": lines,
        "n_rows": len(evidence.story_sets),
        "n_columns": len(probe_sets),
        "lead_limit": limit,
    }
