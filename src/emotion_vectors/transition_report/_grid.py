"""Section 4 exhibit: the full factorial design grid, computed from evidence.

Probe sets down the rows, story sets across the columns, every cell
holding the two reads' ranges across the six layers and how many layers clear
each registered bar. The table is BUILT here rather than typed in markdown, so
no cell of it can drift away from the evidence files, and a probe set an arm
never carried says so instead of being silently blank.

Returns ``{"html": ..., "lines": ..., "rows": ...}``: the notebook displays the
HTML and prints the lines, so every number in section 4 is computed beside the
prose that interprets it.
"""

from __future__ import annotations

from typing import Any

from ._data import (
    NULL_PROBE_SET,
    PROBE_SET_LABEL,
    PROBE_SET_ORDER,
    PROBE_SET_SIZE,
    StorySet,
    TransitionEvidence,
)

TABLE_STYLE = (
    "border-collapse:collapse;font-size:12px;line-height:1.45;border:1px solid #b0b0b0;width:100%"
)
CELL_STYLE = "border:1px solid #d0d0d0;padding:6px 8px;vertical-align:top"
HEAD_STYLE = CELL_STYLE + ";background:#eeeeee;text-align:left;font-weight:600"
NOT_CARRIED = "this probe set is not carried in this arm's shards"


def _identity_summary(story_set: StorySet, probe_set: str, layers: list[int]) -> str | None:
    """One cell's identity read: rank range and how many layers clear the bar.

    Returns None when the arm never carried this probe set, which is what makes
    the "not carried" cells honest rather than blank.
    """
    cells = [story_set.identity_cell(probe_set, layer) for layer in layers]
    scored = [cell for cell in cells if cell is not None]
    if not scored:
        return None
    ranks = [float(cell["median_rank"]) for cell in scored]
    n_pass = sum(1 for cell in scored if cell["passes"])
    worst_p = max(float(cell["n2_p"]) for cell in scored)
    return (
        f"<b>identity</b>: rank {min(ranks):.0f} to {max(ranks):.0f}"
        f" of {PROBE_SET_SIZE[probe_set]}"
        f"; {n_pass} of {len(scored)} layers clear the bar"
        f"; largest shuffle p {worst_p:.4f}"
    )


def _anticipation_summary(story_set: StorySet, probe_set: str, layers: list[int]) -> str | None:
    """One cell's anticipation read: lead range and how many layers pass."""
    cells = [story_set.r1_cell(probe_set, layer) for layer in layers]
    leads = [story_set.lead_in_noise_sd(probe_set, layer) for layer in layers]
    scored = [
        (cell, lead) for cell, lead in zip(cells, leads) if cell is not None and lead is not None
    ]
    if not scored:
        return None
    values = [lead for _, lead in scored]
    n_pass = sum(1 for cell, _ in scored if cell["passes"])
    return (
        f"<b>anticipation</b>: lead {min(values):+.1f} to {max(values):+.1f} x noise"
        f"; {n_pass} of {len(scored)} layers pass"
    )


def _cell_text(story_set: StorySet, probe_set: str, layers: list[int]) -> str:
    """Both reads for one (probe set, story set) cell, or the absence note."""
    identity = _identity_summary(story_set, probe_set, layers)
    anticipation = _anticipation_summary(story_set, probe_set, layers)
    if identity is None and anticipation is None:
        return NOT_CARRIED
    parts = [part for part in (identity, anticipation) if part is not None]
    return "<br>".join(parts)


def _null_cell_text(story_set: StorySet) -> str:
    """The random-direction null row: what N1 actually scored on this arm."""
    n_phases, n_transitions = story_set.null_probe_set_counts()
    if n_phases == 0 and n_transitions == 0:
        return (
            "<b>NOT SCORED</b>: 0 phases, 0 transitions (a random direction has no tagged"
            " emotion to be ranked against; see section 5, point 6)"
        )
    return f"<b>identity</b>: {n_phases} phases; <b>anticipation</b>: {n_transitions} transitions"


def design_grid(evidence: TransitionEvidence) -> dict[str, Any]:
    """Section 4 exhibit: the probe-set x story-set grid, built from evidence.

    Input: the loaded :class:`~._data.TransitionEvidence`. Returns
    ``{"html": table markup for display, "lines": the same content as printed
    text, "rows": {probe set: {story set: cell text}}, "n_columns": the column
    count DERIVED from the evidence}``. Every number in the table is read out of
    the evidence files at call time; nothing here is typed.
    """
    story_sets = list(evidence.story_sets.values())
    rows: dict[str, dict[str, str]] = {}
    for probe_set in PROBE_SET_ORDER:
        rows[probe_set] = {
            each.label: _cell_text(each, probe_set, evidence.layers) for each in story_sets
        }
    rows[NULL_PROBE_SET] = {each.label: _null_cell_text(each) for each in story_sets}

    header = "".join(f"<th style='{HEAD_STYLE}'>{each.tick_label}</th>" for each in story_sets)
    labels = dict(PROBE_SET_LABEL)
    labels[NULL_PROBE_SET] = "random directions (24, the N1 meaninglessness floor)"
    body = ""
    for probe_set, cells in rows.items():
        body += f"<tr><th style='{HEAD_STYLE}'>{labels[probe_set]}</th>"
        body += "".join(f"<td style='{CELL_STYLE}'>{cells[each.label]}</td>" for each in story_sets)
        body += "</tr>"
    html = (
        f"<table style='{TABLE_STYLE}'><tr>"
        f"<th style='{HEAD_STYLE}'>probe set (rows) vs story set (columns)</th>{header}</tr>"
        f"{body}</table>"
    )
    lines = [
        f"{labels[probe_set]} | {story_set_label} | {text.replace('<br>', ' | ')}"
        for probe_set, cells in rows.items()
        for story_set_label, text in cells.items()
    ]
    return {"html": html, "lines": lines, "rows": rows, "n_columns": len(story_sets)}
