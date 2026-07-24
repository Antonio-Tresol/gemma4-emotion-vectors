"""Section 3: the blessed post-fix vector bundles, sized from their manifests.

Each bundle is one extraction run: per-story activation shards plus
per-emotion mean vectors, with a ``manifest.jsonl`` (one row per story:
emotion, token count, error field) and a ``run_config.json`` (reader model,
extracted layers, source corpus). ``s3_bundles_figure`` reads both through
``fetch`` and returns ``(figure, stats)`` with ``stats["lines"]`` holding
the printed record and ``stats["total_errors"]`` feeding the notebook's
zero-errors assert.
"""

from __future__ import annotations

import json
from typing import Any

import plotly.graph_objects as go

from emotion_vectors.artifacts import fetch

# The blessed instrument from 2026-07-22 on: re-extractions after the padding
# bug fix (TREE node Q1.H3.E4b). Unsuffixed predecessors are kept as the
# pre-fix record for the E4b before/after comparison.
POSTFIX_BUNDLE_NAMES = [
    "emotion_vectors_postfix",
    "emotion_vectors_it_postfix",
    "self_story_vectors_it_postfix",
    "neutral_vectors_it_postfix",
]


def _s3_title(total_stories: int, total_errors: int, layer_counts: set[int]) -> str:
    """Verdict title with the headline counts computed from the manifests."""
    layers_text = "/".join(str(count) for count in sorted(layer_counts))
    return (
        f"Are the blessed vector bundles complete? {total_errors} extraction errors across"
        f" {total_stories:,} stories in {len(POSTFIX_BUNDLE_NAMES)} bundles"
        "<br><sup>one row = one post-padding-fix extraction run (TREE Q1.H3.E4b); every column"
        " counted from its manifest.jsonl / run_config.json, never typed</sup>"
        f"<br><sup>each bundle stores per-emotion mean vectors at {layers_text} layers of its"
        " reader model;"
        "<br>grading: a green errors cell (0, PASS) means every story was extracted, a red one"
        " (nonzero, FAIL) would disqualify the bundle as the blessed instrument</sup>"
    )


BundleRow = tuple[str, str, str, int, int, str, int, int]
"""(results/ path, reader model, source corpus, emotions, stories, tokens, layers, errors)."""


def _bundle_rows() -> list[BundleRow]:
    """One :data:`BundleRow` per blessed bundle, counted from its own files.

    ``manifest.jsonl`` gives one row per story (emotion, token count, error
    field); ``run_config.json`` gives the reader model, the source corpus, and
    the extracted layers. Raises loudly if either file is missing.
    """
    rows: list[BundleRow] = []
    for bundle_name in POSTFIX_BUNDLE_NAMES:
        manifest = [json.loads(line) for line in open(fetch(f"{bundle_name}/manifest.jsonl"))]
        run_config = json.load(open(fetch(f"{bundle_name}/run_config.json")))
        ok_entries = [entry for entry in manifest if entry["error"] is None]
        rows.append(
            (
                f"results/{bundle_name}",
                run_config["model"],
                run_config["dataset"],
                len({entry["emotion"] for entry in ok_entries}),
                len(ok_entries),
                f"{sum(entry['n_tokens'] for entry in ok_entries):,}",
                len(run_config["layers"]),
                len(manifest) - len(ok_entries),
            )
        )
    return rows


def _bundles_table(rows: list[BundleRow]) -> go.Figure:
    """The bundle table, with the errors column carrying its own grading key.

    A table cannot hold reference lines, so pass/fail lives in the cell color:
    green for zero failed extractions, red for any nonzero count, which would
    disqualify the bundle as the blessed instrument.
    """
    columns = list(zip(*rows))
    error_fill = ["honeydew" if errors == 0 else "mistyrose" for errors in columns[7]]
    cell_fills = ["white"] * 7 + [error_fill]
    error_text = [f"{errors} PASS" if errors == 0 else f"{errors} FAIL" for errors in columns[7]]
    story_text = [f"{stories:,}" for stories in columns[4]]  # thousands separators, like tokens
    return go.Figure(
        go.Table(
            columnwidth=[1.9, 1.4, 3.1, 0.6, 0.6, 0.7, 0.5, 0.85],
            header=dict(
                values=[
                    "blessed bundle<br>(results/ path)",
                    "reader model<br>(whose activations)",
                    "source corpus<br>(what it read)",
                    "emotions",
                    "stories",
                    "tokens",
                    "layers",
                    "errors<br>(0 = pass)",
                ],
                align="left",
                font=dict(size=11),
                height=34,
            ),
            cells=dict(
                values=[*columns[:4], story_text, *columns[5:7], error_text],
                align="left",
                height=26,
                font=dict(size=11),
                fill_color=cell_fills,
            ),
        )
    )


def s3_bundles_figure() -> tuple[go.Figure, dict[str, Any]]:
    """Section 3 exhibit: size and integrity of the blessed -postfix bundles.

    Inputs: nothing (bundle manifests and run configs resolve through
    ``fetch``). Output: ``(figure, stats)`` where the figure is a table (one
    row per bundle: location, reader model, source corpus, emotions, stories,
    tokens, layers, errors) and ``stats`` carries ``lines`` (printed record),
    ``total_errors`` (for the notebook's assert), ``layer_counts``, and the
    raw ``rows``. Raises if any manifest or run config is missing.
    """
    rows = _bundle_rows()
    total_stories = sum(row[4] for row in rows)
    total_errors = sum(row[7] for row in rows)
    layer_counts = {row[6] for row in rows}

    fig = _bundles_table(rows)
    fig.update_layout(
        title=dict(text=_s3_title(total_stories, total_errors, layer_counts), font=dict(size=13)),
        height=350,
        width=1320,
        margin=dict(t=155, b=10),
    )
    lines = [
        f"{name}: {emotions} emotions, {stories:,} stories, {tokens} tokens,"
        f" {layers} layers, {errors} errors (reader {model}, source {source})"
        for name, model, source, emotions, stories, tokens, layers, errors in rows
    ]
    lines.append(f"bundle totals: {total_stories:,} stories, {total_errors} extraction errors")
    return fig, {
        "lines": lines,
        "total_errors": total_errors,
        "layer_counts": layer_counts,
        "rows": rows,
    }
