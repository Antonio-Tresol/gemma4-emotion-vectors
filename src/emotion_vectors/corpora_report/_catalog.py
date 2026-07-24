"""Section 1: the corpus catalog table.

One table row per corpus, every size column counted from the file itself in
``load_corpora_context`` (never typed). ``s1_catalog_figure`` returns
``(figure, stats)`` with ``stats["lines"]`` holding the printed record.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from ._data import CorporaContext


def _s1_title(ctx: CorporaContext) -> str:
    """Verdict title with the headline sizes computed from the catalog."""
    n_generator_models = len({row.generator.split(" (")[0] for row in ctx.catalog})
    total_texts = sum(row.n_texts for row in ctx.catalog)
    return (
        f"What text does every downstream result rest on? {len(ctx.catalog)} corpora, "
        f"{total_texts:,} texts, {n_generator_models} generator models"
        "<br><sup>one row = one corpus; the emotions and texts columns are counted from the file"
        " in the location column, never typed</sup>"
        "<br><sup>generator = who WROTE the texts (which model later READS them is a separate,"
        " per-experiment choice, named in section 3); role = which experiments consume the"
        " corpus</sup>"
        "<br><sup>this is an inventory, not a verdict: no corpus can pass or fail a census."
        " Cleanliness is graded in section 2, extraction integrity in section 3</sup>"
    )


def s1_catalog_figure(ctx: CorporaContext) -> tuple[go.Figure, dict[str, Any]]:
    """Section 1 exhibit: the corpus catalog.

    Input: the shared :class:`CorporaContext`. Output: ``(figure, stats)``
    where the figure is a table (one row per corpus: name, location,
    generator, role, emotions, texts) and ``stats["lines"]`` is the printed
    record (the totals line plus the combined-corpus shape caveat).
    ``stats["total_texts"]`` carries the headline count.
    """
    columns = [
        [row.label for row in ctx.catalog],
        [row.location for row in ctx.catalog],
        [row.generator for row in ctx.catalog],
        [row.role for row in ctx.catalog],
        [row.n_groups for row in ctx.catalog],
        [f"{row.n_texts:,}" for row in ctx.catalog],
    ]
    fig = go.Figure(
        go.Table(
            # widths track the longest string each column actually holds, so the
            # results/ paths and role sentences do not truncate
            columnwidth=[1.5, 3.3, 1.5, 2.2, 0.55, 0.6],
            header=dict(
                values=[
                    "corpus",
                    "location<br>(results/ path or HF dataset id)",
                    "generator<br>(who WROTE it)",
                    "role<br>(which experiments consume it)",
                    "emotions",
                    "texts",
                ],
                align="left",
                font=dict(size=11),
                height=34,
            ),
            cells=dict(values=columns, align="left", height=26, font=dict(size=11)),
        )
    )
    fig.update_layout(
        title=dict(text=_s1_title(ctx), font=dict(size=13)),
        height=490,
        width=1320,
        margin=dict(t=160, b=10),
    )
    total_texts = sum(row.n_texts for row in ctx.catalog)
    generator_models = sorted({row.generator.split(" (")[0] for row in ctx.catalog})
    lines = [
        f"catalog totals: {len(ctx.catalog)} corpora, {total_texts:,} texts,"
        f" {len(generator_models)} generator models ({', '.join(generator_models)})",
        (
            f"combined (Q3) corpus shape caveat: its file groups stories by emotion triple "
            f"({ctx.combined_triples} triples), so its emotions column counts distinct "
            f"emotions across triples, not rows of the file"
        ),
    ]
    return fig, {
        "lines": lines,
        "total_texts": total_texts,
        "n_corpora": len(ctx.catalog),
        "generator_models": generator_models,
    }
