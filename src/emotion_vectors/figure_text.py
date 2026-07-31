"""Title wrapping shared by every report package's figures.

Plotly never wraps a title, it just runs off the canvas, so every title in the
report packages is wrapped by ``figure_title`` before it is set.

This lived as three byte-identical copies in ``detection_report/_data.py``,
``lineage_report/_data.py`` and ``transition_report/_data.py``. Three copies of
a wrapping rule are three chances for one figure's titles to start wrapping
differently from its neighbours' after somebody tunes a constant, which is the
kind of drift nobody notices in a rendered PNG. One definition here removes
that.
"""

from __future__ import annotations

import textwrap

# Characters that fit per pixel of figure width, measured from rendered PNGs at
# title_font_size 13: the headline renders at full size, the subtitle lines
# inside <sup> render smaller and fit more.
HEADLINE_CHARS_PER_PIXEL = 0.125
SUBTITLE_CHARS_PER_PIXEL = 0.17


def figure_title(headline: str, subtitles: list[str], width_px: int) -> str:
    """Wrap a question-form headline plus subtitle lines to the figure width.

    Inputs: ``headline`` (the question and its answer, headline values already
    formatted in), ``subtitles`` (each becomes one smaller line, in order:
    what one mark is, then the anchors and the evidence file), and
    ``width_px`` (the figure's ``width``, which decides how many characters
    fit on a line). Returns the ``<br>``-joined title string.
    """
    if width_px <= 0:
        raise ValueError(f"figure width must be positive, got {width_px}")
    lines = textwrap.wrap(headline, width=int(width_px * HEADLINE_CHARS_PER_PIXEL))
    for subtitle in subtitles:
        wrapped = textwrap.wrap(subtitle, width=int(width_px * SUBTITLE_CHARS_PER_PIXEL))
        lines += [f"<sup>{line}</sup>" for line in wrapped]
    return "<br>".join(lines)
