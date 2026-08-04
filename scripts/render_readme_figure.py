"""Render the README's hero image from the notebooks' own figure code.

The README should not carry a screenshot. This calls the same figure builders
the report notebooks call, so the picture at the top of the repository cannot
drift from the analysis behind it: re-run this after any change to the figure
packages and the image follows.

Two panels, stacked, because the project has two results worth seeing at once:

  top     s2_circumplex_figure  the geometry, base against instruction-tuned
  bottom  s1_top1_figure        the emotion tracking, and how much it depends
                                on who wrote the stories

The interactive layer slider is stripped from both. A still image cannot be
dragged, and leaving the control in implies otherwise. Layer 33 is each
figure's own default and the layer the published numbers are quoted at.

    uv run python scripts/render_readme_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from emotion_vectors.geometry_report import load_geometry_context, s2_circumplex_figure
from emotion_vectors.taxonomy_report import load_arms, s1_top1_figure

# Under docs/ because GitHub Pages serves this site from that folder, so an
# image anywhere else is a 404 on the live page. One copy feeds both the README
# and the write-up rather than two that can drift.
OUT = Path(__file__).resolve().parent.parent / "docs" / "img" / "hero.png"
PANEL_WIDTH = 1500
PANEL_HEIGHT = 780
SCALE = 2
TRACKING_ARMS = ["it_v2", "it", "deepseek", "deepseek_constant", "base"]

# One line each. The figures' own subtitles run to five lines of small text and
# assume a notebook reader; everything they say is in the README caption.
# Neutral, because the panels are ordered instruct-then-base: a title claiming
# "recovered, then demoted" would run against the order the eye meets them in.
# The README caption names which panel is which and carries the argument.
GEOMETRY_TITLE = "Where each model keeps its emotion circumplex, at layer 33"
TRACKING_TITLE = "How often each emotion is identified while its story is read, at layer 33"


def still(figure, title: str) -> None:
    """Make a notebook figure legible as a still image.

    Two things have to go. The interactive controls, because a still image
    cannot be dragged and leaving the slider in implies otherwise. And the
    notebook subtitle, which is four or five lines of small text written for a
    reader who has the surrounding narrative: in a static export it collides
    with the subplot titles, and some of it is meaningless anyway ("the slider
    picks the layer"). The README caption carries that detail instead, where it
    can be read as prose.

    Direct assignment, not ``update_layout(sliders=[])``: plotly reads the empty
    list there as "no change" and silently keeps the slider.
    """
    figure.layout.sliders = ()
    figure.layout.updatemenus = ()
    figure.frames = ()
    figure.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=27)),
        margin=dict(l=80, r=80, t=130, b=90),
    )


def render(figure, title: str, path: Path) -> None:
    still(figure, title)
    figure.write_image(str(path), width=PANEL_WIDTH, height=PANEL_HEIGHT, scale=SCALE)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    top = OUT.parent / "_geometry.png"
    bottom = OUT.parent / "_tracking.png"

    geometry, _ = s2_circumplex_figure(load_geometry_context())
    render(geometry, GEOMETRY_TITLE, top)
    print(f"  rendered the geometry panel -> {top.name}")

    tracking, _ = s1_top1_figure(load_arms(TRACKING_ARMS))
    render(tracking, TRACKING_TITLE, bottom)
    print(f"  rendered the tracking panel -> {bottom.name}")

    images = [Image.open(top), Image.open(bottom)]
    width = max(image.width for image in images)
    stacked = Image.new("RGB", (width, sum(image.height for image in images)), "white")
    offset = 0
    for image in images:
        stacked.paste(image, ((width - image.width) // 2, offset))
        offset += image.height
    stacked.save(OUT, optimize=True)
    for image in images:
        image.close()
    top.unlink()
    bottom.unlink()

    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {stacked.width}x{stacked.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
