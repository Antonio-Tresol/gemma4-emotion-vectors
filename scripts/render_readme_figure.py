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

OUT = Path(__file__).resolve().parent.parent / "img" / "hero.png"
PANEL_WIDTH = 1500
PANEL_HEIGHT = 760
SCALE = 2
TRACKING_ARMS = ["it_v2", "it", "deepseek", "deepseek_constant", "base"]


def still(figure) -> None:
    """Strip the interactive controls a static export cannot honour.

    Direct assignment, not ``update_layout(sliders=[])``: plotly reads the empty
    list there as "no change" and silently keeps the slider.
    """
    figure.layout.sliders = ()
    figure.layout.updatemenus = ()
    figure.frames = ()
    figure.update_layout(margin=dict(l=70, r=70, t=120, b=60))


def render(figure, path: Path) -> None:
    still(figure)
    figure.write_image(str(path), width=PANEL_WIDTH, height=PANEL_HEIGHT, scale=SCALE)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    top = OUT.parent / "_geometry.png"
    bottom = OUT.parent / "_tracking.png"

    geometry, _ = s2_circumplex_figure(load_geometry_context())
    render(geometry, top)
    print(f"  rendered the geometry panel -> {top.name}")

    tracking, _ = s1_top1_figure(load_arms(TRACKING_ARMS))
    render(tracking, bottom)
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
