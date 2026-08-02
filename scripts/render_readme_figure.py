"""Render the README's hero image from the notebook's own figure code.

The README should not carry a screenshot. This exports the same
`s2_circumplex_figure` the report notebook renders, so the picture at the top
of the repository cannot drift from the analysis behind it: re-run this after
any change to the geometry package and the image follows.

The interactive layer slider is stripped, because a static PNG cannot be
dragged and leaving it in implies otherwise. Layer 33 is the figure's own
default and the layer every published geometry number is quoted at.
"""

from __future__ import annotations

import sys
from pathlib import Path

from emotion_vectors.geometry_report import load_geometry_context, s2_circumplex_figure

OUT = Path(__file__).resolve().parent.parent / "img" / "circumplex.png"


def main() -> int:
    figure, _record = s2_circumplex_figure(load_geometry_context())
    # Drop the slider and the frames it drives: meaningless in a still image.
    # Direct assignment, not update_layout(sliders=[]) - plotly treats the empty
    # list there as "no change" and silently keeps the slider.
    figure.layout.sliders = ()
    figure.layout.updatemenus = ()
    figure.frames = ()
    figure.update_layout(margin=dict(l=70, r=70, t=110, b=60))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.write_image(str(OUT), width=1500, height=760, scale=2)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
