"""Notebook-08 exhibit library: the Q3 transition-tracking first tranche.

Every code cell in ``notebooks/08_transition_tracking_first_reads.ipynb`` is
load-call-show over this package: :func:`load_transition_evidence` resolves the
frozen identity / anticipation evidence JSONs through
``emotion_vectors.artifacts.fetch`` (local ``results/`` first, then the
published Hugging Face datasets), and one builder per section returns
``(figure, stats)`` where ``stats["lines"]`` holds every number the notebook
prints. Keeping the analysis importable means each exhibit can be tested by
importing a function and calling it with parameters instead of executing the
notebook.

Nothing here re-scores: the package only slices the evidence written by the
registered scorer ``scripts/score_q3_gate_r1.py``.

Two things this package refuses to let the notebook get wrong:

* **Layout follows the evidence.** Every grid shape (panel count, heatmap rows
  and columns, dots per slider step) is derived from the story sets and probe
  sets that actually loaded, and each builder returns that shape in its stats so
  the notebook can assert it in plain sight. A typed column count once shipped
  this notebook with a stored exception when a fourth story set arrived.
* **A probe set and a story set never share a name.** "DeepSeek" names one of
  each here, so no label in this package is the bare word:
  a legend says "DeepSeek probes (12, DeepSeek-written stories)" and a row says
  "reading DeepSeek-written stories".
"""

from ._data import (
    CHANCE_FRACTION,
    CHANCE_LIKE_FRACTION,
    IDENTITY_BAR_FRACTION,
    LAYERS,
    NULL_PROBE_SET,
    PRIMARY_LAYER,
    PROBE_SET_COLORS,
    PROBE_SET_LABEL,
    PROBE_SET_ORDER,
    PROBE_SET_SHORT,
    PROBE_SET_SIZE,
    PROBE_SET_TICK_LABEL,
    PROBED_MODEL,
    R1_BAR_IN_NOISE_SD,
    StorySet,
    TransitionEvidence,
    figure_title,
    load_transition_evidence,
    random_null_lines,
)
from ._dissociation import dissociation_figure, dissociation_panels
from ._grid import design_grid
from ._identity import identity_figure, identity_rows
from ._zones import zone_figure, zone_points

__all__ = [
    "CHANCE_FRACTION",
    "CHANCE_LIKE_FRACTION",
    "IDENTITY_BAR_FRACTION",
    "LAYERS",
    "NULL_PROBE_SET",
    "PRIMARY_LAYER",
    "PROBED_MODEL",
    "PROBE_SET_COLORS",
    "PROBE_SET_LABEL",
    "PROBE_SET_ORDER",
    "PROBE_SET_SHORT",
    "PROBE_SET_SIZE",
    "PROBE_SET_TICK_LABEL",
    "R1_BAR_IN_NOISE_SD",
    "StorySet",
    "TransitionEvidence",
    "design_grid",
    "dissociation_figure",
    "dissociation_panels",
    "figure_title",
    "identity_figure",
    "identity_rows",
    "load_transition_evidence",
    "random_null_lines",
    "zone_figure",
    "zone_points",
]
