"""Notebook-10 exhibit library: the sprint-story review figures and printouts.

Every act-level code cell in ``notebooks/10_the_sprint_story.ipynb`` is
load-call-show over this package: the cell loads the evidence files (through
``emotion_vectors.artifacts.fetch``), calls one builder here, prints the
returned record, and shows the returned figure. Each figure builder returns
``(figure, stats)`` where ``stats["lines"]`` carries, in order, exactly the
lines the notebook prints, so the printed record is reproducible by importing
the function and calling it with the same evidence dicts. The two addendum
builders (``act1_addendum_lines``, ``act6_addendum_lines``) have no figure and
return the stats dict alone.

Nothing here re-scores: every function formats and plots numbers already
present in committed evidence files (the Act VI addendum slices the published
per-record score dump, computing only rates, means, and distances over it).
"""

from ._act1_geometry import act1_addendum_lines, act1_geometry_figure
from ._act4_act5 import act4_behavior_figure, act5_lineage_figure
from ._act6_dynamics import (
    BANK_LABEL,
    BANK_SIZE,
    BANKS,
    LAYERS_Q3,
    PRIMARY_LAYER_Q3,
    act6_addendum_lines,
    act6_factorial_figure,
)

__all__ = [
    "BANK_LABEL",
    "BANK_SIZE",
    "BANKS",
    "LAYERS_Q3",
    "PRIMARY_LAYER_Q3",
    "act1_addendum_lines",
    "act1_geometry_figure",
    "act4_behavior_figure",
    "act5_lineage_figure",
    "act6_addendum_lines",
    "act6_factorial_figure",
]
