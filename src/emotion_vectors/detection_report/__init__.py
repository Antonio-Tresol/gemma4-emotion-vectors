"""Notebook-03 exhibit library: the detection-probe campaign.

Every code cell in ``notebooks/03_detection_probe_campaign.ipynb`` is
load-call-show over this package: :func:`load_sweep_context` resolves both
models' probe bundles and swept battery activations through
``emotion_vectors.artifacts.fetch`` (local ``results/`` first, then the
published Hugging Face datasets), and one builder per section returns
``(figure, stats)`` where ``stats["lines"]`` holds every number the notebook
prints. Keeping the analysis importable means each exhibit can be tested by
importing a function and calling it with parameters instead of executing the
notebook.

What deliberately does NOT live here: the registered pass rules. The
dual-battery rule ("at least 8 of 12 on BOTH batteries") and the archival
reproduction asserts are applied in the notebook cells, in plain sight, so a
reader watches the rule being applied to the numbers rather than taking a
package function's word for it. This package computes counts; the notebook
grades them.

Two centering conventions coexist in this notebook and must not be unified;
:mod:`._data` documents which sections use which and why.

Sections 8 and 9 (:mod:`._preferences`) are a later addition and ask a
different question from the detection campaign: they are the two halves of the
source paper's Figure 4, preference prediction and causal steering, folded in
from the paper-parity notebook. They read the scorer's registered verdicts
from ``results/`` rather than recomputing them.
"""

from ._data import (
    BASE_LABEL,
    BATTERY_ROLES,
    BATTERY_SHORT,
    CHANCE_CORRECT,
    IT_LABEL,
    N_EMOTIONS,
    N_SCENARIOS,
    PASS_BAR,
    READOUT_LABELS,
    READOUTS,
    REGISTERED_LAYER,
    TOP_K,
    ModelSweep,
    SweepContext,
    battery_counts,
    figure_title,
    load_sweep_context,
)
from ._intensity import (
    CHANCE_DIRECTIONS,
    REGISTERED_DIRECTIONS,
    TemplateCurve,
    direction_scores,
    intensity_curves,
    intensity_figure,
)
from ._preferences import (
    PAPER_COUPLING_R,
    PAPER_MEAN_SHIFTS,
    PAPER_PROBE_ELO_RANGE,
    ScoredArm,
    instrument_sensitivity_stats,
    load_preference_arms,
    load_steering_arms,
    preference_figure,
    steering_figure,
)
from ._projection import projection_probes, projection_table
from ._scale import (
    ScaleCorpus,
    best_scores,
    convergence_cosines,
    convergence_figure,
    describe_configuration,
    load_scale_corpus,
    scale_figure,
    scale_rows,
)
from ._selfgen import direction_similarity, leakage_report, load_probe_sets, rank_table_lines
from ._sweep import combination_rows, sweep_figure, sweep_grids

__all__ = [
    "BASE_LABEL",
    "BATTERY_ROLES",
    "BATTERY_SHORT",
    "CHANCE_CORRECT",
    "CHANCE_DIRECTIONS",
    "IT_LABEL",
    "N_EMOTIONS",
    "N_SCENARIOS",
    "PAPER_COUPLING_R",
    "PAPER_MEAN_SHIFTS",
    "PAPER_PROBE_ELO_RANGE",
    "PASS_BAR",
    "READOUTS",
    "READOUT_LABELS",
    "REGISTERED_DIRECTIONS",
    "REGISTERED_LAYER",
    "TOP_K",
    "ModelSweep",
    "ScaleCorpus",
    "ScoredArm",
    "SweepContext",
    "TemplateCurve",
    "battery_counts",
    "best_scores",
    "combination_rows",
    "convergence_cosines",
    "convergence_figure",
    "describe_configuration",
    "direction_scores",
    "direction_similarity",
    "figure_title",
    "instrument_sensitivity_stats",
    "intensity_curves",
    "intensity_figure",
    "leakage_report",
    "load_preference_arms",
    "load_probe_sets",
    "load_scale_corpus",
    "load_steering_arms",
    "load_sweep_context",
    "preference_figure",
    "projection_probes",
    "projection_table",
    "rank_table_lines",
    "scale_figure",
    "scale_rows",
    "steering_figure",
    "sweep_figure",
    "sweep_grids",
]
