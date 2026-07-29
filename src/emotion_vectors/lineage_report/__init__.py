"""Notebook-07 exhibit library: the E11/E12 story-source figures.

The question these figures answer is whose stories the probes were built
from, and whether that choice changes what the probes measure.

Every code cell in ``notebooks/07_generator_lineages.ipynb`` is
load-call-show over this package: ``load_evidence`` / ``load_corpora``
resolve the frozen evidence JSONs and the three story corpora through
``emotion_vectors.artifacts.fetch`` (local ``results/`` first, then the
published Hugging Face datasets), and one builder per section returns
``(figure, stats)`` where ``stats["lines"]`` holds every number the
notebook prints. Keeping the analysis importable means each exhibit can be
tested by importing a function and calling it with parameters instead of
executing the notebook.

Nothing here re-scores: the package only slices the evidence written by the
registered scorers (``scripts/score_e11_lineage.py``,
``scripts/score_e12_scale.py``) and the E4b
extraction-format audit (``results/raw_vs_chat_extraction_cosine.json``).
"""

from ._caveat import extraction_format_figure
from ._data import (
    CHANCE_CORRECT,
    GEOMETRY_LAYER,
    N_EMOTIONS,
    PASS_BAR,
    PROBED_MODEL,
    REQUESTED_PER_EMOTION,
    STORY_SOURCE_COLORS,
    STORY_SOURCE_LABELS,
    STORY_SOURCE_ORDER,
    TOP_K,
    StorySourceEvidence,
    load_corpora,
    load_evidence,
    per_layer_counts,
    seed_mean_by_n,
)
from ._e11 import corpus_overview, detection_figure, diversity_figure, geometry_figure
from ._e12 import dose_response_figure, preference_figure
from ._verdict import verdict_stats

__all__ = [
    "CHANCE_CORRECT",
    "GEOMETRY_LAYER",
    "N_EMOTIONS",
    "PASS_BAR",
    "PROBED_MODEL",
    "REQUESTED_PER_EMOTION",
    "STORY_SOURCE_COLORS",
    "STORY_SOURCE_LABELS",
    "STORY_SOURCE_ORDER",
    "TOP_K",
    "StorySourceEvidence",
    "corpus_overview",
    "detection_figure",
    "diversity_figure",
    "dose_response_figure",
    "extraction_format_figure",
    "geometry_figure",
    "load_corpora",
    "load_evidence",
    "per_layer_counts",
    "preference_figure",
    "seed_mean_by_n",
    "verdict_stats",
]
