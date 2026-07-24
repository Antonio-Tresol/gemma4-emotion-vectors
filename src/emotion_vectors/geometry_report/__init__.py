"""Notebook-02 exhibit library: the emotion-circumplex geometry report.

Every code cell in ``notebooks/02_circumplex_geometry.ipynb`` is
load-call-show over this package. ``load_geometry_context`` owns the former
section-1 cell (post-fix bundles, NRC VAD match, one deterministic PCA per
model per layer); one figure builder per section returns ``(figure, stats)``
where ``stats["lines"]`` holds every line the notebook prints, so the printed
record and the figure come from one computation. Keeping the analysis
importable means each exhibit can be tested by importing a function and
calling it with a context instead of executing the notebook.

The two load-bearing verification blocks stay visible in the notebook as
asserts (its verification contract); this package only feeds them:
``e4b_deviations`` for the E4b padding-fix cross-check, and the
``GeometryContext.plane`` record for the instruct valence-best anchor.

Submodules: ``_context`` (shared state), ``_replication`` (sections 1, 2, 5),
``_similarity`` (sections 3, 4, 6), ``_tuning`` (section 7),
``_displacement`` (sections 8-9, TREE Q1.H1.E3/E4), and ``_parity``
(section 10, the logit-lens read of the paper's Table 1, folded in from the
paper-parity notebook).

Number-identity contract: the computations are line-for-line ports of the
formerly inline notebook cells, so re-running the notebook reproduces the
committed printed numbers and figure data exactly (deterministic PCA via
``svd_solver="full"``; fixed k-means/t-SNE seeds).
"""

from ._context import (
    LATE_BAND,
    MID_BAND,
    GeometryContext,
    LayerPlane,
    e4b_deviations,
    load_geometry_context,
    nrc_vad_lexicon_path,
    repo_root,
)
from ._displacement import (
    corpus_mean_story_lengths,
    load_dominance,
    s8_displacement_figure,
    s9_fragmentation_figure,
)
from ._parity import (
    JUDGED_AFFECTIVE,
    LENS_DEFAULT_LAYER,
    LENS_FILES,
    s10_logit_lens_figure,
)
from ._replication import s1_replication_figure, s2_circumplex_figure, s5_loadings_figure
from ._similarity import (
    centered_cosine,
    s3_similarity_figure,
    s4_families_figure,
    s6_rsa_figure,
)
from ._tuning import s7_tuning_stats

__all__ = [
    "JUDGED_AFFECTIVE",
    "LATE_BAND",
    "LENS_DEFAULT_LAYER",
    "LENS_FILES",
    "MID_BAND",
    "GeometryContext",
    "LayerPlane",
    "centered_cosine",
    "corpus_mean_story_lengths",
    "e4b_deviations",
    "load_dominance",
    "load_geometry_context",
    "nrc_vad_lexicon_path",
    "repo_root",
    "s10_logit_lens_figure",
    "s1_replication_figure",
    "s2_circumplex_figure",
    "s3_similarity_figure",
    "s4_families_figure",
    "s5_loadings_figure",
    "s6_rsa_figure",
    "s7_tuning_stats",
    "s8_displacement_figure",
    "s9_fragmentation_figure",
]
