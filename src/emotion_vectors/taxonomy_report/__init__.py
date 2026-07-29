"""Notebook-11 exhibit library: the Q3.H1.E2/E3 category-taxonomy figures.

Every code cell in ``notebooks/11_tracking_taxonomy.ipynb`` is load-call-show
over this package: ``load_arms`` / ``load_vad`` / ``load_has_nonaffect``
resolve the inputs, the data helpers slice the per-record dumps, and one
figure builder per section returns ``(figure, stats)`` where ``stats`` holds
every number the notebook prints. Keeping the analysis importable means each
exhibit can be tested by importing a function and calling it with parameters
instead of executing the notebook.

Scoring conventions are bit-identical to the registered scorer
(``scripts/score_q3_gate_r1.py``) via the shared
``emotion_vectors.q3_conventions`` module, which produced the record dumps
read here (``scripts/dump_q3_records.py``): centered cosine with the
story-set mean and ``norms_centered`` denominator, SEQUENTIAL stories only,
phase means, and the W=16 boundary-referenced anticipation lead. Nothing in
this package re-scores; it only slices records.

Reproducibility: every confidence interval is a cluster bootstrap over
``triple_id`` with ``N_BOOT`` resamples drawn from one shared numpy
``Generator``. The notebook creates ``rng = np.random.default_rng(SEED)``
once and threads it through the section builders in notebook order
(S2, S3, S4, S5), so the bootstrap CIs reproduce exactly.
"""

from ._data import (
    FAMILIES,
    LAYERS,
    N_BOOT,
    PRIMARY_LAYER,
    S3_VALENCE_GAP_EDGES,
    S7_CATEGORIES,
    SEED,
    Arm,
    Arms,
    ConfidenceInterval,
    LayerStats,
    LeadRead,
    RankRead,
    Row,
    Vad,
    cluster_boot_ci,
    layer_slider,
    load_arms,
    load_has_nonaffect,
    load_vad,
    probe_set_columns,
    repo_root,
    tagged_ranks,
    true_leads,
)
from ._s1_s2_emotions import s1_top1_figure, s2_family_figure
from ._s3_s4_distance import s3_affective_distance_figure, s4_confusion_figure
from ._s5_crossarm import cross_arm_lines, s5_position_nonaffect_figure
from ._s6_s7_confusions import confusion, s6_named_confusions, s7_thirds_figure, third_match
from ._s8_geometry import rank_partial, s8_geometry_figure, selfgen_probe_cosines

__all__ = [
    "FAMILIES",
    "LAYERS",
    "N_BOOT",
    "PRIMARY_LAYER",
    "S3_VALENCE_GAP_EDGES",
    "S7_CATEGORIES",
    "SEED",
    "Arm",
    "Arms",
    "ConfidenceInterval",
    "LayerStats",
    "LeadRead",
    "RankRead",
    "Row",
    "Vad",
    "cluster_boot_ci",
    "confusion",
    "cross_arm_lines",
    "layer_slider",
    "load_arms",
    "load_has_nonaffect",
    "load_vad",
    "probe_set_columns",
    "rank_partial",
    "repo_root",
    "s1_top1_figure",
    "s2_family_figure",
    "s3_affective_distance_figure",
    "s4_confusion_figure",
    "s5_position_nonaffect_figure",
    "s6_named_confusions",
    "s7_thirds_figure",
    "s8_geometry_figure",
    "selfgen_probe_cosines",
    "tagged_ranks",
    "third_match",
    "true_leads",
]
