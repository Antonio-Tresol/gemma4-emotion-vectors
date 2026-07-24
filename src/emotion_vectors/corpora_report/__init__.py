"""Notebook-01 exhibit library: the corpora-and-extraction catalog report.

Every code cell in ``notebooks/01_corpora_and_extraction.ipynb`` is
load-call-show over this package. ``load_corpora_context`` owns the loading:
it resolves every story corpus through ``emotion_vectors.artifacts.fetch``
(local ``results/`` first, the published Hugging Face dataset otherwise) and
loads the reference corpus from its Hugging Face rows, so the catalog table,
the leakage audit, and the printed record all count the same files. One
figure builder per section returns ``(figure, stats)`` where
``stats["lines"]`` holds every line the notebook prints, so the printed
record and the figure come from one computation.

The load-bearing verification stays visible in the notebook as asserts (its
verification contract); this package only feeds them: the reference-corpus
and triple-count anchors after ``load_corpora_context``, and the
zero-extraction-errors anchor after ``s3_bundles_figure``.

Submodules: ``_data`` (shared context), ``_catalog`` (section 1),
``_leakage`` (section 2), ``_bundles`` (section 3).

Number-identity contract: the computations are ports of the formerly inline
notebook cells (pure counting, no randomness), so re-running the notebook
reproduces the committed printed numbers and table values exactly.
"""

from ._bundles import POSTFIX_BUNDLE_NAMES, s3_bundles_figure
from ._catalog import s1_catalog_figure
from ._data import (
    REFERENCE_CORPUS_ID,
    CorporaContext,
    CorpusEntry,
    load_corpora_context,
)
from ._leakage import s2_leakage_figure

__all__ = [
    "POSTFIX_BUNDLE_NAMES",
    "REFERENCE_CORPUS_ID",
    "CorporaContext",
    "CorpusEntry",
    "load_corpora_context",
    "s1_catalog_figure",
    "s2_leakage_figure",
    "s3_bundles_figure",
]
