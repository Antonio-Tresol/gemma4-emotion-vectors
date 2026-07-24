"""Shared state for the corpora report: every story corpus, loaded once.

``load_corpora_context`` builds one :class:`CorporaContext` that the three
section exhibits read from, so the catalog table, the leakage audit, and the
printed record are guaranteed to count the same files. Every corpus resolves
through ``emotion_vectors.artifacts.fetch`` (local ``results/`` first, the
published Hugging Face dataset otherwise); the reference corpus, which has no
``results/`` copy, loads straight from its Hugging Face rows.

Fail-loud contract: a missing corpus raises, with one sanctioned exception:
the diverse DeepSeek corpus may live only on Hugging Face, and when it cannot
be fetched the context records a printed degradation naming its dataset repo
instead of silently dropping the row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from emotion_vectors.artifacts import ROUTES, fetch
from emotion_vectors.corpus import load_emotions_data

# The open replication's story corpus (written by gemma-4-4B), the extraction
# source for all corpus-lineage vectors. Not ours; lives only on Hugging Face.
REFERENCE_CORPUS_ID = "snae/emotion_stories_gemma_4_4B"


@dataclass(frozen=True)
class CorpusEntry:
    """One catalog row: a corpus, its provenance, and its measured size.

    ``n_groups`` counts the jsonl rows: one per emotion for every corpus
    except the combined (Q3) corpus, whose rows group stories by emotion
    TRIPLE; there ``n_groups`` counts distinct emotions across triples and
    ``n_triples`` (on the context) counts the rows themselves.
    """

    key: str  # short internal key the leakage section selects rows by
    label: str  # plain-words name shown in every figure
    location: str  # results/-relative path, or the HF dataset id for the reference corpus
    generator: str  # who wrote the texts (model, ours vs the reference authors)
    role: str  # which experiments consume this corpus
    n_groups: int  # emotions covered (see docstring for the combined corpus)
    n_texts: int  # story/dialogue texts in the file
    grouped_path: Path | None  # local grouped jsonl (None: reference corpus, HF rows only)


@dataclass(frozen=True)
class CorporaContext:
    """Everything the three corpora-report sections read.

    Attributes:
        published: the reference corpus as ``{emotion: [story, ...]}``.
        catalog: one :class:`CorpusEntry` per corpus, table order.
        combined_triples: rows (emotion triples) in the combined Q3 corpus.
        diverse_available: whether the diverse DeepSeek corpus resolved.
        lines: the load record the notebook prints (one line per corpus,
            plus the degradation line when a corpus did not resolve).
    """

    published: dict[str, list[str]]
    catalog: list[CorpusEntry]
    combined_triples: int
    diverse_available: bool
    lines: list[str] = field(default_factory=list)

    def entry(self, key: str) -> CorpusEntry:
        """The catalog row with this key; loud KeyError when absent."""
        by_key = {row.key: row for row in self.catalog}
        if key not in by_key:
            raise KeyError(f"no corpus with key {key!r}; have {sorted(by_key)}")
        return by_key[key]


def _grouped_corpus_size(grouped_jsonl: Path) -> tuple[int, int]:
    """(n emotion groups, n texts) for a grouped jsonl whose rows hold one
    emotion each with a plain-string ``stories`` list."""
    rows = [json.loads(line) for line in open(grouped_jsonl)]
    return len(rows), sum(len(row["stories"]) for row in rows)


# Every corpus we generated that is counted the plain way (one jsonl row per
# emotion): (catalog key, plain-words label, results-relative path, generator,
# role). Catalog order is table order.
PLAIN_CORPORA = [
    (
        "self_stories",
        "self stories, instruct",
        "self_stories_it/dialogues_grouped.jsonl",
        "gemma-4-31b-it (ours)",
        "scale test E6; self-generated lineage E10/E11",
    ),
    (
        "dialogues_base",
        "dialogues, base model",
        "dialogue_stories/dialogues_grouped.jsonl",
        "gemma-4-31b base (ours)",
        "dialogue-transfer E3",
    ),
    (
        "dialogues_it",
        "dialogues, instruct",
        "dialogue_stories_it/dialogues_grouped.jsonl",
        "gemma-4-31b-it (ours)",
        "instruct dialogue pilot E5",
    ),
    (
        "neutral",
        "neutral transcripts",
        "neutral_transcripts_it/dialogues_grouped.jsonl",
        "gemma-4-31b-it (ours)",
        "confound projection E7",
    ),
    (
        "deepseek_fixed",
        "DeepSeek, fixed prompt",
        "openrouter_stories/stories_grouped.jsonl",
        "deepseek-v4-pro (ours)",
        "external-generator lineage E11",
    ),
]
# The diverse DeepSeek arm, kept separate because it is the one corpus allowed
# to be missing locally (printed degradation instead of a raise).
DIVERSE_CORPUS = (
    "deepseek_diverse",
    "DeepSeek, diverse prompts",
    "openrouter_stories_diverse/stories_grouped.jsonl",
    "deepseek-v4-pro (ours)",
    "diverse-prompt scale arm E12",
)


def _plain_entry(key: str, label: str, relpath: str, generator: str, role: str) -> CorpusEntry:
    """One catalog row for a plainly grouped corpus, sized from its own file."""
    grouped_path = fetch(relpath)
    n_groups, n_texts = _grouped_corpus_size(grouped_path)
    return CorpusEntry(
        key=key,
        label=label,
        location=f"results/{relpath}",
        generator=generator,
        role=role,
        n_groups=n_groups,
        n_texts=n_texts,
        grouped_path=grouped_path,
    )


def _reference_entry(published: dict[str, list[str]]) -> CorpusEntry:
    """The reference corpus row, sized from its Hugging Face rows (no results/ copy)."""
    return CorpusEntry(
        key="published",
        label="published stories (reference)",
        location=REFERENCE_CORPUS_ID,
        generator="gemma-4-4B (reference authors)",
        role="probe extraction, both reader models",
        n_groups=len(published),
        n_texts=sum(len(stories) for stories in published.values()),
        grouped_path=None,
    )


def _combined_entry() -> tuple[CorpusEntry, int]:
    """The combined (Q3) corpus row plus its triple count.

    This file groups stories by emotion TRIPLE rather than by single emotion,
    so it needs its own counter: ``n_groups`` counts distinct emotions across
    the triples, and the returned int counts the triples (the file's rows).
    """
    combined_path = fetch("combined_stories/stories_grouped.jsonl")
    combined_rows = [json.loads(line) for line in open(combined_path)]
    n_triples = len(combined_rows)
    entry = CorpusEntry(
        key="combined",
        label="combined stories (Q3)",
        location="results/combined_stories/stories_grouped.jsonl",
        generator="gemma-4-31b-it (ours)",
        role=f"Q3 transition trajectories ({n_triples} emotion triples)",
        n_groups=len({emotion for row in combined_rows for emotion in row["emotions"]}),
        n_texts=sum(len(row["stories"]) for row in combined_rows),
        grouped_path=combined_path,
    )
    return entry, n_triples


def load_corpora_context() -> CorporaContext:
    """Load every story corpus in the project and return the shared context.

    Inputs: nothing (all paths resolve through ``fetch``; the reference
    corpus through its Hugging Face dataset id). Output: a
    :class:`CorporaContext` whose ``lines`` are the notebook's load record.
    Raises on any missing corpus except the diverse DeepSeek arm (printed
    degradation, see module docstring).
    """
    lines: list[str] = []
    published = load_emotions_data(REFERENCE_CORPUS_ID, "train")
    catalog: list[CorpusEntry] = [_reference_entry(published)]

    # Table order: reference, our four plainly grouped corpora, the combined
    # (Q3) corpus, then the two external-generator arms.
    for key, label, relpath, generator, role in PLAIN_CORPORA[:4]:
        catalog.append(_plain_entry(key, label, relpath, generator, role))
    combined_entry, combined_triples = _combined_entry()
    catalog.append(combined_entry)
    catalog.append(_plain_entry(*PLAIN_CORPORA[4]))

    # The diverse DeepSeek corpus may live only on HF; degrade loudly, not silently.
    diverse_available = True
    try:
        catalog.append(_plain_entry(*DIVERSE_CORPUS))
    except Exception as fetch_error:
        diverse_available = False
        lines.append(
            f"DEGRADED: diverse DeepSeek corpus not fetchable here "
            f"({type(fetch_error).__name__}); it lives on HF at "
            f"{ROUTES['openrouter_stories_diverse']}"
        )

    lines.extend(
        f"{row.label}: {row.n_groups} emotions, {row.n_texts:,} texts ({row.location})"
        for row in catalog
    )
    return CorporaContext(
        published=published,
        catalog=catalog,
        combined_triples=combined_triples,
        diverse_available=diverse_available,
        lines=lines,
    )
