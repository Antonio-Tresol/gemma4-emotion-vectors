"""Data resolution and shared vocabulary for the notebook-07 exhibit library.

Notebook 07 reports E11 (does probe quality come from generator identity or
generator quality?) and E12 (scale and diversity dose-response), plus the
E4b extraction-format caveat. This module resolves every input through
``emotion_vectors.artifacts.fetch`` (local ``results/`` first, then the
published Hugging Face datasets), so the notebook runs unchanged on any
clone, and pins the shared vocabulary: story-source labels that name the ROLE
of each probe set (whose stories the probes were built from), one fixed color
per story source, and the scoring constants of the registered detection read.

Nothing in this package re-scores; every function slices the frozen
evidence JSONs written by ``scripts/score_e11_lineage.py`` and
``scripts/score_e12_scale.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from emotion_vectors.artifacts import fetch

# figure_title and its two wrapping constants live in emotion_vectors.figure_text.
# Three byte-identical copies used to sit in the three report packages; they are
# re-exported here so this package's own modules keep importing them from _data.
from emotion_vectors.figure_text import (
    HEADLINE_CHARS_PER_PIXEL,
    SUBTITLE_CHARS_PER_PIXEL,
    figure_title,
)

PROBED_MODEL = "google/gemma-4-31b-it"

# Constants of the registered detection read (R1, registered in TREE.md
# under Q1.H2.E11 before scoring).
N_EMOTIONS = 12  # candidate emotions ranked for each scenario
N_SCENARIOS = 12  # scenario prompts in each of the two sets
TOP_K = 3  # a scenario is correct if the target emotion ranks in the top 3
PASS_BAR = 8  # the registered bar: >= 8 of 12 correct on BOTH scenario sets
# top-3 of 12 emotions by luck: 3/12 chance per scenario, so 3 of 12 expected
CHANCE_CORRECT = N_SCENARIOS * TOP_K / N_EMOTIONS
GEOMETRY_LAYER = 33  # the registered geometry read (R2) is defined here only

# Canonical story-source order for every figure, and the ROLE each label
# names: the generator that wrote the probe-building stories.
STORY_SOURCE_ORDER = ["selfgen", "weak_external", "fixed_deepseek", "diverse_deepseek"]
STORY_SOURCE_LABELS = {
    "selfgen": "self-generated (stories by the probed model)",
    "weak_external": "weak external (stories by gemma-4-4B)",
    "fixed_deepseek": "fixed DeepSeek (stories by deepseek-v4-pro)",
    "diverse_deepseek": "diverse DeepSeek (deepseek-v4-pro, persona x setting grid)",
}
STORY_SOURCE_COLORS = {
    "selfgen": "#2ca02c",
    "weak_external": "#7f7f7f",
    "fixed_deepseek": "#1f77b4",
    "diverse_deepseek": "#d62728",
}
# Short two-line versions of the labels, for axis ticks where the full role
# sentence would collide with its neighbours. The role still reads off the
# tick ("who wrote the stories"); the full sentence stays in the printed
# record and in the figure subtitles.
STORY_SOURCE_TICK_LABELS = {
    "selfgen": "self-generated<br>(probed model)",
    "weak_external": "weak external<br>(gemma-4-4B)",
    "fixed_deepseek": "fixed DeepSeek<br>(deepseek-v4-pro)",
    "diverse_deepseek": "diverse DeepSeek<br>(deepseek-v4-pro grid)",
}

# Corpus label -> (results/ subtree, grouped-stories file) for the three
# story corpora shown in section 1. The self-generated corpus keeps its
# historical "dialogues_grouped" filename.
CORPUS_FILES = {
    "self-generated (E10)": ("self_stories_it", "dialogues_grouped.jsonl"),
    "fixed DeepSeek (E11)": ("openrouter_stories", "stories_grouped.jsonl"),
    "diverse DeepSeek (E12)": ("openrouter_stories_diverse", "stories_grouped.jsonl"),
}
# Stories the registered generation recipe ASKED FOR per emotion, per corpus
# (scripts/generate_self_stories.py and scripts/generate_openrouter_stories.py,
# the second with --diverse for the grid arm). Kept counts fall below these
# wherever the leakage and terminal-ending filters dropped a story, which is
# why the dose-response curves stop at the kept minimum rather than at these
# round numbers; corpus_overview prints requested against kept.
REQUESTED_PER_EMOTION = {
    "self-generated (E10)": 256,
    "fixed DeepSeek (E11)": 256,
    "diverse DeepSeek (E12)": 1024,
}


@dataclass(frozen=True)
class StorySourceEvidence:
    """The five frozen evidence JSONs notebook 07 reads.

    Fields (each the parsed dict of one committed results file):
      e11          -- results/e11_lineage.json (R1-R4 for the three E11 arms)
      full_grid    -- results/e12_diverse_fullcorpus_grid.json (diverse arm
                      at full corpus, scored through the same R1-R3 slots)
      scale_curves -- results/e12_scale_curve.json (dose-response points for
                      the fixed and diverse arms, 5 seeds per corpus size)
      matched_n    -- results/e12_pref_matched_n_exploratory.json
      raw_vs_chat  -- results/raw_vs_chat_extraction_cosine.json (E4b audit)
    """

    e11: dict[str, Any]
    full_grid: dict[str, Any]
    scale_curves: dict[str, Any]
    matched_n: dict[str, Any]
    raw_vs_chat: dict[str, Any]


def _read_json(relpath: str) -> dict[str, Any]:
    """Parse one results-relative JSON, downloading from HF if absent."""
    return json.loads(fetch(relpath).read_text())


def load_evidence() -> StorySourceEvidence:
    """Resolve and cross-check the five evidence files.

    Two arms (self-generated and fixed DeepSeek) were scored into BOTH
    e11_lineage.json and e12_diverse_fullcorpus_grid.json, so the notebook
    can quote either file for them. Raises ``ValueError`` if the duplicated
    rows are not bit-identical, on the detection read (R1 passing layers) or
    on the geometry read (R2 cosine and RSA at layer 33): a silent
    disagreement there would let one figure and another figure print
    different values for the same measured quantity.
    """
    evidence = StorySourceEvidence(
        e11=_read_json("e11_lineage.json"),
        full_grid=_read_json("e12_diverse_fullcorpus_grid.json"),
        scale_curves=_read_json("e12_scale_curve.json"),
        matched_n=_read_json("e12_pref_matched_n_exploratory.json"),
        raw_vs_chat=_read_json("raw_vs_chat_extraction_cosine.json"),
    )
    r1_overlap = [  # (row in e11, same row in the full-corpus grid)
        ("selfgen", "selfgen_postfix"),
        ("strong_external", "fixed_deepseek_n256"),
    ]
    for e11_name, grid_name in r1_overlap:
        # "r1_dual_battery" is the key the scorers wrote into the frozen
        # evidence JSONs; it holds the two-scenario-set detection read
        e11_row = evidence.e11["r1_dual_battery"][e11_name]["passing_layers"]
        grid_row = evidence.full_grid["r1_dual_battery"][grid_name]["passing_layers"]
        if e11_row != grid_row:
            raise ValueError(
                f"R1 grids disagree on shared arm {e11_name}/{grid_name}: {e11_row} != {grid_row}"
            )
    # The one probe-set pair measured in both files: self-generated vs fixed
    # DeepSeek. "r2_cross_lineage_layer33" is the frozen key the scorers
    # wrote; it holds the pairwise geometry read at layer 33.
    e11_pair = evidence.e11["r2_cross_lineage_layer33"]["selfgen_vs_strong_external"]
    grid_pair = evidence.full_grid["r2_cross_lineage_layer33"][
        "selfgen_postfix_vs_fixed_deepseek_n256"
    ]
    for field in ("mean_contrast_cos", "rsa_12x12"):
        if e11_pair[field] != grid_pair[field]:
            raise ValueError(
                f"R2 grids disagree on self-generated vs fixed DeepSeek {field}: "
                f"{e11_pair[field]} != {grid_pair[field]}"
            )
    return evidence


def load_corpora() -> dict[str, dict[str, list[str]]]:
    """Load the three story corpora as {corpus label: {emotion: stories}}.

    Every corpus resolves through ``fetch`` and a missing file raises
    (no silent skip): the section-1 exhibit needs all three.
    """
    corpora: dict[str, dict[str, list[str]]] = {}
    for label, (subtree, filename) in CORPUS_FILES.items():
        path = fetch(f"{subtree}/{filename}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        corpora[label] = {row["emotion"]: row["stories"] for row in rows}
        if not corpora[label]:
            raise ValueError(f"{label}: {path} parsed to an empty corpus")
    return corpora


def per_layer_counts(r1_arm: dict[str, Any]) -> dict[int, tuple[int, int]]:
    """One R1 arm's per-layer grid as {layer: (paper, held-out) correct counts}.

    Layer keys arrive as strings in the JSON; this is the single place that
    normalizes them to ints.
    """
    counts = {
        int(layer): (int(pair[0]), int(pair[1])) for layer, pair in r1_arm["per_layer"].items()
    }
    if not counts:
        raise ValueError("R1 arm has an empty per_layer grid")
    return counts


def seed_mean_by_n(points: list[dict[str, Any]], metric: str) -> dict[int, float]:
    """Seed-mean of ``metric`` at each corpus size in a dose-response arm.

    ``points`` is the arm's list of per-(n, seed) records from
    e12_scale_curve.json; the mean at each n averages its 5 seeds (fewer at
    the full-corpus point, where subsampling is a no-op).
    """
    sizes = sorted({point["n"] for point in points})
    means = {}
    for size in sizes:
        values = [point[metric] for point in points if point["n"] == size]
        means[size] = math.fsum(values) / len(values)
    return means
