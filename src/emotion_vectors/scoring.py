"""Battery scoring and corpus quality control — the promoted analysis layer.

"Battery" means the scenario test set: the Anthropic paper's 12
implicit-emotion prompts (Table 2) plus our held-out 12. Scoring asks, for
each scenario, whether its target emotion's probe (per-emotion direction
vector) ranks among the top 3 by cosine similarity.

Every function here produced tree-linked evidence from inside notebooks before
being promoted (the engineering contract's rule: logic that evidence rests on
lives in the package, typed and lint-gated; notebooks narrate and call).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from scipy.stats import rankdata

if TYPE_CHECKING:
    from jaxtyping import Float

# Word stems that count as naming an emotion, for leakage QC of generated text.
# Single source of truth; previously duplicated across three notebooks.
EMOTION_STEMS: Final[dict[str, list[str]]] = {
    "happy": ["happy", "happi"],
    "inspired": ["inspir"],
    "loving": ["love", "loving"],
    "proud": ["proud", "pride"],
    "calm": ["calm"],
    "desperate": ["desperat"],
    "angry": ["angry", "anger"],
    "guilty": ["guilt"],
    "sad": ["sad"],
    "afraid": ["afraid", "fear"],
    "nervous": ["nervous"],
    "surprised": ["surpris"],
}


def leakage(corpus_file: Path, stems: dict[str, list[str]] = EMOTION_STEMS) -> tuple[int, int]:
    """(leaked, total) texts naming their target emotion in a grouped corpus file."""
    total = leaked = 0
    with open(corpus_file) as f:
        for line in f:
            row = json.loads(line)
            emotion_stems = stems.get(row["emotion"])
            if not emotion_stems:
                continue
            for story in row["stories"]:
                total += 1
                leaked += any(stem in story.lower() for stem in emotion_stems)
    return leaked, total


def battery_matrix(
    probe_means: "Float[np.ndarray, 'emotions d_model']",
    activations: "Float[np.ndarray, 'scenarios d_model']",
    center_pool: "Float[np.ndarray, 'pool d_model'] | None" = None,
) -> "Float[np.ndarray, 'emotions scenarios']":
    """Cosine of each centered, unit-norm probe against each scenario activation.

    center_pool defines the contrast baseline (its mean is subtracted). Default
    is the probes themselves; pass the full emotion set when probes are a
    subset of a larger extraction (the sweep convention: center on all 171)."""
    baseline = (center_pool if center_pool is not None else probe_means).mean(axis=0)
    probes = probe_means - baseline
    probes = probes / np.linalg.norm(probes, axis=1, keepdims=True)
    return probes @ activations.T / np.linalg.norm(activations, axis=1)


def top3_count(matrix: "Float[np.ndarray, 'emotions scenarios']") -> int:
    """Scenarios whose target emotion (the diagonal) ranks in the top 3 probes.

    Assumes probe row i is scenario i's target, the battery convention."""
    n = matrix.shape[1]
    return sum(int(n - rankdata(matrix[:, j])[j]) + 1 <= 3 for j in range(n))


def score_battery(
    probe_means: "Float[np.ndarray, 'emotions d_model']",
    activations: "Float[np.ndarray, 'scenarios d_model']",
    center_pool: "Float[np.ndarray, 'pool d_model'] | None" = None,
) -> tuple[int, "Float[np.ndarray, 'emotions scenarios']"]:
    """(top-3 count, full cosine matrix) — the registered battery statistic."""
    matrix = battery_matrix(probe_means, activations, center_pool)
    return top3_count(matrix), matrix
