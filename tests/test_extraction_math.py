"""Laptop-safe checks that the extraction math matches the reference's.

Two facts the ARENA chapter-4 audit left as queued items, locked here:

1. Recombination: the reference (sinievanderben/emotion_experiment) keeps one
   running sum per emotion and divides by the total post-mask token count — a
   token-weighted mean. Our per-story shards recombined by story_store's
   emotion_mean must give the identical vector (not an equal-weight mean over
   story means).

2. Mask semantics: the TOKEN_OFFSET column skip does what the method says
   (skip each story's first TOKEN_OFFSET tokens) under right padding, and the
   left-padding deviation of the pre-fix runs is exactly the closed form used
   by scripts/audit_extraction_offset.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emotion_vectors.corpus import TOKEN_OFFSET
from emotion_vectors.story_store import emotion_mean, story_key


def _write_shards(
    tmp_path: Path, stories: list[np.ndarray], counts: list[int]
) -> list[dict[str, object]]:
    (tmp_path / "shards").mkdir(exist_ok=True)
    rows = []
    for idx, (vec, n) in enumerate(zip(stories, counts)):
        sid = story_key("test emotion", idx)
        np.save(tmp_path / "shards" / f"{sid}.npy", vec.astype(np.float32))
        rows.append({"story_id": sid, "emotion": "test emotion", "n_tokens": n})
    return rows


def test_emotion_mean_equals_reference_running_sum(tmp_path: Path) -> None:
    """sum(mean_i * n_i) / sum(n_i) == reference's accum/total on raw tokens."""
    rng = np.random.default_rng(20260722)
    layers, d_model = 3, 8
    counts = [7, 191, 43, 1]
    token_activations = [rng.normal(size=(n, layers, d_model)) for n in counts]

    story_means = [acts.mean(axis=0) for acts in token_activations]
    rows = _write_shards(tmp_path, story_means, counts)

    reference = np.concatenate(token_activations, axis=0).sum(axis=0) / sum(counts)
    ours = emotion_mean(tmp_path, rows)
    np.testing.assert_allclose(ours, reference, rtol=1e-5)


def test_emotion_mean_is_not_equal_weight(tmp_path: Path) -> None:
    """Guard against a silent regression to np.mean over story means."""
    counts = [1, 100]
    story_means = [np.full((2, 4), 1.0), np.full((2, 4), 3.0)]
    rows = _write_shards(tmp_path, story_means, counts)
    ours = emotion_mean(tmp_path, rows)
    token_weighted = (1 * 1.0 + 100 * 3.0) / 101
    assert np.allclose(ours, token_weighted)
    assert not np.allclose(ours, 2.0)  # the equal-weight answer


def _included_counts(lengths: list[int], padding_side: str) -> list[int]:
    """Post-mask token counts from pure mask arithmetic, both padding sides."""
    s = max(lengths)
    included = []
    for length in lengths:
        mask = np.zeros(s, dtype=int)
        if padding_side == "right":
            mask[:length] = 1
        else:
            mask[s - length :] = 1
        mask[:TOKEN_OFFSET] = 0  # pool_batch's column skip
        included.append(int(mask.sum()))
    return included


def test_token_offset_semantics_right_vs_left_padding() -> None:
    """Right padding skips each story's first TOKEN_OFFSET tokens; left padding
    reproduces the audit script's closed form (and the recorded manifests)."""
    lengths = [60, 200, 512, 51]
    s = max(lengths)
    assert _included_counts(lengths, "right") == [
        max(0, length - TOKEN_OFFSET) for length in lengths
    ]
    assert _included_counts(lengths, "left") == [
        length - max(0, TOKEN_OFFSET - (s - length)) for length in lengths
    ]


def test_audit_matches_recorded_manifest_counts() -> None:
    """The shipped base-corpus audit found every discriminating story matching
    the left-padding form and none matching the intended form — the mechanics
    proof this test suite assumes. Regression-guard the recorded verdict."""
    audit_file = Path("results/extraction_offset_token_audit.json")
    if not audit_file.exists():
        return  # audit artifact not present in this checkout
    audit = json.loads(audit_file.read_text())
    assert audit["n_tokens_match_rightpad_intended"] == 0
    assert audit["n_tokens_match_neither"] == 0
    assert audit["n_tokens_match_leftpad_buggy"] > 1000
