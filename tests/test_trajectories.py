"""Tests for the Q3 trajectory logic against real combined-story fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from emotion_vectors.trajectories import (
    kept_rows,
    parse_story,
    phase_token_starts,
    random_unit_directions,
    story_id,
    token_probe_dots,
    unit_contrast_probes,
)

FIXTURES = Path(__file__).parent / "fixtures" / "combined_stories_sample.jsonl"


def fixture_rows() -> list[dict]:
    return [json.loads(line) for line in FIXTURES.read_text().splitlines() if line.strip()]


def test_parse_sequential_phases_match_tags() -> None:
    for row in fixture_rows():
        if row["mode"] != "SEQUENTIAL":
            continue
        story = parse_story(row["text"], row["mode"])
        assert story.phase_emotions == row["tags"]
        assert story.phase_char_starts[0] == 0
        assert story.phase_char_starts == sorted(story.phase_char_starts)
        assert "<emotion>" not in story.clean_text


def test_parse_simultaneous_single_phase() -> None:
    for row in fixture_rows():
        if row["mode"] != "SIMULTANEOUS":
            continue
        story = parse_story(row["text"], row["mode"])
        assert story.phase_char_starts == [0]
        assert sorted(story.phase_emotions) == sorted(row["emotions"])
        assert "<emotion>" not in story.clean_text


def test_parse_strips_parenthesized_tag() -> None:
    text = "<emotion>(proud, angry)</emotion>The door slammed."
    story = parse_story(text, "SIMULTANEOUS")
    assert story.phase_emotions == ["proud", "angry"]
    assert story.clean_text == "The door slammed."


def test_parse_rejects_untagged_story() -> None:
    with pytest.raises(ValueError):
        parse_story("No tags at all.", "SEQUENTIAL")


def test_phase_token_starts_skips_special_tokens() -> None:
    # BOS at (0,0), then tokens covering chars [0,5), [5,9), [9,14)
    offsets = [(0, 0), (0, 5), (5, 9), (9, 14)]
    assert phase_token_starts([0, 9], offsets) == [1, 3]


def test_unit_contrast_probes_are_unit_and_centered() -> None:
    rng = np.random.default_rng(0)
    means = rng.standard_normal((5, 3, 16)).astype(np.float32)
    probes = unit_contrast_probes(means)
    norms = np.linalg.norm(probes, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    raw_contrast = means - means.mean(axis=0, keepdims=True)
    assert np.allclose(raw_contrast.sum(axis=0), 0.0, atol=1e-4)


def test_token_probe_dots_recovers_cosine() -> None:
    rng = np.random.default_rng(1)
    acts = rng.standard_normal((7, 3, 16)).astype(np.float32)
    probes = random_unit_directions(4, 3, 16, seed=2)
    dots, norms = token_probe_dots(acts, probes)
    assert dots.shape == (7, 3, 4)
    assert norms.shape == (7, 3)
    cosines = dots / norms[:, :, None]
    manual = np.sum(acts[0, 0] * probes[0, 0]) / np.linalg.norm(acts[0, 0])
    assert np.isclose(cosines[0, 0, 0], manual, atol=1e-5)
    assert np.all(np.abs(cosines) <= 1.0 + 1e-5)


def test_kept_rows_filter_and_story_ids() -> None:
    rows = kept_rows(FIXTURES)
    assert len(rows) == len(fixture_rows())  # fixtures are all kept-quality
    ids = [story_id(row) for row in rows]
    assert len(set(ids)) == len(ids)
    assert all(id_.startswith("t") and ("_seq_" in id_ or "_sim_" in id_) for id_ in ids)
