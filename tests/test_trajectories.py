"""Tests for the Q3 trajectory logic against real combined-story fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from emotion_vectors.trajectories import (
    circumplex_weights,
    kept_rows,
    parse_story,
    phase_token_starts,
    random_unit_directions,
    story_id,
    token_probe_dots,
    transition_windows,
    unit_contrast_probes,
)
from emotion_vectors.trajectory_plots import (
    barycentric,
    circumplex_figure,
    cosine_3d_figure,
    layer_ternaries,
    lines_figure,
    probe_heatmap_figure,
    smooth,
    speed_figure,
    story_cosines,
    ternary_figure,
    transition_locked_figure,
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


def test_story_cosines_centered_matches_manual() -> None:
    rng = np.random.default_rng(3)
    acts = rng.standard_normal((10, 2, 8)).astype(np.float32)
    probes = random_unit_directions(3, 2, 8, seed=4)
    dots, norms = token_probe_dots(acts, probes)
    centered_acts = acts - acts.mean(axis=0, keepdims=True)
    shard = {
        "dots": dots.astype(np.float16),
        "norms": norms.astype(np.float16),
        "norms_centered": np.linalg.norm(centered_acts, axis=-1).astype(np.float16),
    }
    labels = ["selfgen:a", "selfgen:b", "selfgen:c"]
    cos = story_cosines(shard, labels, ["a", "b", "c"], layer_pos=1)
    manual = np.sum(centered_acts[0, 1] * probes[0, 1]) / np.linalg.norm(centered_acts[0, 1])
    assert cos.shape == (10, 3)
    assert np.isclose(cos[0, 0], manual, atol=2e-3)  # fp16 substrate tolerance
    bary = barycentric(cos)
    assert np.allclose(bary.sum(axis=1), 1.0, atol=1e-6)
    assert smooth(cos, window=4).shape == cos.shape


def test_trajectory_figures_build() -> None:
    rng = np.random.default_rng(5)
    cos = rng.standard_normal((30, 3)).astype(np.float32) * 0.02
    fig1 = ternary_figure(cos, ["a", "b", "c"], [0, 10, 20])
    fig2 = lines_figure(cos, ["a", "b", "c"], [0, 10, 20])
    fig3 = layer_ternaries([cos, cos], ["layer 6", "layer 33"], ["a", "b", "c"])
    assert fig1.data and fig2.data and fig3.data


def test_transition_windows_alignment() -> None:
    cos = np.zeros((20, 3), dtype=np.float32)
    cos[:, 1] = np.arange(20)  # phase-2 emotion counts tokens
    incoming, outgoing = transition_windows(cos, [0, 10, 18], window=3)
    assert incoming.shape == (2, 7) and outgoing.shape == (2, 7)
    assert np.allclose(incoming[0], np.arange(7, 14))  # col 1 around t=10
    assert np.allclose(outgoing[0], 0.0)  # col 0 is flat zero
    assert np.isnan(incoming[1][-1])  # t=18+3 runs off the 20-token story
    assert np.allclose(outgoing[1][:5], np.arange(15, 20))  # col 1 exits at t=18


def test_circumplex_weights_reconstruct_pca_projection() -> None:
    rng = np.random.default_rng(7)
    means = rng.standard_normal((9, 16)).astype(np.float32)
    contrast = means - means.mean(axis=0, keepdims=True)
    units = contrast / np.linalg.norm(contrast, axis=-1, keepdims=True)
    _, _, vt = np.linalg.svd(contrast, full_matrices=False)
    token = rng.standard_normal(16).astype(np.float32)
    weights = circumplex_weights(means)
    via_dots = weights @ (units @ token)
    direct = vt[:2] @ token
    assert np.allclose(via_dots, direct, atol=1e-4)


def test_new_figures_build() -> None:
    rng = np.random.default_rng(9)
    cos = rng.standard_normal((40, 3)).astype(np.float32) * 0.02
    incoming, outgoing = transition_windows(cos, [0, 15, 30], window=5)
    figs = [
        transition_locked_figure(incoming, outgoing, window=5, n_boot=50),
        probe_heatmap_figure(
            rng.standard_normal((40, 12)).astype(np.float32),
            [f"e{i}" for i in range(12)],
            [0, 15, 30],
        ),
        circumplex_figure(
            rng.standard_normal((40, 2)).astype(np.float32), [0, 15, 30], ["a", "b", "c"]
        ),
        cosine_3d_figure(cos, ["a", "b", "c"], [0, 15, 30]),
        speed_figure(np.abs(rng.standard_normal(39)).astype(np.float32), [0, 15, 30]),
    ]
    assert all(fig.data for fig in figs)
