"""Q3 per-token trajectory logic: tag parsing, token alignment, probe dots.

The combined-story corpus (scripts/combined_story_gen/) marks emotion phases
with ``<emotion>label</emotion>`` tags: SEQUENTIAL stories carry one tag at
each phase start (including position 0), SIMULTANEOUS stories one tag listing
all three emotions. This module turns a tagged story into (clean text, phase
spans), aligns phase boundaries to token indices via the tokenizer's offset
mapping, and projects per-token activations onto unit probe directions.

Substrate convention (open-science policy): the extraction script stores raw
per-token dot products against unit probes plus per-token activation norms —
cosines and any centering convention (e.g. the E9 scenario-mean centering)
are transformations applied in analysis code, reproducible from the shards.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from einops import einsum
from jaxtyping import Float

TAG_RE = re.compile(r"\s*<emotion>\s*\(?([^<)]*?)\)?\s*</emotion>\s*")


def kept_rows(stories_path: Path) -> list[dict]:
    """Kept-story filter mirroring the generator's QC: no error, no leaked
    emotion words, tags consistent with the assigned emotions."""
    rows = []
    for line in stories_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("error") is None
            and not row.get("leaked_words")
            and row.get("tags_match_emotions", True)
        ):
            rows.append(row)
    return rows


def story_id(row: dict) -> str:
    """Stable shard id: triple, mode, permutation, and a content hash.

    sha1 is a naming device here, not a security primitive: it de-duplicates
    story text into a short stable filename. The algorithm cannot be changed
    without renaming every published shard, so the flag is what moves.
    """
    digest = hashlib.sha1(row["text"].encode(), usedforsecurity=False).hexdigest()[:8]
    return f"t{row['triple_id']:03d}_{row['mode'][:3].lower()}_p{row['perm_idx']}_{digest}"


@dataclass(frozen=True)
class ParsedStory:
    """A tag-stripped story with phase boundaries in clean-text coordinates."""

    clean_text: str
    phase_emotions: list[str]  # one per phase (SEQUENTIAL) or all three (SIMULTANEOUS)
    phase_char_starts: list[int]  # clean-text char offset where each phase begins
    mode: str


def parse_story(text: str, mode: str) -> ParsedStory:
    """Strip ``<emotion>`` tags and record where each phase starts.

    SEQUENTIAL: each tag opens a phase at its own position; the labels are the
    per-phase emotions. SIMULTANEOUS: a single tag lists all emotions and the
    whole story is one phase starting at 0.
    """
    labels: list[str] = []
    starts: list[int] = []
    clean_parts: list[str] = []
    cursor = 0
    clean_len = 0
    for match in TAG_RE.finditer(text):
        clean_parts.append(text[cursor : match.start()])
        clean_len += match.start() - cursor
        labels.append(match.group(1).strip())
        starts.append(clean_len)
        cursor = match.end()
    clean_parts.append(text[cursor:])
    clean_text = "".join(clean_parts).strip()
    if not labels:
        raise ValueError("story has no <emotion> tags")
    if mode == "SIMULTANEOUS":
        emotions = [label.strip() for label in labels[0].split(",") if label.strip()]
        return ParsedStory(clean_text, emotions, [0], mode)
    return ParsedStory(clean_text, labels, starts, mode)


def phase_token_starts(char_starts: list[int], offsets: list[tuple[int, int]]) -> list[int]:
    """First token index of each phase, given the tokenizer offset mapping.

    Special tokens (BOS) map to (0, 0) offsets and never win; a phase starting
    at char c begins at the first token whose span ends past c.
    """
    token_starts: list[int] = []
    for char_start in char_starts:
        # first non-special token whose char span reaches past the phase start;
        # fall back to the last token if the phase starts at/after the story end
        phase_first_token = len(offsets) - 1
        for token_idx, (span_start, span_end) in enumerate(offsets):
            is_real_token = span_end > span_start  # special tokens map to (0, 0)
            if is_real_token and span_end > char_start:
                phase_first_token = token_idx
                break
        token_starts.append(phase_first_token)
    return token_starts


def unit_contrast_probes(
    means: Float[np.ndarray, "emotions layers d_model"],
) -> Float[np.ndarray, "emotions layers d_model"]:
    """Difference-of-means probe directions, unit-normalized per (emotion, layer).

    Contrast = per-emotion mean minus the grand mean over emotions — the same
    convention as the battery probes (centering pool = the emotion set given).
    """
    contrast = means - means.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(contrast, axis=-1, keepdims=True)
    return contrast / np.clip(norms, 1e-8, None)


def random_unit_directions(
    n_dirs: int, n_layers: int, d_model: int, seed: int
) -> Float[np.ndarray, "dirs layers d_model"]:
    """Fixed-seed random unit directions — the trajectory null control."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_dirs, n_layers, d_model)).astype(np.float32)
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)


def token_probe_dots(
    acts: Float[np.ndarray, "tokens layers d_model"],
    probes: Float[np.ndarray, "probes layers d_model"],
) -> tuple[Float[np.ndarray, "tokens layers probes"], Float[np.ndarray, "tokens layers"]]:
    """(raw dots onto unit probes, per-token activation norms).

    cosine(token, probe) = dots / norms — left to analysis code so centering
    conventions stay explicit transformations of the stored substrate.
    """
    dots = einsum(
        acts, probes, "tokens layers d_model, probes layers d_model -> tokens layers probes"
    )
    norms = np.linalg.norm(acts, axis=-1)
    return dots, norms


def transition_windows(
    cosines: Float[np.ndarray, "tokens phases"],
    phase_starts: list[int],
    window: int,
) -> tuple[Float[np.ndarray, "trans width"], Float[np.ndarray, "trans width"]]:
    """(incoming, outgoing) cosine windows around each phase transition.

    Column j of ``cosines`` must be phase j's emotion. For transition k (the
    start of phase k, k >= 1), incoming = column k, outgoing = column k-1,
    both sliced to [start - window, start + window] and NaN-padded where the
    slice runs off the story. Row order matches transitions in story order.
    """
    width = 2 * window + 1
    incoming, outgoing = [], []
    for entered_phase, start in enumerate(phase_starts[1:], start=1):
        left_phase = entered_phase - 1
        for column, bucket in ((entered_phase, incoming), (left_phase, outgoing)):
            # copy cosines[start - window : start + window + 1] into a NaN-padded
            # row, clipping the slice where it runs off either end of the story
            row = np.full(width, np.nan, dtype=np.float32)
            window_lo, window_hi = start - window, start + window + 1
            clipped_lo, clipped_hi = max(window_lo, 0), min(window_hi, len(cosines))
            row[clipped_lo - window_lo : clipped_hi - window_lo] = cosines[
                clipped_lo:clipped_hi, column
            ]
            bucket.append(row)
    empty = np.empty((0, width), dtype=np.float32)
    return (
        np.stack(incoming) if incoming else empty,
        np.stack(outgoing) if outgoing else empty,
    )


def circumplex_weights(
    means: Float[np.ndarray, "emotions d_model"],
) -> Float[np.ndarray, "two emotions"]:
    """Weights expressing the PC1/PC2 circumplex axes over the unit probes.

    PCA axes of the centered emotion means lie in the span of the contrast
    directions, so a token's projection onto PC1/PC2 equals a weighted sum of
    its stored probe dots: proj = weights @ dots. This is what lets the
    circumplex view be computed from published shards alone.
    """
    contrast = means - means.mean(axis=0, keepdims=True)
    # PC1/PC2 = top-2 right singular vectors of the centered means
    _, _, right_singular = np.linalg.svd(contrast, full_matrices=False)
    pca_axes = right_singular[:2]  # [two, d_model]
    contrast_norms = np.linalg.norm(contrast, axis=-1, keepdims=True)
    unit_contrasts = contrast / np.clip(contrast_norms, 1e-8, None)
    # least-squares solve for weights with unit_contrasts.T @ weights.T ~= pca_axes.T,
    # i.e. express each PCA axis as a combination of the unit probe directions
    weights, *_ = np.linalg.lstsq(unit_contrasts.T, pca_axes.T, rcond=None)
    return weights.T
