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
    """Stable shard id: triple, mode, permutation, and a content hash."""
    digest = hashlib.sha1(row["text"].encode()).hexdigest()[:8]
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
        emotions = [e.strip() for e in labels[0].split(",") if e.strip()]
        return ParsedStory(clean_text, emotions, [0], mode)
    return ParsedStory(clean_text, labels, starts, mode)


def phase_token_starts(char_starts: list[int], offsets: list[tuple[int, int]]) -> list[int]:
    """First token index of each phase, given the tokenizer offset mapping.

    Special tokens (BOS) map to (0, 0) offsets and never win; a phase starting
    at char c begins at the first token whose span ends past c.
    """
    token_starts: list[int] = []
    for char_start in char_starts:
        idx = next(
            (i for i, (s, e) in enumerate(offsets) if e > char_start and e > s),
            len(offsets) - 1,
        )
        token_starts.append(idx)
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
