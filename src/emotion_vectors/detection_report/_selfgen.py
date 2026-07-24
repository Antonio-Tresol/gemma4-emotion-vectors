"""Section 3: probes rebuilt from the probed model's own stories and dialogues.

Section 1's probes came from stories written by a different, smaller model
(gemma-4-4B). This module loads the two self-generated probe sets, measures how
far their directions sit from the corpus probes, reports the leakage quality
check on the generated texts, and formats the ranking table the notebook
prints. The registered pass rule itself is applied in the notebook cell, in
plain sight; nothing here decides pass or fail.

Every score in this section centers on the 12 probes themselves, not on the
171-word extraction: the self-generated sets only ever extracted the 12
battery emotions, so no 171-word pool exists for them (see :mod:`._data`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float

from emotion_vectors.artifacts import fetch
from emotion_vectors.scoring import leakage

from ._data import IT_LABEL, REGISTERED_LAYER, ModelSweep, SweepContext

# probe-set key -> (what it is in plain words, results subtree of its vectors)
SELF_GENERATED_SETS = {
    "self-story": ("stories the probed model wrote itself", "self_story_vectors_it"),
    "dialogue": ("dialogues the probed model wrote itself", "dialogue_vectors_it"),
}
CORPUS_SET_LABEL = "probes from gemma-4-4B stories (section 1's probes)"
# corpus label -> the grouped-text file the leakage check reads
LEAKAGE_CORPORA = {
    "self-story": "self_stories_it/dialogues_grouped.jsonl",
    "dialogue": "dialogue_stories_it/dialogues_grouped.jsonl",
}


def _load_vector_dir(
    subtree: str, probe_order: list[str], layers: list[int]
) -> Float[np.ndarray, "battery_emotions layers d_model"]:
    """Stack one extraction directory into [battery_emotions layers d_model].

    One ``<emotion>/layer_<n>_resid.npy`` per emotion and layer; a missing
    emotion directory or layer file raises rather than being skipped.
    """
    root = Path(fetch(subtree))
    stacks = []
    for emotion in probe_order:
        emotion_dir = root / emotion.replace(" ", "_")
        if not emotion_dir.is_dir():
            raise FileNotFoundError(f"{subtree}: no extraction directory for '{emotion}'")
        stacks.append(
            np.stack([np.load(emotion_dir / f"layer_{layer}_resid.npy") for layer in layers])
        )
    return np.stack(stacks)


def load_probe_sets(
    ctx: SweepContext,
) -> dict[str, Float[np.ndarray, "battery_emotions layers d_model"]]:
    """The three instruct-model probe sets, in battery-emotion row order.

    Returns {"corpus": gemma-4-4B story probes, "self-story": probes from the
    model's own stories, "dialogue": probes from its own dialogues}, each
    [battery_emotions layers d_model] over the shared layer grid.
    """
    model: ModelSweep = ctx.model(IT_LABEL)
    corpus = np.stack([model.means[row] for row in model.probe_index])
    sets = {"corpus": corpus}
    for key, (_description, subtree) in SELF_GENERATED_SETS.items():
        sets[key] = _load_vector_dir(subtree, ctx.probe_order, ctx.layers)
        if sets[key].shape != corpus.shape:
            raise ValueError(
                f"{key}: shape {sets[key].shape} differs from the corpus set's {corpus.shape}"
            )
    return sets


def _contrast_cosines(
    reference: Float[np.ndarray, "battery_emotions d_model"],
    other: Float[np.ndarray, "battery_emotions d_model"],
) -> Float[np.ndarray, " battery_emotions"]:
    """Per-emotion cosine between two probe sets' contrast directions.

    Each set's own mean is removed first, so this compares what makes each
    emotion different from the others rather than the shared offset that
    every probe carries.
    """
    centered_reference = reference - reference.mean(0)
    centered_other = other - other.mean(0)
    return (centered_reference * centered_other).sum(1) / (
        np.linalg.norm(centered_reference, axis=1) * np.linalg.norm(centered_other, axis=1)
    )


def direction_similarity(
    probe_sets: dict[str, Float[np.ndarray, "battery_emotions layers d_model"]],
    layers: list[int],
) -> dict[str, Any]:
    """How far the self-generated probe directions sit from the corpus probes.

    Inputs: ``load_probe_sets`` output and the shared layer grid. Returns
    ``stats`` with ``lines`` (the printed record: the registered-layer mean
    and minimum per set, then the mean at every layer) and ``per_layer``
    {set: {layer: mean cosine}}, which section 5 reuses as its grading anchor
    for what a genuinely different probe set looks like.
    """
    corpus = probe_sets["corpus"]
    names = [key for key in probe_sets if key != "corpus"]
    per_layer: dict[str, dict[int, float]] = {name: {} for name in names}
    lines = [f"model: {IT_LABEL} (all probe sets and battery activations)"]
    registered_pos = layers.index(REGISTERED_LAYER)
    for name in names:
        cosines = _contrast_cosines(
            corpus[:, registered_pos, :], probe_sets[name][:, registered_pos, :]
        )
        lines.append(
            f"probe-direction cosine vs 4B-corpus probes ({name}, layer {REGISTERED_LAYER}): "
            f"mean {cosines.mean():.3f}, min {cosines.min():.3f}"
        )
    for layer_pos, layer in enumerate(layers):
        for name in names:
            cosines = _contrast_cosines(corpus[:, layer_pos, :], probe_sets[name][:, layer_pos, :])
            per_layer[name][int(layer)] = float(cosines.mean())
    lines.append("\nmean probe-direction cosine vs 4B-corpus probes, all layers:")
    lines.append(f"{'layer':>5s} {'self-story':>11s} {'dialogue':>9s}")
    for layer in layers:
        marker = "  *" if int(layer) == REGISTERED_LAYER else ""
        lines.append(
            f"{int(layer):5d} {per_layer['self-story'][int(layer)]:11.3f}"
            f" {per_layer['dialogue'][int(layer)]:9.3f}{marker}"
        )
    return {"lines": lines, "per_layer": per_layer}


def leakage_report() -> dict[str, Any]:
    """Quality check: do the model's own texts simply name the target emotion?

    Returns ``stats`` with ``lines`` (the printed record) and ``share``
    {set: fraction of texts naming their emotion}. A high share would mean
    the probes are reading a word, not a state.
    """
    lines, share = [], {}
    for name, relpath in LEAKAGE_CORPORA.items():
        leaked, total = leakage(Path(fetch(relpath)))
        if total == 0:
            raise ValueError(f"{name}: {relpath} holds no texts to check")
        share[name] = leaked / total
        lines.append(
            f"leakage QC ({name} corpus): {leaked}/{total} texts name their emotion"
            f" ({share[name]:.1%})"
        )
    return {"lines": lines, "share": share}


def rank_table_lines(
    rows: list[tuple[int, int, str, str, int, str]], per_set: int = 4
) -> list[str]:
    """Format the best few combinations per probe set as the printed table.

    ``rows`` are (paper, held-out, set, format, layer, readout) sorted best
    first; at most ``per_set`` rows per probe set are shown, so one strong set
    cannot crowd the others out of the table.
    """
    lines = [
        f"probe sets scored on {IT_LABEL} battery activations:",
        f"  corpus     = {CORPUS_SET_LABEL}",
    ]
    for key, (description, _subtree) in SELF_GENERATED_SETS.items():
        lines.append(f"  {key:10s} = probes from {description}")
    lines.append(f"{'paper':>6s} {'heldout':>8s}  set         fmt    layer readout")
    shown: dict[str, int] = {}
    for row in rows:
        if shown.get(row[2], 0) < per_set:
            shown[row[2]] = shown.get(row[2], 0) + 1
            lines.append(
                f"  {row[0]:2d}/12    {row[1]:2d}/12  {row[2]:11s} {row[3]:6s} {row[4]:3d}  {row[5]}"
            )
    return lines
