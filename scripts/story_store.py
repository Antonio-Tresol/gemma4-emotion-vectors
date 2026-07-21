"""On-disk store for per-story pooled activations — the resumability layer.

The reference pipeline accumulates one running sum per emotion in memory and
saves only at the end, so a crash loses the whole run. This store instead keeps
one `[n_layers, d_model]` fp32 shard per story plus a JSONL manifest row with
its post-mask token count, and recombines them as
sum(story_mean * story_tokens) / sum(story_tokens) — the identical
token-weighted mean the reference computes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jaxtyping import Float


def story_key(emotion: str, idx: int) -> str:
    safe = emotion.replace(" ", "_").replace("/", "-")
    return f"{safe}__{idx:04d}"


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_manifest(path: Path) -> dict[str, dict[str, object]]:
    """Last row per story_id wins, so resumed runs supersede earlier errors."""
    rows: dict[str, dict[str, object]] = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["story_id"]] = row
    return rows


def pending_stories(
    emotions_data: dict[str, list[str]],
    manifest: dict[str, dict[str, object]],
    shards_dir: Path,
) -> dict[str, list[tuple[int, str]]]:
    """Stories not yet extracted, grouped by emotion in reference order. A
    story counts as done only with an error-free manifest row, a matching text
    hash, and its shard file present on disk."""
    pending: dict[str, list[tuple[int, str]]] = {}
    for emotion, stories in emotions_data.items():
        for idx, text in enumerate(stories):
            sid = story_key(emotion, idx)
            row = manifest.get(sid)
            done = (
                row is not None
                and row.get("error") is None
                and row.get("sha1") == sha1(text)
                and (shards_dir / f"{sid}.npy").exists()
            )
            if not done:
                pending.setdefault(emotion, []).append((idx, text))
    return pending


def emotion_mean(out_dir: Path, rows: list[dict[str, object]]) -> "Float[np.ndarray, 'l d']":
    """Token-weighted mean over one emotion's story shards, float64 accumulation:
    sum(story_mean * story_tokens) / sum(story_tokens) — the reference's mean."""
    acc, n_tok = None, 0
    for row in rows:
        vec = np.load(out_dir / "shards" / f"{row['story_id']}.npy").astype(np.float64)
        acc = vec * row["n_tokens"] + (0 if acc is None else acc)
        n_tok += row["n_tokens"]
    return (acc / max(n_tok, 1)).astype(np.float32)


def aggregate(out_dir: Path, logger: logging.Logger) -> None:
    """Recombine per-story shards into the reference's outputs: per-emotion
    layer_<N>_resid.npy plus a combined emotion_vectors.json."""
    config = json.loads((out_dir / "run_config.json").read_text())
    layers: list[int] = config["layers"]
    manifest = load_manifest(out_dir / "manifest.jsonl")
    ok = [r for r in manifest.values() if r.get("error") is None]
    failed = len(manifest) - len(ok)
    if failed:
        logger.warning(f"aggregating with {failed} failed stories excluded — not a full corpus")
    by_emotion: dict[str, list[dict[str, object]]] = {}
    for row in ok:
        by_emotion.setdefault(row["emotion"], []).append(row)

    combined: dict[str, dict[str, list[float]]] = {}
    for emotion, rows in by_emotion.items():
        mean = emotion_mean(out_dir, rows)  # [n_layers, d_model]
        emo_dir = out_dir / emotion.replace(" ", "_").replace("/", "-")
        emo_dir.mkdir(parents=True, exist_ok=True)
        combined[emotion] = {}
        for pos, layer_idx in enumerate(layers):
            np.save(emo_dir / f"layer_{layer_idx}_resid.npy", mean[pos])
            combined[emotion][str(layer_idx)] = mean[pos].tolist()
    (out_dir / "emotion_vectors.json").write_text(json.dumps(combined))
    logger.info(f"aggregated {len(by_emotion)} emotions x {len(layers)} layers -> {out_dir}")
