"""Shared conventions for Q3 transition-tracking reads (TREE Q3.H1.E1).

Both the registered scorer (scripts/score_q3_gate_r1.py) and the
category-analysis record dump (scripts/dump_q3_records.py) must apply the
exact same story selection, story-set mean, and centered-cosine
definition. Housing them here makes that identity true by construction
instead of by parallel maintenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from einops import rearrange
from jaxtyping import Float


def manifest_rows(traj_dir: Path) -> tuple[list[dict[str, object]], int]:
    """SEQUENTIAL rows whose shard file exists; count of orphaned rows.
    (23/5888 shards were lost to a mid-run restart on the primary substrate —
    recorded, skipped, and reported rather than crashed on.)"""
    rows = [json.loads(line) for line in (traj_dir / "manifest.jsonl").read_text().splitlines()]
    sequential = [row for row in rows if row.get("error") is None and row["mode"] == "SEQUENTIAL"]
    shard_ids_on_disk = {path.stem for path in (traj_dir / "shards").glob("*.npz")}
    kept = [row for row in sequential if row["story_id"] in shard_ids_on_disk]
    return kept, len(sequential) - len(kept)


def story_set_mean(
    traj_dir: Path, rows: list[dict[str, object]]
) -> Float[np.ndarray, "layers probes"]:
    """Token-weighted mean of raw dots over the story set: [layers, probes]."""
    total = None
    n_tokens = 0
    for row in rows:
        shard = np.load(traj_dir / "shards" / f"{row['story_id']}.npz")
        dots = shard["dots"].astype(np.float64)
        total = dots.sum(axis=0) if total is None else total + dots.sum(axis=0)
        n_tokens += dots.shape[0]
    return (total / n_tokens).astype(np.float32)


def centered_cos(
    shard: np.lib.npyio.NpzFile, mean_dots: Float[np.ndarray, "layers probes"]
) -> Float[np.ndarray, "tokens layers probes"]:
    """Centered cosine per (token, layer, probe): story-set mean subtracted
    from the stored dots, norms_centered denominator."""
    dots: Float[np.ndarray, "tokens layers probes"] = shard["dots"].astype(np.float32)
    norms: Float[np.ndarray, "tokens layers"] = shard["norms_centered"].astype(np.float32)
    # the denominator is one norm per (token, layer), so it needs a trailing
    # probe axis to divide the dots. Spelling that out beats an anonymous None.
    denominator = rearrange(np.clip(norms, 1e-6, None), "tokens layers -> tokens layers 1")
    return (dots - mean_dots) / denominator
