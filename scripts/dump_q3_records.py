"""Q3 category-analysis substrate: per-phase and per-transition record dump.

The registered gate/R1 scorer (score_q3_gate_r1.py) aggregates over all
phases and transitions, which hides the category structure Peyton asked
about: which emotions get tracked, which designed triple families
(A_superposition .. F_valence_spread) are easy or hard, whether detection
follows affective distance, and what the model confuses with what.

This script does NO aggregation. It applies the exact same conventions as
the registered scorer (centered cosine with the story-set mean and
norms_centered denominator; SEQUENTIAL stories only; per-phase mean
cosine; W=16 boundary-referenced lead) and dumps the full score matrices
plus every piece of per-record metadata. All slicing then happens
laptop-side inside the analysis notebook, so every number in the notebook
is computed in-cell from this substrate.

Outputs <out-prefix>.npz and <out-prefix>_meta.json:

  phase_scores      [phases, layers, probes]  mean centered cosine per phase
  trans_lead        [transitions, layers, probes]  W-window lead at boundary
  (meta json)       per-phase: story_id, triple_id, category, mode,
                    phase_index, emotion, phase length in tokens;
                    per-transition: the same plus from/to emotions and
                    transition index; probe labels; layer list; conventions.

Run on the pod (shards live there):

    .venv/bin/python scripts/dump_q3_records.py \
        --traj-dir results/combined_trajectories_it_v2 \
        --out-prefix results/q3_records_it_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from jaxtyping import Float

from emotion_vectors.q3_conventions import centered_cos, manifest_rows, story_set_mean

LAYERS = [6, 15, 24, 33, 42, 51]
W = 16
MIN_PHASE_TOKENS = 4  # same skip rule as the registered scorer


def dump(
    traj_dir: Path,
    rows: list[dict[str, object]],
    mean_dots: Float[np.ndarray, "layers probes"],
) -> tuple[dict[str, Float[np.ndarray, "records layers probes"]], dict[str, object]]:
    """One pass over the shards; returns (arrays, metadata lists)."""
    phase_scores, phase_meta = [], []
    trans_lead, trans_meta = [], []

    for row in rows:
        shard = np.load(traj_dir / "shards" / f"{row['story_id']}.npz")
        cos = centered_cos(shard, mean_dots)
        n_tokens = cos.shape[0]
        starts = list(row["phase_token_starts"]) + [n_tokens]
        common = {
            "story_id": row["story_id"],
            "triple_id": row["triple_id"],
            "category": row["category"],
        }

        # one record per phase: mean cosine over the phase's tokens
        for k, emotion in enumerate(row["phase_emotions"]):
            phase_start, phase_end = starts[k], starts[k + 1]
            if phase_end - phase_start < MIN_PHASE_TOKENS:
                continue
            phase_scores.append(cos[phase_start:phase_end].mean(axis=0))
            phase_meta.append(
                {
                    **common,
                    "phase_index": k,
                    "emotion": emotion,
                    "n_tokens": phase_end - phase_start,
                }
            )

        # one record per transition: lead = [b-W, b) window minus [b-2W, b-W)
        for k in range(1, len(row["phase_emotions"])):
            boundary = starts[k]
            if boundary < 2 * W or n_tokens - boundary < W:
                continue
            near_window = cos[boundary - W : boundary].mean(axis=0)
            earlier_window = cos[boundary - 2 * W : boundary - W].mean(axis=0)
            trans_lead.append(near_window - earlier_window)
            trans_meta.append(
                {
                    **common,
                    "transition_index": k,
                    "from_emotion": row["phase_emotions"][k - 1],
                    "to_emotion": row["phase_emotions"][k],
                }
            )

    arrays = {
        "phase_scores": np.stack(phase_scores).astype(np.float32),
        "trans_lead": np.stack(trans_lead).astype(np.float32),
    }
    meta = {"phase_records": phase_meta, "transition_records": trans_meta}
    return arrays, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="smoke: first N stories")
    args = parser.parse_args()

    rows, n_orphaned = manifest_rows(args.traj_dir)
    if args.limit:
        rows = rows[: args.limit]
    mean_dots = story_set_mean(args.traj_dir, rows)
    arrays, meta = dump(args.traj_dir, rows, mean_dots)

    meta["probe_labels"] = json.loads((args.traj_dir / "probe_labels.json").read_text())
    meta["layers"] = LAYERS
    meta["n_stories"] = len(rows)
    meta["n_orphaned_manifest_rows"] = n_orphaned
    meta["conventions"] = (
        "centered cosine (story-set mean, norms_centered); SEQUENTIAL only; "
        f"phases < {MIN_PHASE_TOKENS} tokens skipped; lead windows W={W} "
        "boundary-referenced, same as score_q3_gate_r1.py"
    )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{args.out_prefix}.npz", **arrays)
    Path(f"{args.out_prefix}_meta.json").write_text(json.dumps(meta, indent=1))
    print(
        f"{len(meta['phase_records'])} phase records, "
        f"{len(meta['transition_records'])} transition records, "
        f"{arrays['phase_scores'].shape[2]} probes -> {args.out_prefix}.npz"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
