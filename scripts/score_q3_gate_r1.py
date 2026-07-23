"""Q3.H1.E1 scoring, first tranche: gate read G and anticipation read R1.

Implements the registered conventions verbatim (TREE Q3.H1.E1, registered
2026-07-22 before any true-probe scoring; substrate amendment 2026-07-23):

- Centered cosine primary: per-(layer, probe) STORY-SET mean subtracted from
  the stored dots, norms_centered denominator. SEQUENTIAL stories only.
- G (gate): per-phase rank of the tagged emotion's mean centered cosine
  among the 171 corpus probes; bar median rank <= 17 AND N2 p < 0.001.
- R1: anticipation lead, W=16 — mean cos over [b-W, b) minus the matched
  earlier window [b-2W, b-W) (the calibration script's exact definition),
  boundary-referenced (the cue-referenced twin needs the judge bulk pass);
  bar N2 p < 0.001 AND mean lead >= 0.5x the layer's calibrated noise sd.
- N1: the same statistics on the 24 random-direction probes.
- N2: permuted emotion assignment, 10,000 shuffles (instant via
  precomputed per-phase / per-transition statistics over ALL probes).
- Control subtraction (amendment): G and R1 on the constant-emotion arm;
  the true-arm statistic must exceed the control arm's 95th percentile.

Run on the pod (shards are local there):

    .venv/bin/python scripts/score_q3_gate_r1.py \
        --traj-dir results/combined_trajectories \
        --control-dir results/combined_trajectories_deepseek_constant \
        --calibration results/trajectory_instrument_calibration_it.json \
        --out results/q3_gate_r1_it.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LAYERS = [6, 15, 24, 33, 42, 51]
W = 16
N_SHUFFLES = 10_000
PRIMARY_LAYER = 33


def manifest_rows(traj_dir: Path) -> tuple[list[dict], int]:
    """SEQUENTIAL rows whose shard file exists; count of orphaned rows.
    (23/5888 shards were lost to a mid-run restart on the primary substrate —
    recorded, skipped, and reported rather than crashed on.)"""
    rows = [json.loads(line) for line in (traj_dir / "manifest.jsonl").read_text().splitlines()]
    seq = [r for r in rows if r.get("error") is None and r["mode"] == "SEQUENTIAL"]
    have = {p.stem for p in (traj_dir / "shards").glob("*.npz")}
    kept = [r for r in seq if r["story_id"] in have]
    return kept, len(seq) - len(kept)


def probe_groups(traj_dir: Path) -> dict[str, list[int]]:
    labels = json.loads((traj_dir / "probe_labels.json").read_text())
    groups: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab.split(":", 1)[0], []).append(i)
    groups["_labels"] = labels  # type: ignore[assignment]
    return groups


def story_set_mean(traj_dir: Path, rows: list[dict]) -> np.ndarray:
    """Token-weighted mean of raw dots over the story set: [layers, probes]."""
    total = None
    n = 0
    for r in rows:
        d = np.load(traj_dir / "shards" / f"{r['story_id']}.npz")
        dots = d["dots"].astype(np.float64)
        total = dots.sum(axis=0) if total is None else total + dots.sum(axis=0)
        n += dots.shape[0]
    return (total / n).astype(np.float32)


def collect(traj_dir: Path, rows: list[dict], mean_dots: np.ndarray) -> dict:
    """Per-phase mean centered cosines and per-transition leads, all probes.

    phase_stats: [phases, layers, probes]; phase_target: probe-label index
    of the tagged emotion is resolved later per bank. trans_lead likewise
    [transitions, layers, probes] with the incoming emotion recorded."""
    phase_stats, phase_emotion, trans_lead, trans_incoming = [], [], [], []
    for r in rows:
        d = np.load(traj_dir / "shards" / f"{r['story_id']}.npz")
        cos = (d["dots"].astype(np.float32) - mean_dots) / np.clip(
            d["norms_centered"].astype(np.float32)[:, :, None], 1e-6, None
        )
        T = cos.shape[0]
        starts = list(r["phase_token_starts"]) + [T]
        for k, emotion in enumerate(r["phase_emotions"]):
            a, b = starts[k], starts[k + 1]
            if b - a < 4:
                continue
            phase_stats.append(cos[a:b].mean(axis=0))
            phase_emotion.append(emotion)
        for k in range(1, len(r["phase_emotions"])):
            b = starts[k]
            if b < 2 * W or T - b < W:
                continue
            lead = cos[b - W : b].mean(axis=0) - cos[b - 2 * W : b - W].mean(axis=0)
            trans_lead.append(lead)
            trans_incoming.append(r["phase_emotions"][k])
    return {
        "phase_stats": np.stack(phase_stats),
        "phase_emotion": phase_emotion,
        "trans_lead": np.stack(trans_lead),
        "trans_incoming": trans_incoming,
    }


def emotion_index(labels: list[str], bank: str, emotion: str) -> int | None:
    try:
        return labels.index(f"{bank}:{emotion}")
    except ValueError:
        return None


def gate_read(c: dict, groups: dict, bank: str, rng: np.random.Generator) -> dict:
    """Per-layer median rank of the tagged emotion within the bank + N2."""
    labels, idx = groups["_labels"], groups[bank]
    keep = [
        i for i, e in enumerate(c["phase_emotion"]) if emotion_index(labels, bank, e) is not None
    ]
    if not keep:
        return {"n_phases": 0}
    stats = c["phase_stats"][keep][:, :, idx]  # [phases, layers, bank]
    targets = np.array(
        [idx.index(emotion_index(labels, bank, c["phase_emotion"][i])) for i in keep]
    )
    ranks = (stats > stats[np.arange(len(keep)), :, targets][:, :, None]).sum(axis=2) + 1
    out = {"n_phases": len(keep), "per_layer": {}}
    for li, layer in enumerate(LAYERS):
        observed = float(np.median(ranks[:, li]))
        null = np.median(
            np.take_along_axis(
                np.argsort(np.argsort(-stats[:, li], axis=1), axis=1) + 1,
                rng.integers(0, len(idx), size=(len(keep), N_SHUFFLES)),
                axis=1,
            ),
            axis=0,
        )
        p = float((null <= observed).mean())
        out["per_layer"][layer] = {
            "median_rank": observed,
            "n2_p": p,
            "bar_rank": len(idx) // 10 or 1,
            "passes": observed <= max(len(idx) // 10, 1) and p < 0.001,
        }
    return out


def r1_read(c: dict, groups: dict, bank: str, noise_sd: dict, rng: np.random.Generator) -> dict:
    labels, idx = groups["_labels"], groups[bank]
    keep = [
        i for i, e in enumerate(c["trans_incoming"]) if emotion_index(labels, bank, e) is not None
    ]
    if not keep:
        return {"n_transitions": 0}
    leads = c["trans_lead"][keep][:, :, idx]  # [trans, layers, bank]
    targets = np.array(
        [idx.index(emotion_index(labels, bank, c["trans_incoming"][i])) for i in keep]
    )
    true_leads = leads[np.arange(len(keep)), :, targets]  # [trans, layers]
    out = {"n_transitions": len(keep), "per_layer": {}}
    shuffle_targets = rng.integers(0, len(idx), size=(N_SHUFFLES, len(keep)))
    for li, layer in enumerate(LAYERS):
        observed = float(true_leads[:, li].mean())
        null_means = leads[np.arange(len(keep))[None, :], li, shuffle_targets].mean(axis=1)
        p = float((null_means >= observed).mean())
        sd = noise_sd.get(str(layer))
        out["per_layer"][layer] = {
            "mean_lead": observed,
            "n2_p": p,
            "noise_sd": sd,
            "bar_lead": None if sd is None else 0.5 * sd,
            "passes": p < 0.001 and (sd is not None and observed >= 0.5 * sd),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="smoke: first N stories")
    args = parser.parse_args()
    calib = json.loads(args.calibration.read_text())
    noise_sd = {layer: v["cos_sd"] for layer, v in calib["random_probe"].items()}
    rng = np.random.default_rng(20260723)

    def run(traj_dir: Path) -> dict:
        rows, n_orphaned = manifest_rows(traj_dir)
        if args.limit:
            rows = rows[: args.limit]
        groups = probe_groups(traj_dir)
        mean_dots = story_set_mean(traj_dir, rows)
        c = collect(traj_dir, rows, mean_dots)
        banks = [b for b in ("corpus", "selfgen", "deepseek", "random") if b in groups]
        return {
            "n_stories": len(rows),
            "n_orphaned_manifest_rows": n_orphaned,
            "gate_G": {b: gate_read(c, groups, b, rng) for b in banks},
            "r1_anticipation": {b: r1_read(c, groups, b, noise_sd, rng) for b in banks},
        }

    result = {
        "experiment": "Q3.H1.E1 gate G + R1 (boundary-referenced tranche)",
        "conventions": "centered cosine (story-set mean, norms_centered); W=16; "
        "N2 10k label shuffles; N1 = random bank scored identically; "
        "cue-referenced R1 twin PENDING the judge bulk pass",
        "primary_cell": {"layer": PRIMARY_LAYER, "bank": "corpus"},
        "true_arm": run(args.traj_dir),
    }
    if args.control_dir and args.control_dir.exists():
        result["control_arm"] = run(args.control_dir)
    args.out.write_text(json.dumps(result, indent=2))
    for read in ("gate_G", "r1_anticipation"):
        for bank, r in result["true_arm"][read].items():
            cell = r.get("per_layer", {}).get(PRIMARY_LAYER)
            if cell:
                print(f"{read} {bank:8s} L33: {cell}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
