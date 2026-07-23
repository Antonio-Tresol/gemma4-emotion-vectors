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
    sequential = [row for row in rows if row.get("error") is None and row["mode"] == "SEQUENTIAL"]
    shard_ids_on_disk = {path.stem for path in (traj_dir / "shards").glob("*.npz")}
    kept = [row for row in sequential if row["story_id"] in shard_ids_on_disk]
    return kept, len(sequential) - len(kept)


def probe_groups(traj_dir: Path) -> dict[str, list[int]]:
    """Probe-column indices grouped by bank; labels look like '<bank>:<emotion>'."""
    labels = json.loads((traj_dir / "probe_labels.json").read_text())
    groups: dict[str, list[int]] = {}
    for col, label in enumerate(labels):
        bank = label.split(":", 1)[0]
        groups.setdefault(bank, []).append(col)
    groups["_labels"] = labels  # type: ignore[assignment]
    return groups


def story_set_mean(traj_dir: Path, rows: list[dict]) -> np.ndarray:
    """Token-weighted mean of raw dots over the story set: [layers, probes]."""
    total = None
    n_tokens = 0
    for row in rows:
        shard = np.load(traj_dir / "shards" / f"{row['story_id']}.npz")
        dots = shard["dots"].astype(np.float64)
        total = dots.sum(axis=0) if total is None else total + dots.sum(axis=0)
        n_tokens += dots.shape[0]
    return (total / n_tokens).astype(np.float32)


def collect(traj_dir: Path, rows: list[dict], mean_dots: np.ndarray) -> dict:
    """Per-phase mean centered cosines and per-transition leads, all probes.

    phase_stats: [phases, layers, probes]; phase_target: probe-label index
    of the tagged emotion is resolved later per bank. trans_lead likewise
    [transitions, layers, probes] with the incoming emotion recorded."""
    phase_stats, phase_emotion, trans_lead, trans_incoming = [], [], [], []
    for row in rows:
        shard = np.load(traj_dir / "shards" / f"{row['story_id']}.npz")

        # centered cosine per (token, layer, probe): story-set mean subtracted
        # from the stored dots, norms_centered denominator
        cos = (shard["dots"].astype(np.float32) - mean_dots) / np.clip(
            shard["norms_centered"].astype(np.float32)[:, :, None], 1e-6, None
        )
        n_tokens = cos.shape[0]
        starts = list(row["phase_token_starts"]) + [n_tokens]

        # per-phase mean cosine (phases shorter than 4 tokens are skipped)
        for k, emotion in enumerate(row["phase_emotions"]):
            phase_start, phase_end = starts[k], starts[k + 1]
            if phase_end - phase_start < 4:
                continue
            phase_stats.append(cos[phase_start:phase_end].mean(axis=0))
            phase_emotion.append(emotion)

        # per-transition lead: window [b-W, b) minus [b-2W, b-W) at boundary b;
        # skipped when either window would run off the story
        for k in range(1, len(row["phase_emotions"])):
            boundary = starts[k]
            if boundary < 2 * W or n_tokens - boundary < W:
                continue
            near_window = cos[boundary - W : boundary].mean(axis=0)
            earlier_window = cos[boundary - 2 * W : boundary - W].mean(axis=0)
            trans_lead.append(near_window - earlier_window)
            trans_incoming.append(row["phase_emotions"][k])
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


def gate_read(collected: dict, groups: dict, bank: str, rng: np.random.Generator) -> dict:
    """Per-layer median rank of the tagged emotion within the bank + N2."""
    labels, bank_cols = groups["_labels"], groups[bank]

    # phases whose tagged emotion has a probe in this bank
    kept = [
        i
        for i, emotion in enumerate(collected["phase_emotion"])
        if emotion_index(labels, bank, emotion) is not None
    ]
    if not kept:
        return {"n_phases": 0}
    n_phases = len(kept)
    scores = collected["phase_stats"][kept][:, :, bank_cols]  # [phases, layers, bank_probes]
    target_probe = np.array(
        [bank_cols.index(emotion_index(labels, bank, collected["phase_emotion"][i])) for i in kept]
    )

    # rank of the tagged emotion within the bank, per (phase, layer):
    # 1 + how many other probes scored above it
    target_score = scores[np.arange(n_phases), :, target_probe]  # [phases, layers]
    n_probes_above_target = (scores > target_score[:, :, None]).sum(axis=2)
    ranks = n_probes_above_target + 1

    bar_rank = max(len(bank_cols) // 10, 1)  # top decile of the bank
    out = {"n_phases": n_phases, "per_layer": {}}
    for layer_pos, layer in enumerate(LAYERS):
        observed_median_rank = float(np.median(ranks[:, layer_pos]))

        # N2 null: what median rank does a randomly assigned WRONG emotion
        # get? Precompute every probe's within-phase rank once, so each of
        # the 10,000 shuffles is just an integer lookup.
        descending_order = np.argsort(-scores[:, layer_pos], axis=1)
        rank_of_each_probe = np.argsort(descending_order, axis=1) + 1  # [phases, bank_probes]
        wrong_emotion_picks = rng.integers(0, len(bank_cols), size=(n_phases, N_SHUFFLES))
        wrong_emotion_ranks = np.take_along_axis(rank_of_each_probe, wrong_emotion_picks, axis=1)
        null_median_ranks = np.median(wrong_emotion_ranks, axis=0)  # [shuffles]

        p_value = float((null_median_ranks <= observed_median_rank).mean())
        out["per_layer"][layer] = {
            "median_rank": observed_median_rank,
            "n2_p": p_value,
            "bar_rank": bar_rank,
            "passes": observed_median_rank <= bar_rank and p_value < 0.001,
        }
    return out


def r1_read(
    collected: dict, groups: dict, bank: str, noise_sd: dict, rng: np.random.Generator
) -> dict:
    labels, bank_cols = groups["_labels"], groups[bank]

    # transitions whose incoming emotion has a probe in this bank
    kept = [
        i
        for i, emotion in enumerate(collected["trans_incoming"])
        if emotion_index(labels, bank, emotion) is not None
    ]
    if not kept:
        return {"n_transitions": 0}
    n_transitions = len(kept)
    leads = collected["trans_lead"][kept][:, :, bank_cols]  # [transitions, layers, bank_probes]
    incoming_probe = np.array(
        [bank_cols.index(emotion_index(labels, bank, collected["trans_incoming"][i])) for i in kept]
    )

    # each transition's lead for its actual incoming emotion
    true_leads = leads[np.arange(n_transitions), :, incoming_probe]  # [transitions, layers]

    # N2 null: mean lead when every transition is assigned a random wrong
    # emotion instead; one row of picks per shuffle
    wrong_emotion_picks = rng.integers(0, len(bank_cols), size=(N_SHUFFLES, n_transitions))

    out = {"n_transitions": n_transitions, "per_layer": {}}
    for layer_pos, layer in enumerate(LAYERS):
        observed_mean_lead = float(true_leads[:, layer_pos].mean())
        leads_at_layer = leads[:, layer_pos]  # [transitions, bank_probes]
        wrong_leads = leads_at_layer[np.arange(n_transitions)[None, :], wrong_emotion_picks]
        null_mean_leads = wrong_leads.mean(axis=1)  # [shuffles]
        p_value = float((null_mean_leads >= observed_mean_lead).mean())

        sd = noise_sd.get(str(layer))
        bar_lead = None if sd is None else 0.5 * sd
        out["per_layer"][layer] = {
            "mean_lead": observed_mean_lead,
            "n2_p": p_value,
            "noise_sd": sd,
            "bar_lead": bar_lead,
            "passes": p_value < 0.001 and bar_lead is not None and observed_mean_lead >= bar_lead,
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
        collected = collect(traj_dir, rows, mean_dots)
        banks = [bank for bank in ("corpus", "selfgen", "deepseek", "random") if bank in groups]
        return {
            "n_stories": len(rows),
            "n_orphaned_manifest_rows": n_orphaned,
            "gate_G": {bank: gate_read(collected, groups, bank, rng) for bank in banks},
            "r1_anticipation": {
                bank: r1_read(collected, groups, bank, noise_sd, rng) for bank in banks
            },
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
        for bank, bank_result in result["true_arm"][read].items():
            cell = bank_result.get("per_layer", {}).get(PRIMARY_LAYER)
            if cell:
                print(f"{read} {bank:8s} L33: {cell}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
