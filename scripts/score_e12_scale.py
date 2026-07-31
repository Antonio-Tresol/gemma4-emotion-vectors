"""Q1.H2.E12 scoring: scale-and-diversity dose-response in the strong lineage.

Registered reads (TREE Q1.H2.E12, locked before the diverse corpus existed):
(R1) E11's dual-battery centered-cosine grid as a function of n per emotion,
     via seeded subsampling of per-story residual shards — passing-layer count
     (pass = >= 8/12 on BOTH batteries) and best-layer counts, 5 seeds per n.
(R2) contrast cosine to the post-fix self-gen probes at layer 33, vs n.
(R3) at matched n=256, diverse-prompt arm vs fixed-prompt arm passing layers.

Each arm is a directory of per-story shards named <emotion>__<idx>.npy with
shape [20 layers, 5376]; the E11 scorer supplies the readout conventions.

    .venv/bin/python scripts/score_e12_scale.py \
        --arm fixed=results/openrouter_vectors_it_shards \
        --arm diverse=results/openrouter_vectors_diverse_shards \
        --results-root results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from jaxtyping import Float

sys.path.insert(0, str(Path(__file__).parent))
from score_e11_lineage import BATTERY, battery_counts, contrast_probes_12pool, load_battery_means

from emotion_vectors.probe_prompts import HELDOUT_SCENARIOS, SCENARIOS

N_GRID = [8, 16, 32, 64, 128, 256, 512, 1024]
N_SEEDS = 5


def load_shards(shard_dir: Path) -> dict[str, Float[np.ndarray, "stories layers d_model"]]:
    """Per-emotion stacks of per-story mean residuals, [stories, 20, 5376]."""
    # shard filename convention: <emotion>__<idx>.npy
    files_by_emotion: dict[str, list[Path]] = defaultdict(list)
    for shard_file in sorted(shard_dir.glob("*.npy")):
        emotion = shard_file.stem.rsplit("__", 1)[0]
        files_by_emotion[emotion].append(shard_file)

    stacks = {}
    for emotion, files in files_by_emotion.items():
        stacks[emotion] = np.stack([np.load(shard_file) for shard_file in files])
    return stacks


def grid_read(
    probes_by_layer: Float[np.ndarray, "emotions layers d_model"],
    sweep: dict[str, object],
    layers: list[int],
) -> dict[str, object]:
    """E11's R1 on one probe bundle: per-layer dual-battery counts + passes."""
    # The sweep's activation rows are ordered: all paper-battery scenarios,
    # then all held-out scenarios. Rebuild that order with each row's target.
    order: list[tuple[str, str]] = []
    for kind, battery in (("scenario", SCENARIOS), ("heldout", HELDOUT_SCENARIOS)):
        for _name, target, _prompt in battery:
            order.append((kind, target))
    rows = {
        battery_kind: [i for i, (kind, _) in enumerate(order) if kind == battery_kind]
        for battery_kind in ("scenario", "heldout")
    }
    targets = {battery_kind: [order[i][1] for i in rows[battery_kind]] for battery_kind in rows}

    acts = sweep["chat_last"].astype(np.float32)
    per_layer, passing_layers = {}, []
    for layer_pos, layer in enumerate(layers):
        probes = contrast_probes_12pool(probes_by_layer[:, layer_pos])
        paper_count = battery_counts(acts[rows["scenario"], layer_pos], probes, targets["scenario"])
        held_count = battery_counts(acts[rows["heldout"], layer_pos], probes, targets["heldout"])
        per_layer[layer] = [paper_count, held_count]
        if paper_count >= 8 and held_count >= 8:
            passing_layers.append(layer)
    best_layer, best_counts = max(per_layer.items(), key=lambda kv: sum(kv[1]))
    return {
        "per_layer": per_layer,
        "passing_layers": passing_layers,
        "best": {"layer": best_layer, "counts": best_counts},
    }


def curve_for_arm(
    stacks: dict[str, Float[np.ndarray, "stories layers d_model"]],
    sweep,
    layers,
    selfgen_probes,
    layer33_pos,
) -> dict[str, object]:
    n_avail = min(len(stacks[emotion]) for emotion in BATTERY)
    curve: dict[str, list] = {"n_available_min": n_avail, "points": []}

    # seeded subsample points, plus one deterministic full-corpus anchor
    # (seed -1) so the top of the curve survives dropped-story off-by-a-few
    jobs = [(n, seed) for n in N_GRID if n <= n_avail for seed in range(N_SEEDS)]
    jobs.append((n_avail, -1))

    for n, seed in jobs:
        if seed == -1:
            # full-corpus anchor: mean over every story, no sampling
            means = np.stack([stacks[emotion].mean(0) for emotion in BATTERY])
        else:
            # subsample n stories per emotion (without replacement), then mean
            rng = np.random.default_rng(1000 * n + seed)
            per_emotion_means = []
            for emotion in BATTERY:
                stack = stacks[emotion]
                chosen = rng.choice(len(stack), n, replace=False)
                per_emotion_means.append(stack[chosen].mean(0))
            means = np.stack(per_emotion_means)

        r1 = grid_read(means, sweep, layers)
        probes33 = contrast_probes_12pool(means[:, layer33_pos])
        cos_selfgen = float(np.sum(probes33 * selfgen_probes, axis=1).mean())
        curve["points"].append(
            {
                "n": n,
                "seed": seed,
                "n_passing_layers": len(r1["passing_layers"]),
                "passing_layers": r1["passing_layers"],
                "best": r1["best"],
                "mean_contrast_cos_to_selfgen_L33": cos_selfgen,
            }
        )
    return curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=SHARD_DIR")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--selfgen",
        type=Path,
        default=Path("results/self_story_vectors_it_postfix/emotion_means.npz"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/e12_scale_curve.json"))
    args = parser.parse_args()
    sweep = np.load(args.results_root / "probe_sweep_it/activations.npz", allow_pickle=True)
    layers = [int(x) for x in sweep["layers"]]
    layer33_pos = layers.index(33)
    selfgen_probes = contrast_probes_12pool(load_battery_means(args.selfgen)[:, layer33_pos])

    result = {"experiment": "Q1.H2.E12", "n_grid": N_GRID, "n_seeds": N_SEEDS, "arms": {}}
    for spec in args.arm:
        name, shard_dir = spec.split("=", 1)
        stacks = load_shards(Path(shard_dir))
        result["arms"][name] = curve_for_arm(stacks, sweep, layers, selfgen_probes, layer33_pos)

        # console summary: per n, the range of passing-layer counts across seeds
        points = result["arms"][name]["points"]
        for n in sorted({point["n"] for point in points}):
            pass_counts = [point["n_passing_layers"] for point in points if point["n"] == n]
            cosines = [
                point["mean_contrast_cos_to_selfgen_L33"] for point in points if point["n"] == n
            ]
            print(
                f"{name:8s} n={n:5d}: passing layers {min(pass_counts)}-{max(pass_counts)} "
                f"(mean {np.mean(pass_counts):.1f}) | cos-to-selfgen {np.mean(cosines):.3f}"
            )
    args.out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
