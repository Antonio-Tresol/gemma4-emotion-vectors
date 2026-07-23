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

sys.path.insert(0, str(Path(__file__).parent))
from score_e11_lineage import BATTERY, battery_counts, contrast_probes_12pool, load_battery_means

from emotion_vectors.probe_prompts import HELDOUT_SCENARIOS, SCENARIOS

N_GRID = [8, 16, 32, 64, 128, 256, 512, 1024]
N_SEEDS = 5


def load_shards(shard_dir: Path) -> dict[str, np.ndarray]:
    """Per-emotion stacks of per-story mean residuals, [stories, 20, 5376]."""
    by_emotion: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(shard_dir.glob("*.npy")):
        by_emotion[f.stem.rsplit("__", 1)[0]].append(f)
    return {e: np.stack([np.load(f) for f in files]) for e, files in by_emotion.items()}


def grid_read(probes_by_layer: np.ndarray, sweep: dict, layers: list[int]) -> dict:
    """E11's R1 on one probe bundle: per-layer dual-battery counts + passes."""
    order = [
        (kind, target)
        for kind, battery in (("scenario", SCENARIOS), ("heldout", HELDOUT_SCENARIOS))
        for _, target, _ in battery
    ]
    acts = sweep["chat_last"].astype(np.float32)
    rows = {
        k: [i for i, (kind, _) in enumerate(order) if kind == k] for k in ("scenario", "heldout")
    }
    targets = {k: [order[i][1] for i in rows[k]] for k in rows}
    per_layer, passes = {}, []
    for lp, layer in enumerate(layers):
        probes = contrast_probes_12pool(probes_by_layer[:, lp])
        paper = battery_counts(acts[rows["scenario"], lp], probes, targets["scenario"])
        held = battery_counts(acts[rows["heldout"], lp], probes, targets["heldout"])
        per_layer[layer] = [paper, held]
        if paper >= 8 and held >= 8:
            passes.append(layer)
    best = max(per_layer.items(), key=lambda kv: sum(kv[1]))
    return {
        "per_layer": per_layer,
        "passing_layers": passes,
        "best": {"layer": best[0], "counts": best[1]},
    }


def curve_for_arm(stacks: dict[str, np.ndarray], sweep, layers, selfgen, l33) -> dict:
    n_avail = min(len(stacks[e]) for e in BATTERY)
    curve: dict[str, list] = {"n_available_min": n_avail, "points": []}
    # seeded subsample points, plus one deterministic full-corpus anchor
    # (seed -1) so the top of the curve survives dropped-story off-by-a-few
    jobs = [(n, seed) for n in N_GRID if n <= n_avail for seed in range(N_SEEDS)]
    jobs.append((n_avail, -1))
    for n, seed in jobs:
        if seed == -1:
            means = np.stack([stacks[e].mean(0) for e in BATTERY])
        else:
            rng = np.random.default_rng(1000 * n + seed)
            means = np.stack(
                [stacks[e][rng.choice(len(stacks[e]), n, replace=False)].mean(0) for e in BATTERY]
            )
        r1 = grid_read(means, sweep, layers)
        probes33 = contrast_probes_12pool(means[:, l33])
        cos_selfgen = float(np.sum(probes33 * selfgen, axis=1).mean())
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
    l33 = layers.index(33)
    selfgen = contrast_probes_12pool(load_battery_means(args.selfgen)[:, l33])
    result = {"experiment": "Q1.H2.E12", "n_grid": N_GRID, "n_seeds": N_SEEDS, "arms": {}}
    for spec in args.arm:
        name, shard_dir = spec.split("=", 1)
        stacks = load_shards(Path(shard_dir))
        result["arms"][name] = curve_for_arm(stacks, sweep, layers, selfgen, l33)
        pts = result["arms"][name]["points"]
        for n in sorted({p["n"] for p in pts}):
            ns = [p["n_passing_layers"] for p in pts if p["n"] == n]
            cs = [p["mean_contrast_cos_to_selfgen_L33"] for p in pts if p["n"] == n]
            print(
                f"{name:8s} n={n:5d}: passing layers {min(ns)}-{max(ns)} (mean {np.mean(ns):.1f}) | cos-to-selfgen {np.mean(cs):.3f}"
            )
    args.out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
