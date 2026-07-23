"""Q1.H2.E11 scoring: self-gen vs strong-external vs weak-external probes.

The registered reads (TREE Q1.H2.E11, locked before any story existed):
(R1) E10's dual-battery centered-cosine grid — chat_last readout, all 20
     layers, matched 12-pool centering, target-in-top-3 counts on the paper
     and held-out batteries; a configuration passes at >= 8/12 on BOTH.
(R2) cross-lineage contrast cosines and 12x12 representational similarity
     analysis (RSA) at layer 33.
(R3) preference probe-Elo max |r| at the four measured layers on the FIXED
     chat preference data.

Post-fix convention (agreed with the E4b session 2026-07-22): all three
arms score from post-fix extractions — pass the bundles explicitly:

    .venv/bin/python scripts/score_e11_lineage.py \
        --selfgen results/self_story_vectors_it_postfix/emotion_means.npz \
        --weak results/emotion_vectors_it_postfix/emotion_means.npz \
        --strong results/openrouter_vectors_it/emotion_means.npz
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

from emotion_vectors.probe_prompts import HELDOUT_SCENARIOS, SCENARIOS

BATTERY = "happy inspired loving proud calm desperate angry guilty sad afraid nervous surprised".split()


def r4_diversity_exploratory(corpora: dict[str, Path | None], seed: int = 0) -> dict:
    """EXPLORATORY (registered as such, fenced from R1-R3): within-corpus mean
    pairwise lexical similarity — 5-gram Jaccard over a seeded sample of up to
    30 stories per battery emotion. Separates 'distribution match' from
    'scaffold degeneracy' as candidate mechanisms across the three lineages."""
    out: dict[str, object] = {"read": "mean pairwise 5-gram Jaccard, <=30 stories/emotion sample"}
    for name, path in corpora.items():
        if path is None or not path.exists():
            out[name] = None
            continue
        grouped: dict[str, list[str]] = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("stories") and row["emotion"] in BATTERY:
                grouped[row["emotion"]] = row["stories"]
        rng = random.Random(seed)
        sims: list[float] = []
        for stories in grouped.values():
            sample = rng.sample(stories, min(30, len(stories)))
            grams = [
                {" ".join(ws[i : i + 5]) for ws in [s.lower().split()] for i in range(len(ws) - 4)}
                for s in sample
            ]
            for a, b in itertools.combinations(grams, 2):
                union = len(a | b)
                if union:
                    sims.append(len(a & b) / union)
        out[name] = {"mean_pairwise_jaccard": float(np.mean(sims)), "n_pairs": len(sims)}
    return out


def load_battery_means(path: Path) -> np.ndarray:
    """[12, 20, 5376] float32 means for the battery emotions, any bundle shape."""
    z = np.load(path, allow_pickle=True)
    means = z["means"].astype(np.float32)
    if means.ndim == 4:  # e6-style n-buckets: take the largest
        means = means[-1]
    emotions = [str(e) for e in z["emotions"]]
    idx = [emotions.index(e) for e in BATTERY]
    return means[idx]


def contrast_probes_12pool(means: np.ndarray) -> np.ndarray:
    """Center on the 12-emotion pool (E10's matched convention), unit-normalize."""
    contrast = means - means.mean(axis=0, keepdims=True)
    return contrast / np.clip(np.linalg.norm(contrast, axis=-1, keepdims=True), 1e-8, None)


def battery_counts(acts: np.ndarray, probes: np.ndarray, targets: list[str]) -> int:
    """Target-in-top-3 count under the centered-cosine readout (E9 convention:
    activations centered on the scenario set's own mean)."""
    centered = acts - acts.mean(axis=0, keepdims=True)
    cos = centered @ probes.T
    cos /= np.clip(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8, None)
    count = 0
    for row, target in zip(cos, targets):
        top3 = [BATTERY[j] for j in np.argsort(row)[::-1][:3]]
        count += target in top3
    return count


def r1_grid(bundles: dict[str, np.ndarray], sweep: dict) -> dict:
    order = [(kind, name, target) for kind, battery in (("scenario", SCENARIOS), ("heldout", HELDOUT_SCENARIOS)) for name, target, _ in battery]
    paper_rows = [i for i, (k, _, _) in enumerate(order) if k == "scenario"]
    held_rows = [i for i, (k, _, _) in enumerate(order) if k == "heldout"]
    paper_targets = [order[i][2] for i in paper_rows]
    held_targets = [order[i][2] for i in held_rows]
    acts_all = sweep["chat_last"].astype(np.float32)  # [prompts, 20, 5376]
    layers = [int(x) for x in sweep["layers"]]
    out: dict[str, dict] = {}
    for name, means in bundles.items():
        per_layer = {}
        passes = []
        for lp, layer in enumerate(layers):
            probes = contrast_probes_12pool(means[:, lp])
            paper = battery_counts(acts_all[paper_rows, lp], probes, paper_targets)
            held = battery_counts(acts_all[held_rows, lp], probes, held_targets)
            per_layer[layer] = [paper, held]
            if paper >= 8 and held >= 8:
                passes.append(layer)
        best = max(per_layer.items(), key=lambda kv: sum(kv[1]))
        out[name] = {"per_layer": per_layer, "passing_layers": passes, "best": {"layer": best[0], "counts": best[1]}}
    return out


def r2_cross(bundles: dict[str, np.ndarray], layer_pos: int) -> dict:
    names = list(bundles)
    probes = {n: contrast_probes_12pool(bundles[n][:, layer_pos]) for n in names}
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            per_emotion = np.sum(probes[a] * probes[b], axis=1)
            sim_a = probes[a] @ probes[a].T
            sim_b = probes[b] @ probes[b].T
            tri = np.triu_indices(len(BATTERY), k=1)
            rsa = float(np.corrcoef(sim_a[tri], sim_b[tri])[0, 1])
            out[f"{a}_vs_{b}"] = {
                "mean_contrast_cos": float(per_emotion.mean()),
                "range": [float(per_emotion.min()), float(per_emotion.max())],
                "rsa_12x12": rsa,
            }
    return out


def r3_preferences(bundles: dict[str, np.ndarray], sweep_layers: list[int]) -> dict:
    pref = np.load(PREF_DIR / "preferences.npz", allow_pickle=True)
    scores = json.loads((PREF_DIR / "scores.json").read_text())
    # rows ordered like activities.py, the same order feel_feats rows use
    elo = np.array([row["elo"] for row in scores["elo_per_activity"]], dtype=np.float64)
    feel = pref["feel_feats"].astype(np.float32)  # [64 activities, 4 layers, 5376]
    pref_layers = [int(x) for x in pref["layers"]]
    out = {}
    for name, means in bundles.items():
        best = {"abs_r": 0.0, "layer": None, "emotion": None}
        for pl, layer in enumerate(pref_layers):
            lp = sweep_layers.index(layer)
            probes = contrast_probes_12pool(means[:, lp])
            acts = feel[:, pl]
            cos = (acts / np.clip(np.linalg.norm(acts, axis=1, keepdims=True), 1e-8, None)) @ probes.T
            for e_i, emotion in enumerate(BATTERY):
                r = float(np.corrcoef(cos[:, e_i], elo)[0, 1])
                if abs(r) > best["abs_r"]:
                    best = {"abs_r": abs(r), "r": r, "layer": layer, "emotion": emotion}
        out[name] = best
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfgen", type=Path, required=True)
    parser.add_argument("--weak", type=Path, required=True)
    parser.add_argument("--strong", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/e11_lineage.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results"),
                        help="where the sweep/preference inputs live (the main clone's results/)")
    parser.add_argument("--selfgen-stories", type=Path, default=None)
    parser.add_argument("--weak-stories", type=Path, default=None)
    parser.add_argument("--strong-stories", type=Path, default=None)
    args = parser.parse_args()
    global SWEEP, PREF_DIR
    SWEEP = args.results_root / "probe_sweep_it/activations.npz"
    PREF_DIR = args.results_root / "preferences_it_chat_fixed"
    bundles = {
        "selfgen": load_battery_means(args.selfgen),
        "weak_external": load_battery_means(args.weak),
        "strong_external": load_battery_means(args.strong),
    }
    sweep = np.load(SWEEP, allow_pickle=True)
    layers = [int(x) for x in sweep["layers"]]
    result = {
        "experiment": "Q1.H2.E11",
        "probe_files": {k: str(v) for k, v in
                        {"selfgen": args.selfgen, "weak": args.weak, "strong": args.strong}.items()},
        "convention": "post-fix bundles on all arms; 12-pool probe centering; "
        "scenario-set activation centering; chat_last readout",
        "r1_dual_battery": r1_grid(bundles, sweep),
        "r2_cross_lineage_layer33": r2_cross(bundles, layers.index(33)),
        "r3_preference_probe_elo": r3_preferences(bundles, layers),
        "r4_diversity_EXPLORATORY": r4_diversity_exploratory(
            {"selfgen": args.selfgen_stories, "weak_external": args.weak_stories,
             "strong_external": args.strong_stories}
        ),
    }
    args.out.write_text(json.dumps(result, indent=2))
    for name, r1 in result["r1_dual_battery"].items():
        print(f"{name:16s} best layer {r1['best']['layer']}: {r1['best']['counts']} | passing: {r1['passing_layers']}")
    print(json.dumps(result["r2_cross_lineage_layer33"], indent=2))
    print("R3:", {k: round(v["abs_r"], 3) for k, v in result["r3_preference_probe_elo"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
