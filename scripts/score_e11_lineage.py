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
from jaxtyping import Float

from emotion_vectors.probe_prompts import HELDOUT_SCENARIOS, SCENARIOS

BATTERY = (
    "happy inspired loving proud calm desperate angry guilty sad afraid nervous surprised".split()
)


def r4_diversity_exploratory(corpora: dict[str, Path | None], seed: int = 0) -> dict[str, object]:
    """EXPLORATORY (registered as such, fenced from R1-R3): within-corpus mean
    pairwise lexical similarity — 5-gram Jaccard over a seeded sample of up to
    30 stories per battery emotion. Separates 'distribution match' from
    'scaffold degeneracy' as candidate mechanisms across the three lineages."""
    out: dict[str, object] = {"read": "mean pairwise 5-gram Jaccard, <=30 stories/emotion sample"}
    for name, path in corpora.items():
        if path is None or not path.exists():
            out[name] = None
            continue

        # stories per battery emotion, in the corpus file's own line order
        stories_by_emotion: dict[str, list[str]] = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("stories") and row["emotion"] in BATTERY:
                stories_by_emotion[row["emotion"]] = row["stories"]

        rng = random.Random(seed)
        similarities: list[float] = []
        for stories in stories_by_emotion.values():
            sample = rng.sample(stories, min(30, len(stories)))

            # each story becomes its set of word 5-grams
            gram_sets: list[set[str]] = []
            for story in sample:
                words = story.lower().split()
                grams = {" ".join(words[i : i + 5]) for i in range(len(words) - 4)}
                gram_sets.append(grams)

            # Jaccard similarity for every within-emotion story pair
            for grams_a, grams_b in itertools.combinations(gram_sets, 2):
                union = len(grams_a | grams_b)
                if union:
                    similarities.append(len(grams_a & grams_b) / union)

        out[name] = {
            "mean_pairwise_jaccard": float(np.mean(similarities)),
            "n_pairs": len(similarities),
        }
    return out


def load_battery_means(path: Path) -> Float[np.ndarray, "emotions layers d_model"]:
    """[12, 20, 5376] float32 means for the battery emotions, any bundle shape."""
    bundle = np.load(path, allow_pickle=True)
    means = bundle["means"].astype(np.float32)
    if means.ndim == 4:  # e6-style n-buckets: take the largest
        means = means[-1]
    emotions = [str(e) for e in bundle["emotions"]]
    battery_rows = [emotions.index(e) for e in BATTERY]
    return means[battery_rows]


def contrast_probes_12pool(
    means: Float[np.ndarray, "emotions d_model"],
) -> Float[np.ndarray, "emotions d_model"]:
    """Center on the 12-emotion pool (E10's matched convention), unit-normalize."""
    contrast = means - means.mean(axis=0, keepdims=True)
    return contrast / np.clip(np.linalg.norm(contrast, axis=-1, keepdims=True), 1e-8, None)


def battery_counts(
    acts: Float[np.ndarray, "scenarios d_model"],
    probes: Float[np.ndarray, "emotions d_model"],
    targets: list[str],
) -> int:
    """Target-in-top-3 count under the centered-cosine readout (E9 convention:
    activations centered on the scenario set's own mean)."""
    centered = acts - acts.mean(axis=0, keepdims=True)
    cos = centered @ probes.T
    cos /= np.clip(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8, None)
    count = 0
    for cos_row, target in zip(cos, targets):
        descending = np.argsort(cos_row)[::-1]
        top3 = [BATTERY[j] for j in descending[:3]]
        count += target in top3
    return count


def r1_grid(
    bundles: dict[str, Float[np.ndarray, "emotions layers d_model"]], sweep: dict[str, object]
) -> dict[str, object]:
    # The sweep's activation rows are ordered: all paper-battery scenarios,
    # then all held-out scenarios. Rebuild that order with each row's target.
    order: list[tuple[str, str, str]] = []
    for kind, battery in (("scenario", SCENARIOS), ("heldout", HELDOUT_SCENARIOS)):
        for name, target, _prompt in battery:
            order.append((kind, name, target))
    paper_rows = [i for i, (kind, _, _) in enumerate(order) if kind == "scenario"]
    held_rows = [i for i, (kind, _, _) in enumerate(order) if kind == "heldout"]
    paper_targets = [order[i][2] for i in paper_rows]
    held_targets = [order[i][2] for i in held_rows]

    acts_all = sweep["chat_last"].astype(np.float32)  # [prompts, 20, 5376]
    layers = [int(x) for x in sweep["layers"]]

    out: dict[str, dict] = {}
    for name, means in bundles.items():
        per_layer = {}
        passing_layers = []
        for layer_pos, layer in enumerate(layers):
            probes = contrast_probes_12pool(means[:, layer_pos])
            paper_count = battery_counts(acts_all[paper_rows, layer_pos], probes, paper_targets)
            held_count = battery_counts(acts_all[held_rows, layer_pos], probes, held_targets)
            per_layer[layer] = [paper_count, held_count]
            if paper_count >= 8 and held_count >= 8:
                passing_layers.append(layer)
        best_layer, best_counts = max(per_layer.items(), key=lambda kv: sum(kv[1]))
        out[name] = {
            "per_layer": per_layer,
            "passing_layers": passing_layers,
            "best": {"layer": best_layer, "counts": best_counts},
        }
    return out


def r2_cross(
    bundles: dict[str, Float[np.ndarray, "emotions layers d_model"]], layer_pos: int
) -> dict[str, object]:
    names = list(bundles)
    probes = {name: contrast_probes_12pool(bundles[name][:, layer_pos]) for name in names}
    out = {}
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            # per-emotion cosine between the two lineages' matching probes
            # (probes are unit-norm, so the row-wise dot IS the cosine)
            per_emotion_cos = np.sum(probes[name_a] * probes[name_b], axis=1)

            # RSA: correlate the two lineages' 12x12 within-battery similarity
            # structures over the off-diagonal upper triangle
            sim_a = probes[name_a] @ probes[name_a].T
            sim_b = probes[name_b] @ probes[name_b].T
            upper_tri = np.triu_indices(len(BATTERY), k=1)
            rsa = float(np.corrcoef(sim_a[upper_tri], sim_b[upper_tri])[0, 1])

            out[f"{name_a}_vs_{name_b}"] = {
                "mean_contrast_cos": float(per_emotion_cos.mean()),
                "range": [float(per_emotion_cos.min()), float(per_emotion_cos.max())],
                "rsa_12x12": rsa,
            }
    return out


def r3_preferences(
    bundles: dict[str, Float[np.ndarray, "emotions layers d_model"]], sweep_layers: list[int]
) -> dict[str, object]:
    pref = np.load(PREF_DIR / "preferences.npz", allow_pickle=True)
    scores = json.loads((PREF_DIR / "scores.json").read_text())
    # rows ordered like activities.py, the same order feel_feats rows use
    elo = np.array([row["elo"] for row in scores["elo_per_activity"]], dtype=np.float64)
    feel = pref["feel_feats"].astype(np.float32)  # [64 activities, 4 layers, 5376]
    pref_layers = [int(x) for x in pref["layers"]]
    out = {}
    for name, means in bundles.items():
        best = {"abs_r": 0.0, "layer": None, "emotion": None}
        for pref_layer_pos, layer in enumerate(pref_layers):
            sweep_layer_pos = sweep_layers.index(layer)
            probes = contrast_probes_12pool(means[:, sweep_layer_pos])

            # cosine of each activity's feel-activation with each emotion probe
            acts = feel[:, pref_layer_pos]
            acts_unit = acts / np.clip(np.linalg.norm(acts, axis=1, keepdims=True), 1e-8, None)
            cos = acts_unit @ probes.T  # [activities, emotions]

            # track the (layer, emotion) cell with the largest |r| against Elo
            for emotion_pos, emotion in enumerate(BATTERY):
                r = float(np.corrcoef(cos[:, emotion_pos], elo)[0, 1])
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
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="where the sweep/preference inputs live (the main clone's results/)",
    )
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
    probe_files = {"selfgen": args.selfgen, "weak": args.weak, "strong": args.strong}
    result = {
        "experiment": "Q1.H2.E11",
        "probe_files": {name: str(path) for name, path in probe_files.items()},
        "convention": "post-fix bundles on all arms; 12-pool probe centering; "
        "scenario-set activation centering; chat_last readout",
        "r1_dual_battery": r1_grid(bundles, sweep),
        "r2_cross_lineage_layer33": r2_cross(bundles, layers.index(33)),
        "r3_preference_probe_elo": r3_preferences(bundles, layers),
        "r4_diversity_EXPLORATORY": r4_diversity_exploratory(
            {
                "selfgen": args.selfgen_stories,
                "weak_external": args.weak_stories,
                "strong_external": args.strong_stories,
            }
        ),
    }
    args.out.write_text(json.dumps(result, indent=2))
    for name, r1 in result["r1_dual_battery"].items():
        print(
            f"{name:16s} best layer {r1['best']['layer']}: {r1['best']['counts']} | passing: {r1['passing_layers']}"
        )
    print(json.dumps(result["r2_cross_lineage_layer33"], indent=2))
    print("R3:", {k: round(v["abs_r"], 3) for k, v in result["r3_preference_probe_elo"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
