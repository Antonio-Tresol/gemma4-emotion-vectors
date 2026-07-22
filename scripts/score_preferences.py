"""Q1.H3.E1 scoring — Bradley-Terry Elo per activity, probe-Elo correlations.

Laptop-side. Reads the pod collection (results/preferences_it/preferences.npz
and activities.jsonl), scores against the registered predictions, and writes
results/preferences_it/scores.json — the evidence file the tree links.

Operational reads, declared here before the collection finished (this file is
committed while the pod run is still in its A/B pass):

  P1 (sanity): mean Elo over the paper's positive categories (helpful,
      engaging, social, self_curiosity) exceeds mean Elo over the negative
      ones (misaligned, unsafe). Aversive and neutral are reported, not
      scored (the paper shows them mid-scale).
  P2 (probe-Elo, registered max |r| >= 0.5): per layer, Pearson r between
      each of the 171 probe activations (cosine of the centered probe against
      the activity-token features, sweep convention: center on all 171) and
      the 64 Elo scores. Valence organization is read as Pearson correlation
      between the per-probe r values and NRC valence, with a 10,000-shuffle
      permutation p; "valence-organized" means p < 0.05 with positive sign.

    uv run python scripts/score_preferences.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import pearsonr

if TYPE_CHECKING:
    from jaxtyping import Float

from emotion_vectors.analysis import load_nrc_vad
from emotion_vectors.preferences import bradley_terry, elo, soft_wins, win_matrix
from emotion_vectors.scoring import battery_matrix

PREF_DIR = Path("results/preferences_it")
PROBE_BUNDLE = Path("results/emotion_vectors_it_means.npz")  # the -it extraction, bundled
LEXICON = Path("data/lexicons/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt")
POSITIVE = ("helpful", "engaging", "social", "self_curiosity")
NEGATIVE = ("misaligned", "unsafe")
RNG = np.random.default_rng(20260722)
N_PERMUTATIONS = 10_000


def probe_elo_block(
    layer: int,
    layer_pos_probe: int,
    layer_pos_feel: int,
    scores: "Float[np.ndarray, 'activities']",
    feel: "Float[np.ndarray, 'activities layers d_model']",
    means: "Float[np.ndarray, 'emotions layers d_model']",
    valence: "Float[np.ndarray, 'emotions']",
) -> dict[str, object]:
    """Per-probe Elo correlations at one layer, plus the valence-organization read."""
    activations = battery_matrix(
        means[:, layer_pos_probe, :], feel[:, layer_pos_feel, :].astype(np.float64)
    )
    r = np.array([pearsonr(activations[k], scores).statistic for k in range(len(activations))])
    organization = pearsonr(r, valence).statistic
    perm = np.array(
        [abs(pearsonr(r, RNG.permutation(valence)).statistic) for _ in range(N_PERMUTATIONS)]
    )
    best_idx = int(np.argmax(np.abs(r)))
    return {
        "layer": layer,
        "max_abs_r": round(float(np.max(np.abs(r))), 4),
        "argmax_probe_index": best_idx,
        "valence_organization_r": round(float(organization), 4),
        "valence_organization_perm_p": float((perm >= abs(organization)).mean()),
        "per_probe_r": [round(float(x), 4) for x in r],
        "best_probe_activation_per_activity": [round(float(x), 5) for x in activations[best_idx]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pref-dir", type=Path, default=PREF_DIR)
    parser.add_argument("--experiment", default="Q1.H3.E1")
    args = parser.parse_args()
    pref_dir = args.pref_dir
    bundle = np.load(pref_dir / "preferences.npz")
    meta = [json.loads(line) for line in open(pref_dir / "activities.jsonl")]
    categories = [m["category"] for m in meta]

    p_first = soft_wins(bundle["ab_logits"].astype(np.float64))
    wins = win_matrix(bundle["pair_i"], bundle["pair_j"], p_first, len(meta))
    scores = elo(bradley_terry(wins))

    by_category = {
        c: round(float(np.mean([s for s, cc in zip(scores, categories) if cc == c])), 1)
        for c in dict.fromkeys(categories)
    }
    pos_mean = float(np.mean([by_category[c] for c in POSITIVE]))
    neg_mean = float(np.mean([by_category[c] for c in NEGATIVE]))

    probe_bundle = np.load(PROBE_BUNDLE, allow_pickle=True)
    emotions = list(map(str, probe_bundle["emotions"]))
    probe_layers = list(map(int, probe_bundle["layers"]))
    means = probe_bundle["means"].astype(np.float64)
    vad = load_nrc_vad(LEXICON)
    matched = [i for i, e in enumerate(emotions) if e.lower() in vad]
    valence = np.array([vad[emotions[i].lower()][0] for i in matched])
    feel = bundle["feel_feats"]
    blocks = [
        probe_elo_block(
            int(layer),
            probe_layers.index(int(layer)),
            pos,
            scores,
            feel,
            means[matched],
            valence,
        )
        for pos, layer in enumerate(bundle["layers"])
    ]
    best = max(blocks, key=lambda b: b["max_abs_r"])

    out = {
        "experiment": args.experiment,
        "n_activities": len(meta),
        "n_ordered_pairs": int(bundle["ab_logits"].shape[0]),
        "elo_by_category": by_category,
        "elo_per_activity": [
            {"category": c, "activity": m["activity"], "elo": round(float(s), 1)}
            for m, c, s in zip(meta, categories, scores)
        ],
        "p1_positive_mean": round(pos_mean, 1),
        "p1_negative_mean": round(neg_mean, 1),
        "p1_pass": pos_mean > neg_mean,
        "probe_elo_by_layer": blocks,
        "p2_registered_bar": 0.5,
        "p2_max_abs_r": best["max_abs_r"],
        "p2_best_layer": best["layer"],
        "p2_valence_organization_r": best["valence_organization_r"],
        "p2_valence_organization_perm_p": best["valence_organization_perm_p"],
        "p2_pass": bool(
            best["max_abs_r"] >= 0.5
            and best["valence_organization_r"] > 0
            and best["valence_organization_perm_p"] < 0.05
        ),
        "p2_best_probe_emotion": emotions[matched[best["argmax_probe_index"]]],
        "probe_emotions_matched": [emotions[i] for i in matched],
    }
    out_file = pref_dir / "scores.json"
    out_file.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"P1 {'PASS' if out['p1_pass'] else 'FAIL'}: positive {pos_mean:.0f} vs negative {neg_mean:.0f}"
    )
    print(
        f"P2 {'PASS' if out['p2_pass'] else 'FAIL'}: max |r|={best['max_abs_r']} at layer "
        f"{best['layer']}, organization r={best['valence_organization_r']} "
        f"(perm p={best['valence_organization_perm_p']})"
    )
    print(f"-> {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
