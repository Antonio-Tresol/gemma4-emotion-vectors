"""What occupies instruct PC1/PC2 after the valence demotion — Q1.H1.E3.

The demotion diagnostic (check_pc_demotion.py) showed valence moving from
it-PC1 to it-PC3 while PC1's variance share doubled, but never asked what
PC1 and PC2 became. Registered reads (TREE Q1.H1.E3, named before
computing): per-PC correlations against NRC VAD valence / arousal /
dominance, top/bottom emotions per PC, and cross-model alignment (score
correlations, principal angles, direction cosines) that separates a
rotated survivor of base structure from structure instruction tuning
added. Laptop-safe. Writes results/it_pc_structure.json.

    uv run python scripts/analyze_it_pc_structure.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from jaxtyping import Float
from scipy.linalg import subspace_angles
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from emotion_vectors.artifacts import fetch

PRIMARY_LAYER = 33
N_PCS = 5
N_EXTREME = 10
LEXICON = Path("data/lexicons/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt")


def load_vad3(lexicon_file: Path) -> dict[str, tuple[float, float, float]]:
    """{term: (valence, arousal, dominance)} — all three NRC VAD columns."""
    vad: dict[str, tuple[float, float, float]] = {}
    with open(lexicon_file) as f:
        next(f)  # header: term, valence, arousal, dominance
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                vad[parts[0].lower()] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return vad


def fit_pca(
    means_layer: Float[np.ndarray, "emotions d_model"],
) -> tuple[PCA, Float[np.ndarray, "emotions pcs"]]:
    """check_pc_demotion.py convention: mean-center the 171 means, then PCA."""
    centered = means_layer - means_layer.mean(axis=0)
    pca = PCA(n_components=N_PCS).fit(centered)
    return pca, pca.transform(centered)


def pc_rows(
    scores: Float[np.ndarray, "emotions pcs"],
    evr: Float[np.ndarray, " pcs"],
    matched: list[int],
    targets: dict[str, Float[np.ndarray, " matched"]],
) -> list[dict[str, float]]:
    rows = []
    for k in range(N_PCS):
        row: dict[str, float] = {"pc": k + 1, "evr": round(float(evr[k]), 4)}
        for name, values in targets.items():
            stat = pearsonr(scores[matched, k], values)
            row[f"r_{name}"] = round(float(stat.statistic), 4)
            row[f"p_{name}"] = float(stat.pvalue)
        rows.append(row)
    return rows


def extremes(
    scores: Float[np.ndarray, "emotions pcs"], emotions: list[str]
) -> list[dict[str, list[str]]]:
    out = []
    for k in range(N_PCS):
        order = np.argsort(scores[:, k])
        out.append(
            {
                "pc": k + 1,
                "low": [emotions[i] for i in order[:N_EXTREME]],
                "high": [emotions[i] for i in order[-N_EXTREME:][::-1]],
            }
        )
    return out


def story_length_read(
    emotions: list[str],
    scores_by_model: dict[str, Float[np.ndarray, "emotions pcs"]],
) -> dict[str, object]:
    """Read 4, the nuisance probe: per-emotion mean story length (chars) from
    the extraction corpus vs PC scores. Story count is constant (9/emotion)
    so only length varies. Guarded: returns a note if the corpus is absent."""
    try:
        ds = load_dataset("snae/emotion_stories_gemma_4_4B", split="train")
    except Exception as err:  # noqa: BLE001 — any fetch failure degrades to a note
        return {"unavailable": f"{type(err).__name__}: {err}"}
    length: dict[str, list[int]] = defaultdict(list)
    for row in ds:
        length[row["emotion"]].extend(len(story) for story in row["stories"])
    mean_len = np.array([float(np.mean(length[e])) for e in emotions])
    out: dict[str, object] = {
        "mean_story_len_chars_min_max": [
            round(float(mean_len.min())),
            round(float(mean_len.max())),
        ],
        "n_stories_per_emotion": 9,
    }
    for model, scores in scores_by_model.items():
        stats = [pearsonr(scores[:, k], mean_len) for k in range(N_PCS)]
        out[model] = {
            "r_mean_story_len_per_pc": [round(float(s.statistic), 4) for s in stats],
            "p_per_pc": [float(s.pvalue) for s in stats],
        }
    return out


def main() -> int:
    base = np.load(fetch("emotion_vectors/emotion_means.npz"), allow_pickle=True)
    it = np.load(fetch("emotion_vectors_it_means.npz"), allow_pickle=True)
    emotions = list(map(str, base["emotions"]))
    assert emotions == list(map(str, it["emotions"]))
    layers = list(base["layers"])
    vad = load_vad3(LEXICON)
    matched = [i for i, e in enumerate(emotions) if e.lower() in vad]
    targets = {
        name: np.array([vad[emotions[i].lower()][col] for i in matched])
        for col, name in enumerate(("valence", "arousal", "dominance"))
    }

    result: dict[str, object] = {
        "primary_layer": PRIMARY_LAYER,
        "n_pcs": N_PCS,
        "n_matched": len(matched),
        "conventions": "mean-centered 171 means, sklearn PCA, correlations on NRC-matched "
        "emotions; inherited from check_pc_demotion.py",
    }

    # primary layer: profiles, extremes, cross-model alignment
    layer_pos = layers.index(PRIMARY_LAYER)
    base_means = base["means"].astype(np.float64)[:, layer_pos, :]
    it_means = it["means"].astype(np.float64)[:, layer_pos, :]
    base_pca, base_scores = fit_pca(base_means)
    it_pca, it_scores = fit_pca(it_means)
    result["base"] = {
        "per_pc": pc_rows(base_scores, base_pca.explained_variance_ratio_, matched, targets),
        "extremes": extremes(base_scores, emotions),
    }
    result["it"] = {
        "per_pc": pc_rows(it_scores, it_pca.explained_variance_ratio_, matched, targets),
        "extremes": extremes(it_scores, emotions),
    }

    # alignment: score correlations are basis-free (the primary read);
    # direction cosines compare weight-space axes across two different models
    # (same d_model, different weights) and carry that caveat by construction
    score_corr = [
        [
            round(float(pearsonr(it_scores[:, i], base_scores[:, j]).statistic), 4)
            for j in range(N_PCS)
        ]
        for i in range(N_PCS)
    ]
    direction_cos = [
        [round(float(it_pca.components_[i] @ base_pca.components_[j]), 4) for j in range(N_PCS)]
        for i in range(N_PCS)
    ]
    angles = subspace_angles(base_pca.components_[:3].T, it_pca.components_[:3].T)
    result["alignment"] = {
        "score_corr_it_by_base": score_corr,
        "direction_cos_it_by_base": direction_cos,
        "principal_angles_deg_top3": [round(float(np.degrees(a)), 1) for a in angles],
    }

    result["story_length_read"] = story_length_read(
        emotions, {"base": base_scores, "it": it_scores}
    )

    # 20-layer grid: compact per-layer |r| profile for both models
    grid: dict[str, dict[str, object]] = {}
    for pos, layer in enumerate(layers):
        row: dict[str, object] = {}
        for name, bundle in (("base", base), ("it", it)):
            pca, scores = fit_pca(bundle["means"].astype(np.float64)[:, pos, :])
            row[name] = {
                "evr": [round(float(v), 4) for v in pca.explained_variance_ratio_],
                **{
                    f"abs_r_{tname}": [
                        round(abs(float(pearsonr(scores[matched, k], tvals).statistic)), 4)
                        for k in range(N_PCS)
                    ]
                    for tname, tvals in targets.items()
                },
            }
        grid[str(int(layer))] = row
    result["per_layer"] = grid

    Path("results/it_pc_structure.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("base", "it", "alignment")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
