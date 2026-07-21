"""Where does valence live in each model's emotion-vector PCA? — E4 geometry diagnostic.

The -it geometry sweep showed PC1-valence collapsing after layer 6 while PC1's
explained variance grew. This checks whether valence structure vanished or was
demoted: correlate NRC valence against the top-10 PC scores for both models at
the base model's peak layer, and project the -it means onto the BASE model's
PC1 (valence) direction. Laptop-safe. Writes results/geometry_pc_demotion.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from emotion_vectors.analysis import load_nrc_vad

LAYER = 33
N_PCS = 10


def pc_valence_profile(
    means: "np.ndarray", matched: list[int], valence: "np.ndarray"
) -> tuple[list[float], list[float]]:
    centered = means - means.mean(axis=0)
    pca = PCA(n_components=N_PCS).fit(centered)
    scores = pca.transform(centered)
    rs = [abs(float(pearsonr(scores[matched, k], valence).statistic)) for k in range(N_PCS)]
    return rs, [round(float(v), 4) for v in pca.explained_variance_ratio_]


def main() -> int:
    base = np.load("results/emotion_vectors/emotion_means.npz", allow_pickle=True)
    it = np.load("results/emotion_vectors_it_means.npz", allow_pickle=True)
    emotions = list(map(str, base["emotions"]))
    layer_pos = list(base["layers"]).index(LAYER)
    vad = load_nrc_vad(Path("data/lexicons/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"))
    matched = [i for i, e in enumerate(emotions) if e.lower() in vad]
    valence = np.array([vad[emotions[i].lower()][0] for i in matched])

    result: dict[str, object] = {"layer": LAYER, "n_pcs": N_PCS}
    for name, bundle in (("base", base), ("it", it)):
        means = bundle["means"].astype(np.float64)[:, layer_pos, :]
        rs, evr = pc_valence_profile(means, matched, valence)
        result[name] = {
            "abs_r_valence_per_pc": [round(r, 4) for r in rs],
            "explained_variance_ratio": evr,
            "best_pc": int(np.argmax(rs)) + 1,
            "best_r": round(max(rs), 4),
        }

    base_means = base["means"].astype(np.float64)[:, layer_pos, :]
    it_means = it["means"].astype(np.float64)[:, layer_pos, :]
    base_pc1 = PCA(n_components=1).fit(base_means - base_means.mean(0)).components_[0]
    proj = (it_means - it_means.mean(0)) @ base_pc1
    result["it_on_base_pc1_abs_r_valence"] = round(
        abs(float(pearsonr(proj[matched], valence).statistic)), 4
    )
    Path("results/geometry_pc_demotion.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
