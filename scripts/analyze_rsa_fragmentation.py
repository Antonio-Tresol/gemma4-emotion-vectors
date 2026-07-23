"""Why does instruct cross-layer RSA fragment while base stays coherent — Q1.H1.E4.

Registered hypothesis (TREE Q1.H1.E4, before computing): the fragmentation is
carried by the dominant non-affective components E3 identified — if the top
PCs change regime across depth while the underlying affective structure stays
put, RSA fragments even though the circumplex survives. Three reads, all
predicted in the tree node: (R1) top-component ablation, (R2) PC1 axis
continuity across depth, (R3) cross-model RSA. Conventions identical to
notebook 02 section 6: centered-cosine similarity on the post-fix bundles,
upper-triangle Pearson. Laptop-safe. Writes results/rsa_fragmentation.json.

    uv run python scripts/analyze_rsa_fragmentation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from jaxtyping import Float
from sklearn.decomposition import PCA

from emotion_vectors.artifacts import fetch

SOURCES = {
    "it": "emotion_vectors_it_postfix/emotion_means.npz",
    "base": "emotion_vectors_postfix/emotion_means.npz",
}
N_ABLATE_MAX = 2
MID_BAND = (12, 24)  # the instruct block that disagrees with the late band
LATE_BAND = (30, 57)


def sim_triu(
    centered: Float[np.ndarray, "emotions d_model"],
) -> Float[np.ndarray, " pairs"]:
    """Upper triangle of the centered-cosine similarity matrix (section 6's
    convention: rows already grand-mean centered, then row-normalized)."""
    normed = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    sim = normed @ normed.T
    return sim[np.triu_indices(sim.shape[0], k=1)]


def band_means(rsa: Float[np.ndarray, "layers layers"], layers: list[int]) -> dict[str, float]:
    late = [p for p, layer in enumerate(layers) if LATE_BAND[0] <= layer <= LATE_BAND[1]]
    mid = [p for p, layer in enumerate(layers) if MID_BAND[0] <= layer <= MID_BAND[1]]
    within_late = rsa[np.ix_(late, late)][np.triu_indices(len(late), k=1)].mean()
    mid_to_late = rsa[np.ix_(mid, late)].mean()
    return {
        "within_late_mean": round(float(within_late), 4),
        "mid_to_late_mean": round(float(mid_to_late), 4),
    }


def main() -> int:
    bundles = {name: np.load(fetch(path), allow_pickle=True) for name, path in SOURCES.items()}
    layers = [int(layer) for layer in bundles["it"]["layers"]]

    # per (model, layer): centered means, top-PC directions, ablated variants
    triu = {}  # (model, layer, k_removed) -> flattened similarity
    pc1 = {}  # (model, layer) -> top PC direction
    for name, bundle in bundles.items():
        means = bundle["means"].astype(np.float64)
        for pos, layer in enumerate(layers):
            centered = means[:, pos, :] - means[:, pos, :].mean(axis=0)
            pca = PCA(n_components=N_ABLATE_MAX, svd_solver="full").fit(centered)
            pc1[name, layer] = pca.components_[0]
            scores = pca.transform(centered)
            for k_removed in range(N_ABLATE_MAX + 1):
                ablated = centered - scores[:, :k_removed] @ pca.components_[:k_removed]
                triu[name, layer, k_removed] = sim_triu(ablated)

    result: dict[str, object] = {
        "layers": layers,
        "conventions": "centered-cosine similarity, post-fix bundles, upper-triangle Pearson; "
        "identical to notebook 02 section 6; ablation removes the top-k per-layer PCs",
        "bands": {"mid": list(MID_BAND), "late": list(LATE_BAND)},
    }

    # R1: within-model RSA at each ablation depth
    for name in bundles:
        per_k = {}
        for k_removed in range(N_ABLATE_MAX + 1):
            flat = np.stack([triu[name, layer, k_removed] for layer in layers])
            rsa = np.corrcoef(flat)
            per_k[str(k_removed)] = {
                "rsa": [[round(float(v), 4) for v in row] for row in rsa],
                **band_means(rsa, layers),
            }
        result[name] = {"ablation": per_k}

    # R2: PC1 continuity — adjacent layers, and every layer against layer 33
    for name in bundles:
        adjacent = [
            round(abs(float(pc1[name, a] @ pc1[name, b])), 4)
            for a, b in zip(layers[:-1], layers[1:])
        ]
        vs_33 = [round(abs(float(pc1[name, layer] @ pc1[name, 33])), 4) for layer in layers]
        result[name]["pc1_continuity"] = {"adjacent_abs_cos": adjacent, "vs_layer33_abs_cos": vs_33}

    # R3: cross-model RSA, all 400 layer pairs (unablated similarities)
    cross = np.array(
        [
            [float(np.corrcoef(triu["it", li, 0], triu["base", lj, 0])[0, 1]) for lj in layers]
            for li in layers
        ]
    )
    late = [p for p, layer in enumerate(layers) if LATE_BAND[0] <= layer <= LATE_BAND[1]]
    mid = [p for p, layer in enumerate(layers) if MID_BAND[0] <= layer <= MID_BAND[1]]
    result["cross_model"] = {
        "rsa_it_rows_base_cols": [[round(float(v), 4) for v in row] for row in cross],
        "it_late_x_base_late_mean": round(float(cross[np.ix_(late, late)].mean()), 4),
        "it_mid_x_base_late_mean": round(float(cross[np.ix_(mid, late)].mean()), 4),
        "it_mid_x_base_mid_mean": round(float(cross[np.ix_(mid, mid)].mean()), 4),
    }

    # POST-HOC (disclosed, decided after seeing R3's unablated 0.29): does removing
    # the instruct top components restore base-likeness in the late band?
    post_hoc = {}
    for k_removed in (1, 2):
        cross_k = np.array(
            [
                [
                    float(np.corrcoef(triu["it", li, k_removed], triu["base", lj, 0])[0, 1])
                    for lj in layers
                ]
                for li in layers
            ]
        )
        post_hoc[f"it_k{k_removed}_late_x_base_late_mean"] = round(
            float(cross_k[np.ix_(late, late)].mean()), 4
        )
    result["cross_model_ablated_post_hoc"] = post_hoc

    Path("results/rsa_fragmentation.json").write_text(json.dumps(result, indent=1) + "\n")
    for name in bundles:
        rows = {k: result[name]["ablation"][k] for k in ("0", "1", "2")}
        print(
            f"{name}: "
            + " | ".join(
                f"k={k} late {v['within_late_mean']:.3f} mid-to-late {v['mid_to_late_mean']:.3f}"
                for k, v in rows.items()
            )
        )
    print("cross-model:", {k: v for k, v in result["cross_model"].items() if "mean" in k})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
