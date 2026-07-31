"""E4b GPU-half scoring: before/after comparison of pre-fix vs post-fix vectors.

Before = the shipped pre-fix extractions (left-padding TOKEN_OFFSET deviation,
token-level blast radius in results/extraction_offset_token_audit*.json).
After = the post-fix re-extractions from the pod (*_postfix dirs).

Reads REGISTERED before any post-fix vector was seen (recorded 2026-07-22 in
the rerun-plan message and encoded here before results sync):

  (R1) per-emotion cosine(before, after), raw AND centered-contrast (contrast =
       vector minus the emotion-set mean — the object the battery actually
       uses; centering pool = each lineage's own emotion set).
  (R2) geometry re-read on the corpus lineages: PC1-valence |r| per layer via
       the C1 pipeline (analysis.layer_geometry + NRC VAD), before vs after.
  (R3) neutral bundle: per-story cosines plus principal angles between the
       50%-variance PC subspaces (the object E7 projects out), layer 33.

Verdict ladder, bars registered before scoring, applied to the WORKING layers
{24, 30, 33, 36, 39, 42, 57} on centered contrasts (stricter than the loose
"per-emotion cosine" wording of the plan message, chosen before data):
  min contrast cosine >= 0.995 AND base C1 peak-|r| delta <= 0.02
      -> no_downstream_reruns
  min contrast cosine >= 0.98  -> tier1_recompute_reads (laptop rescoring only)
  else                         -> tier2_gpu_reruns_on_the_table

Usage (after syncing *_postfix dirs from the pod into results/):
    uv run python scripts/compare_extraction_postfix.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import PCA

from emotion_vectors.analysis import layer_geometry, load_emotion_means, load_nrc_vad

if TYPE_CHECKING:
    from jaxtyping import Float

LEXICON = Path("data/lexicons/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt")
WORKING_LAYERS = [24, 30, 33, 36, 39, 42, 57]
BAR_NO_RERUNS = 0.995
BAR_TIER1 = 0.98
BAR_GEOMETRY_DELTA = 0.02
OUT_FILE = Path("results/e4b_extraction_impact.json")


def cosine_rows(
    a: "Float[np.ndarray, 'emotions d_model']", b: "Float[np.ndarray, 'emotions d_model']"
) -> "Float[np.ndarray, 'emotions']":
    a, b = a.astype(np.float64), b.astype(np.float64)  # e7 bundle is fp16; norms overflow
    num = (a * b).sum(axis=-1)
    return num / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def load_before_npz(
    path: Path,
) -> tuple[list[str], list[int], Float[np.ndarray, "emotions layers d_model"]]:
    z = np.load(path, allow_pickle=True)
    return [str(e) for e in z["emotions"]], [int(x) for x in z["layers"]], z["means"]


def load_before_e6_n256(
    path: Path,
) -> tuple[list[str], list[int], Float[np.ndarray, "emotions layers d_model"]]:
    z = np.load(path, allow_pickle=True)
    bucket = int(np.argmax(z["n_buckets"] == 256))
    assert int(z["n_buckets"][bucket]) == 256, "n=256 bucket not found in e6_scale_means"
    return [str(e) for e in z["emotions"]], [int(x) for x in z["layers"]], z["means"][bucket]


def align(
    before: tuple[list[str], list[int], Float[np.ndarray, "emotions layers d_model"]],
    after: tuple[list[str], list[int], Float[np.ndarray, "emotions layers d_model"]],
) -> tuple[
    list[str],
    list[int],
    Float[np.ndarray, "emotions layers d_model"],
    Float[np.ndarray, "emotions layers d_model"],
]:
    """Common (emotions, layers) intersection, both means reordered to match."""
    emotions = sorted(set(before[0]) & set(after[0]))
    layers = [layer for layer in before[1] if layer in set(after[1])]
    b_idx = {e: i for i, e in enumerate(before[0])}
    a_idx = {e: i for i, e in enumerate(after[0])}
    bl = {layer: i for i, layer in enumerate(before[1])}
    al = {layer: i for i, layer in enumerate(after[1])}
    b = before[2][[b_idx[e] for e in emotions]][:, [bl[x] for x in layers]]
    a = after[2][[a_idx[e] for e in emotions]][:, [al[x] for x in layers]]
    return emotions, layers, b, a


def lineage_report(
    before: tuple[list[str], list[int], Float[np.ndarray, "emotions layers d_model"]],
    after: tuple[list[str], list[int], Float[np.ndarray, "emotions layers d_model"]],
) -> dict[str, object]:
    emotions, layers, b, a = align(before, after)
    per_layer: dict[str, dict[str, float]] = {}
    working_min_contrast = 1.0
    for pos, layer in enumerate(layers):
        raw = cosine_rows(b[:, pos], a[:, pos])
        contrast = cosine_rows(
            b[:, pos] - b[:, pos].mean(axis=0), a[:, pos] - a[:, pos].mean(axis=0)
        )
        per_layer[str(layer)] = {
            "raw_cos_mean": round(float(raw.mean()), 6),
            "raw_cos_min": round(float(raw.min()), 6),
            "contrast_cos_mean": round(float(contrast.mean()), 6),
            "contrast_cos_min": round(float(contrast.min()), 6),
            "contrast_cos_argmin": emotions[int(contrast.argmin())],
        }
        if layer in WORKING_LAYERS:
            working_min_contrast = min(working_min_contrast, float(contrast.min()))
    return {
        "n_emotions_compared": len(emotions),
        "working_min_contrast_cos": round(working_min_contrast, 6),
        "per_layer": per_layer,
    }


def geometry_compare(
    before: tuple[list[str], list[int], Float[np.ndarray, "emotions layers d_model"]],
    after: tuple[list[str], list[int], Float[np.ndarray, "emotions layers d_model"]],
) -> dict[str, object]:
    """C1's read on both vector sets: PC1-valence |r| per layer, peak deltas."""
    vad = load_nrc_vad(LEXICON)
    emotions, layers, b, a = align(before, after)
    matched = [i for i, e in enumerate(emotions) if e.lower() in vad]
    valence = np.array([vad[emotions[i].lower()][0] for i in matched])
    arousal = np.array([vad[emotions[i].lower()][1] for i in matched])
    rows: dict[str, dict[str, float]] = {}
    for pos, layer in enumerate(layers):
        gb = layer_geometry(b[:, pos], matched, valence, arousal)
        ga = layer_geometry(a[:, pos], matched, valence, arousal)
        rows[str(layer)] = {
            "pc1_valence_abs_r_before": round(abs(gb["pc1_valence"]["pearson_r"]), 4),
            "pc1_valence_abs_r_after": round(abs(ga["pc1_valence"]["pearson_r"]), 4),
        }
    peak_before = max(rows.values(), key=lambda r: r["pc1_valence_abs_r_before"])
    peak_after = max(rows.values(), key=lambda r: r["pc1_valence_abs_r_after"])
    return {
        "per_layer": rows,
        "peak_abs_r_before": peak_before["pc1_valence_abs_r_before"],
        "peak_abs_r_after": peak_after["pc1_valence_abs_r_after"],
        "peak_abs_r_delta": round(
            abs(peak_after["pc1_valence_abs_r_after"] - peak_before["pc1_valence_abs_r_before"]), 4
        ),
    }


def neutral_compare(before_bundle: Path, after_dir: Path) -> dict[str, object]:
    """Per-story cosines plus principal angles of the 50%-variance PC subspace
    (layer 33) — the object project_out_neutral removes."""
    z = np.load(before_bundle, allow_pickle=True)
    vectors_before, layers = z["vectors"], [int(x) for x in z["layers"]]
    shard_files = sorted((after_dir / "shards").glob("neutral__*.npy"))
    vectors_after = np.stack([np.load(f) for f in shard_files])
    n = min(len(vectors_before), len(vectors_after))
    pos33 = layers.index(33)
    story_cos = cosine_rows(vectors_before[:n, pos33], vectors_after[:n, pos33])

    def basis(
        vectors: Float[np.ndarray, "emotions d_model"],
    ) -> Float[np.ndarray, "components d_model"]:
        centered = vectors - vectors.mean(axis=0)
        pca = PCA(n_components=min(centered.shape)).fit(centered)
        k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.5)) + 1
        return pca.components_[:k]

    b_basis, a_basis = basis(vectors_before[:, pos33]), basis(vectors_after[:n, pos33])
    principal_cos = np.linalg.svd(b_basis @ a_basis.T, compute_uv=False)
    return {
        "n_stories_compared": int(n),
        "story_cos_mean_L33": round(float(story_cos.mean()), 6),
        "story_cos_min_L33": round(float(story_cos.min()), 6),
        "n_pcs_before": int(b_basis.shape[0]),
        "n_pcs_after": int(a_basis.shape[0]),
        "principal_cosines_L33": [round(float(c), 4) for c in principal_cos],
    }


def verdict(working_min: float, geometry_delta: float | None) -> str:
    if working_min >= BAR_NO_RERUNS and (
        geometry_delta is None or geometry_delta <= BAR_GEOMETRY_DELTA
    ):
        return "no_downstream_reruns"
    if working_min >= BAR_TIER1:
        return "tier1_recompute_reads"
    return "tier2_gpu_reruns_on_the_table"


def main() -> int:
    lineages: dict[str, object] = {}

    base_before = load_before_npz(Path("results/emotion_vectors/emotion_means.npz"))
    base_after = load_emotion_means(Path("results/emotion_vectors_postfix"))
    lineages["corpus_base"] = lineage_report(base_before, base_after)
    lineages["corpus_base"]["geometry"] = geometry_compare(base_before, base_after)

    it_before = load_before_npz(Path("results/emotion_vectors_it_means.npz"))
    it_after = load_emotion_means(Path("results/emotion_vectors_it_postfix"))
    lineages["corpus_it"] = lineage_report(it_before, it_after)
    lineages["corpus_it"]["geometry"] = geometry_compare(it_before, it_after)

    selfgen_before = load_before_e6_n256(Path("results/e6_scale_means.npz"))
    selfgen_after = load_emotion_means(Path("results/self_story_vectors_it_postfix"))
    lineages["selfgen_n256_it"] = lineage_report(selfgen_before, selfgen_after)

    lineages["neutral_it"] = neutral_compare(
        Path("results/e7_neutral_bundle.npz"), Path("results/neutral_vectors_it_postfix")
    )

    verdicts = {
        name: verdict(
            rep["working_min_contrast_cos"],
            rep.get("geometry", {}).get("peak_abs_r_delta") if isinstance(rep, dict) else None,
        )
        for name, rep in lineages.items()
        if "working_min_contrast_cos" in rep
    }
    result: dict[str, object] = {
        "registered_bars": {
            "working_layers": WORKING_LAYERS,
            "no_reruns_min_contrast_cos": BAR_NO_RERUNS,
            "tier1_min_contrast_cos": BAR_TIER1,
            "geometry_peak_abs_r_delta_max": BAR_GEOMETRY_DELTA,
            "note": "bars registered before any post-fix vector was seen",
        },
        "verdicts": verdicts,
        "lineages": lineages,
    }
    OUT_FILE.write_text(json.dumps(result, indent=2) + "\n")
    for name, v in verdicts.items():
        rep = lineages[name]
        print(f"{name}: min contrast cos (working layers) {rep['working_min_contrast_cos']} -> {v}")
    print(f"neutral: {lineages['neutral_it']}")
    print(f"-> {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
