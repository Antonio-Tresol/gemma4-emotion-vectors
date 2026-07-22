"""Circumplex geometry check for extracted emotion vectors — Q1.H1.

Tests the literature expectation (notes/emotion-vectors-brief.md): after
mean-centering the per-emotion vectors (difference-of-means contrasts), the
first principal component should track NRC VAD valence (Anthropic report
r = 0.81 at the best layer; the open replication reached r = 0.83) and the
second arousal (r = 0.66). Laptop-safe: reads the aggregated layer_<N>_resid
files that extraction produced, no torch.

Lexicon: NRC VAD (Mohammad), downloaded from
https://saifmohammad.com/WebPages/nrc-vad.html into data/lexicons/ (gitignored
per the lexicon's redistribution terms; values in [-1, 1] in v2).

Decisions recorded here rather than hidden in code: PCA is fit on all
extracted emotions; correlations use only the lexicon-matched subset (the
unmatched labels are listed in the output). PCA component signs are arbitrary,
so the claim-relevant quantity is |r|; signed r is reported as computed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from jaxtyping import Float


def safe_name(emotion: str) -> str:
    return emotion.replace(" ", "_").replace("/", "-")


def load_emotion_means(
    results_dir: Path,
) -> tuple[list[str], list[int], "Float[np.ndarray, 'emotions layers d_model']"]:
    """(emotions, layers, means) from extraction's aggregated outputs."""
    config = json.loads((results_dir / "run_config.json").read_text())
    layers: list[int] = config["layers"]
    emotions = sorted(
        {json.loads(line)["emotion"] for line in open(results_dir / "manifest.jsonl")}
    )
    means = np.stack(
        [
            np.stack(
                [
                    np.load(results_dir / safe_name(emotion) / f"layer_{layer}_resid.npy")
                    for layer in layers
                ]
            )
            for emotion in emotions
        ]
    )
    return emotions, layers, means


def load_nrc_vad(lexicon_file: Path) -> dict[str, tuple[float, float]]:
    """{term: (valence, arousal)} from the tab-separated NRC VAD lexicon."""
    vad: dict[str, tuple[float, float]] = {}
    with open(lexicon_file) as f:
        next(f)  # header: term, valence, arousal, dominance
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                vad[parts[0].lower()] = (float(parts[1]), float(parts[2]))
    return vad


def layer_geometry(
    means: "Float[np.ndarray, 'emotions d_model']",
    matched_idx: list[int],
    valence: "Float[np.ndarray, 'matched']",
    arousal: "Float[np.ndarray, 'matched']",
) -> dict[str, object]:
    """PCA on all emotions' mean-centered vectors; correlations on the
    lexicon-matched subset. Component signs are arbitrary — judge by |r|."""
    pca = PCA(n_components=2)
    scores = pca.fit_transform(means)  # centers internally: difference-of-means
    pc1, pc2 = scores[matched_idx, 0], scores[matched_idx, 1]
    out: dict[str, object] = {
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_]
    }
    for name, component, target in (
        ("pc1_valence", pc1, valence),
        ("pc2_arousal", pc2, arousal),
        ("pc1_arousal", pc1, arousal),
        ("pc2_valence", pc2, valence),
    ):
        pearson = stats.pearsonr(component, target)
        spearman = stats.spearmanr(component, target)
        out[name] = {
            "pearson_r": round(float(pearson.statistic), 4),
            "pearson_p": float(pearson.pvalue),
            "spearman_r": round(float(spearman.statistic), 4),
        }
    return out


def analyze(results_dir: Path, lexicon_file: Path, out_file: Path) -> dict[str, object]:
    """Full sweep: geometry per layer, peak layers by |r|, coverage report."""
    emotions, layers, means = load_emotion_means(results_dir)
    vad = load_nrc_vad(lexicon_file)
    matched_idx = [i for i, e in enumerate(emotions) if e.lower() in vad]
    unmatched = [e for e in emotions if e.lower() not in vad]
    valence = np.array([vad[emotions[i].lower()][0] for i in matched_idx])
    arousal = np.array([vad[emotions[i].lower()][1] for i in matched_idx])

    per_layer = {
        str(layer): layer_geometry(means[:, pos, :], matched_idx, valence, arousal)
        for pos, layer in enumerate(layers)
    }
    result: dict[str, object] = {
        "n_emotions": len(emotions),
        "n_matched_to_nrc_vad": len(matched_idx),
        "unmatched_emotions": unmatched,
        "lexicon": lexicon_file.name,
        "literature_reference": "Anthropic: PC1-valence r=0.81, PC2-arousal r=0.66; "
        "open replication up to r=0.83 (notes/emotion-vectors-brief.md)",
        "peak_valence_layer": max(
            per_layer, key=lambda k: abs(per_layer[k]["pc1_valence"]["pearson_r"])
        ),
        "peak_arousal_layer": max(
            per_layer, key=lambda k: abs(per_layer[k]["pc2_arousal"]["pearson_r"])
        ),
        "per_layer": per_layer,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2) + "\n")
    return result


def project_out_neutral(
    probe_means: "Float[np.ndarray, 'emotions d_model']",
    neutral_vectors: "Float[np.ndarray, 'stories d_model']",
    variance_threshold: float = 0.5,
) -> tuple["Float[np.ndarray, 'emotions d_model']", int]:
    """The paper's confound-removal step, absent from the reference code (a
    deviation we inherited, caught 2026-07-21): PCA the pooled activations of
    emotionally neutral transcripts, take the top components up to
    variance_threshold cumulative explained variance, and project them out of
    the probe vectors. Returns (projected probes, n_components_removed)."""
    centered = neutral_vectors - neutral_vectors.mean(axis=0)
    pca = PCA(n_components=min(centered.shape)).fit(centered)
    k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), variance_threshold)) + 1
    basis = pca.components_[:k]
    return probe_means - (probe_means @ basis.T) @ basis, k
