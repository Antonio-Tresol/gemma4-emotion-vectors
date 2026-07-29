"""Shared section-1 state for the notebook-02 geometry report.

Owns what the notebook's former section-1 cell computed: both models'
post-fix mean-activation bundles, the NRC VAD match, and one deterministic
PCA per model per layer. Every other submodule (and every section exhibit)
reads from the :class:`GeometryContext` built here; nothing in this module
draws.

Number-identity contract: :func:`load_geometry_context` is a line-for-line
port of the formerly inline loader (commit 73e3e3d), so re-running the
notebook reproduces the committed printed numbers exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float
from sklearn.decomposition import PCA

from emotion_vectors.analysis import load_nrc_vad
from emotion_vectors.artifacts import fetch  # local results/ first, HF otherwise

# One layer's PCA record, as built by _fit_layer_plane:
# keys pca (fitted sklearn PCA), scores [emotions pcs] (sign-oriented),
# vb / ab (valence- / arousal-best component index), r_vb / r_ab,
# rv / ra (per-component |r| lists), evr (explained variance ratios).
LayerPlane = dict[str, Any]

# RSA band edges shared by sections 6 and 9: the mid block and the late block
# that the instruct model's fragmentation separates.
MID_BAND: tuple[int, int] = (12, 24)
LATE_BAND: tuple[int, int] = (30, 57)

IT_LABEL = "gemma-4-31b-it"  # the instruct model under study
BASE_LABEL = "gemma-4-31b (base)"  # the untuned comparison model
# post-fix bundles: the blessed instrument since the E4b padding-bug fix
BUNDLE_SOURCES = {
    IT_LABEL: "emotion_vectors_it_postfix/emotion_means.npz",
    BASE_LABEL: "emotion_vectors_postfix/emotion_means.npz",
}
# base model's peak layer; every slider starts here but scrubs all 20 layers
DEFAULT_LAYER = 33
N_COMPONENTS = 10  # PCA components kept per layer; every rv/ra/evr list has this length


@dataclass(frozen=True)
class GeometryContext:
    """Everything the section exhibits need from the section-1 state.

    ``models[label]`` holds the per-emotion mean activations
    [emotions layers d_model]; ``plane[label][layer]`` the fitted PCA record
    (:data:`LayerPlane`). ``matched`` indexes the emotions found in the NRC
    lexicon, and ``valence`` / ``arousal`` carry the NRC human ratings for
    exactly those emotions; ``default_layer`` is where sliders and
    single-layer reads sit.
    """

    it_label: str
    base_label: str
    models: dict[str, Float[np.ndarray, "emotions layers d_model"]]
    plane: dict[str, dict[int, LayerPlane]]
    emotions: list[str]
    layers: list[int]
    matched: list[int]
    valence: Float[np.ndarray, " matched"]
    arousal: Float[np.ndarray, " matched"]
    default_layer: int

    @property
    def labels(self) -> tuple[str, str]:
        """(instruct, base), the panel order used by every paired figure."""
        return (self.it_label, self.base_label)


def repo_root() -> Path:
    """Checkout root, found by walking up from the current directory.

    Repo-relative inputs (the NRC VAD lexicon) resolve the same way from the
    repo root and from notebooks/, with no machine absolute paths anywhere.
    """
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"no pyproject.toml above {here}")


def nrc_vad_lexicon_path() -> Path:
    """The checked-in NRC VAD lexicon (third-party license, populated manually)."""
    return repo_root() / "data/lexicons/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"


def _fit_layer_plane(
    layer_means: Float[np.ndarray, "emotions d_model"],
    matched: list[int],
    valence: Float[np.ndarray, " matched"],
    arousal: Float[np.ndarray, " matched"],
) -> LayerPlane:
    """One layer's PCA record: top-10 components, valence-/arousal-best picks.

    ``svd_solver="full"`` keeps the fit deterministic. The arousal-best pick
    excludes the valence-best component so the circumplex plane always spans
    two distinct axes. The two selected components are sign-oriented so their
    correlate increases rightward/upward.
    """
    pca = PCA(n_components=N_COMPONENTS, svd_solver="full")  # full SVD: deterministic
    scores = pca.fit_transform(layer_means)
    r_valence = [np.corrcoef(scores[matched, k], valence)[0, 1] for k in range(N_COMPONENTS)]
    r_arousal = [np.corrcoef(scores[matched, k], arousal)[0, 1] for k in range(N_COMPONENTS)]
    valence_best = int(np.argmax(np.abs(r_valence)))  # valence-best of the top 10 components
    arousal_best = int(
        np.argmax([abs(r) if k != valence_best else -1.0 for k, r in enumerate(r_arousal)])
    )
    if r_valence[valence_best] < 0:
        scores[:, valence_best] *= -1
    if r_arousal[arousal_best] < 0:
        scores[:, arousal_best] *= -1
    return dict(
        pca=pca,
        scores=scores,
        vb=valence_best,
        ab=arousal_best,
        r_vb=abs(r_valence[valence_best]),
        r_ab=abs(r_arousal[arousal_best]),
        rv=[abs(r) for r in r_valence],
        ra=[abs(r) for r in r_arousal],
        evr=pca.explained_variance_ratio_,
    )


def load_geometry_context(default_layer: int = DEFAULT_LAYER) -> GeometryContext:
    """Load both post-fix bundles and the NRC VAD lexicon, fit one PCA per
    model per layer, and return the context every exhibit reads from."""
    models: dict[str, Float[np.ndarray, "emotions layers d_model"]] = {}
    emotions: list[str] = []
    layers: list[int] = []
    for label, means_path in BUNDLE_SOURCES.items():
        bundle = np.load(fetch(means_path), allow_pickle=True)
        bundle_emotions = [str(emotion) for emotion in bundle["emotions"]]
        bundle_layers = [int(layer) for layer in bundle["layers"]]
        if not emotions:
            emotions, layers = bundle_emotions, bundle_layers
        assert (bundle_emotions, bundle_layers) == (emotions, layers), (
            "bundles must share emotions and layers"
        )
        models[label] = bundle["means"].astype(np.float64)  # stored float32; upcast before math
    vad = load_nrc_vad(nrc_vad_lexicon_path())
    matched = [pos for pos, emotion in enumerate(emotions) if emotion.lower() in vad]
    valence = np.array([vad[emotions[pos].lower()][0] for pos in matched])
    arousal = np.array([vad[emotions[pos].lower()][1] for pos in matched])
    plane = {
        label: {
            layer: _fit_layer_plane(means[:, layer_pos, :], matched, valence, arousal)
            for layer_pos, layer in enumerate(layers)
        }
        for label, means in models.items()
    }
    return GeometryContext(
        it_label=IT_LABEL,
        base_label=BASE_LABEL,
        models=models,
        plane=plane,
        emotions=emotions,
        layers=layers,
        matched=matched,
        valence=valence,
        arousal=arousal,
        default_layer=default_layer,
    )


def e4b_deviations(ctx: GeometryContext) -> dict[str, float]:
    """Per model, the max over layers of |recomputed PC1-valence |r| minus the
    E4b padding-fix audit's post-fix value|.

    The notebook asserts these stay below 5e-4 — its verification contract
    that the bundles on disk are the audited post-fix instrument
    (``results/e4b_extraction_impact.json``). Keys come base first, matching
    the notebook's printed record.
    """
    audit = json.loads(fetch("e4b_extraction_impact.json").read_text())
    deviations: dict[str, float] = {}
    # "lineages" is the frozen key in e4b_extraction_impact.json: one block
    # per story source that was re-extracted
    for label, story_source in ((ctx.base_label, "corpus_base"), (ctx.it_label, "corpus_it")):
        audited = audit["lineages"][story_source]["geometry"]["per_layer"]
        deviations[label] = max(
            abs(ctx.plane[label][layer]["rv"][0] - audited[str(layer)]["pc1_valence_abs_r_after"])
            for layer in ctx.layers
        )
    return deviations
