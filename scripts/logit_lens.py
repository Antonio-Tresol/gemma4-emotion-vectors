"""Logit lens over emotion vectors — the paper's Table 1 replication.

Projects each emotion's centered contrast vector through the unembedding
matrix and records the top and bottom tokens. The paper reports emotion-word
neighborhoods (sad -> grief, tears, lonely). Simplifications documented in the
output: we apply the final RMSNorm's learned scale explicitly (Gemma's 1+w
convention, see NOTE below), and Gemma's logit softcapping is ignored; both affect
magnitudes, not top-k ordering materially.

    # single layer, from per-emotion dirs:
    uv run python scripts/logit_lens.py
    # every layer in a means bundle, one JSON:
    uv run python scripts/logit_lens.py --all-layers
        --means-bundle results/emotion_vectors_it_means.npz
        --out results/logit_lens_it_all_layers.json

The ``--all-layers`` sweep reads the all-layer means bundle (present locally and
on HF), selects the 12 battery emotions and centres them on their own mean — the
same convention as the committed single-layer files — so ``by_layer['33']`` and
``by_layer['57']`` reproduce ``logit_lens_*_L33*.json`` / ``*_L57.json`` up to
top-k tie reordering (the bundle is fp16); a real mismatch there means the other
layers should not be trusted. It feeds the layer scrubber in
``notebooks/explore_layers.ipynb``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from emotion_vectors.probe_prompts import SCENARIOS

if TYPE_CHECKING:
    from jaxtyping import Float

try:
    from emotion_vectors.extraction import load_model_bf16, setup_logger
except ModuleNotFoundError as exc:  # torch lives in the gpu extra; this is a GPU entry point
    raise SystemExit(f"missing {exc.name}: this entry point needs `uv sync --extra gpu`") from exc

BATTERY = [t for _, t, _ in SCENARIOS]  # the 12 implicit-emotion battery emotions
NOTE = "final-RMSNorm scale applied (gemma 1+w convention); softcapping ignored"
# committed single-layer files the sweep should reproduce, per model
REPRO = {
    "google/gemma-4-31b-it": {
        33: "results/logit_lens_it_L33_normed.json",
        57: "results/logit_lens_it_L57.json",
    },
    "google/gemma-4-31b": {
        33: "results/logit_lens_base_L33.json",
        57: "results/logit_lens_base_L57.json",
    },
}


def emotion_means_from_dirs(
    results_dir: Path, emotions: list[str], layer: int
) -> "Float[np.ndarray, 'emotions d_model']":
    return np.stack(
        [np.load(results_dir / e.replace(" ", "_") / f"layer_{layer}_resid.npy") for e in emotions]
    )


def _final_norm(lm):
    """Gemma's final RMSNorm weight, wherever it lives in the wrapper hierarchy."""
    for attr in ("model.norm", "model.language_model.norm", "model.language_model.model.norm"):
        obj = lm.model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj.weight.detach().float()
        except AttributeError:
            continue
    return None


def lens_table(probes, emotions, w_unembed, final_norm, tokenizer, top_k):
    """Top/bottom-token table for a stack of already-centered probe directions."""
    import torch  # noqa: PLC0415

    table: dict[str, dict[str, list[str]]] = {}
    with torch.inference_mode():
        for emotion, probe in zip(emotions, probes):
            direction = torch.from_numpy(np.asarray(probe)).float().to(w_unembed.device)
            if final_norm is not None:
                direction = direction * (1.0 + final_norm)  # gemma norm: weight is (1 + w)
            logits = w_unembed @ direction
            table[emotion] = {
                "up": [
                    tokenizer.decode([t]).strip()
                    for t in torch.topk(logits, top_k).indices.tolist()
                ],
                "down": [
                    tokenizer.decode([t]).strip()
                    for t in torch.topk(-logits, top_k).indices.tolist()
                ],
            }
    return table


def _repro_check(model, by_layer, logger):
    """Compare the sweep's 33/57 tables against the committed single-layer files."""
    for layer, path in REPRO.get(model, {}).items():
        ref_path, key = Path(path), str(layer)
        if not ref_path.exists() or key not in by_layer:
            continue
        ref = json.loads(ref_path.read_text())["table"]
        got = by_layer[key]
        exact = sum(
            set(got[e]["up"]) == set(ref[e]["up"]) and set(got[e]["down"]) == set(ref[e]["down"])
            for e in ref
        )
        verdict = "OK" if exact >= len(ref) - 2 else "MISMATCH — do not trust other layers"
        logger.info(
            f"repro check L{layer} vs {ref_path.name}: {exact}/{len(ref)} emotions exact — {verdict}"
        )


def sweep_all_layers(args, logger) -> int:
    bundle = np.load(args.means_bundle, allow_pickle=True)
    emotions = list(map(str, bundle["emotions"]))
    layers = [int(l) for l in bundle["layers"]]
    idx = [emotions.index(e) for e in BATTERY]  # battery rows in the bundle's own order
    means = bundle["means"][idx].astype(np.float32)  # [12, layers, d_model]

    lm, _ = load_model_bf16(args.model, logger.info)
    w_unembed = lm.model.get_output_embeddings().weight.detach().float()
    final_norm = _final_norm(lm)

    by_layer: dict[str, dict] = {}
    for lp, layer in enumerate(layers):
        m = means[:, lp, :]
        probes = m - m.mean(axis=0)  # centre on the 12-battery mean, same as committed files
        by_layer[str(layer)] = lens_table(
            probes, BATTERY, w_unembed, final_norm, lm.tokenizer, args.top_k
        )
        logger.info(f"layer {layer}: {[by_layer[str(layer)][e]['up'][0] for e in BATTERY[:4]]} ...")

    _repro_check(args.model, by_layer, logger)
    args.out.write_text(
        json.dumps(
            {
                "model": args.model,
                "note": NOTE,
                "centering": "12-battery mean",
                "means_bundle": str(args.means_bundle),
                "emotions": BATTERY,
                "layers": layers,
                "by_layer": by_layer,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(f"written -> {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--results-dir", type=Path, default=Path("results/emotion_vectors_it"))
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/logit_lens_it.json"))
    parser.add_argument(
        "--all-layers", action="store_true", help="sweep every layer in --means-bundle"
    )
    parser.add_argument(
        "--means-bundle", type=Path, default=Path("results/emotion_vectors_it_means.npz")
    )
    args = parser.parse_args()

    logger = setup_logger(args.out.parent / "logit_lens.log")
    if args.all_layers:
        return sweep_all_layers(args, logger)

    means = emotion_means_from_dirs(args.results_dir, BATTERY, args.layer)
    probes = means - means.mean(axis=0)
    lm, _ = load_model_bf16(args.model, logger.info)
    w_unembed = lm.model.get_output_embeddings().weight.detach().float()  # [vocab, d_model]
    table = lens_table(probes, BATTERY, w_unembed, _final_norm(lm), lm.tokenizer, args.top_k)
    for emotion in BATTERY:
        logger.info(f"{emotion}: up={table[emotion]['up']} down={table[emotion]['down']}")

    args.out.write_text(
        json.dumps(
            {
                "model": args.model,
                "layer": args.layer,
                "probes": str(args.results_dir),
                "note": NOTE,
                "table": table,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
