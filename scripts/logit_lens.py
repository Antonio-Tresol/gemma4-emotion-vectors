"""Logit lens over emotion vectors — the paper's Table 1 replication.

Projects each emotion's centered contrast vector through the unembedding
matrix and records the top and bottom tokens. The paper reports emotion-word
neighborhoods (sad -> grief, tears, lonely). Simplifications documented in the
output: we apply the final norm's scale implicitly by using raw directions
(cosine-style lens), and Gemma's logit softcapping is ignored; both affect
magnitudes, not top-k ordering materially.

    uv run python scripts/logit_lens.py  # pod, after vectors exist
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


def emotion_means_from_dirs(
    results_dir: Path, emotions: list[str], layer: int
) -> "Float[np.ndarray, 'emotions d_model']":
    return np.stack(
        [np.load(results_dir / e.replace(" ", "_") / f"layer_{layer}_resid.npy") for e in emotions]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--results-dir", type=Path, default=Path("results/emotion_vectors_it"))
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/logit_lens_it.json"))
    args = parser.parse_args()

    logger = setup_logger(args.out.parent / "logit_lens.log")
    emotions = [t for _, t, _ in SCENARIOS]
    means = emotion_means_from_dirs(args.results_dir, emotions, args.layer)
    probes = means - means.mean(axis=0)

    lm, _ = load_model_bf16(args.model, logger.info)
    import torch  # noqa: PLC0415

    w_unembed = lm.model.get_output_embeddings().weight.detach().float()  # [vocab, d_model]
    table: dict[str, dict[str, list[str]]] = {}
    with torch.inference_mode():
        for emotion, probe in zip(emotions, probes):
            logits = w_unembed @ torch.from_numpy(probe).float().to(w_unembed.device)
            top = torch.topk(logits, args.top_k).indices.tolist()
            bottom = torch.topk(-logits, args.top_k).indices.tolist()
            table[emotion] = {
                "up": [lm.tokenizer.decode([t]).strip() for t in top],
                "down": [lm.tokenizer.decode([t]).strip() for t in bottom],
            }
            logger.info(f"{emotion}: up={table[emotion]['up']} down={table[emotion]['down']}")

    args.out.write_text(
        json.dumps(
            {
                "model": args.model,
                "layer": args.layer,
                "probes": str(args.results_dir),
                "note": "raw-direction lens; final-norm scale and logit softcapping ignored",
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
