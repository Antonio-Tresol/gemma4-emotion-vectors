"""Q1.H2.E1 activation collection — GPU entry point.

Runs the registered prompt battery (emotion_vectors.probe_prompts: the paper's
12 implicit-emotion scenarios and 6 numerical-intensity templates) through the
model and saves the residual-stream activation at the final pre-response token
for the requested layers. No scoring happens here — cosines against the
emotion vectors and the registered-prediction checks are laptop-side
(notebooks/03_probe_validation.ipynb), so the expensive step stays minimal.

    uv run python scripts/run_probe_validation.py  # on the pod
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from emotion_vectors.probe_prompts import build_prompts

try:
    from emotion_vectors.extraction import get_layer, load_model_bf16, make_hook, setup_logger
except ModuleNotFoundError as exc:  # torch lives in the gpu extra; this is a GPU entry point
    raise SystemExit(f"missing {exc.name}: this entry point needs `uv sync --extra gpu`") from exc


def last_token_activations(
    lm: object, text: str, layers: list[int]
) -> "np.ndarray":  # [n_layers, d_model]
    """One prompt, no padding: the last position is the pre-response token."""
    import torch  # noqa: PLC0415

    inputs = lm.tokenizer(text, return_tensors="pt").to(lm.model.device)
    captured: dict[int, torch.Tensor] = {}
    hooks = [get_layer(lm.model, i).register_forward_hook(make_hook(captured, i)) for i in layers]
    try:
        with torch.inference_mode():
            lm.model(**inputs)
    finally:
        for h in hooks:
            h.remove()
    return np.stack([captured[i][0, -1].cpu().numpy() for i in layers])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-31b")
    parser.add_argument("--layers", type=int, nargs="+", default=[33, 39])
    parser.add_argument("--out-dir", type=Path, default=Path("results/probe_validation"))
    args = parser.parse_args()

    logger = setup_logger(args.out_dir / "collect.log")
    prompts = build_prompts()
    logger.info(f"{len(prompts)} prompts, layers {args.layers}, model {args.model}")
    lm, _ = load_model_bf16(args.model, logger.info)

    acts = np.stack(
        [last_token_activations(lm, p["text"], args.layers) for p in prompts]
    )  # [n_prompts, n_layers, d_model]
    np.savez_compressed(args.out_dir / "activations.npz", acts=acts, layers=np.array(args.layers))
    (args.out_dir / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
    logger.info(f"saved {acts.shape} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
