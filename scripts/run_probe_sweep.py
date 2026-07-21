"""Q1.H2.E2 collection — layer x readout x format sweep of the probe battery.

Collects residual activations for every battery prompt (the paper's 12
scenarios, our 12 held-out confirmation scenarios, and the 37 numerical
template points) across ALL swept layers, under two formats (plain
Human:/Assistant: text and the model's chat template, if it has one), keeping
three readouts per prompt: the last token, the mean over all tokens, and the
mean over the scenario-content tokens (located via tokenizer offset mapping).

Scoring is laptop-side (notebooks/04_probe_sweep.ipynb) against the selection
rule registered in TREE.md: a (layer, readout, format) combination selected on
the paper's 12 scenarios counts only if it also passes on the held-out 12.

    uv run python scripts/run_probe_sweep.py  # on the pod
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from emotion_vectors.probe_prompts import PROMPT_FORMAT, build_prompts

if TYPE_CHECKING:
    from jaxtyping import Bool, Float

try:
    from emotion_vectors.extraction import get_layer, load_model_bf16, make_hook, setup_logger
except ModuleNotFoundError as exc:  # torch lives in the gpu extra; this is a GPU entry point
    raise SystemExit(f"missing {exc.name}: this entry point needs `uv sync --extra gpu`") from exc

READOUTS = ("last", "mean_all", "mean_content")


def format_text(tokenizer: object, text: str, fmt: str) -> str:
    if fmt == "plain":
        return PROMPT_FORMAT.format(text=text)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def content_mask(
    tokenizer: object, formatted: str, raw: str, n_tokens: int
) -> "Bool[np.ndarray, 't']":
    """Token positions overlapping the raw scenario text; all-True fallback."""
    mask = np.zeros(n_tokens, dtype=bool)
    start = formatted.find(raw)
    if start >= 0:
        end = start + len(raw)
        offsets = tokenizer(formatted, return_offsets_mapping=True)["offset_mapping"]
        for i, (a, b) in enumerate(offsets[:n_tokens]):
            if b > start and a < end:
                mask[i] = True
    if not mask.any():
        mask[:] = True
    return mask


def collect_one(
    lm: object, formatted: str, raw: str, layers: list[int]
) -> dict[str, "Float[np.ndarray, 'l d']"]:
    import torch  # noqa: PLC0415

    inputs = lm.tokenizer(formatted, return_tensors="pt").to(lm.model.device)
    captured: dict[int, torch.Tensor] = {}
    hooks = [get_layer(lm.model, i).register_forward_hook(make_hook(captured, i)) for i in layers]
    try:
        with torch.inference_mode():
            lm.model(**inputs)
    finally:
        for h in hooks:
            h.remove()
    acts = np.stack([captured[i][0].cpu().numpy() for i in layers])  # [L, T, d]
    cmask = content_mask(lm.tokenizer, formatted, raw, acts.shape[1])
    return {
        "last": acts[:, -1],
        "mean_all": acts.mean(axis=1),
        "mean_content": acts[:, cmask].mean(axis=1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-31b")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(0, 60, 3)))
    parser.add_argument("--out-dir", type=Path, default=Path("results/probe_sweep"))
    args = parser.parse_args()

    logger = setup_logger(args.out_dir / "collect.log")
    prompts = build_prompts()
    lm, _ = load_model_bf16(args.model, logger.info)
    formats = ["plain"] + (["chat"] if lm.tokenizer.chat_template else [])
    if len(formats) == 1:
        logger.warning("tokenizer has no chat template — chat arm skipped, reported as such")
    logger.info(f"{len(prompts)} prompts x {formats} x {len(args.layers)} layers")

    arrays: dict[str, list[np.ndarray]] = {f"{f}_{r}": [] for f in formats for r in READOUTS}
    for i, p in enumerate(prompts):
        for fmt in formats:
            readouts = collect_one(
                lm, format_text(lm.tokenizer, p["text"], fmt), p["text"], args.layers
            )
            for r in READOUTS:
                arrays[f"{fmt}_{r}"].append(readouts[r])
        if (i + 1) % 10 == 0 or i + 1 == len(prompts):
            logger.info(f"{i + 1}/{len(prompts)} prompts collected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "activations.npz",
        layers=np.array(args.layers),
        formats=np.array(formats),
        **{k: np.stack(v).astype(np.float16) for k, v in arrays.items()},
    )
    (args.out_dir / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
    logger.info(
        f"saved {len(prompts)} x {len(args.layers)} layers x {list(arrays)} -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
