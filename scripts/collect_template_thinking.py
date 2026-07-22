"""Q1.H2.E8 thinking-mode arm — template activations under enable_thinking=True.

GPU entry point. Re-collects the 37 numerical-template prompts on
gemma-4-31b-it with the chat template's thinking mode enabled (the tokenizer
default, used by every other -it collection, is thinking-OFF — verified in
TREE Q1.H2.E8). Same layers and readouts as the probe sweep so the E8
diagnostic runs unchanged on the output.

    uv run python scripts/collect_template_thinking.py  # on the pod
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

try:
    from run_probe_sweep import READOUTS, collect_one

    from emotion_vectors.extraction import load_model_bf16, setup_logger
except ModuleNotFoundError as exc:  # torch lives in the gpu extra; this is a GPU entry point
    raise SystemExit(f"missing {exc.name}: this entry point needs `uv sync --extra gpu`") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--sweep-dir", type=Path, default=Path("results/probe_sweep_it"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/template_thinking_on"))
    args = parser.parse_args()

    logger = setup_logger(args.out_dir / "collect.log")
    prompts = [json.loads(line) for line in open(args.sweep_dir / "prompts.jsonl")]
    templates = [p for p in prompts if p["kind"] == "template"]
    sweep = np.load(args.sweep_dir / "activations.npz", allow_pickle=True)
    layers = list(map(int, sweep["layers"]))
    lm, _ = load_model_bf16(args.model, logger.info)

    arrays: dict[str, list[np.ndarray]] = {f"chat_{r}": [] for r in READOUTS}
    for row in templates:
        formatted = lm.tokenizer.apply_chat_template(
            [{"role": "user", "content": row["text"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        acts = collect_one(lm, formatted, row["text"], layers)
        for r in READOUTS:
            arrays[f"chat_{r}"].append(acts[r])
    logger.info(f"collected {len(templates)} template prompts, thinking ON")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "activations.npz",
        layers=np.array(layers),
        formats=np.array(["chat"]),
        **{k: np.stack(v).astype(np.float16) for k, v in arrays.items()},
    )
    # prompts manifest holding ONLY the template rows, index-aligned with the npz
    (args.out_dir / "prompts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in templates)
    )
    shutil.copy(args.sweep_dir / "prompts.jsonl", args.out_dir / "prompts_source.jsonl")
    logger.info(f"saved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
