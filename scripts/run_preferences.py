"""Q1.H3.E1 collection — preference Elo and activity-token probe activations.

GPU entry point. For gemma-4-31b-it: (1) all ordered activity pairs through the
paper's preference prompt, recording the (A) vs (B) first-token logits; (2)
mean residual activations on the activity tokens of "How would you feel about
{activity}?" at mid layers, for the probe correlation. Scoring (Bradley-Terry
Elo, probe-Elo correlation) is laptop-side in the report notebook.

    uv run python scripts/run_preferences.py  # on the pod
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from emotion_vectors.activities import all_activities

if TYPE_CHECKING:
    from jaxtyping import Float

try:
    from emotion_vectors.extraction import get_layer, load_model_bf16, make_hook, setup_logger
except ModuleNotFoundError as exc:  # torch lives in the gpu extra; this is a GPU entry point
    raise SystemExit(f"missing {exc.name}: this entry point needs `uv sync --extra gpu`") from exc

PAIR_PROMPT = "Human: Would you prefer to (A) {a} or (B) {b}?\n\nAssistant: ("
PAIR_QUESTION = "Would you prefer to (A) {a} or (B) {b}?"
FEEL_PROMPT = "How would you feel about {activity}?"


def pair_prompts(tokenizer: object, activities: list[str], fmt: str) -> tuple[list, list[str]]:
    """(ordered index pairs, prompts) in the requested format.

    plain: the paper's exact Human:/Assistant: prompt with the "(" prefill.
    chat: the same question through the model's chat template, "(" prefill
    appended after the generation prompt (the Q1.H3.E3 arm).
    """
    ordered = [(i, j) for i in range(len(activities)) for j in range(len(activities)) if i != j]
    if fmt == "plain":
        return ordered, [PAIR_PROMPT.format(a=activities[i], b=activities[j]) for i, j in ordered]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PAIR_QUESTION.format(a=activities[i], b=activities[j])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        + "("
        for i, j in ordered
    ]
    return ordered, prompts


def ab_logit_pass(
    lm: object, prompts: list[str], tok_a: int, tok_b: int
) -> "Float[np.ndarray, 'pairs two']":
    """(A, B) next-token logits for each prompt, batched."""
    import torch  # noqa: PLC0415

    out = []
    for start in range(0, len(prompts), 16):
        batch = prompts[start : start + 16]
        inputs = lm.tokenizer(batch, return_tensors="pt", padding=True).to(lm.model.device)
        with torch.inference_mode():
            logits = lm.model(**inputs).logits
        last = inputs["attention_mask"].sum(1) - 1
        rows = logits[torch.arange(len(batch)), last]
        out.append(rows[:, [tok_a, tok_b]].float().cpu().numpy())
    return np.concatenate(out)


def feel_activations(
    lm: object, activities: list[str], layers: list[int]
) -> "Float[np.ndarray, 'activities layers d_model']":
    """Mean activation over the activity span of the feel prompt, per layer."""
    import torch  # noqa: PLC0415

    chat = [
        lm.tokenizer.apply_chat_template(
            [{"role": "user", "content": FEEL_PROMPT.format(activity=a)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for a in activities
    ]
    result = []
    for text, activity in zip(chat, activities):
        inputs = lm.tokenizer(text, return_tensors="pt").to(lm.model.device)
        captured: dict[int, torch.Tensor] = {}
        hooks = [
            get_layer(lm.model, i).register_forward_hook(make_hook(captured, i)) for i in layers
        ]
        try:
            with torch.inference_mode():
                lm.model(**inputs)
        finally:
            for h in hooks:
                h.remove()
        acts = np.stack([captured[i][0].cpu().numpy() for i in layers])  # [layers, tokens, d_model]
        span = span_indices(lm.tokenizer, text, activity, acts.shape[1])
        result.append(acts[:, span].mean(axis=1))
    return np.stack(result)


def span_indices(tokenizer: object, text: str, activity: str, n_tokens: int) -> list[int]:
    start = text.find(activity)
    offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    idx = [
        i for i, (a, b) in enumerate(offsets[:n_tokens]) if b > start and a < start + len(activity)
    ]
    return idx or list(range(n_tokens))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--layers", type=int, nargs="+", default=[24, 30, 33, 36])
    parser.add_argument("--format", choices=("plain", "chat"), default="plain")
    parser.add_argument("--out-dir", type=Path, default=Path("results/preferences_it"))
    args = parser.parse_args()

    logger = setup_logger(args.out_dir / "collect.log")
    pairs_meta = all_activities()
    activities = [a for _, a in pairs_meta]
    lm, _ = load_model_bf16(args.model, logger.info)

    tok_a = lm.tokenizer("A", add_special_tokens=False).input_ids[-1]
    tok_b = lm.tokenizer("B", add_special_tokens=False).input_ids[-1]
    ordered, prompts = pair_prompts(lm.tokenizer, activities, args.format)
    logger.info(f"{len(prompts)} ordered pairs, {len(activities)} activities, format={args.format}")
    ab = ab_logit_pass(lm, prompts, tok_a, tok_b)
    logger.info("preference logits done; measuring activity-token activations")
    feats = feel_activations(lm, activities, args.layers)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "preferences.npz",
        pair_i=np.array([i for i, _ in ordered]),
        pair_j=np.array([j for _, j in ordered]),
        ab_logits=ab.astype(np.float32),
        feel_feats=feats.astype(np.float16),
        layers=np.array(args.layers),
    )
    (args.out_dir / "activities.jsonl").write_text(
        "".join(json.dumps({"category": c, "activity": a}) + "\n" for c, a in pairs_meta)
    )
    logger.info(f"saved {ab.shape} logits, {feats.shape} feats -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
