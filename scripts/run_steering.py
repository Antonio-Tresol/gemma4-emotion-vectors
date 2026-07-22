"""Q1.H3.E2 collection — steered preference passes (paper Figure 4, right half).

GPU entry point. For each steered emotion: add alpha * unit(centered emotion
vector) to the layer-33 residual output at every position during the
chat-format A/B preference pass, record the (A, B) logits. Baseline Elo comes
from the unsteered chat run (results/preferences_it_chat). Design and
predictions registered in TREE.md Q1.H3.E2 before this ran.

    # calibration (happy/angry exemplars, 512-pair subset, alphas 2/4/8):
    uv run python scripts/run_steering.py --calibrate
    # full run at the chosen alpha:
    uv run python scripts/run_steering.py --alpha 4
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
    import torch
    from run_preferences import ab_logit_pass, pair_prompts

    from emotion_vectors.extraction import get_layer, load_model_bf16, setup_logger
except ModuleNotFoundError as exc:  # torch lives in the gpu extra; this is a GPU entry point
    raise SystemExit(f"missing {exc.name}: this entry point needs `uv sync --extra gpu`") from exc

BATTERY = [
    "happy",
    "inspired",
    "loving",
    "proud",
    "calm",
    "desperate",
    "angry",
    "guilty",
    "sad",
    "afraid",
    "nervous",
    "surprised",
]
LAYER = 33


def steering_direction(bundle_file: Path, emotion: str) -> "Float[np.ndarray, 'd_model']":
    """Unit-norm centered emotion vector at the steering layer (171-mean baseline)."""
    bundle = np.load(bundle_file, allow_pickle=True)
    emotions = list(map(str, bundle["emotions"]))
    layer_pos = list(map(int, bundle["layers"])).index(LAYER)
    means = bundle["means"][:, layer_pos, :].astype(np.float64)
    v = means[emotions.index(emotion)] - means.mean(axis=0)
    return v / np.linalg.norm(v)


def median_residual_rms(lm: object, prompts: list[str]) -> float:
    """Median L2-per-sqrt(d) of the layer-33 residual over a few prompts."""
    norms = []
    for text in prompts[:8]:
        inputs = lm.tokenizer(text, return_tensors="pt").to(lm.model.device)
        captured: dict[int, torch.Tensor] = {}

        def hook(_m: object, _i: object, out: object) -> None:
            h = out[0] if isinstance(out, tuple) else out
            captured[0] = h.detach()

        handle = get_layer(lm.model, LAYER).register_forward_hook(hook)
        try:
            with torch.inference_mode():
                lm.model(**inputs)
        finally:
            handle.remove()
        h = captured[0].float()
        norms.append(h.norm(dim=-1).median().item() / (h.shape[-1] ** 0.5))
    return float(np.median(norms))


def steered_pass(
    lm: object,
    prompts: list[str],
    tok_a: int,
    tok_b: int,
    direction: "Float[np.ndarray, 'd_model']",
    scale: float,
) -> "Float[np.ndarray, 'pairs two']":
    """A/B logits with alpha-scaled steering added to the layer residual output."""
    vec = torch.tensor(direction * scale, dtype=torch.bfloat16, device=lm.model.device)

    def hook(_m: object, _i: object, out: object) -> object:
        if isinstance(out, tuple):
            return (out[0] + vec,) + out[1:]
        return out + vec

    handle = get_layer(lm.model, LAYER).register_forward_hook(hook)
    try:
        return ab_logit_pass(lm, prompts, tok_a, tok_b)
    finally:
        handle.remove()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--bundle", type=Path, default=Path("results/emotion_vectors_it_means.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/steering_it"))
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--alpha", type=float, default=4.0)
    args = parser.parse_args()

    logger = setup_logger(args.out_dir / "collect.log")
    activities = [a for _, a in all_activities()]
    lm, _ = load_model_bf16(args.model, logger.info)
    tok_a = lm.tokenizer("A", add_special_tokens=False).input_ids[-1]
    tok_b = lm.tokenizer("B", add_special_tokens=False).input_ids[-1]
    ordered, prompts = pair_prompts(lm.tokenizer, activities, "chat")
    rms = median_residual_rms(lm, prompts)
    logger.info(f"median residual rms at layer {LAYER}: {rms:.3f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.calibrate:
        rng = np.random.default_rng(20260722)
        subset = sorted(rng.choice(len(prompts), 512, replace=False))
        sub_prompts = [prompts[i] for i in subset]
        report = {"rms": rms, "subset": [int(i) for i in subset], "runs": {}}
        for emotion in ("happy", "angry"):
            direction = steering_direction(args.bundle, emotion)
            for alpha in (2.0, 4.0, 8.0):
                logits = steered_pass(lm, sub_prompts, tok_a, tok_b, direction, alpha * rms)
                key = f"{emotion}_a{alpha:g}"
                np.save(args.out_dir / f"calib_{key}.npy", logits.astype(np.float32))
                gap = float(np.abs(logits[:, 0] - logits[:, 1]).mean())
                report["runs"][key] = {"mean_abs_gap": round(gap, 4)}
                logger.info(f"calib {key}: mean |A-B gap| {gap:.3f}")
        (args.out_dir / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
        return 0

    for emotion in BATTERY:
        out_file = args.out_dir / f"steered_{emotion}.npy"
        if out_file.exists():
            logger.info(f"{emotion}: exists, skipping")
            continue
        direction = steering_direction(args.bundle, emotion)
        logits = steered_pass(lm, prompts, tok_a, tok_b, direction, args.alpha * rms)
        np.save(out_file, logits.astype(np.float32))
        logger.info(f"{emotion}: saved {logits.shape} (alpha={args.alpha:g} x rms)")
    (args.out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "alpha": args.alpha,
                "rms": rms,
                "layer": LAYER,
                "format": "chat",
                "pair_i": [i for i, _ in ordered],
                "pair_j": [j for _, j in ordered],
            }
        )
        + "\n"
    )
    logger.info("steering collection complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
