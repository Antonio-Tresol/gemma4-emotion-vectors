"""Infra investigation — can vLLM serve as the activation-extraction engine?

GPU entry point, pod-only (needs the NVMe vllm env, see
scripts/pod/setup_vllm_env.sh and the Blackwell survival guide in
notes/vllm-parallel-inference-template.md). Three questions, answered in one
run and written to results/vllm_activation_bench.json:

  1. Feasibility: with enforce_eager and the in-process engine, do forward
     hooks on vLLM's model modules fire and capture the residual stream?
  2. Numerics: exact parity check — the 37 template prompts from the probe
     sweep, one at a time, last-token layer-33 vector vs the HF-path values
     stored in results/probe_sweep_it/activations.npz (expect cosine > 0.99;
     bf16 kernel-order differences bound the gap).
  3. Speed: wall-clock for a ~256-prompt prefill-only batched pass with the
     hook armed (the HF comparison number comes from the extraction logs:
     1539 stories in 7.8 min = ~0.30 s/story on the same card).

    /root/vllm-env/bin/python scripts/bench_vllm_activations.py  # on the pod
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

# Blackwell survival settings (see the vllm notes): TRITON_ATTN backend, no
# flashinfer sampler, in-process V0 engine so hooks live in our process.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "google/gemma-4-31b-it"
LAYER = 33
SWEEP_DIR = Path("results/probe_sweep_it")
OUT = Path("results/vllm_activation_bench.json")


def find_layers(model: object) -> list[object]:
    """Locate the decoder layer list on a vLLM-wrapped model."""
    for path in ("model.layers", "language_model.model.layers", "model.model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return list(obj)  # decoder blocks; torch.nn.Module, typed loosely (vllm internals)
    raise RuntimeError(f"no layer list found on {type(model).__name__}")


def main() -> int:
    import torch  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    report: dict[str, object] = {"model": MODEL, "layer": LAYER}
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        disable_log_stats=True,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    layers = find_layers(runner.model)
    report["n_layers_found"] = len(layers)

    captured: list[torch.Tensor] = []

    def hook(_m: object, _i: object, out: object) -> None:
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach().float().cpu())

    params = SamplingParams(max_tokens=1, temperature=0.0)

    # 1. feasibility
    handle = layers[LAYER].register_forward_hook(hook)
    try:
        llm.generate(["The quick brown fox"], params, use_tqdm=False)
    finally:
        handle.remove()
    report["hooks_fire"] = bool(captured)
    report["captured_shape"] = list(captured[0].shape) if captured else None
    if not captured:
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print("hooks did NOT fire; vLLM path infeasible under this engine config")
        return 0

    # 2. exact numerics parity on the sweep's template prompts, one at a time
    prompts = [json.loads(line) for line in open(SWEEP_DIR / "prompts.jsonl")]
    sweep = np.load(SWEEP_DIR / "activations.npz", allow_pickle=True)
    layer_pos = list(map(int, sweep["layers"])).index(LAYER)
    hf_last = sweep["chat_last"].astype(np.float64)[:, layer_pos, :]
    tokenizer = llm.get_tokenizer()
    cosines = []
    for idx, row in enumerate(prompts):
        if row["kind"] != "template":
            continue
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["text"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        captured.clear()
        handle = layers[LAYER].register_forward_hook(hook)
        try:
            llm.generate(formatted, params, use_tqdm=False)
        finally:
            handle.remove()
        v = captured[0].reshape(-1, captured[0].shape[-1])[-1].numpy().astype(np.float64)
        h = hf_last[idx]
        cosines.append(float(np.dot(v, h) / (np.linalg.norm(v) * np.linalg.norm(h))))
    report["numerics_cosine_min"] = round(min(cosines), 5)
    report["numerics_cosine_mean"] = round(float(np.mean(cosines)), 5)
    report["numerics_n_prompts"] = len(cosines)

    # 3. batched prefill-only speed with the hook armed
    bench = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["text"]}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
        for _ in range(7)
    ][:256]
    captured.clear()
    handle = layers[LAYER].register_forward_hook(hook)
    try:
        t0 = time.perf_counter()
        llm.generate(bench, params, use_tqdm=False)
        vllm_s = time.perf_counter() - t0
    finally:
        handle.remove()
    report["vllm_hooked_prefill_s"] = round(vllm_s, 2)
    report["vllm_hooked_prefill_n"] = len(bench)
    report["vllm_s_per_prompt"] = round(vllm_s / len(bench), 4)
    report["hf_reference_s_per_story"] = 0.30  # 1539 stories / 7.8 min, extraction log
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
