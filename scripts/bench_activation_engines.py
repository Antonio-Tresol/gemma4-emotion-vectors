"""Definitive activation-extraction engine comparison — HF loop vs vLLM.

GPU entry point, pod-only. One arm per invocation (each loads the model;
run sequentially, HF arm in the project venv, vLLM arms in /root/vllm-env):

    .venv/bin/python scripts/bench_activation_engines.py --arm hf
    /root/vllm-env/bin/python scripts/bench_activation_engines.py --arm hooks
    /root/vllm-env/bin/python scripts/bench_activation_engines.py --arm official

All arms process the SAME 256 stories (first 256 of the self-generated -it
corpus, truncated to 512 tokens, the extraction pipeline's cap) and capture
all 20 extraction layers (0,3,...,57), so the numbers are what a real
corpus-scale job pays. Timing excludes model load, includes moving
activations off-GPU (hf/hooks) or safetensors writes (official — that is
its real cost). Appends one record per arm to
results/activation_engine_bench.json.

Context: the quick feasibility bench (results/vllm_activation_bench.json)
already established hooks fire and numerics are exact (cosine >= 0.99999).
This bench answers the remaining question: throughput at real workload
shape, for the research-engineering record.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CORPUS = Path("results/self_stories_it/dialogues_grouped.jsonl")
OUT = Path("results/activation_engine_bench.json")
MODEL = "google/gemma-4-31b-it"
LAYERS = list(range(0, 60, 3))
N_STORIES = 256
MAX_LENGTH = 512  # the extraction pipeline's cap (REFERENCE_MAX_LENGTH)


def load_stories() -> list[str]:
    """First N_STORIES stories from the grouped corpus, deterministic order."""
    stories: list[str] = []
    with open(CORPUS) as f:
        for line in f:
            row = json.loads(line)
            stories.extend(row["stories"])
            if len(stories) >= N_STORIES:
                break
    return stories[:N_STORIES]


def append_record(record: dict[str, object]) -> None:
    existing = json.loads(OUT.read_text()) if OUT.exists() else []
    existing = [r for r in existing if r["arm"] != record["arm"]]
    existing.append(record)
    OUT.write_text(json.dumps(existing, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def count_tokens(tokenizer: object, stories: list[str]) -> int:
    return sum(len(tokenizer(s, truncation=True, max_length=MAX_LENGTH).input_ids) for s in stories)


def truncate_texts(tokenizer: object, stories: list[str]) -> list[str]:
    """Decode-after-truncate so vLLM sees the same 512-token cap as the HF path."""
    out = []
    for s in stories:
        ids = tokenizer(
            s, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False
        ).input_ids
        out.append(tokenizer.decode(ids))
    return out


def arm_hf(stories: list[str]) -> dict[str, object]:
    """The production extraction path: run_batch with 20-layer capture."""
    from emotion_vectors.corpus import REFERENCE_BATCH_SIZE  # noqa: PLC0415
    from emotion_vectors.extraction import load_model_bf16, run_batch  # noqa: PLC0415

    lm, _ = load_model_bf16(MODEL, print)
    n_tokens = count_tokens(lm.tokenizer, stories)
    t0 = time.perf_counter()
    for start in range(0, len(stories), REFERENCE_BATCH_SIZE):
        run_batch(lm, stories[start : start + REFERENCE_BATCH_SIZE], LAYERS, MAX_LENGTH)
    elapsed = time.perf_counter() - t0
    return {
        "arm": "hf",
        "engine": "transformers, production run_batch, batch 4",
        "n_stories": len(stories),
        "n_tokens": n_tokens,
        "seconds": round(elapsed, 2),
        "tokens_per_s": round(n_tokens / elapsed),
    }


def arm_hooks(stories: list[str]) -> dict[str, object]:
    """vLLM with 20 forward hooks under enforce_eager (Route A)."""
    import os  # noqa: PLC0415

    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from bench_vllm_activations import find_layers  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=MAX_LENGTH + 8,
        gpu_memory_utilization=0.90,
        disable_log_stats=True,
    )
    layers = find_layers(llm.llm_engine.model_executor.driver_worker.model_runner.model)
    captured: list[object] = []

    def hook(_m: object, _i: object, out: object) -> None:
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach().float().cpu())

    texts = truncate_texts(llm.get_tokenizer(), stories)
    n_tokens = count_tokens(llm.get_tokenizer(), stories)
    handles = [layers[i].register_forward_hook(hook) for i in LAYERS]
    try:
        t0 = time.perf_counter()
        llm.generate(texts, SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
        elapsed = time.perf_counter() - t0
    finally:
        for h in handles:
            h.remove()
    return {
        "arm": "hooks",
        "engine": "vllm eager + 20 forward hooks (Route A)",
        "n_stories": len(stories),
        "n_tokens": n_tokens,
        "seconds": round(elapsed, 2),
        "tokens_per_s": round(n_tokens / elapsed),
        "n_captures": len(captured),
    }


def arm_official(stories: list[str]) -> dict[str, object]:
    """vLLM's extract_hidden_states API with the example connector (Route B)."""
    import os  # noqa: PLC0415
    import shutil  # noqa: PLC0415

    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    # corpus-scale activation dumps do not fit the pod's NVMe root disk (a
    # 256-story run is ~25 GB; the root disk filled at file 203 on the first
    # attempt), so the realistic target is the large network volume
    dump_dir = Path("/workspace/cambria/hidden_states_bench")
    shutil.rmtree(dump_dir, ignore_errors=True)
    dump_dir.mkdir(parents=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=MAX_LENGTH + 8,
        gpu_memory_utilization=0.90,
        disable_log_stats=True,
        enable_chunked_prefill=False,  # documented incompatibility
        # schema verified against the installed vllm 0.22.1 source
        # (vllm/config/speculative.py: draft_model_config["hf_config"] kwargs
        # feed ExtractHiddenStatesConfig)
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,  # asserted == 1 by the extract proposer
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": LAYERS},
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(dump_dir)},
        },
    )
    texts = truncate_texts(llm.get_tokenizer(), stories)
    n_tokens = count_tokens(llm.get_tokenizer(), stories)
    t0 = time.perf_counter()
    llm.generate(texts, SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
    elapsed = time.perf_counter() - t0
    n_files = len(list(dump_dir.glob("**/*.safetensors")))
    return {
        "arm": "official",
        "engine": "vllm extract_hidden_states + ExampleHiddenStatesConnector (Route B)",
        "n_stories": len(stories),
        "n_tokens": n_tokens,
        "seconds": round(elapsed, 2),
        "tokens_per_s": round(n_tokens / elapsed),
        "n_safetensors_files": n_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("hf", "hooks", "official"), required=True)
    args = parser.parse_args()
    stories = load_stories()
    record = {"hf": arm_hf, "hooks": arm_hooks, "official": arm_official}[args.arm](stories)
    append_record(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
