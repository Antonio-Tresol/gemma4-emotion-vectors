"""Estimate wall-clock time for Q1 activation extraction on the pod.

Mirrors the reference pipeline (sinievanderben/emotion_experiment,
extract_emotion_vectors.py) as closely as possible so the estimate is robust:
the same {emotion: [stories]} loading, the same plain-tokenizer call with
truncation at max_length, the same batch-of-4 padding arithmetic — and in
benchmark mode it runs the production run_batch from cbai_cambria.pipeline,
so the measurement is of the exact code the extraction executes, plus the
per-story disk writes our adaptation needs for resumability (the reference
accumulates in memory and saves once at the end — a crash loses the whole run).

Two modes:

  analytic (default, laptop-friendly): tokenises the real corpus and reports
      time under a range of assumed prefill throughputs, plus the disk budget
      for resumable per-story writes.

  --benchmark (pod, needs the gpu extra): loads the model in bf16 (the
      reference's float32 default needs ~124 GB for the 31B model and does not
      fit the 96 GB card), times real batches through the reference loop, and
      extrapolates from measured numbers, reporting forward and write stages
      separately.

Run:
    uv run python scripts/estimate_extraction_time.py
    uv run python scripts/estimate_extraction_time.py --benchmark --sample-batches 6
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Final

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from cbai_cambria.corpus import (
    REFERENCE_BATCH_SIZE,
    REFERENCE_MAX_LENGTH,
    human,
    load_emotions_data,
)

FP32_BYTES: Final[int] = 4
ASSUMED_D_MODEL: Final[int] = 6144  # fallback when the gated config is unreachable
ASSUMED_WRITE_MB_S: Final[float] = 200.0


def story_token_lengths(
    emotions_data: dict[str, list[str]], model_name: str, max_length: int
) -> tuple[dict[str, list[int]], str]:
    """Per-story truncated token counts using the reference tokenizer call."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        method = f"tokenizer:{model_name}, truncated at {max_length}"

        def length(text: str) -> int:
            return min(len(tokenizer(text).input_ids), max_length)
    except Exception as exc:
        print(f"note: tokenizer unavailable ({type(exc).__name__}); words x 1.3 proxy")
        method = "word-proxy x1.3"

        def length(text: str) -> int:
            return min(int(len(text.split()) * 1.3), max_length)

    return ({e: [length(s) for s in stories] for e, stories in emotions_data.items()}, method)


def padded_token_total(lengths_by_emotion: dict[str, list[int]], batch_size: int) -> int:
    """Compute-relevant tokens: each batch pads to its longest story, and the
    reference batches within each emotion's story list, in order."""
    total = 0
    for lengths in lengths_by_emotion.values():
        for start in range(0, len(lengths), batch_size):
            batch = lengths[start : start + batch_size]
            total += len(batch) * max(batch)
    return total


def detect_d_model(model_name: str) -> tuple[int, str]:
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        d = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
        return int(d), "from config"
    except Exception:
        return ASSUMED_D_MODEL, "ASSUMED — verify from the model config on the pod"


def disk_budget(n_stories: int, n_layers: int, d_model: int) -> dict[str, str]:
    """Resumable-adaptation writes: one pooled vector per story per layer."""
    per_story = n_layers * d_model * FP32_BYTES
    total = n_stories * per_story
    return {
        "per_story": f"{per_story / 1e6:.1f} MB",
        "corpus_total": f"{total / 1e9:.2f} GB",
        "write_time_at_assumed_throughput": human(total / (ASSUMED_WRITE_MB_S * 1e6)),
    }


def analytic_report(padded_tokens: int, args: argparse.Namespace) -> dict[str, object]:
    scenarios: dict[str, object] = {}
    for tps in args.throughputs:
        compute = padded_tokens / tps * args.overhead
        scenarios[f"{tps}_tok_per_s"] = {
            "forward": human(compute),
            "with_model_load": human(compute + args.load_seconds),
        }
        print(
            f"  @{tps:>5} tok/s: {human(compute):>8} forward, "
            f"{human(compute + args.load_seconds):>8} incl. one model load"
        )
    return scenarios


def bench_sample(emotions_data: dict[str, list[str]], seed: int, n_batches: int) -> list[str]:
    """A random story sample sized to n_batches reference-sized batches."""
    rng = random.Random(seed)
    all_stories = [s for stories in emotions_data.values() for s in stories]
    return rng.sample(all_stories, min(n_batches * REFERENCE_BATCH_SIZE, len(all_stories)))


def benchmark(emotions_data: dict[str, list[str]], args: argparse.Namespace) -> dict[str, float]:
    """Time the production pipeline: run_batch (tokenize -> forward with hooks
    -> pool) from cbai_cambria.pipeline, then the per-story shard writes."""
    # The one sanctioned lazy import: analytic mode must run on machines
    # without the gpu extra, so the torch-importing pipeline loads only here.
    from cbai_cambria.pipeline import load_model_bf16, run_batch  # noqa: PLC0415

    lm, load_s = load_model_bf16(args.model)
    layers = list(range(0, args.n_layers_total, args.layer_stride))
    sample = bench_sample(emotions_data, args.seed, args.sample_batches)
    out_dir = Path(args.out).parent / "benchmark_scratch"
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = {"forward_s": 0.0, "write_s": 0.0, "tokens": 0.0}
    for start in range(0, len(sample), REFERENCE_BATCH_SIZE):
        texts = sample[start : start + REFERENCE_BATCH_SIZE]
        t0 = time.monotonic()
        result = run_batch(lm, texts, layers, REFERENCE_MAX_LENGTH)
        stages["forward_s"] += time.monotonic() - t0
        stages["tokens"] += float(result.raw_tokens)
        t0 = time.monotonic()
        for story_idx, pooled in enumerate(result.means):
            np.save(out_dir / f"story_{start + story_idx}.npy", pooled)
        stages["write_s"] += time.monotonic() - t0
    stages["model_load_s"] = load_s
    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="snae/emotion_stories_gemma_4_4B")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="google/gemma-4-31b")
    parser.add_argument("--throughputs", type=int, nargs="+", default=[1000, 3000, 6000])
    parser.add_argument("--overhead", type=float, default=1.2)
    parser.add_argument("--load-seconds", type=float, default=300)
    parser.add_argument("--n-layers-total", type=int, default=60)
    parser.add_argument(
        "--layer-stride", type=int, default=3, help="sweep every Nth layer (plan: every 3rd of 60)"
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sample-batches", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--out", type=Path, default=Path("results/extraction_time_estimate.json"))
    return parser.parse_args()


def corpus_report(
    args: argparse.Namespace, emotions_data: dict[str, list[str]]
) -> tuple[dict[str, object], int]:
    """Tokenise the corpus, print the analytic summary, and build the base
    result record. Returns (result, padded_token_total)."""
    n_stories = sum(len(v) for v in emotions_data.values())
    print(
        f"corpus: {len(emotions_data)} emotions, {n_stories} stories "
        f"(reference-style loading from the 'stories' lists)"
    )
    lengths, method = story_token_lengths(emotions_data, args.model, REFERENCE_MAX_LENGTH)
    raw_total = sum(sum(v) for v in lengths.values())
    padded = padded_token_total(lengths, REFERENCE_BATCH_SIZE)
    print(
        f"tokens: {raw_total:,} raw, {padded:,} padded in reference batches of "
        f"{REFERENCE_BATCH_SIZE} ({method})"
    )
    n_layers = len(range(0, args.n_layers_total, args.layer_stride))
    d_model, d_src = detect_d_model(args.model)
    disk = disk_budget(n_stories, n_layers, d_model)
    print(
        f"resumable writes: {n_layers} layers x d_model {d_model} ({d_src}) -> "
        f"{disk['per_story']}/story, {disk['corpus_total']} corpus, "
        f"~{disk['write_time_at_assumed_throughput']} at {ASSUMED_WRITE_MB_S:.0f} MB/s"
    )
    result: dict[str, object] = {
        "dataset": args.dataset,
        "n_emotions": len(emotions_data),
        "n_stories": n_stories,
        "raw_tokens": raw_total,
        "padded_tokens": padded,
        "token_method": method,
        "n_layers_swept": n_layers,
        "d_model": d_model,
        "d_model_source": d_src,
        "resumable_disk": disk,
        "seed": args.seed,
        "note": "reference default fp32 needs ~124 GB for 31B; run bf16 on the 96 GB card",
    }
    return result, padded


def benchmark_report(stages: dict[str, float], padded: int) -> dict[str, object]:
    """Extrapolate measured stage times to the full corpus."""
    tps = stages["tokens"] / stages["forward_s"]
    write_frac = stages["write_s"] / max(stages["forward_s"], 1e-9)
    est = padded / tps * (1 + write_frac)
    print(
        f"\nmeasured {tps:.0f} tok/s; writes add {write_frac:.1%}; "
        f"full run ~{human(est)} + {human(stages['model_load_s'])} load"
    )
    return {
        **{k: round(v, 2) for k, v in stages.items()},
        "tokens_per_s": round(tps, 1),
        "estimated_total": human(est + stages["model_load_s"]),
    }


def main() -> int:
    args = parse_args()
    emotions_data = load_emotions_data(args.dataset, args.split)
    result, padded = corpus_report(args, emotions_data)
    if args.benchmark:
        result["benchmark"] = benchmark_report(benchmark(emotions_data, args), padded)
    else:
        result["scenarios"] = analytic_report(padded, args)
        print("\nrun with --benchmark on the pod for measured numbers")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
