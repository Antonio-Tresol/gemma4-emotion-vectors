"""Estimate wall-clock time for Q1 activation extraction on the pod.

Two modes:

  analytic (default, laptop-friendly): measures the real corpus (token counts
      via the Gemma tokenizer when available, a words-based proxy otherwise)
      and reports time under a range of assumed prefill throughputs. Use this
      to budget before renting anything.

  --benchmark (pod, needs the gpu extra): loads the model, times real forward
      passes with hidden-state capture over a seeded sample of stories, and
      extrapolates from the MEASURED tokens/sec. Trust this one.

Run:
    uv run python scripts/estimate_extraction_time.py
    uv run python scripts/estimate_extraction_time.py --benchmark --sample 8
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Final

TOKENS_PER_WORD_PROXY: Final[float] = 1.3  # typical English BPE ratio; reported when used
TEXT_COLUMN_CANDIDATES: Final[tuple[str, ...]] = ("story", "text", "content", "completion")


def find_text_column(column_names: list[str], sample_row: dict[str, object]) -> str:
    for name in TEXT_COLUMN_CANDIDATES:
        if name in column_names:
            return name
    string_cols = [c for c in column_names if isinstance(sample_row[c], str)]
    return max(string_cols, key=lambda c: len(str(sample_row[c])))


def count_tokens(texts: list[str], tokenizer_name: str) -> tuple[list[int], str]:
    """Per-story token counts, with the method used ('tokenizer' or 'word-proxy')."""
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415 — heavy, optional
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return [len(tokenizer(t).input_ids) for t in texts], f"tokenizer:{tokenizer_name}"
    except Exception as exc:  # gated model / offline / missing token
        print(f"note: tokenizer unavailable ({type(exc).__name__}); using "
              f"words x {TOKENS_PER_WORD_PROXY} proxy")
        return [int(len(t.split()) * TOKENS_PER_WORD_PROXY) for t in texts], "word-proxy"


def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def analytic_report(total_tokens: int, n_stories: int, args: argparse.Namespace) -> dict[str, object]:
    scenarios = {}
    for tps in args.throughputs:
        compute = total_tokens / tps * args.overhead
        scenarios[f"{tps}_tok_per_s"] = {
            "forward_passes": human(compute),
            "with_model_load": human(compute + args.load_seconds),
        }
        print(f"  @{tps:>5} tok/s prefill: {human(compute):>8} compute, "
              f"{human(compute + args.load_seconds):>8} incl. one model load")
    return scenarios


def benchmark_tokens_per_second(texts: list[str], args: argparse.Namespace) -> float:
    import torch  # noqa: PLC0415 — gpu extra only
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"loading {args.model} (bf16)...")
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    print(f"model loaded in {human(time.monotonic() - t0)}")
    rng = random.Random(args.seed)
    sample = rng.sample(texts, min(args.sample, len(texts)))
    timed_tokens = 0
    t0 = time.monotonic()
    with torch.no_grad():
        for text in sample:
            batch = tokenizer(text, return_tensors="pt").to(model.device)
            model(**batch, output_hidden_states=True)
            timed_tokens += int(batch.input_ids.shape[1])
    elapsed = time.monotonic() - t0
    tps = timed_tokens / elapsed
    print(f"measured: {timed_tokens} tokens in {human(elapsed)} -> {tps:.0f} tok/s "
          f"(hidden states captured, batch size 1)")
    return tps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="snae/emotion_stories_gemma_4_4B")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="google/gemma-4-31b",
                        help="tokenizer source; the model itself loads only with --benchmark")
    parser.add_argument("--throughputs", type=int, nargs="+", default=[1000, 3000, 6000],
                        help="prefill tok/s scenarios for analytic mode")
    parser.add_argument("--overhead", type=float, default=1.2,
                        help="multiplier for hook/IO overhead on top of pure forward passes")
    parser.add_argument("--load-seconds", type=float, default=300,
                        help="assumed one-off model load time from the volume")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sample", type=int, default=8, help="stories to time in benchmark mode")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--out", type=Path, default=Path("results/extraction_time_estimate.json"))
    args = parser.parse_args()

    from datasets import load_dataset  # noqa: PLC0415 — after argparse so --help stays instant
    ds = load_dataset(args.dataset, split=args.split)
    column = find_text_column(ds.column_names, ds[0])
    texts = list(ds[column])
    lengths, method = count_tokens(texts, args.model)
    total = sum(lengths)
    print(f"corpus: {len(texts)} stories ({column!r}), {total:,} tokens total "
          f"(mean {total / len(texts):.0f}/story, method={method})")

    result: dict[str, object] = {
        "dataset": args.dataset, "n_stories": len(texts), "total_tokens": total,
        "token_method": method, "overhead_multiplier": args.overhead, "seed": args.seed,
    }
    if args.benchmark:
        tps = benchmark_tokens_per_second(texts, args)
        compute = total / tps * args.overhead
        result["measured_tokens_per_second"] = round(tps, 1)
        result["estimated_total"] = human(compute)
        print(f"\nfull extraction estimate: {human(compute)} "
              f"(+ one model load, measured above)")
    else:
        result["scenarios"] = analytic_report(total, len(texts), args)
        print("\nrun with --benchmark on the pod for a measured estimate")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
