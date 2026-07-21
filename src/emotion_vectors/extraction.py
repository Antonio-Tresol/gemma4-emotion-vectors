"""The extraction pipeline — reference-faithful math, Q1.H1.E1. GPU-only:
imports torch at module level, so it needs the `gpu` dependency extra.

Mirrors sinievanderben/emotion_experiment extract_emotion_vectors.py: same
tokenizer call (padding to the batch max, truncation), same forward-hook
capture of each target layer's output, same model-family layer lookup, same
pooling mask (padding zeroed, first TOKEN_OFFSET tokens zeroed). Deliberate
deviations from the reference:

  - bf16 load instead of fp32 (fp32 needs ~124 GB for the 31B model and does
    not fit the 96 GB card; hooks cast activations to fp32, as the reference's
    hooks already do).
  - torch.inference_mode() instead of no_grad.
  - per-story persistence via story_store: the reference keeps one running sum
    per emotion in memory and saves only at the end, so a crash loses the
    whole run. We save each story's pooled mean plus its post-mask token
    count, and recombine as sum(mean_i * n_i) / sum(n_i) — the identical
    token-weighted mean — so the run is resumable and per-story vectors
    support within-emotion variance analyses the reference cannot do.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, reduce
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotion_vectors.corpus import TOKEN_OFFSET, human
from emotion_vectors.story_store import (
    append_jsonl,
    load_manifest,
    pending_stories,
    sha1,
    story_key,
)

LOG_EVERY_BATCHES = 10


@dataclass(frozen=True)
class LoadedModel:
    """Tokenizer and model travel together through the pipeline."""

    tokenizer: object
    model: object


@dataclass(frozen=True)
class RunSettings:
    """One extraction run's parameters, echoed verbatim into run_config.json."""

    model: str
    dataset: str
    split: str
    out_dir: Path
    layers: list[int]
    d_model: int
    batch_size: int
    max_length: int
    seed: int
    smoke: bool
    git_commit: str

    @property
    def shards_dir(self) -> Path:
        return self.out_dir / "shards"

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / "manifest.jsonl"

    def to_config(self) -> dict[str, object]:
        config = {k: v for k, v in asdict(self).items() if k != "out_dir"}
        config["token_offset"] = TOKEN_OFFSET
        return config


@dataclass(frozen=True)
class BatchResult:
    """Pooled activations for one batch of stories."""

    means: Float[np.ndarray, "batch layers d_model"]
    token_counts: list[int]  # post-mask tokens per story: the aggregation weights
    raw_tokens: int  # attention-mask total, for throughput accounting


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extract")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_file), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def get_layer(model: object, idx: int) -> object:
    """The reference's _get_layer, condensed: model-family layer lookup."""
    for fn in (
        lambda m, i: m.model.layers[i],
        lambda m, i: m.model.language_model.model.layers[i],
        lambda m, i: m.model.language_model.layers[i],
        lambda m, i: m.language_model.model.layers[i],
    ):
        try:
            return fn(model, idx)
        except (AttributeError, IndexError):
            continue
    raise AttributeError(f"cannot locate layer {idx}")


def load_model_bf16(
    model_name: str, log: Callable[[str], None] = print
) -> tuple[LoadedModel, float]:
    """(loaded model, load_seconds). bf16, not the reference's fp32:
    fp32 needs ~124 GB for the 31B model and does not fit the 96 GB card."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"loading {model_name} (bf16)...")
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    load_s = time.monotonic() - t0
    log(f"model loaded in {human(load_s)}")
    return LoadedModel(tokenizer=tokenizer, model=model), load_s


def make_hook(
    captured: dict[int, Float[Tensor, "batch seq d_model"]], idx: int
) -> Callable[[object, object, object], None]:
    def _hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        captured[idx] = hidden.detach().float()  # reference casts to fp32 here

    return _hook


def pool_batch(
    captured: dict[int, Float[Tensor, "batch seq d_model"]],
    attention_mask: Int[Tensor, "batch seq"],
    layers: list[int],
) -> tuple[Float[np.ndarray, "batch layers d_model"], list[int]]:
    """Reference pooling, kept per-story: mask padding and the first
    TOKEN_OFFSET tokens, then mean over the surviving token positions."""
    mask = attention_mask.clone()
    mask[:, :TOKEN_OFFSET] = 0  # reference: early tokens are narrative framing
    mask_f = rearrange(mask, "batch seq -> batch seq 1").float()
    counts = reduce(mask.float(), "batch seq -> batch", "sum")
    sums = torch.stack(
        [reduce(captured[i] * mask_f, "batch seq d_model -> batch d_model", "sum") for i in layers]
    )
    denom = rearrange(counts.clamp(min=1.0), "batch -> 1 batch 1")
    means = (
        rearrange(sums / denom, "layers batch d_model -> batch layers d_model")
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return means, [int(c) for c in counts.tolist()]


def run_batch(lm: LoadedModel, texts: list[str], layers: list[int], max_length: int) -> BatchResult:
    """Tokenize -> forward with hooks -> pool. The reference loop, per batch."""
    inputs = lm.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(lm.model.device)
    captured: dict[int, Float[Tensor, "batch seq d_model"]] = {}
    hooks = [get_layer(lm.model, i).register_forward_hook(make_hook(captured, i)) for i in layers]
    try:
        with torch.inference_mode():
            lm.model(**inputs)
    finally:
        for h in hooks:
            h.remove()
    means, counts = pool_batch(captured, inputs["attention_mask"], layers)
    return BatchResult(means, counts, int(inputs["attention_mask"].sum().item()))


def record_failure(
    settings: RunSettings, emotion: str, batch: list[tuple[int, str]], exc: Exception
) -> None:
    for idx, text in batch:
        append_jsonl(
            settings.manifest_path,
            {
                "story_id": story_key(emotion, idx),
                "emotion": emotion,
                "story_index": idx,
                "sha1": sha1(text),
                "error": repr(exc),
            },
        )


def record_batch(
    settings: RunSettings, emotion: str, batch: list[tuple[int, str]], result: BatchResult
) -> None:
    for (idx, text), vec, count in zip(batch, result.means, result.token_counts):
        sid = story_key(emotion, idx)
        np.save(settings.shards_dir / f"{sid}.npy", vec)
        append_jsonl(
            settings.manifest_path,
            {
                "story_id": sid,
                "emotion": emotion,
                "story_index": idx,
                "sha1": sha1(text),
                "n_tokens": count,
                "error": None,
            },
        )


def log_progress(logger: logging.Logger, done: int, total: int, t0: float) -> None:
    rate = done / max(time.monotonic() - t0, 1e-9)
    eta = (total - done) / max(rate, 1e-9)
    logger.info(
        f"{done}/{total} stories | {rate:.2f} stories/s | "
        f"elapsed {human(time.monotonic() - t0)} | ETA {human(eta)}"
    )


def extract_loop(
    lm: LoadedModel,
    pending: dict[str, list[tuple[int, str]]],
    settings: RunSettings,
    logger: logging.Logger,
) -> None:
    total = sum(len(v) for v in pending.values())
    done, n_batches, t0 = 0, 0, time.monotonic()
    for emotion, items in pending.items():
        for start in range(0, len(items), settings.batch_size):
            batch = items[start : start + settings.batch_size]
            texts = [text for _, text in batch]
            try:
                result = run_batch(lm, texts, settings.layers, settings.max_length)
            except Exception as exc:
                logger.error(f"[{emotion}] batch at {start} failed: {exc!r}")
                record_failure(settings, emotion, batch, exc)
                continue
            record_batch(settings, emotion, batch, result)
            done += len(batch)
            n_batches += 1
            if n_batches % LOG_EVERY_BATCHES == 0 or done == total:
                log_progress(logger, done, total, t0)


def run_extraction(
    settings: RunSettings, emotions_data: dict[str, list[str]], logger: logging.Logger
) -> None:
    settings.shards_dir.mkdir(parents=True, exist_ok=True)
    config = settings.to_config()
    (settings.out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    logger.info(f"config: {json.dumps(config)}")
    manifest = load_manifest(settings.manifest_path)
    pending = pending_stories(emotions_data, manifest, settings.shards_dir)
    n_total = sum(len(v) for v in emotions_data.values())
    n_pending = sum(len(v) for v in pending.values())
    logger.info(f"{n_total} stories, {n_pending} pending ({n_total - n_pending} already on disk)")
    if n_pending:
        torch.manual_seed(settings.seed)
        lm, _ = load_model_bf16(settings.model, logger.info)
        get_layer(lm.model, settings.layers[-1])  # fail fast on layer lookup, before the loop
        extract_loop(lm, pending, settings, logger)
