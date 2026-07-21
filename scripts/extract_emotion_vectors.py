"""Extract emotion vectors from the residual stream — Q1.H1.E1 pipeline.

Reference-faithful math (sinievanderben/emotion_experiment,
extract_emotion_vectors.py): same corpus loading, same tokenizer call (padding
to the batch max, truncation at 512), same forward-hook capture of each target
layer's output, same pooling mask (padding zeroed, first TOKEN_OFFSET=50 tokens
zeroed), same final token-weighted per-emotion mean. Deliberate deviations:

  - bf16 load instead of fp32 (fp32 needs ~124 GB for the 31B model and does
    not fit the 96 GB card; hooks cast activations to fp32, as the reference's
    hooks already do).
  - torch.inference_mode() instead of no_grad.
  - per-story persistence: the reference keeps one running sum per emotion in
    memory and saves only at the end, so a crash loses the whole run. We save
    each story's pooled mean plus its post-mask token count, and recombine as
    sum(mean_i * n_i) / sum(n_i) — the identical token-weighted mean — so the
    run is resumable and per-story vectors support within-emotion variance
    analyses the reference cannot do.

Companions: extraction_common.py (reference-shared paths, also used by the
time estimator) and hf_publish.py (dataset-repo publishing).

Usage:
    uv run python scripts/extract_emotion_vectors.py --smoke   # 3 emotions x 4
    uv run python scripts/extract_emotion_vectors.py           # full corpus
    uv run python scripts/extract_emotion_vectors.py --publish # + HF upload
    uv run python scripts/extract_emotion_vectors.py --publish-only

Resumable: a story is skipped iff its shard file and manifest row both exist
with a matching text hash. Kill it mid-run and re-run; it continues.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from extraction_common import (
    REFERENCE_BATCH_SIZE,
    REFERENCE_MAX_LENGTH,
    TOKEN_OFFSET,
    LoadedModel,
    detect_model_geometry,
    get_layer,
    human,
    load_emotions_data,
    load_model_bf16,
)
from hf_publish import check_hf_auth, publish
from story_store import aggregate, append_jsonl, load_manifest, pending_stories, sha1, story_key

if TYPE_CHECKING:
    from jaxtyping import Float, Int
    from torch import Tensor

SMOKE_EMOTIONS = 3
SMOKE_STORIES_PER_EMOTION = 4
LOG_EVERY_BATCHES = 10


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

    means: "Float[np.ndarray, 'b l d']"
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


def make_hook(
    captured: dict[int, "Tensor"], idx: int
) -> "Callable[[object, object, object], None]":
    def _hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        captured[idx] = hidden.detach().float()  # reference casts to fp32 here

    return _hook


def pool_batch(
    captured: dict[int, "Float[Tensor, 'b s d']"],
    attention_mask: "Int[Tensor, 'b s']",
    layers: list[int],
) -> tuple["Float[np.ndarray, 'b l d']", list[int]]:
    """Reference pooling, kept per-story: mask padding and the first
    TOKEN_OFFSET tokens, then mean over the surviving token positions."""
    import torch  # noqa: PLC0415
    from einops import rearrange, reduce  # noqa: PLC0415

    mask = attention_mask.clone()
    mask[:, :TOKEN_OFFSET] = 0  # reference: early tokens are narrative framing
    mask_f = rearrange(mask, "b s -> b s 1").float()
    counts = reduce(mask.float(), "b s -> b", "sum")
    sums = torch.stack([reduce(captured[i] * mask_f, "b s d -> b d", "sum") for i in layers])
    denom = rearrange(counts.clamp(min=1.0), "b -> 1 b 1")
    means = rearrange(sums / denom, "l b d -> b l d").cpu().numpy().astype(np.float32)
    return means, [int(c) for c in counts.tolist()]


def run_batch(lm: LoadedModel, texts: list[str], layers: list[int], max_length: int) -> BatchResult:
    """Tokenize -> forward with hooks -> pool. The reference loop, per batch."""
    import torch  # noqa: PLC0415

    inputs = lm.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(lm.model.device)
    captured: dict[int, torch.Tensor] = {}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-31b")
    parser.add_argument("--dataset", default="snae/emotion_stories_gemma_4_4B")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default results/emotion_vectors (or _smoke)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="default: every --layer-stride'th layer from the config",
    )
    parser.add_argument("--layer-stride", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=REFERENCE_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=REFERENCE_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"{SMOKE_EMOTIONS} emotions x {SMOKE_STORIES_PER_EMOTION} stories",
    )
    parser.add_argument("--publish", action="store_true", help="upload to HF after extraction")
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--hf-repo", default=None, help="dataset repo id; default <you>/emotion-…")
    return parser.parse_args()


def default_out_dir(args: argparse.Namespace) -> Path:
    return args.output_dir or Path(
        "results/emotion_vectors_smoke" if args.smoke else "results/emotion_vectors"
    )


def load_corpus(args: argparse.Namespace) -> dict[str, list[str]]:
    emotions_data = load_emotions_data(args.dataset, args.split)
    if args.smoke:
        emotions_data = {
            emotion: emotions_data[emotion][:SMOKE_STORIES_PER_EMOTION]
            for emotion in list(emotions_data)[:SMOKE_EMOTIONS]
        }
    return emotions_data


def build_settings(args: argparse.Namespace, out_dir: Path) -> RunSettings:
    d_model, n_layers = detect_model_geometry(args.model)
    layers = args.layers or list(range(0, n_layers, args.layer_stride))
    bad = [i for i in layers if i >= n_layers]
    if bad:
        raise SystemExit(f"layers {bad} out of range: model has {n_layers} layers")
    return RunSettings(
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        out_dir=out_dir,
        layers=layers,
        d_model=d_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        smoke=args.smoke,
        git_commit=git_commit(),
    )


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
        import torch  # noqa: PLC0415

        torch.manual_seed(settings.seed)
        lm, _ = load_model_bf16(settings.model, logger.info)
        get_layer(lm.model, settings.layers[-1])  # fail fast on layer lookup, before the loop
        extract_loop(lm, pending, settings, logger)


def main() -> int:
    args = parse_args()
    out_dir = default_out_dir(args)
    logger = setup_logger(out_dir / "extract.log")
    if args.publish or args.publish_only:  # fail fast: auth check before the model load
        check_hf_auth(logger)
    if not args.publish_only:
        run_extraction(build_settings(args, out_dir), load_corpus(args), logger)
        aggregate(out_dir, logger)
    if args.publish or args.publish_only:
        publish(out_dir, args.hf_repo, args.smoke, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
