"""Q3.H1.E1 collection: per-token probe trajectories over combined stories.

For every kept story in the combined-story corpus, run a teacher-forced
forward pass through gemma-4-31b-it, capture the residual stream at a band of
layers (early / early-middle / mid-late — the layer-contrast read), and store
per-story shards of raw per-token dot products onto unit probe directions
plus per-token activation norms. Probes: the 171 corpus-lineage contrasts,
the 12 self-generated n=256 contrasts (the E10 winner), and 24 fixed-seed
random unit directions (the null control). Cosines and centering conventions
are analysis-side transformations of these shards (open-science substrate).

    .venv/bin/python scripts/combined_story_gen/extract_trajectories.py \
        --stories results/combined_stories/stories_raw.jsonl --smoke

Resumable: shards already in the manifest are skipped on relaunch.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from emotion_vectors.extraction import (
    LoadedModel,
    get_layer,
    git_commit,
    load_model_bf16,
    make_hook,
    setup_logger,
)
from emotion_vectors.story_store import append_jsonl, sha1
from emotion_vectors.trajectories import (
    kept_rows,
    parse_story,
    phase_token_starts,
    random_unit_directions,
    story_id,
    token_probe_dots,
    unit_contrast_probes,
)

MODEL = "google/gemma-4-31b-it"
DEFAULT_LAYERS = [6, 15, 24, 33, 42, 51]  # early -> mid-late bands
N_RANDOM = 24
RANDOM_SEED = 20260722
CORPUS_PROBES = Path("results/emotion_vectors_it_means.npz")
SELFGEN_PROBES = Path("results/e6_scale_means.npz")  # bucket -1 = n=256


@dataclass(frozen=True)
class ProbeBank:
    """Unit probe directions at the run's layers, with their labels."""

    directions: np.ndarray  # [probes, layers, d_model] float32
    labels: list[str]  # "corpus:afraid", "selfgen:happy", "random:07", ...


def load_probe_bank(layers: list[int]) -> ProbeBank:
    corpus = np.load(CORPUS_PROBES)
    layer_idx = [list(corpus["layers"]).index(layer) for layer in layers]
    corpus_dirs = unit_contrast_probes(corpus["means"].astype(np.float32))[:, layer_idx]
    selfgen = np.load(SELFGEN_PROBES, allow_pickle=True)
    selfgen_dirs = unit_contrast_probes(selfgen["means"][-1].astype(np.float32))[:, layer_idx]
    d_model = corpus_dirs.shape[-1]
    random_dirs = random_unit_directions(N_RANDOM, len(layers), d_model, RANDOM_SEED)
    labels = (
        [f"corpus:{e}" for e in corpus["emotions"]]
        + [f"selfgen:{e}" for e in selfgen["emotions"]]
        + [f"random:{i:02d}" for i in range(N_RANDOM)]
    )
    return ProbeBank(np.concatenate([corpus_dirs, selfgen_dirs, random_dirs]), labels)


def forward_tokens(
    lm: LoadedModel, texts: list[str], layers: list[int], max_length: int
) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[int], np.ndarray]:
    """(acts [batch, seq, layers, d_model], per-row offsets, valid lengths, ids)."""
    encoded = lm.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = [[tuple(pair) for pair in row] for row in encoded.pop("offset_mapping").tolist()]
    inputs = encoded.to(lm.model.device)
    captured: dict[int, torch.Tensor] = {}
    hooks = [get_layer(lm.model, i).register_forward_hook(make_hook(captured, i)) for i in layers]
    try:
        with torch.inference_mode():
            lm.model(**inputs)
    finally:
        for hook in hooks:
            hook.remove()
    acts = torch.stack([captured[i] for i in layers], dim=2).cpu().numpy()
    lengths = [int(n) for n in inputs["attention_mask"].sum(dim=1).tolist()]
    return acts, offsets, lengths, inputs["input_ids"].cpu().numpy()


def process_batch(batch: list[dict], lm: LoadedModel, bank: ProbeBank, args) -> int:
    """Extract, project, and shard one batch; returns tokens processed."""
    parsed = [parse_story(row["text"], row["mode"]) for row in batch]
    acts, offsets, lengths, ids = forward_tokens(
        lm, [p.clean_text for p in parsed], args.layers, args.max_length
    )
    for i, (row, story) in enumerate(zip(batch, parsed)):
        n_tokens = lengths[i]
        dots, norms = token_probe_dots(acts[i, :n_tokens], bank.directions)
        starts = phase_token_starts(story.phase_char_starts, offsets[i][:n_tokens])
        sid = story_id(row)
        np.savez_compressed(
            args.out_dir / "shards" / f"{sid}.npz",
            dots=dots.astype(np.float16),
            norms=norms.astype(np.float16),
            token_ids=ids[i, :n_tokens],
            phase_token_starts=np.array(starts),
        )
        append_jsonl(
            args.out_dir / "manifest.jsonl",
            {
                "story_id": sid,
                "triple_id": row["triple_id"],
                "mode": row["mode"],
                "emotions": row["emotions"],
                "phase_emotions": story.phase_emotions,
                "phase_token_starts": starts,
                "category": row["category"],
                "n_tokens": n_tokens,
                "text_sha1": sha1(row["text"]),
                "error": None,
            },
        )
    return sum(lengths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/combined_trajectories"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true", help="first 8 stories only")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    (args.out_dir / "shards").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.out_dir / "extract.log")
    rows = kept_rows(args.stories)
    if args.smoke:
        rows = rows[:8]
    manifest = args.out_dir / "manifest.jsonl"
    done = (
        {json.loads(line)["story_id"] for line in manifest.read_text().splitlines() if line}
        if manifest.exists()
        else set()
    )
    pending = [row for row in rows if story_id(row) not in done]
    config = {
        "model": args.model,
        "stories": str(args.stories),
        "layers": args.layers,
        "n_random": N_RANDOM,
        "random_seed": RANDOM_SEED,
        "probe_files": [str(CORPUS_PROBES), str(SELFGEN_PROBES)],
        "max_length": args.max_length,
        "smoke": args.smoke,
        "git_commit": git_commit(),
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    logger.info("config: %s", json.dumps(config))
    logger.info("%d kept stories, %d pending (%d resumed)", len(rows), len(pending), len(done))
    bank = load_probe_bank(args.layers)
    (args.out_dir / "probe_labels.json").write_text(json.dumps(bank.labels))
    lm, _ = load_model_bf16(args.model, logger.info)
    t0 = time.monotonic()
    tokens = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        tokens += process_batch(batch, lm, bank, args)
        done_n = start + len(batch)
        rate = tokens / max(time.monotonic() - t0, 1e-6)
        logger.info("%d/%d stories | %.0f tok/s", done_n, len(pending), rate)
    logger.info("DONE: %d stories, %d tokens", len(pending), tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
