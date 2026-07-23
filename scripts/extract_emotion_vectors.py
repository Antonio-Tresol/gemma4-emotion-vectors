"""Extract emotion vectors from the residual stream — Q1.H1.E1 entry point.

Thin CLI over emotion_vectors.extraction (reference-faithful extraction; see its
docstring for the math and the deliberate deviations from the reference),
emotion_vectors.story_store (resumable per-story persistence and token-weighted
aggregation), and emotion_vectors.hf_publish (private dataset-repo upload).

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
from pathlib import Path

from emotion_vectors.corpus import (
    REFERENCE_BATCH_SIZE,
    REFERENCE_MAX_LENGTH,
    detect_model_geometry,
    load_emotions_data,
)
from emotion_vectors.hf_publish import check_hf_auth, publish
from emotion_vectors.story_store import aggregate

try:
    from emotion_vectors.extraction import RunSettings, git_commit, run_extraction, setup_logger
except ModuleNotFoundError as exc:  # torch lives in the gpu extra; this is a GPU entry point
    raise SystemExit(f"missing {exc.name}: this entry point needs `uv sync --extra gpu`") from exc

SMOKE_EMOTIONS = 3
SMOKE_STORIES_PER_EMOTION = 4


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
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="wrap each story as a user turn via the model's chat template before "
        "extraction (raw-vs-chat audit arm; raw text is the reference's convention)",
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
    if args.chat_template:
        emotions_data = {
            emotion: wrap_chat(args.model, stories) for emotion, stories in emotions_data.items()
        }
    return emotions_data


def wrap_chat(model_name: str, stories: list[str]) -> list[str]:
    """Each story as a user turn under the model's chat template. The rendered
    string starts with the BOS text, which the pipeline's tokenizer call would
    add again (add_special_tokens default) — strip it to avoid a double BOS."""
    from transformers import AutoTokenizer  # noqa: PLC0415  (only needed for this audit arm)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    wrapped = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": story}], tokenize=False, add_generation_prompt=False
        )
        for story in stories
    ]
    bos = tokenizer.bos_token or ""
    return [w.removeprefix(bos) for w in wrapped]


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
        chat_template=args.chat_template,
    )


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
