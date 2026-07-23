"""Token-level blast radius of the left-padding TOKEN_OFFSET bug — Q1.H3.E4b, laptop half.

The pre-fix extractions (results/emotion_vectors and every other set with a
git_commit older than a2a88dd) tokenized with Gemma-4's default LEFT padding
while pool_batch zeroed the first TOKEN_OFFSET *columns*. This script needs no
GPU because the deviation is pure masking: re-tokenize the corpus, rebuild the
extraction batches (per-emotion, story_index order, reference batch size),
and predict each story's post-mask token count under both semantics:

  left-pad (buggy):    included_i = L_i - max(0, TOKEN_OFFSET - (S - L_i))
  right-pad (intended): included_i = max(0, L_i - TOKEN_OFFSET)

where L_i is the story's tokenized length (truncated at max_length) and S the
batch max. The manifest's recorded n_tokens then adjudicates which semantics
actually ran, story by story, and the leaked-framing share per emotion is

  leak_share = sum_i (included_actual_i - included_intended_i) / sum_i included_actual_i

— the fraction of each pooled mean's token mass that the registered method
said to skip. The activation-space cosine (GPU half) still needs a post-fix
re-extraction; this bounds and mechanically confirms what it will measure.

Usage:
    uv run python scripts/audit_extraction_offset.py \
        [--results-dir results/emotion_vectors] [--out results/extraction_offset_token_audit.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from transformers import AutoTokenizer

from emotion_vectors.corpus import TOKEN_OFFSET, load_emotions_data
from emotion_vectors.story_store import story_key


def tokenized_lengths(tokenizer, stories: list[str], max_length: int) -> list[int]:
    """Per-story token counts as the extraction saw them (specials included,
    truncation applied), without building padded tensors."""
    enc = tokenizer(stories, truncation=True, max_length=max_length)
    return [len(ids) for ids in enc["input_ids"]]


def predict_batch(lengths: list[int]) -> tuple[list[int], list[int]]:
    """(left-pad buggy counts, right-pad intended counts) for one batch."""
    s = max(lengths)
    buggy = [length - max(0, TOKEN_OFFSET - (s - length)) for length in lengths]
    intended = [max(0, length - TOKEN_OFFSET) for length in lengths]
    return buggy, intended


@dataclass
class Tally:
    """Adjudication counters. Rows where the two predictions coincide (the
    batch-max story, or ties) cannot discriminate; only rows where they
    differ count toward a match category."""

    ambiguous: int = 0
    match_buggy: int = 0
    match_intended: int = 0
    match_neither: int = 0
    mismatches: list[dict[str, object]] = field(default_factory=list)

    def record(self, sid: str, rec: int, buggy: int, intended: int) -> None:
        if buggy == intended:
            self.ambiguous += 1
        elif rec == buggy:
            self.match_buggy += 1
        elif rec == intended:
            self.match_intended += 1
        else:
            self.match_neither += 1
        if rec not in (buggy, intended) and len(self.mismatches) < 20:
            self.mismatches.append(
                {"story_id": sid, "recorded": rec, "buggy": buggy, "intended": intended}
            )

    @property
    def n_scored(self) -> int:
        return self.ambiguous + self.match_buggy + self.match_intended + self.match_neither


def load_recorded_counts(results_dir: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in open(results_dir / "manifest.jsonl") if line.strip()]
    return {r["story_id"]: r["n_tokens"] for r in rows if r.get("error") is None}


def load_corpus(config: dict[str, object]) -> dict[str, list[str]]:
    dataset = str(config["dataset"])
    if Path(dataset).exists():
        rows = (json.loads(line) for line in open(dataset) if line.strip())
        return {row["emotion"]: row["stories"] for row in rows}
    return load_emotions_data(dataset, str(config["split"]))


def audit_emotion(
    emotion: str,
    lengths: list[int],
    batch_size: int,
    recorded: dict[str, int] | None,
    tally: Tally,
) -> dict[str, float]:
    """One emotion's leak accounting; mutates the shared tally."""
    leaked = actual_total = intended_total = 0
    for start in range(0, len(lengths), batch_size):
        buggy, intended = predict_batch(lengths[start : start + batch_size])
        for offset, (b, i) in enumerate(zip(buggy, intended)):
            sid = story_key(emotion, start + offset)
            rec = b if recorded is None else recorded.get(sid)
            if rec is None:
                continue
            tally.record(sid, rec, b, i)
            leaked += rec - i  # tokens included that the method said to skip
            actual_total += rec
            intended_total += i
    return {
        "n_stories": len(lengths),
        "tokens_pooled": actual_total,
        "tokens_intended": intended_total,
        "leaked_framing_tokens": leaked,
        "leak_share": round(leaked / actual_total, 4) if actual_total else 0.0,
    }


def audit(results_dir: Path, out_file: Path, predict_only: bool = False) -> dict[str, object]:
    """predict_only: no manifest to adjudicate against (extraction dir not
    synced back, e.g. the E6 n=256 set) — compute the predicted leak share
    from the corpus alone; the recorded counts are taken to be the left-pad
    form, which every adjudicated set matched exactly."""
    config = json.loads((results_dir / "run_config.json").read_text())
    recorded = None if predict_only else load_recorded_counts(results_dir)
    emotions_data = load_corpus(config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)

    tally = Tally()
    per_emotion = {
        emotion: audit_emotion(
            emotion,
            tokenized_lengths(tokenizer, stories, config["max_length"]),
            config["batch_size"],
            recorded,
            tally,
        )
        for emotion, stories in emotions_data.items()
    }

    shares = [e["leak_share"] for e in per_emotion.values()]
    ambiguous, match_buggy = tally.ambiguous, tally.match_buggy
    match_intended, match_neither = tally.match_intended, tally.match_neither
    mismatches, n_scored = tally.mismatches, tally.n_scored
    result: dict[str, object] = {
        "results_dir": str(results_dir),
        "model": config["model"],
        "dataset": config["dataset"],
        "extraction_git_commit": config["git_commit"],
        "token_offset": TOKEN_OFFSET,
        "batch_size": config["batch_size"],
        "predict_only": predict_only,
        "n_stories_scored": n_scored,
        "n_predictions_coincide_ambiguous": None if predict_only else ambiguous,
        "n_tokens_match_leftpad_buggy": None if predict_only else match_buggy,
        "n_tokens_match_rightpad_intended": None if predict_only else match_intended,
        "n_tokens_match_neither": None if predict_only else match_neither,
        "mismatch_examples": mismatches,
        "leak_share_mean": round(sum(shares) / len(shares), 4) if shares else 0.0,
        "leak_share_min": min(shares) if shares else 0.0,
        "leak_share_max": max(shares) if shares else 0.0,
        "per_emotion": per_emotion,
        "note": (
            "leak_share = fraction of each emotion's pooled token mass that the "
            "registered method (skip first TOKEN_OFFSET story tokens) says to "
            "exclude but the left-padding bug included. Activation-space impact "
            "needs the GPU re-extraction (E4b, pod)."
        ),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/emotion_vectors"))
    parser.add_argument(
        "--out", type=Path, default=Path("results/extraction_offset_token_audit.json")
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="no manifest available: predicted leak share only, no adjudication "
        "(run_config still read from --results-dir; the corpus file it names is "
        "loaded as it exists NOW)",
    )
    args = parser.parse_args()
    result = audit(args.results_dir, args.out, predict_only=args.predict_only)
    adjudication = (
        "predict-only (no manifest adjudication)"
        if args.predict_only
        else (
            f"{result['n_tokens_match_leftpad_buggy']} match left-pad(buggy), "
            f"{result['n_tokens_match_rightpad_intended']} match right-pad(intended), "
            f"{result['n_tokens_match_neither']} neither"
        )
    )
    print(
        f"{result['n_stories_scored']} stories: {adjudication} | "
        f"leak share mean {result['leak_share_mean']} "
        f"[{result['leak_share_min']}, {result['leak_share_max']}] -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
