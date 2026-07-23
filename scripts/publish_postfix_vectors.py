"""Publish the E4b post-fix vector re-extractions to Hugging Face with lineage cards.

Each *_postfix results dir goes to its own private dataset repo whose name is
the pre-fix predecessor's plus a `-postfix` suffix (mirroring the local dir
naming, so artifacts.fetch routes 1:1). A LINEAGE.md is written into the dir
before upload stating exactly how this set differs from its predecessor —
naming discipline so pre-fix and post-fix activations can never be confused.

Run on the machine that holds the shards (the pod, normally):
    uv run python scripts/publish_postfix_vectors.py [--only <dir-suffix>]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from emotion_vectors.hf_publish import check_hf_auth, publish

FIX_COMMIT = "a2a88dd"
IMPACT_FILE = "results/e4b_extraction_impact.json"

# (results dir, HF repo, pre-fix predecessor repo, token-audit file with the
# predecessor's measured/predicted leaked-framing share)
SETS: list[tuple[str, str, str, str]] = [
    (
        "results/emotion_vectors_postfix",
        "abotresol/emotion-vectors-gemma-4-31b-postfix",
        "abotresol/emotion-vectors-gemma-4-31b",
        "results/extraction_offset_token_audit.json",
    ),
    (
        "results/emotion_vectors_it_postfix",
        "abotresol/emotion-vectors-gemma-4-31b-it-postfix",
        "abotresol/emotion-vectors-gemma-4-31b-it",
        "results/extraction_offset_token_audit.json",  # same corpus; -it differs by 1 BOS token
    ),
    (
        "results/self_story_vectors_it_postfix",
        "abotresol/emotion-selfstory-vectors-gemma-4-31b-it-postfix",
        "abotresol/emotion-selfstory-vectors-gemma-4-31b-it",
        "results/extraction_offset_token_audit_selfgen_n256.json",
    ),
    (
        "results/neutral_vectors_it_postfix",
        "abotresol/neutral-vectors-gemma-4-31b-it-postfix",
        "abotresol/neutral-vectors-gemma-4-31b-it",
        "",  # neutral set has no per-emotion audit; covered by the impact JSON
    ),
]


def leak_line(audit_file: str) -> str:
    path = Path(audit_file)
    if not audit_file or not path.exists():
        return "leaked-framing share: see the impact JSON referenced below."
    audit = json.loads(path.read_text())
    kind = "predicted" if audit.get("predict_only") else "measured"
    return (
        f"{kind} leaked-framing share of pooled token mass: mean "
        f"{audit['leak_share_mean']:.1%} (range {audit['leak_share_min']:.1%}"
        f"-{audit['leak_share_max']:.1%}); adjudication and per-emotion detail in "
        f"`{audit_file}` in the project repo."
    )


def write_lineage(out_dir: Path, repo: str, predecessor: str, audit_file: str) -> None:
    config = json.loads((out_dir / "run_config.json").read_text())
    (out_dir / "LINEAGE.md").write_text(
        f"""# Lineage — {repo}

**Supersedes** `{predecessor}` for all probe and analysis work from 2026-07-22 on.

The predecessor was extracted while the Gemma-4 tokenizer silently padded LEFT,
so the pipeline's TOKEN_OFFSET=50 narrative-framing skip zeroed mostly-pad
columns instead of each story's first 50 tokens (project experiment
Q1.H3.E4). For the predecessor, {leak_line(audit_file)}

This set re-extracts the identical corpus with forced right padding
(fix commit `{FIX_COMMIT}`; this extraction ran at commit
`{config["git_commit"]}`), so the registered pooling semantics — skip each
story's first {config["token_offset"]} tokens, mean over the rest — hold
exactly. Everything else (corpus, model `{config["model"]}`, layers, batch
size {config["batch_size"]}, max length {config["max_length"]}, seed
{config["seed"]}, bf16 weights / fp32 activations, token-weighted per-emotion
means) is unchanged from the predecessor.

Quantified before/after impact (per-emotion cosines, contrast-direction
cosines, geometry re-read): `{IMPACT_FILE}` in the project repo.

The predecessor repo stays available unmodified: it is the substrate behind
all pre-2026-07-22 results and the "before" side of the E4b comparison.
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="publish just dirs containing this substring")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("publish_postfix")
    check_hf_auth(logger)
    for dir_name, repo, predecessor, audit_file in SETS:
        if args.only and args.only not in dir_name:
            continue
        out_dir = Path(dir_name)
        if not (out_dir / "run_config.json").exists():
            logger.warning(f"skipping {dir_name}: no run_config.json (extraction incomplete?)")
            continue
        write_lineage(out_dir, repo, predecessor, audit_file)
        publish(out_dir, repo, smoke=False, logger=logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
