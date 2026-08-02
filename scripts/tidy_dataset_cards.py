"""Clean the published dataset cards: pointers on the corpora, jargon out of all.

Two jobs, one pass over the Hub:

1. The nine story corpora now also live, unioned and browsable, in
   `abotresol/emotion-story-corpora`. Nothing was deleted, so each original card
   gains one line at the top saying where the consolidated copy is. A reader
   landing on any of them should not have to guess which repo to use.

2. Twenty-one of twenty-nine cards still carried words this project invented:
   bank, battery, lineage, arm, substrate, blessed, tranche, readout. The repo,
   the notebooks and the write-up were cleaned of these; the cards are the most
   public surface of the lot and were missed.

Replacements say the thing instead of naming it. Where a word means different
things in different places, the phrase around it decides: "battery" is twelve
EMOTIONS in the corpus cards and twelve SCENARIOS in the detection ones, so the
longest, most specific patterns are applied first and the bare word last.

    uv run python scripts/tidy_dataset_cards.py           # show the diff
    uv run python scripts/tidy_dataset_cards.py --apply   # push it
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

HF_USER = "abotresol"
CONSOLIDATED = f"{HF_USER}/emotion-story-corpora"

# The nine corpora now also available, unioned, in the consolidated dataset.
CORPUS_REPOS = [
    "emotion-stories-gemma-4-31b-it",
    "emotion-stories-deepseek-v4-pro",
    "emotion-stories-deepseek-v4-pro-diverse",
    "emotion-dialogues-gemma-4-31b",
    "emotion-dialogues-gemma-4-31b-it",
    "neutral-transcripts-gemma-4-31b-it",
    "emotion-combined-stories-gemma-4-31b-it",
    "emotion-combined-stories-deepseek-v4-pro",
    "emotion-combined-stories-deepseek-v4-pro-constant-control",
]

POINTER = (
    f"> **Also available, browsable:** this corpus is one `source` inside\n"
    f"> [`{CONSOLIDATED}`](https://huggingface.co/datasets/{CONSOLIDATED}),\n"
    f"> which unions every story corpus from this project into two Parquet subsets\n"
    f"> with a working dataset viewer. This repository stays as the original,\n"
    f"> unchanged, so existing links and citations keep resolving.\n"
)

# Longest and most context-bearing first; the bare words last.
REPLACEMENTS: list[tuple[str, str]] = [
    ("scenario-test battery", "twelve test scenarios"),
    ("the paper's battery", "the paper's twelve scenarios"),
    ("held-out battery", "held-out scenarios"),
    ("dual-battery", "both scenario sets"),
    ("battery emotions", "the twelve emotions"),
    ("battery emotion", "one of the twelve emotions"),
    ("battery stories", "stories for the twelve emotions"),
    ("battery contrasts", "twelve-emotion contrasts"),
    ("probe bank", "vector set"),
    ("probe lineage", "story source"),
    ("lineage gap", "story-source gap"),
    ("corpus-lineage", "external-corpus"),
    ("self-generated-lineage", "self-generated"),
    ("base-lineage", "base-model"),
    ("third-lineage", "third story source"),
    ("dialogue-lineage", "dialogue-corpus"),
    ("blessed instrument", "the set every published result uses"),
    ("blessed sets", "the sets every published result uses"),
    ("blessed bundles", "the bundles every published result uses"),
    ("blessed", "current"),
    ("strong-generator arm", "strong-writer condition"),
    ("dose-response arm", "dose-response condition"),
    ("control arm", "control condition"),
    ("trajectory arm", "trajectory condition"),
    ("substrate text", "story text"),
    ("trajectory substrate", "trajectory stories"),
    ("primary substrate", "primary story set"),
    ("substrate", "story set"),
    ("readout", "score"),
    ("tranche", "batch"),
    ("the two lineages", "the two story sources"),
    ("lineages", "story sources"),
    ("lineage", "story source"),
    ("battery", "twelve emotions"),
    # Bare forms last, so every phrase above gets its specific wording first and
    # only the leftovers fall through to these.
    ("readouts", "scores"),
    ("banks", "vector sets"),
    ("bank", "vector set"),
    ("arms", "conditions"),
    ("arm", "condition"),
]

JARGON = re.compile(
    r"\b(bank|banks|battery|batteries|lineage|lineages|arm|arms|substrate|substrates"
    r"|blessed|readout|readouts|tranche|tranches)\b",
    re.I,
)


def clean(card: str) -> str:
    """Apply the replacements to prose only, never to an identifier.

    Three things are masked out first, because in each of them a word is a value
    rather than a description, and rewriting it would break a real reference:
    the YAML frontmatter, fenced code blocks, and `inline code spans` — which is
    where the file names live (`LINEAGE.md`, `e5_cross_lineage_n256.json`).

    "lineage" also carries two distinct meanings across these cards. In a heading
    like "Data lineage" it means provenance; in "corpus-lineage" it means which
    corpus wrote the stories. Those are separated explicitly below rather than
    collapsed into one replacement.
    """
    protected = re.compile(r"(^---\n.*?\n---\n|```.*?```|`[^`\n]+`)", re.S | re.M)
    parts = protected.split(card)
    for index, part in enumerate(parts):
        if part.startswith(("---\n", "```", "`")):
            continue  # an identifier, not prose
        # provenance sense first, so it never falls through to "story source"
        for old, new in (
            ("Data lineage", "Data provenance"),
            ("Lineage caveats", "Provenance caveats"),
            ("LINEAGE:", "PROVENANCE:"),
        ):
            part = part.replace(old, new)
        for old, new in REPLACEMENTS:
            part = re.sub(rf"\b{re.escape(old)}\b", new, part, flags=re.I)
        parts[index] = part
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="push the rewritten cards")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api = HfApi(token=os.environ.get("HF_TOKEN") if args.apply else None)

    changed = 0
    for dataset in sorted(d.id for d in api.list_datasets(author=HF_USER)):
        name = dataset.split("/")[1]
        if name == CONSOLIDATED.split("/")[1] or name == "gemma3-refusal-axis-data":
            continue  # the new one is already clean; the other is unrelated prior work
        try:
            path = api.hf_hub_download(dataset, "README.md", repo_type="dataset")
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"  SKIP {name}: {type(exc).__name__}")
            continue
        original = Path(path).read_text(encoding="utf-8")

        card = clean(original)
        if name in CORPUS_REPOS and CONSOLIDATED not in card:
            # after the YAML frontmatter and the H1, before the body
            match = re.search(r"^(---\n.*?\n---\n)?(#[^\n]*\n)", card, flags=re.S)
            insert = match.end() if match else 0
            card = card[:insert] + "\n" + POINTER + card[insert:]

        if card == original:
            print(f"  ok    {name}")
            continue
        changed += 1
        before = len(JARGON.findall(original))
        after = len(JARGON.findall(card))
        pointer = " +pointer" if name in CORPUS_REPOS else ""
        print(f"  FIX   {name}: jargon {before} -> {after}{pointer}")
        if args.apply:
            api.upload_file(
                path_or_fileobj=card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=dataset,
                repo_type="dataset",
                commit_message="Card: plain wording, and point at the consolidated corpora dataset",
            )

    print(f"\n{changed} cards {'updated' if args.apply else 'would change'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
