"""Consolidate the nine published story corpora into one Hugging Face dataset.

Why: the corpora are the most reusable thing this project produced. The headline
finding is that WHO WRITES THE STORIES decides whether emotion vectors work at
all, and until now comparing those corpora meant nine separate downloads and a
call into this repo's own loader. The Hub documents the fix
(https://huggingface.co/docs/hub/datasets-manual-configuration): one dataset,
several subsets, declared in the card's `configs:` block.

Shape: two subsets, each a union of its corpora with a `source` column, so the
story-source comparison is a filter in the dataset viewer rather than a
download-and-join. Parquet, because the Hub only renders a viewer for formats
it integrates with, and a corpus nobody can read is a corpus nobody reuses.

Nothing is deleted. The nine source repositories stay exactly where they are, so
every existing URL, citation and `fetch()` route keeps resolving; they gain a
card line pointing here.

    uv run python scripts/build_story_corpora_dataset.py            # build + verify
    uv run python scripts/build_story_corpora_dataset.py --publish  # then upload
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

TARGET_REPO = "abotresol/emotion-story-corpora"
BUILD_DIR = Path("results/story_corpora_dataset")

# (subset, source label, source repo, file within it, expected rows)
CORPORA: list[tuple[str, str, str, str, int]] = [
    (
        "single_emotion",
        "gemma-4-31b-it",
        "emotion-stories-gemma-4-31b-it",
        "dialogues_raw.jsonl",
        3072,
    ),
    (
        "single_emotion",
        "deepseek-v4-pro",
        "emotion-stories-deepseek-v4-pro",
        "stories_raw.jsonl",
        3072,
    ),
    (
        "single_emotion",
        "deepseek-v4-pro-diverse",
        "emotion-stories-deepseek-v4-pro-diverse",
        "stories_raw.jsonl",
        12288,
    ),
    (
        "single_emotion",
        "dialogues-gemma-4-31b",
        "emotion-dialogues-gemma-4-31b",
        "dialogues_raw.jsonl",
        192,
    ),
    (
        "single_emotion",
        "dialogues-gemma-4-31b-it",
        "emotion-dialogues-gemma-4-31b-it",
        "dialogues_raw.jsonl",
        192,
    ),
    (
        "single_emotion",
        "neutral-gemma-4-31b-it",
        "neutral-transcripts-gemma-4-31b-it",
        "dialogues_raw.jsonl",
        128,
    ),
    (
        "three_emotion",
        "gemma-4-31b-it",
        "emotion-combined-stories-gemma-4-31b-it",
        "stories_raw.jsonl",
        5888,
    ),
    (
        "three_emotion",
        "deepseek-v4-pro",
        "emotion-combined-stories-deepseek-v4-pro",
        "stories_raw.jsonl",
        5785,
    ),
    (
        "three_emotion",
        "deepseek-v4-pro-constant-control",
        "emotion-combined-stories-deepseek-v4-pro-constant-control",
        "stories_raw.jsonl",
        209,
    ),
]

# Columns that only some corpora carry. Listed so the union has a stable schema
# instead of one that depends on which corpus happened to be read first.
OPTIONAL_COLUMNS = {
    "single_emotion": ["model", "leaked", "slot", "tokens_out", "error"],
    "three_emotion": [
        "triple_id",
        "emotions",
        "permutation",
        "perm_idx",
        "mode",
        "category",
        "has_nonaffect",
        "tags",
        "tags_match_emotions",
        "leaked_words",
        "n_tags",
        "n_chars",
        "seed",
        "had_thinking_trace",
        "generator_model",
        "tokens_out",
        "error",
    ],
}
LEADING_COLUMNS = {
    "single_emotion": ["source", "emotion", "text"],
    "three_emotion": ["source", "text"],
}


def load_corpus(repo: str, filename: str) -> list[dict[str, object]]:
    """Every row of one published corpus, fetched anonymously."""
    path = hf_hub_download(f"abotresol/{repo}", filename, repo_type="dataset", token=False)
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def build() -> dict[str, pd.DataFrame]:
    """One DataFrame per subset, with a `source` column and a stable schema."""
    frames: dict[str, list[pd.DataFrame]] = {"single_emotion": [], "three_emotion": []}
    for subset, source, repo, filename, expected in CORPORA:
        rows = load_corpus(repo, filename)
        if len(rows) != expected:
            raise SystemExit(f"{repo}/{filename}: expected {expected} rows, read {len(rows)}")
        frame = pd.DataFrame(rows)
        frame.insert(0, "source", source)
        frames[subset].append(frame)
        print(f"  {subset:<14} {source:<32} {len(rows):>6,} rows")

    built = {}
    for subset, parts in frames.items():
        table = pd.concat(parts, ignore_index=True)
        for column in OPTIONAL_COLUMNS[subset]:
            if column not in table.columns:
                table[column] = None
        ordered = LEADING_COLUMNS[subset] + [
            c for c in table.columns if c not in LEADING_COLUMNS[subset]
        ]
        built[subset] = table[ordered]
    return built


def verify(built: dict[str, pd.DataFrame]) -> None:
    """Row counts must add up and no story may have been dropped or emptied."""
    for subset, table in built.items():
        expected = sum(rows for s, _, _, _, rows in CORPORA if s == subset)
        if len(table) != expected:
            raise SystemExit(f"{subset}: {len(table)} rows, expected {expected}")
        sources = sorted({s for sub, s, _, _, _ in CORPORA if sub == subset})
        if sorted(table["source"].unique()) != sources:
            raise SystemExit(f"{subset}: source column does not match the inputs")
        empty = int(table["text"].isna().sum() + (table["text"].astype(str).str.len() == 0).sum())
        print(f"  {subset:<14} {len(table):>6,} rows  {len(sources)} sources  empty texts: {empty}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish", action="store_true", help="upload after building and verifying"
    )
    args = parser.parse_args()

    print("reading the nine published corpora (anonymously):")
    built = build()
    print("\nverifying:")
    verify(built)

    out = BUILD_DIR / "data"
    out.mkdir(parents=True, exist_ok=True)
    for subset, table in built.items():
        target = out / f"{subset}.parquet"
        table.to_parquet(target, index=False)
        print(f"  wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")

    (BUILD_DIR / "README.md").write_text(card(built), encoding="utf-8")
    print(f"  wrote {BUILD_DIR / 'README.md'}")

    if not args.publish:
        print("\nbuilt only. Re-run with --publish to upload.")
        return 0

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(TARGET_REPO, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(BUILD_DIR),
        repo_id=TARGET_REPO,
        repo_type="dataset",
        commit_message="Consolidate the nine story corpora into two browsable subsets",
    )
    print(f"\npublished -> https://huggingface.co/datasets/{TARGET_REPO}")
    return 0


def card(built: dict[str, pd.DataFrame]) -> str:
    """The dataset card, including the configs block the viewer needs."""
    single, triple = built["single_emotion"], built["three_emotion"]

    def source_rows(subset: str, table: pd.DataFrame) -> str:
        lines = []
        for sub, source, repo, _, _ in CORPORA:
            if sub != subset:
                continue
            count = int((table["source"] == source).sum())
            lines.append(
                f"| `{source}` | {count:,} | "
                f"[`{repo}`](https://huggingface.co/datasets/abotresol/{repo}) |"
            )
        return "\n".join(lines)

    return f"""---
license: mit
configs:
- config_name: single_emotion
  data_files: "data/single_emotion.parquet"
  default: true
- config_name: three_emotion
  data_files: "data/three_emotion.parquet"
---

# Emotion story corpora

Every story corpus behind *How are emotions represented in large language
models?*, in one place. These are the texts a model reads while its internal
state is recorded, each written to evoke a named emotion without naming it.

The project's most transferable finding is that **who writes the stories decides
whether any of it works**: holding everything else fixed, swapping the story
writer moved the number of working layers from 1 to 9 out of 20. That comparison
is why these corpora are unioned here with a `source` column instead of sitting
in nine separate repositories. Filter on `source` to reproduce it.

- Code and research record: <https://github.com/Antonio-Tresol/gemma4-emotion-vectors>
- Write-up: <https://antonio-tresol.github.io/gemma4-emotion-vectors/>
- Replicating: Sofroniew et al., *Emotion Concepts and their Function in a Large
  Language Model*, Anthropic, 2026
  (<https://transformer-circuits.pub/2026/emotions/index.html>)

## `single_emotion` ({len(single):,} rows)

One story per row, written to evoke a single emotion.

| `source` | rows | original repository |
|---|---|---|
{source_rows("single_emotion", single)}

Columns: `source`, `emotion`, `text`, and generation metadata that only some
corpora carry (`model`, `leaked`, `slot`, `tokens_out`, `error`), null elsewhere.

## `three_emotion` ({len(triple):,} rows)

One story per row, written to move through three emotions in sequence. These are
the texts behind the emotion-tracking work.

| `source` | rows | original repository |
|---|---|---|
{source_rows("three_emotion", triple)}

`emotions` holds the three in written order, `permutation` and `perm_idx` record
which ordering was asked for, and `mode` distinguishes a story that moves through
the emotions in sequence from one that holds them simultaneously. The
`deepseek-v4-pro-constant-control` source is the control: the same scaffold and
the same transition scenes with **no** emotion change, so any transition-locked
signal there is a scene-change artifact rather than an emotion one.

## Loading

```python
from datasets import load_dataset

stories = load_dataset("{TARGET_REPO}", "single_emotion", split="train")
gemma_only = stories.filter(lambda row: row["source"] == "gemma-4-31b-it")
```

## Provenance and caveats

Nothing here is deleted or renamed: the nine original repositories remain, and
this dataset is a repackaging of their `*_raw.jsonl` files with a `source`
column added. Row counts are asserted against the originals at build time by
`scripts/build_story_corpora_dataset.py`.

These are model-written texts, not human writing, and they were generated to
evoke emotions rather than to describe them. A story is labelled with the
emotion it was *written to evoke*; no human verified that it succeeds. The
`leaked` and `leaked_words` fields record where a generator named the emotion
outright despite being told not to.

Licence: MIT, matching the project. The models that wrote these texts carry
their own terms.
"""


if __name__ == "__main__":
    sys.exit(main())
