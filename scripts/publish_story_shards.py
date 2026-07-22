"""Publish per-story activation shards — the substrate under every vector.

Open-science policy (2026-07-22): the published record must contain the
measurement substrate, not just our transformations of it. Per-emotion means
are one transformation deep (token-weighted recombination); the substrate is
one `[layers, d_model]` fp32 mean per story plus the manifest row carrying
its token count, from which every downstream vector is reproducible in
numpy alone. This uploads `shards/` + `manifest.jsonl` + `run_config.json`
into each extraction dataset repo.

Provenance caveat carried on each card: shards predate the padding-side fix
(TREE Q1.H3.E4) — the TOKEN_OFFSET skip applied fully only to the longest
story per batch of 4; the impact check quantifies the deviation. They are
the substrate of what was actually measured.

    .venv/bin/python scripts/publish_story_shards.py  # on the pod (HF_TOKEN)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

try:  # laptop convenience; the pod launcher sources .env instead
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv() -> None:
        return None


DIRS: list[tuple[str, str]] = [
    ("emotion_vectors", "emotion-vectors-gemma-4-31b"),
    ("emotion_vectors_it", "emotion-vectors-gemma-4-31b-it"),
    ("self_story_vectors_it", "emotion-selfstory-vectors-gemma-4-31b-it"),
    ("dialogue_vectors", "emotion-dialogue-vectors-gemma-4-31b"),
    ("dialogue_vectors_it", "emotion-dialogue-vectors-gemma-4-31b-it"),
    ("neutral_vectors_it", "neutral-vectors-gemma-4-31b-it"),
]

CARD_APPENDIX = """

## Per-story substrate (added 2026-07-22)

`shards/{story_id}.npy` — one `[n_layers, d_model]` fp32 mean per story;
`manifest.jsonl` — one row per story with its post-mask token count and
content hash. The published per-emotion means are the token-weighted
recombination of these; any alternative pooling (equal-weight, subsets,
bootstrap) is reproducible from the shards in numpy alone. Provenance
caveat: shards predate the padding-side fix (project TREE Q1.H3.E4) — the
first-50-token skip fully applied only to the longest story per batch of 4.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    load_dotenv()
    api = HfApi()
    user = api.whoami()["name"]
    for subdir, suffix in DIRS:
        src = args.results / subdir
        if not (src / "shards").exists():
            print(f"skip {subdir}: no shards/")
            continue
        repo = f"{user}/{suffix}"
        api.upload_folder(
            folder_path=str(src / "shards"),
            repo_id=repo,
            repo_type="dataset",
            path_in_repo="shards",
        )
        for extra in ("manifest.jsonl", "run_config.json", "extract.log"):
            if (src / extra).exists():
                api.upload_file(
                    path_or_fileobj=(src / extra).read_bytes(),
                    path_in_repo=extra,
                    repo_id=repo,
                    repo_type="dataset",
                )
        try:
            readme = api.hf_hub_download(repo, "README.md", repo_type="dataset")
            text = Path(readme).read_text()
        except Exception:
            text = f"# {suffix}\n"
        if "Per-story substrate" not in text:
            api.upload_file(
                path_or_fileobj=(text + CARD_APPENDIX).encode(),
                path_in_repo="README.md",
                repo_id=repo,
                repo_type="dataset",
            )
        print(f"substrate published -> {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
