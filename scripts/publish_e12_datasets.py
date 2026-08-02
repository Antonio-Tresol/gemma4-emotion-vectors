"""Publish the E12 corpora and vector sets (with per-story shards) to HF.

Per the open-science substrate policy: shards go up so the dose-response
subsampling reproduces from public data with no GPU inference. Repos match
the routes in emotion_vectors.artifacts. All public.

    .venv/bin/python scripts/publish_e12_datasets.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

# One results tree, resolved from this file. There used to be two, because the
# vectors sat in the main clone while the corpora sat in a worktree, and the
# main clone's path was hardcoded. That is a machine-local absolute path in
# committed code: it breaks for every other person and leaks a home directory
# into a public repository.
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    load_dotenv()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    user = api.whoami()["name"]

    jobs = [
        # (repo, [(local path, path in repo)])
        (
            f"{user}/emotion-stories-deepseek-v4-pro-diverse",
            [
                (RESULTS / "openrouter_stories_diverse/stories_raw.jsonl", "stories_raw.jsonl"),
                (
                    RESULTS / "openrouter_stories_diverse/stories_grouped.jsonl",
                    "stories_grouped.jsonl",
                ),
                (RESULTS / "openrouter_stories_diverse/run_config.json", "run_config.json"),
            ],
        ),
        (
            f"{user}/emotion-deepseek-vectors-gemma-4-31b-it",
            [
                (RESULTS / "openrouter_vectors_it_means.npz", "emotion_means.npz"),
                (RESULTS / "openrouter_vectors_it_shards", "shards"),
            ],
        ),
        (
            f"{user}/emotion-deepseek-diverse-vectors-gemma-4-31b-it",
            [
                (RESULTS / "openrouter_vectors_diverse_it_means.npz", "emotion_means.npz"),
                (RESULTS / "openrouter_vectors_diverse_it_shards", "shards"),
            ],
        ),
    ]
    for repo, items in jobs:
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        for local, in_repo in items:
            print(f"uploading {local} -> {repo}/{in_repo}", flush=True)
            if local.is_dir():
                api.upload_folder(
                    folder_path=str(local), repo_id=repo, repo_type="dataset", path_in_repo=in_repo
                )
            else:
                api.upload_file(
                    path_or_fileobj=str(local),
                    repo_id=repo,
                    repo_type="dataset",
                    path_in_repo=in_repo,
                )
        api.update_repo_settings(repo, private=False, repo_type="dataset")
        print(f"DONE {repo}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
