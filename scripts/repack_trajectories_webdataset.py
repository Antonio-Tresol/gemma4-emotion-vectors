"""Repack the per-token trajectory shards as WebDataset tars.

The problem: each trajectory dataset holds 5,888 separate `.npz` files in a
single `shards/` directory. That is under the Hub's 10,000-entries-per-folder
ceiling, but only just, and it is why these repositories have no dataset viewer
and why fetching them file-by-file gets rate-limited (HTTP 429 in practice).
The Hub's advice is explicit: "merge data into fewer files", and it names
WebDataset as the format for exactly this case, many binary samples.

Why WebDataset and not Parquet: a shard holds `dots`, a
[tokens, layers, probes] float16 tensor whose first axis differs per story.
Flattening that into Parquet list columns would inflate float16 to float32 and
produce a "viewer" nobody can read anyway. A tar preserves the bytes exactly,
so the repacked data is provably the same data.

Nothing is deleted. This publishes a parallel `-webdataset` repository; the
original stays, so every existing URL, card and `fetch()` route keeps working.

    uv run python scripts/repack_trajectories_webdataset.py --repo <name>
    uv run python scripts/repack_trajectories_webdataset.py --repo <name> --publish
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

HF_USER = "abotresol"
BUILD_ROOT = Path("results/trajectory_webdataset")
# ~1 GB per tar: big enough that the file count collapses, small enough that a
# failed download does not cost the whole set.
SHARDS_PER_TAR = 1500


def _download_with_backoff(repo_id: str, *, attempts: int = 8) -> str:
    """Pull the shards, waiting out Hugging Face's rate limiter.

    `snapshot_download` has no retry of its own, so a single HTTP 429 anywhere
    in a 5,888-file pull aborts the lot. The limiter applies to bulk fetching
    whether or not a token is present: an authenticated run failed here exactly
    as the anonymous one did, and the run that appeared to succeed had simply
    been served from cache. So the fix is patience and fewer workers, not
    credentials.

    Already-downloaded files are skipped on each retry, so progress accumulates
    across attempts rather than restarting.
    """
    delay = 30
    for attempt in range(1, attempts + 1):
        try:
            return snapshot_download(
                repo_id,
                repo_type="dataset",
                allow_patterns=["shards/*.npz"],
                max_workers=2,
            )
        except Exception as exc:  # noqa: BLE001 — any transport failure is retryable here
            if attempt == attempts:
                raise
            print(f"    attempt {attempt} stopped ({type(exc).__name__}); waiting {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 600)
    raise RuntimeError("unreachable")


def build(repo_name: str, *, limit: int | None) -> tuple[Path, dict[str, object]]:
    """Download the shards once, pack them into tars, and verify byte equality."""
    repo_id = f"{HF_USER}/{repo_name}"
    out_dir = BUILD_ROOT / repo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = hf_hub_download(repo_id, "manifest.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(manifest_path, encoding="utf-8")]
    story_ids = [row["story_id"] for row in rows]
    if limit:
        story_ids = story_ids[:limit]

    print(f"  fetching {len(story_ids):,} shards (one snapshot, not one request each)")
    snapshot_root = Path(_download_with_backoff(repo_id))

    present = [sid for sid in story_ids if (snapshot_root / "shards" / f"{sid}.npz").exists()]
    missing = len(story_ids) - len(present)
    if missing:
        print(f"  NOTE {missing} manifest rows have no shard on the Hub; recorded, not packed")

    tars, digests = [], {}
    for start in range(0, len(present), SHARDS_PER_TAR):
        batch = present[start : start + SHARDS_PER_TAR]
        tar_path = out_dir / f"trajectories-{start // SHARDS_PER_TAR:04d}.tar"
        with tarfile.open(tar_path, "w") as tar:
            for story_id in batch:
                source = snapshot_root / "shards" / f"{story_id}.npz"
                payload = source.read_bytes()
                digests[story_id] = hashlib.sha256(payload).hexdigest()
                # snapshot_download hands back symlinks into the blob cache, and
                # tar.add() would faithfully store the LINK rather than the data,
                # producing tars that unpack to nothing outside this machine.
                # Write the bytes explicitly so every member is a regular file.
                # WebDataset keys a sample by the basename before the extension.
                info = tarfile.TarInfo(name=f"{story_id}.npz")
                info.size = len(payload)
                info.mtime = 0  # reproducible: the same shards give the same tar
                tar.addfile(info, io.BytesIO(payload))
        tars.append(tar_path)
        print(
            f"    wrote {tar_path.name} ({len(batch):,} samples, {tar_path.stat().st_size / 1e9:.2f} GB)"
        )

    verify(tars, digests)
    for name in ("manifest.jsonl", "run_config.json"):
        try:
            source = hf_hub_download(repo_id, name, repo_type="dataset")
            (out_dir / name).write_bytes(Path(source).read_bytes())
        except Exception:  # noqa: BLE001 — a repo without run_config is fine
            pass
    return out_dir, {"repo_id": repo_id, "n_samples": len(present), "n_tars": len(tars)}


def verify(tars: list[Path], digests: dict[str, str]) -> None:
    """Every packed member must be byte-identical to the shard it came from."""
    checked = 0
    for tar_path in tars:
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                story_id = member.name.removesuffix(".npz")
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise SystemExit(f"{tar_path.name}: {member.name} is not a regular file")
                if hashlib.sha256(extracted.read()).hexdigest() != digests[story_id]:
                    raise SystemExit(f"{tar_path.name}: {member.name} does not match its source")
                checked += 1
    print(f"  verified {checked:,} samples byte-identical to their original shards")


def card(repo_name: str, stats: dict[str, object]) -> str:
    original = f"https://huggingface.co/datasets/{stats['repo_id']}"
    return f"""---
license: mit
---

# {repo_name}-webdataset

The same per-token trajectory data as
[`{stats["repo_id"]}`]({original}), packed as
[WebDataset](https://github.com/webdataset/webdataset) tars instead of
{stats["n_samples"]:,} separate `.npz` files.

Nothing changed but the packaging. Each tar member is the original `.npz`,
byte for byte; the build script checks every sample's SHA-256 against the file
it came from before publishing. The original repository stays where it is.

**Why:** {stats["n_samples"]:,} files in one directory is near the Hub's
10,000-entries-per-folder ceiling, and fetching them one request at a time gets
rate-limited. {stats["n_tars"]} tars download in {stats["n_tars"]} requests.

## Contents

| file | what it is |
|---|---|
| `trajectories-*.tar` | {stats["n_samples"]:,} samples, keyed by story id |
| `manifest.jsonl` | one row per story: id, emotions, phase boundaries, mode |
| `run_config.json` | model, layers, seed and pooling settings for the run |

Each sample is one story's `.npz` holding `dots`
`[tokens, layers, probes]` float16, `norms` and `norms_centered`
`[tokens, layers]`, `speed` `[tokens-1, layers]`, `token_ids`, and
`phase_token_starts`.

`dots` is the raw dot product and is **not** the quantity the project scores.
The score is a centered cosine: subtract the token-weighted mean of the dots
over the whole story set first, then divide by `norms_centered`.
`emotion_vectors.q3_conventions` in the
[project repository](https://github.com/Antonio-Tresol/gemma4-emotion-vectors)
is the reference implementation.

## Loading

```python
import io, tarfile
import numpy as np
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "{HF_USER}/{repo_name}-webdataset",
    "trajectories-0000.tar",
    repo_type="dataset",
)
with tarfile.open(path) as tar:
    member = tar.getmembers()[0]
    shard = np.load(io.BytesIO(tar.extractfile(member).read()))
    print(member.name, shard["dots"].shape)
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="dataset name without the user prefix")
    parser.add_argument("--limit", type=int, default=None, help="pack only the first N shards")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    print(f"repacking {args.repo}")
    out_dir, stats = build(args.repo, limit=args.limit)
    (out_dir / "README.md").write_text(card(args.repo, stats), encoding="utf-8")

    if not args.publish:
        print(f"\nbuilt in {out_dir}. Re-run with --publish to upload.")
        return 0

    api = HfApi(token=os.environ["HF_TOKEN"])
    target = f"{HF_USER}/{args.repo}-webdataset"
    api.create_repo(target, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=target,
        repo_type="dataset",
        commit_message="Repack the per-story shards as WebDataset tars (bytes unchanged)",
    )
    print(f"\npublished -> https://huggingface.co/datasets/{target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
