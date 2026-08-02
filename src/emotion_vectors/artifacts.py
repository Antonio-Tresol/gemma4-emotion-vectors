"""Artifact resolver — local results/ first, Hugging Face otherwise.

Open-science contract: the report notebooks read every input through
``fetch(relpath)``, so they run identically for us (files already in
``results/``) and for replicators (files pulled from the published datasets
into ``results/``, materializing the same layout). Routes map each results
subtree to its dataset repo; everything unrouted lives in the
experiment-artifacts repo under the same relative path.

The NRC VAD lexicon is deliberately NOT fetchable (third-party license);
obtain it from saifmohammad.com and place it under data/lexicons/.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

# Every dataset this resolves is PUBLIC and needs no credential. The .env is
# loaded only so a maintainer pushing new artifacts has HF_TOKEN available;
# nothing a reader fetches requires it. Keep it that way: a token silently
# present is how the anonymous path stops being exercised, and how a private
# repo goes unnoticed.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

HF_USER = "abotresol"
ARTIFACTS_REPO = f"{HF_USER}/emotion-vectors-experiment-artifacts"
RESULTS = Path(__file__).resolve().parents[2] / "results"

# results/ subtree -> dataset repo holding it (subtree layout matches repo root)
ROUTES: dict[str, str] = {
    "dialogue_stories": f"{HF_USER}/emotion-dialogues-gemma-4-31b",
    "dialogue_stories_it": f"{HF_USER}/emotion-dialogues-gemma-4-31b-it",
    "self_stories_it": f"{HF_USER}/emotion-stories-gemma-4-31b-it",
    "neutral_transcripts_it": f"{HF_USER}/neutral-transcripts-gemma-4-31b-it",
    "emotion_vectors": f"{HF_USER}/emotion-vectors-gemma-4-31b",
    "emotion_vectors_it": f"{HF_USER}/emotion-vectors-gemma-4-31b-it",
    "self_story_vectors_it": f"{HF_USER}/emotion-selfstory-vectors-gemma-4-31b-it",
    "dialogue_vectors": f"{HF_USER}/emotion-dialogue-vectors-gemma-4-31b",
    "dialogue_vectors_it": f"{HF_USER}/emotion-dialogue-vectors-gemma-4-31b-it",
    "neutral_vectors_it": f"{HF_USER}/neutral-vectors-gemma-4-31b-it",
    "combined_stories": f"{HF_USER}/emotion-combined-stories-gemma-4-31b-it",
    "combined_trajectories": f"{HF_USER}/emotion-combined-trajectories-gemma-4-31b-it",
    "combined_trajectories_base": f"{HF_USER}/emotion-combined-trajectories-gemma-4-31b",
    # Post-padding-fix re-extractions (E4b): the blessed sets for all probe work
    # from 2026-07-22 on. The unsuffixed repos above are their pre-fix
    # predecessors, kept for the E4b before/after comparison and for
    # reproducing pre-fix results; each -postfix repo carries a LINEAGE.md.
    "emotion_vectors_postfix": f"{HF_USER}/emotion-vectors-gemma-4-31b-postfix",
    "emotion_vectors_it_postfix": f"{HF_USER}/emotion-vectors-gemma-4-31b-it-postfix",
    "self_story_vectors_it_postfix": f"{HF_USER}/emotion-selfstory-vectors-gemma-4-31b-it-postfix",
    "neutral_vectors_it_postfix": f"{HF_USER}/neutral-vectors-gemma-4-31b-it-postfix",
    "openrouter_stories": f"{HF_USER}/emotion-stories-deepseek-v4-pro",
    # E12 scale-and-diversity arms (registered Q1.H2.E12): the diverse-prompt
    # corpus plus both DeepSeek vector sets, per-story shards included so the
    # dose-response subsampling reproduces without any GPU inference.
    "openrouter_stories_diverse": f"{HF_USER}/emotion-stories-deepseek-v4-pro-diverse",
    "openrouter_vectors_it": f"{HF_USER}/emotion-deepseek-vectors-gemma-4-31b-it",
    "openrouter_vectors_diverse_it": f"{HF_USER}/emotion-deepseek-diverse-vectors-gemma-4-31b-it",
    # Q3 substrate arms added 2026-07-23: DeepSeek-written stories, the
    # constant-emotion control, and the v2 re-extraction of the primary
    # corpus (post-fix corpus probes + the fixed-DeepSeek bank).
    "combined_stories_deepseek": f"{HF_USER}/emotion-combined-stories-deepseek-v4-pro",
    "combined_stories_deepseek_constant": f"{HF_USER}/emotion-combined-stories-deepseek-v4-pro-constant-control",
    "combined_trajectories_deepseek": f"{HF_USER}/emotion-combined-trajectories-deepseek-stories-gemma-4-31b-it",
    "combined_trajectories_deepseek_constant": f"{HF_USER}/emotion-combined-trajectories-constant-control-gemma-4-31b-it",
    "combined_trajectories_it_v2": f"{HF_USER}/emotion-combined-trajectories-gemma-4-31b-it-v2",
}


def repo_relative(path: Path | str) -> str:
    """A path as it should be RECORDED, not as it happened to be typed.

    Provenance fields in `results/` are read by other people on other machines.
    Writing `str(path)` there records whatever was passed on the command line,
    which for an absolute path bakes in one person's home directory: it is
    meaningless to a replicator, it breaks if the checkout moves, and it leaks
    a filesystem layout into a public repository. This returns the path
    relative to the repository root when it lies inside it, and leaves anything
    outside untouched, since that genuinely is external.
    """
    resolved = Path(path).expanduser()
    root = RESULTS.parent
    try:
        return str(resolved.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _route(relpath: str) -> tuple[str, str]:
    """(repo_id, path inside repo) for a results-relative path."""
    head = relpath.split("/", 1)[0]
    if head in ROUTES:
        rest = relpath.split("/", 1)[1] if "/" in relpath else ""
        return ROUTES[head], rest
    return ARTIFACTS_REPO, relpath


def fetch(relpath: str) -> Path:
    """Resolve a results-relative path, downloading from Hugging Face if the
    local file is absent. Directories are materialized recursively. Returns
    the local path under results/ either way."""
    local = RESULTS / relpath
    if local.exists():
        return local
    repo, sub = _route(relpath)
    looks_like_dir = "." not in Path(relpath).name
    if looks_like_dir:
        if _materialize_from_webdataset(repo=repo, target=local):
            return local
        pattern = f"{sub}/**" if sub else "**"
        # place the snapshot so repo-relative sub lands exactly at results/relpath
        base = RESULTS / relpath[: len(relpath) - len(sub)].rstrip("/") if sub else local
        snapshot_download(
            repo,
            repo_type="dataset",
            allow_patterns=[pattern],
            local_dir=str(base),
        )
    else:
        try:
            downloaded = hf_hub_download(repo, sub or relpath, repo_type="dataset")
        except EntryNotFoundError:
            # The vector repos publish one .npy per (emotion, layer) but not
            # always the stacked bundle the analysis code asks for. Every number
            # in that bundle is public either way, so rebuild it rather than
            # dead-end a reader on a 404. See _rebuild_emotion_means.
            if Path(relpath).name != "emotion_means.npz":
                raise
            _rebuild_emotion_means(repo=repo, out_path=local)
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(Path(downloaded).read_bytes())
    if not local.exists():
        raise FileNotFoundError(f"{relpath}: not in local results/ and not found in {repo}")
    return local


def _materialize_from_webdataset(*, repo: str, target: Path) -> bool:
    """Fill `target` from the repo's WebDataset tars. True if it did.

    The trajectory datasets hold 5,888 per-story `.npz` files. Downloading them
    one request each is what a replicator without a token cannot do: Hugging
    Face returns HTTP 429 partway through, so the reproduction simply fails.
    The same data is also published as `<repo>-webdataset`, a handful of tars,
    which arrives in a handful of requests.

    Unpacking reproduces the original `shards/<story_id>.npz` layout exactly, so
    everything downstream, including `q3_conventions.manifest_rows`, is unaware
    that anything changed. Falls back by returning False whenever the tars are
    absent or unreadable, because a faster path is not worth a new failure mode.
    """
    tar_repo = f"{repo}-webdataset"
    try:
        names = list_repo_files(tar_repo, repo_type="dataset")
    except Exception:  # noqa: BLE001 — no tars published for this set yet
        return False
    tar_names = sorted(name for name in names if name.endswith(".tar"))
    if not tar_names:
        return False

    shard_dir = target / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    print(f"{tar_repo}: fetching {len(tar_names)} tar(s) instead of thousands of single files")
    for tar_name in tar_names:
        path = hf_hub_download(tar_repo, tar_name, repo_type="dataset")
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                payload = tar.extractfile(member)
                if payload is not None:
                    (shard_dir / Path(member.name).name).write_bytes(payload.read())

    # the manifest and run config live beside the tars and are needed with them
    for name in ("manifest.jsonl", "run_config.json"):
        if name in names:
            source = hf_hub_download(tar_repo, name, repo_type="dataset")
            (target / name).write_bytes(Path(source).read_bytes())
    print(f"  unpacked {len(list(shard_dir.glob('*.npz'))):,} shards into {target}")
    return True


def _rebuild_emotion_means(*, repo: str, out_path: Path) -> None:
    """Assemble a stacked emotion_means.npz from a repo's per-emotion shards.

    The vector datasets publish `<emotion>/layer_<N>_resid.npy`, one mean
    residual vector per (emotion, layer). The stacked `emotion_means.npz` that
    the analysis code loads is a repackaging of exactly those numbers, and some
    repos carry the shards without it. Rebuilding is therefore lossless, and it
    was verified bit-identical against a locally held bundle before this path
    was written.

    Emotion order is the sorted directory names and the layer order comes from
    the repo's own run_config.json, which is how the original bundle was built.
    """
    files = list_repo_files(repo, repo_type="dataset")
    emotions = sorted({name.split("/")[0] for name in files if "/layer_" in name})
    if not emotions:
        raise FileNotFoundError(f"{repo}: no per-emotion shards to rebuild emotion_means.npz from")
    config = json.loads(
        Path(hf_hub_download(repo, "run_config.json", repo_type="dataset")).read_text()
    )
    layers = [int(layer) for layer in config["layers"]]

    print(f"{repo}: emotion_means.npz is not published; rebuilding it from the per-emotion shards")
    print(f"  {len(emotions)} emotions x {len(layers)} layers, downloaded once and cached locally")
    snapshot_root = Path(
        snapshot_download(repo, repo_type="dataset", allow_patterns=["*/layer_*_resid.npy"])
    )
    stacked = np.stack(
        [
            np.stack(
                [np.load(snapshot_root / emotion / f"layer_{layer}_resid.npy") for layer in layers]
            )
            for emotion in emotions
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, emotions=np.array(emotions), layers=np.array(layers), means=stacked)
    print(f"  wrote {out_path} {stacked.shape}")
