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

from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download

# Notebook kernels don't inherit a sourced .env; the datasets are private
# until sprint end, so resolve the project .env (repo root) for HF_TOKEN.
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
}


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
        downloaded = hf_hub_download(repo, sub or relpath, repo_type="dataset")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(Path(downloaded).read_bytes())
    if not local.exists():
        raise FileNotFoundError(
            f"{relpath}: not local and not found in {repo} (private until sprint end — "
            "authenticate with HF_TOKEN, or check the dataset card)"
        )
    return local
