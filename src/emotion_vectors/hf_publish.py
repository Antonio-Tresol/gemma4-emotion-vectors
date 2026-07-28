"""Publish extraction outputs as a private Hugging Face dataset repository.

Split out of the extraction script so publishing concerns (auth, repo naming,
dataset card) stay separate from the GPU pipeline. Reads run_config.json from
the output directory, so it can re-publish any completed run (--publish-only).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_BASENAME = "emotion-vectors-gemma-4-31b"


def check_hf_auth(logger: logging.Logger) -> None:
    """Fail fast — called before the model load so a bad token costs seconds."""
    logger.info(f"HF auth ok as {HfApi().whoami()['name']}")


def write_readme(out_dir: Path, config: dict[str, object]) -> None:
    # The frontmatter is not decoration: Hugging Face reads it for the licence
    # and for search. Cards published without it left 20 public datasets with no
    # stated terms, which a downloader has no way to resolve.
    (out_dir / "README.md").write_text(
        f"""---
license: mit
task_categories:
  - feature-extraction
tags:
  - interpretability
  - emotion
  - gemma
---

# Emotion vectors — {config["model"]}

Per-story pooled residual-stream activations and per-emotion mean vectors,
extracted with [gemma4-emotion-vectors](https://github.com/Antonio-Tresol/gemma4-emotion-vectors)
`scripts/extract_emotion_vectors.py` (reference-faithful adaptation of
sinievanderben/emotion_experiment `extract_emotion_vectors.py`).

- Corpus: `{config["dataset"]}` (split `{config["split"]}`)
- Layers: {config["layers"]}
- Pooling: mean over non-pad tokens after position {config["token_offset"]},
  truncation at {config["max_length"]}, batch size {config["batch_size"]}, bf16 model,
  fp32 activations.
- `shards/<emotion>__<idx>.npy`: one `[layers, d_model]` fp32 array per story.
- `manifest.jsonl`: per-story metadata (emotion, index, text sha1, token count).
- `<emotion>/layer_<N>_resid.npy` and `emotion_vectors.json`: token-weighted
  per-emotion means, the reference's output format.
- `run_config.json`: full extraction config, seed {config["seed"]},
  commit `{config["git_commit"]}`.

Re-extracting needs the model weights and a GPU with enough memory for a 31B
model in bf16. Analysis does not: the per-emotion means here are enough to redo
the geometry and detection work on a laptop.

```
uv run python scripts/extract_emotion_vectors.py
```

## Licence

MIT, matching the project repository. The model weights and the story corpora
carry their own licences.
"""
    )


def publish(out_dir: Path, repo_arg: str | None, smoke: bool, logger: logging.Logger) -> None:
    """Upload the whole output directory to a private dataset repo. Smoke runs
    get a '-smoke' suffix so they never pollute the real dataset."""
    api = HfApi()
    name = repo_arg or (DEFAULT_REPO_BASENAME + ("-smoke" if smoke else ""))
    if "/" not in name:
        name = f"{api.whoami()['name']}/{name}"
    api.create_repo(name, repo_type="dataset", private=True, exist_ok=True)
    write_readme(out_dir, json.loads((out_dir / "run_config.json").read_text()))
    api.upload_folder(folder_path=str(out_dir), repo_id=name, repo_type="dataset")
    logger.info(f"published (private) -> https://huggingface.co/datasets/{name}")
