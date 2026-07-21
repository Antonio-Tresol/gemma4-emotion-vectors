"""Shared pieces of the Q1 extraction pipeline (reference-faithful paths).

Both the time estimator and the real extraction script mirror the reference
implementation (sinievanderben/emotion_experiment, extract_emotion_vectors.py):
same corpus loading into {emotion: [stories]}, same model-family layer lookup,
same bf16 adaptation of the reference's fp32 default (which needs ~124 GB for
the 31B model and does not fit the 96 GB card). Keeping them here means the
estimator measures exactly the code path the extraction runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final

TOKEN_OFFSET: Final[int] = 50  # reference: pooling skips the first 50 tokens
REFERENCE_BATCH_SIZE: Final[int] = 4  # reference default
REFERENCE_MAX_LENGTH: Final[int] = 512


def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def load_emotions_data(dataset: str, split: str) -> dict[str, list[str]]:
    """Reference's loader, verbatim in behaviour: {emotion: [story, ...]}."""
    from datasets import load_dataset  # noqa: PLC0415

    rows = load_dataset(dataset, split=split)
    emotions_data: dict[str, list[str]] = {}
    for entry in rows:
        if entry.get("stories"):
            emotions_data.setdefault(entry["emotion"], []).extend(entry["stories"])
    return emotions_data


def get_layer(model: object, idx: int) -> object:
    """The reference's _get_layer, condensed: model-family layer lookup."""
    for fn in (
        lambda m, i: m.model.layers[i],
        lambda m, i: m.model.language_model.model.layers[i],
        lambda m, i: m.model.language_model.layers[i],
        lambda m, i: m.language_model.model.layers[i],
    ):
        try:
            return fn(model, idx)
        except (AttributeError, IndexError):
            continue
    raise AttributeError(f"cannot locate layer {idx}")


def detect_model_geometry(model_name: str) -> tuple[int, int]:
    """(d_model, n_layers) from the config — reference reads hidden_size with a
    text_config fallback for Gemma 4's nested multimodal config."""
    from transformers import AutoConfig  # noqa: PLC0415

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_cfg = cfg if getattr(cfg, "hidden_size", None) else cfg.text_config
    return int(text_cfg.hidden_size), int(text_cfg.num_hidden_layers)


def load_model_bf16(
    model_name: str, log: Callable[[str], None] = print
) -> tuple[object, object, float]:
    """(tokenizer, model, load_seconds). bf16, not the reference's fp32:
    fp32 needs ~124 GB for the 31B model and does not fit the 96 GB card."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"loading {model_name} (bf16)...")
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    load_s = time.monotonic() - t0
    log(f"model loaded in {human(load_s)}")
    return tokenizer, model, load_s
