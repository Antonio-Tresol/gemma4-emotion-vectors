"""Corpus loading and model geometry

Mirrors the reference implementation (sinievanderben/emotion_experiment,
extract_emotion_vectors.py): the same {emotion: [stories]} loading and the
same nested-config handling for Gemma 4. Everything here runs on base
dependencies only; anything needing torch lives in emotion_vectors.extraction.
"""

from __future__ import annotations

from typing import Final

from datasets import load_dataset
from transformers import AutoConfig

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
    rows = load_dataset(dataset, split=split)
    emotions_data: dict[str, list[str]] = {}
    for entry in rows:
        if entry.get("stories"):
            emotions_data.setdefault(entry["emotion"], []).extend(entry["stories"])
    return emotions_data


def detect_model_geometry(model_name: str) -> tuple[int, int]:
    """(d_model, n_layers) from the config — reference reads hidden_size with a
    text_config fallback for Gemma 4's nested multimodal config."""
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_cfg = cfg if getattr(cfg, "hidden_size", None) else cfg.text_config
    return int(text_cfg.hidden_size), int(text_cfg.num_hidden_layers)
