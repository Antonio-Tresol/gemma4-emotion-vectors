"""cbai_cambria — library code for the emotion-vectors project (Q1/Q3).

Modules:
    extraction_common — reference-shared paths: corpus loading, layer lookup,
        bf16 model loading (mirrors sinievanderben/emotion_experiment).
    pipeline — the extraction pipeline: hook capture, TOKEN_OFFSET pooling,
        batch loop, run settings.
    story_store — per-story shard persistence, manifest, resumability, and
        token-weighted aggregation into per-emotion means.
    hf_publish — publishing outputs as a private Hugging Face dataset repo.

Entry points live in scripts/ and stay thin; anything imported from more than
one place belongs here.
"""
