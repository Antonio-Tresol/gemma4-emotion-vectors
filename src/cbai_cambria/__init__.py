"""cbai_cambria — library code for the emotion-vectors project (Q1/Q3).

Modules, split along the dependency boundary (laptop-safe vs GPU):
    corpus — corpus loading, reference constants, model geometry
        (mirrors sinievanderben/emotion_experiment; base deps only).
    story_store — per-story shard persistence, manifest, resumability, and
        token-weighted aggregation into per-emotion means (numpy only).
    hf_publish — publishing outputs as a private Hugging Face dataset repo.
    pipeline — the extraction itself: model loading, hook capture,
        TOKEN_OFFSET pooling, batch loop. Imports torch at module level, so it
        needs the `gpu` extra; laptop-side code imports it lazily or not at all.

Entry points live in scripts/ and stay thin; anything imported from more than
one place belongs here. Analysis code for Q1/Q3 gets its own module when it
exists.
"""
