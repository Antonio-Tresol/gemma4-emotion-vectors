# RunPod setup — RTX PRO 6000 Blackwell (96 GB)

Decisions 2026-07-20. d_model = 5376, read from the model config by
scripts/estimate_extraction_time.py.

**Adaptation requirement found in the reference code**: extract_emotion_vectors.py
loads the model in float32 — ~124 GB for the 31B model, which does NOT fit the
96 GB card. Our adaptation loads bf16 (the hooks already cast to float32 for
pooling, so the accumulation math is unchanged). The reference is also not
resumable (in-memory accumulation, one save at the end) — our adaptation
writes per-story pooled vectors incrementally: 0.4 MB/story, ~0.66 GB for the
corpus at 20 swept layers, ~3 s of write time. Measured corpus: 1,539 stories,
276k raw / 315k padded tokens in reference batches of 4.

## Pod

- Card: RTX PRO 6000 Blackwell, 96 GB — fits the ~62 GB bf16 weights with
  headroom for activations and KV cache. (Do NOT substitute the RTX 6000 Ada:
  48 GB does not fit the model unquantised, and quantisation is ruled out for
  activation work.)
- Network volume: **200 GB**, mounted at /workspace. Set `HF_HOME=/workspace/hf`
  so the model download survives pod stops.
- Env: clone the repo, `uv sync --extra gpu` (CUDA torch + accelerate live in
  the gpu extra; the base group is laptop-light). Secrets in `.env`
  (gitignored): `HF_TOKEN`, `RUNPOD_API_KEY`.
- Team tooling: the RunPod plugin (skills + hosted MCP) auto-installs from
  project settings on repo trust; `runpodctl` per machine if wanted.

## What to cache (storage budget)

| Artefact | Size (order of magnitude) | Verdict |
|---|---|---|
| Q1 extraction: per-story per-layer mean vectors, 1,539 stories × 60 layers | ~2 GB | store |
| Emotion contrast vectors + PCA outputs | MBs | store |
| Q3 per-token cosine trajectories (scalars) | MBs | store |
| Q3 per-token residual streams, ~100 stories × ~500 tokens, 2-3 peak layers, bf16 | ~1-2 GB | store — lets us recompute similarities against any vector without re-running the model |
| Q3 per-token residual streams, all 60 layers | ~40-60 GB | only if a post-hoc layer sweep is actually planned |
| Full-corpus token-level cache (1,539 stories × all layers) | ~800 GB | NEVER — this is the blowup the mean-pool-on-the-fly decision exists to avoid |

## Ritual

Runs follow the experiment-engineering contract: incremental JSONL to
/workspace/results, resumable (kill-and-rerun must skip completed stories),
seeds and model revision logged, ETA in the log file. Sync results back to the
repo's results/ before stopping the pod; the volume is durable but the repo is
the record.
