# Parallel inference with a local vLLM — Gemma 4 31B (team template)

A reusable pattern for generating **a lot** of text with a local, offline
vLLM engine, using `google/gemma-4-31B-it` as the model. The running example
is **E5**: regenerate the emotion-story corpus with the probed model itself
(see `TREE.md` Q1.H2.E5). The engineering rules are general — reuse this for
any bulk local-generation job.

This doc assumes the repo's engineering contract (`.claude/skills/
experiment-engineering`). Where a rule below traces to it, it says so. Nothing
here is a claim or a result — it is method. Real runs still get registered in
`TREE.md` and logged in `RESEARCH_LOG.md` by the single writer (the
orchestrator), never by a subagent or a pod session directly.

---

## The one thing to internalise

> **You hand vLLM the *entire* list of prompts once. vLLM does the batching.**

vLLM runs a continuous-batching scheduler that packs many sequences onto the
GPU and keeps it saturated as sequences finish at different lengths. Your job
is to give it a big list and get out of the way. The classic mistake — porting
a `transformers` `for prompt in prompts: model.generate(prompt)` loop — throws
all of that away and serialises the work. Don't. (Contract: *"use the fast
inference path when generating at volume; naive `model.generate` loops are
dramatically slower."*)

So most of the transformers-side batching advice **inverts** here:

| transformers (`model.generate`) | local vLLM (`LLM.generate`) |
|---|---|
| You choose `batch_size`, loop batches | You pass all prompts once; scheduler batches |
| You sort by length, longest-first | Scheduler handles mixed lengths; don't pre-sort |
| You halve batch size on OOM | You cap KV via `gpu_memory_utilization` / `max_num_seqs` |
| `torch.inference_mode()` | vLLM manages this internally |

---

## Blackwell survival guide (RTX PRO 6000, SM 12.x) — measured 2026-07-21

Three independent failures stand between `uv pip install vllm` and a working
engine on this card. Each was hit, diagnosed, and fixed on the pod; the exact
symptoms are listed so the next person greps this file instead of the night's
logs. The runnable version is `scripts/pod/setup_vllm_env.sh`.

| # | Symptom in the log | Cause | Fix |
|---|---|---|---|
| 1 | `Stale file handle (os error 116)` mid-install | The venv lives on the /workspace network volume; large wheel writes get corrupted | Create the vLLM venv on **local NVMe**: `uv venv /root/vllm-env --python 3.14` |
| 2 | `Failed to get device capability: SM 12.x requires CUDA >= 12.9` then `EngineCore failed to start` ... `FlashInfer requires GPUs with sm75 or higher` | FlashInfer's capability probe misreads SM 12.x and treats the card as ancient | `export VLLM_ATTENTION_BACKEND=TRITON_ATTN` |
| 3 | `flashinfer/jit/sampling.py` in the traceback after fix 2 | FlashInfer's JIT **sampler** is a second, independent entry point | `uv pip uninstall flashinfer-python` from the vLLM venv (vLLM falls back to torch-native sampling) and belt-and-suspenders `export VLLM_USE_FLASHINFER_SAMPLER=0` |

Also required: torch built for the driver's CUDA (driver reports 13.0 here;
`torch 2.11+cu130` came with the vllm wheel and works — verify with
`python -c "import torch; print(torch.version.cuda)"`). `enforce_eager=True`
was tried for failure 2 and does NOT help (the failure is in capability
detection, not compilation); it is kept in our generation script only to skip
cudagraph capture time for short jobs, with `max_model_len=2048` bounding the
KV cache far below the model's 262k default.

Measured payoff: 2,424 stories x ~200 tokens in ~9 minutes (one
continuously-batched pass), where the `transformers` fallback loop measured
~20 stories/minute (~2.5 hours). Fallback stays wired in the launcher: every
generation call tries vLLM and drops to `--backend hf` on nonzero exit.

## Setup

**Dependency.** vLLM is a heavy CUDA package — it belongs in the pod-only
`gpu` extra, next to torch, not in the base deps (laptop-side tools must still
import without it). Add it there:

```toml
# pyproject.toml — [project.optional-dependencies]
gpu = [
    "accelerate>=1.14.0",
    "torch>=2.13.0",
    "vllm>=0.11",        # pod-only; pins its own torch — install into the gpu extra
]
```

```bash
# on the RunPod box:
uv sync --extra gpu
```

vLLM pins a specific torch; let it resolve rather than fighting it. If uv
conflicts, install vLLM into its own venv on the pod and treat the generator
as a standalone step whose only output is a JSONL file — the extraction
pipeline reads that file, not the vLLM process.

**Model & hardware.** `google/gemma-4-31B-it` is ~58 GB in bf16 → fits a
single 96 GB card (the box we already use) with room for KV cache. It is a
**native multimodal VLM with a thinking/reasoning mode** — see the gotchas
below; for plain story text you want thinking *off* and images unused.

---

## Minimal working example (offline, batched)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="google/gemma-4-31B-it",
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
    max_model_len=2048,        # prompt + max_tokens; keep TIGHT — it sizes the KV cache
    seed=20260721,
)

sampling = SamplingParams(
    temperature=0.9, top_p=0.95,
    max_tokens=300,
    n=8,                       # 8 independent samples per prompt, shared prefill — cheap volume
    seed=20260721,
)

# ONE call, whole list. gemma-4-31B-it has a chat template → use .chat().
conversations = [
    [{"role": "user", "content": f"Write a short (~150 word) third-person story "
      f"in which the main character feels {emotion}. Do not name the emotion or "
      f"any obvious synonym anywhere in the text."}]
    for emotion in ["happy", "afraid", "calm", "sad"]
]
outputs = llm.chat(conversations, sampling)

for conv_out in outputs:                 # one per conversation, input order preserved
    for sample in conv_out.outputs:      # n samples
        print(sample.text.strip())
```

`.chat()` applies the model's chat template for you (base Gemma 4 has none —
but the **-it** checkpoint does; that instruct chat template is the whole point
of the E5 pivot). If you need raw control, apply the template yourself and call
`llm.generate(list_of_strings, sampling)` instead.

---

## The production pattern (what you actually run for a corpus)

vLLM's `.chat()/.generate()` is **blocking and all-or-nothing**: it returns
only when the whole list finishes. That fights the contract's *resumable-by-
construction* rule — a crash at 90 % loses everything. Resolve the tension by
**chunking the work coarsely** (per emotion, or per few-hundred prompts) so one
`.chat()` call still gets a big batch for the scheduler, but a crash costs at
most one chunk. Write results to JSONL **as each chunk finishes**.

```python
"""E5 corpus generation — promoted (pipeline) mode. Resumable, logged, QC'd.

    uv run --extra gpu python scripts/generate_story_corpus.py \
        --emotions-from snae/emotion_stories_gemma_4_4B \
        --per-emotion 64 --out-dir results/story_corpus_it
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

from datasets import load_dataset

# vLLM (and torch) live in the pod-only `gpu` extra. Import at MODULE TOP — a
# guarded top-level import satisfies ruff PLC0415 and still lets laptop-side
# tooling import this module with a friendly error instead of an ImportError.
# This is the repo's one sanctioned lazy-import exception; see pyproject.toml
# and scripts/generate_dialogue_stories.py. Do NOT scatter imports into functions.
try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError as exc:
    raise SystemExit(f"missing {exc.name}: this entry point needs `uv sync --extra gpu`") from exc


def load_emotion_list(source: str) -> list[str]:
    # DO reuse the reference corpus's emotion keys so the self-gen corpus
    # covers the SAME emotions the extraction/analysis code expects.
    rows = load_dataset(source, split="train")
    return sorted({r["emotion"] for r in rows})

def already_done(raw_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if line.strip():
                e = json.loads(line)["emotion"]
                counts[e] = counts.get(e, 0) + 1
    return counts

def leaks_emotion(text: str, emotion: str) -> bool:
    # QC: drop generations that name the target emotion (or its stem).
    stem = emotion.lower().rstrip("dy").rstrip("e")          # happy→happ, afraid→afrai...
    return bool(re.search(rf"\b{re.escape(emotion.lower())}", text.lower())) or stem in text.lower()

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--emotions-from", default="snae/emotion_stories_gemma_4_4B")
    p.add_argument("--model", default="google/gemma-4-31B-it")
    p.add_argument("--per-emotion", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260721)
    p.add_argument("--smoke", action="store_true", help="2 emotions, n=2 — prove the path first")
    p.add_argument("--out-dir", type=Path, default=Path("results/story_corpus_it"))
    args = p.parse_args()

    # ---- FAIL FAST: cheap checks before the 58 GB load ----------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logger(args.out_dir / "generate.log")   # file + stdout, timestamps
    emotions = load_emotion_list(args.emotions_from)
    if args.smoke:
        emotions, args.per_emotion = emotions[:2], 2
    log.info(f"config: {vars(args)} | git={_git_commit()} | {len(emotions)} emotions")

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=2048, seed=args.seed)

    raw_path = args.out_dir / "stories_raw.jsonl"
    counts = already_done(raw_path)               # RESUMABLE: pick up where we left off
    t0 = _now()
    for i, emotion in enumerate(emotions):
        need = args.per_emotion - counts.get(emotion, 0)
        if need <= 0:
            continue
        sampling = SamplingParams(temperature=0.9, top_p=0.95,
                                  max_tokens=args.max_new_tokens, n=need,
                                  seed=args.seed + i)     # vary seed per emotion
        conv = [[{"role": "user", "content": _story_instruction(emotion)}]]
        try:
            (out,) = llm.chat(conv, sampling)
        except Exception as exc:                   # ERROR HANDLING: one emotion can't kill the run
            log.error(f"[{emotion}] failed: {exc!r}")
            _append(raw_path, {"emotion": emotion, "text": None, "error": repr(exc)})
            continue
        kept = 0
        with open(raw_path, "a") as f:             # STRUCTURED, INCREMENTAL: written as we go
            for s in out.outputs:
                text = s.text.strip()
                row = {"emotion": emotion, "text": text, "seed": args.seed + i,
                       "leaked": leaks_emotion(text, emotion), "n_chars": len(text)}
                if len(text) < 100 or row["leaked"]:   # QC: too short / names the emotion → drop
                    continue
                f.write(json.dumps(row) + "\n"); kept += 1
        counts[emotion] = counts.get(emotion, 0) + kept
        _eta(log, i + 1, len(emotions), t0)        # LOGGING WITH ETA
        log.info(f"[{emotion}] +{kept} (kept) → {counts[emotion]}/{args.per_emotion}")

    _assemble_grouped(raw_path, args.out_dir / "stories_grouped.jsonl", args.per_emotion)
    return 0
```

The helpers (`_setup_logger`, `_git_commit`, `_now`, `_eta`, `_append`,
`_story_instruction`, `_assemble_grouped`) mirror the ones already in
`scripts/generate_dialogue_stories.py` and `src/emotion_vectors/extraction.py`
— reuse those, don't reinvent. `_assemble_grouped` writes the final artifact in
the format the extraction pipeline reads (next section).

---

## Plugging into the pipeline (get the output format right)

`emotion_vectors.corpus.load_emotions_data` reads either an HF dataset id **or
a local `.jsonl`** whose rows are `{"emotion": ..., "stories": [ ... ]}`. So the
generator's final artifact is one JSONL line per emotion:

```json
{"emotion": "happy", "stories": ["...story 1...", "...story 2...", "..."]}
{"emotion": "afraid", "stories": ["...", "..."]}
```

Then extraction consumes it unchanged:

```bash
uv run --extra gpu python scripts/extract_emotion_vectors.py \
    --model google/gemma-4-31B-it \
    --dataset results/story_corpus_it/stories_grouped.jsonl
```

**Keep two files:** `stories_raw.jsonl` (one row per story, with `seed`,
`leaked`, `n_chars`, `error` — the resumable log and the audit trail) and
`stories_grouped.jsonl` (the assembled corpus). The raw file is the evidence;
the grouped file is derived and regenerable from it.

---

## Sampling for volume and diversity

- **`n=K` is the cheap-volume lever.** K samples per prompt share one prefill,
  so 64 stories/emotion costs far less than 64 separate prompts. Each sample
  has independent sampling RNG, so they genuinely differ.
- **`n` alone gives *sampling* diversity, not *scenario* diversity** — all K
  follow the identical instruction. For a corpus that spans situations, also
  **vary the prompt scaffold** (a few different scene setups per emotion) and
  spread K across them.
- **Dedup near-duplicates.** High-temperature batch sampling still produces the
  occasional twin. Drop exact dupes by normalised-text hash; for near-dupes,
  cosine on cheap sentence embeddings with a threshold. Log how many you
  dropped — a corpus silently padded with copies inflates n without adding
  signal (this directly bears on C2: probe directions are noise-limited at
  n~9/emotion, so *effective* diversity is what buys stability, not raw count).
- Reasonable starting point for stories: `temperature=0.9, top_p=0.95`.

---

## Gotchas specific to Gemma 4 31B -it

- **Thinking / reasoning mode.** The -it chat template can emit a reasoning
  trace (Gemma 4 ships a `gemma4` reasoning parser for the server path). For
  corpus text you want the *answer only*. **Verify** whether the chat template
  turns thinking on by default and whether it takes a disable flag
  (analogous to other instruct models' `enable_thinking=False`); regardless,
  add a defensive strip of any `<think>…</think>` / reasoning preamble in QC and
  eyeball raw outputs before trusting the corpus. Don't assume the field you
  get back is clean prose.
- **It's a VLM.** You're using it text-only, which is fine — just don't pass
  image placeholders, and keep `max_model_len` sized for text so you don't
  reserve KV for vision you never use.
- **Known open issues** on the checkpoint at time of writing: a character-
  duplication report and a missing-`reasoning`-field report in vLLM discussions.
  Pin a known-good vLLM version and smoke-test output quality before a long run.

---

## Memory / avoiding an OOM at hour three

- **`gpu_memory_utilization`** (0.85–0.92) is the master dial: it's the fraction
  of VRAM vLLM claims for weights **+ KV cache**. Higher = more concurrent
  sequences = more throughput, until it OOMs. Start 0.90 on the 96 GB card.
- **`max_model_len` is the other half of the KV budget.** Set it to *prompt +
  max_tokens* and no more. A needlessly large context reserves KV you don't use
  and starves concurrency. For ~150-word stories, 2048 is plenty.
- **`max_num_seqs`** caps how many sequences run concurrently — lower it if you
  see OOM during decode rather than at load.
- **Multi-GPU:** `tensor_parallel_size=N` shards the model across N GPUs. Only
  reach for it if the model + KV genuinely won't fit; a single 96 GB card
  already holds 31B-it comfortably, so default to 1.
- **FP4 option:** an `NVFP4` quant exists for Blackwell (`unsloth/
  gemma-4-31B-it-NVFP4`) — smaller footprint, more KV headroom, faster decode.
  Consider it if you need throughput or want to fit a bigger batch; validate
  output quality against bf16 on a sample before committing the corpus to it.

---

## Reproducibility caveat (read before you promise a teammate "same seed = same output")

vLLM is **not bitwise-deterministic across runs by default.** Continuous
batching means a sequence's kernels depend on what else is in the batch, so
even with `seed` fixed you can get different text if batch composition changes
(different `--per-emotion`, resumed run, different GPU). `seed` gives you
best-effort reproducibility, not a guarantee.

The contract's resolution: **the JSONL file is the record, not the seed.** Fix
and log the seed anyway (contract rule 6), but treat the written `stories_raw.
jsonl` as ground truth — that's what extraction consumes and what the tree
links, and it's reproducible by *re-reading*, not by *re-rolling*.

---

## Dos and Don'ts

**Do**
- Pass the whole prompt list to one `.chat()`/`.generate()` call per chunk; let
  the scheduler batch.
- Chunk coarsely (per emotion) and write JSONL after each chunk → resumable.
- Smoke-test first: `--smoke` (2 emotions, n=2, tiny) proves load + template +
  QC + output format before you spend the full GPU-hour.
- Size `max_model_len` to the task; keep `gpu_memory_utilization` ~0.90.
- Reuse the emotion list from the reference corpus so keys line up downstream.
- QC every generation: drop emotion-word leaks, too-short text, near-dupes —
  and **read raw outputs** before believing the corpus.
- Log to a file with an ETA; echo config, seed, git commit at start.
- Keep the vLLM generator as a standalone step whose only contract with the
  rest of the repo is the grouped JSONL file.

**Don't**
- ❌ Loop `llm.generate(one_prompt)` per item — that serialises and wastes the
  engine. This is the #1 mistake porting from `transformers`.
- ❌ Spawn threads / `multiprocessing` / multiple `LLM` objects to "parallelise"
  a single GPU — the engine already parallelises; extra processes just fight for
  VRAM. (`tensor_parallel_size` is the *only* right way to use >1 GPU.)
- ❌ Generate everything in one giant blocking call with no incremental writes —
  a crash at 90 % loses the run.
- ❌ Pre-sort by length or hand-tune batch size — that's transformers thinking.
- ❌ Assume `seed` gives identical text across runs — it doesn't; the JSONL is
  the record.
- ❌ Trust the metric before looking at the text — reasoning-trace leakage,
  emotion-word leakage, and duplicate stories are all invisible to a token count.
- ❌ Let the pod session write to `TREE.md` / `RESEARCH_LOG.md`. Report the
  numbers back; the orchestrator records them (single-writer rule, AGENTS.md).

---

## Pre-flight checklist

- [ ] `vllm` added to the `gpu` extra; `uv sync --extra gpu` clean on the pod.
- [ ] `--smoke` run: model loads, chat template applies, output is clean prose
      (no `<think>` leak), JSONL rows well-formed, grouped file readable by
      `load_emotions_data`.
- [ ] Emotion list pulled from the reference corpus (same keys downstream).
- [ ] Thinking-mode behaviour of the -it template checked and handled.
- [ ] QC filters on (leakage, length, dedup); drop counts logged.
- [ ] Seed + git commit + config in the log; JSONL written incrementally.
- [ ] Resume verified: kill mid-run, re-run, it continues rather than restarts.
- [ ] Findings reported to the orchestrator for TREE.md / RESEARCH_LOG.md — not
      written from the pod.
```

---

## Activation extraction with vLLM (investigated 2026-07-22)

Two routes exist; both were investigated because the queued expensive items
(Figure 1 corpus sweep, Q3 per-token trajectories) are prefill-heavy
activation jobs where the HF loop's measured rate is ~0.30 s/story.

**Route A — forward hooks (works today, benched).** With `enforce_eager=True`
and the in-process V0 engine (`VLLM_USE_V1=0`,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`), the torch modules are reachable at
`llm.llm_engine.model_executor.driver_worker.model_runner.model` and standard
`register_forward_hook` fires. Costs the CUDA-graph speedup. Bench:
`scripts/bench_vllm_activations.py` (feasibility, exact last-token parity vs
the HF-path sweep values, hooked prefill throughput); results in
`results/vllm_activation_bench.json`.

**Route B — official hidden-states extraction (vLLM >= 0.18; pod has
0.22.1).** First-class API reusing the EAGLE draft pathway: `speculative_config
{"method": "extract_hidden_states", "eagle_aux_hidden_state_layer_ids": [...]}`
plus `kv_transfer_config {"kv_connector": "ExampleHiddenStatesConnector",
"kv_role": "kv_producer", "shared_storage_path": ...}`. Writes safetensors
`[prompt_seq_len, num_layers, hidden]` per request, prompt tokens only
(`max_tokens=1`), selectable layers. No eager requirement, so CUDA graphs
stay. Caveats from the docs and the 2026-03-30 vLLM blog post: chunked
prefill must be disabled; the example connector does blocking disk writes
(fine for our batch sizes, bad at scale). Untested on Blackwell + our
survival settings — test before relying on it.

**Bench verdicts (2026-07-22).** Two benches, and the second overturned the
first — capture width decides everything:

| arm (256 stories, 512-token cap) | layers captured | persisted? | tok/s |
|---|---|---|---|
| HF production loop (`run_batch`, batch 4) | 20, pooled in-batch | no (means only) | 1,014 |
| vLLM + forward hooks, eager | 1, raw | no | ~3,400 |
| vLLM + forward hooks, eager | 20, raw | no | 528 |
| vLLM `extract_hidden_states` (official) | 20, full per-token | yes, safetensors to /workspace | 977 |

(`results/vllm_activation_bench.json`, `results/activation_engine_bench.json`)

- Numerics are exact on the hook route: cosine to the HF-path values min
  0.99999 over 37 prompts. The official route was not numerics-checked here
  (its output is per-token safetensors; check before first scientific use).
- Hooks win big at 1-3 layers (~2-3x) and LOSE at 20 (blocking per-layer
  `.cpu()` copies dominate; pooling inside the hook would help but was not
  benched).
- The official API matches the HF loop's throughput WHILE persisting the
  full per-token activations (~25 GB for this batch) — the other arms pay
  nothing for persistence. For corpus-scale per-token dumps it is strictly
  the best option. Working 0.22.1 config (cost three schema iterations):
  `speculative_config={"method": "extract_hidden_states",
  "num_speculative_tokens": 1, "draft_model_config": {"hf_config":
  {"eagle_aux_hidden_state_layer_ids": [...]}}}` plus
  `kv_transfer_config={"kv_connector": "ExampleHiddenStatesConnector",
  "kv_role": "kv_producer", "kv_connector_extra_config":
  {"shared_storage_path": ...}}`, `enable_chunked_prefill=False`.
  Dumps do NOT fit the pod's NVMe root disk — write to /workspace.

Decision rule: pooled means at any scale -> HF loop (simplest, production
path). A few layers, no persistence -> vLLM hooks. Corpus-scale per-token
dumps (Figure 1 sweep, Q3 trajectories) -> official API, numerics check
first. Canonical copy of this knowledge: research-harness
`experiment-engineering` skill.
