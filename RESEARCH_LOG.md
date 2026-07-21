# CBAI sprint — research log

## Project summary

A 2-3 day sprint replicating Anthropic's emotion-concepts finding on
Gemma-4-31B — difference-of-means emotion vectors from the residual stream,
circumplex geometry check, and the cosine-similarity control-prompt validation
the open replication skipped (Q1, the gate) — then, only if the gate passes,
testing whether the model can detect the emotional direction it is being
steered toward and whether misidentifications are graded in valence-arousal
space (Q2). Method, assets, and pitfalls distilled in
notes/emotion-vectors-brief.md. The harness self-evaluation that occupied day
one lives in the harness repo's own tree and log.

---

# Log

Newest entry first. Every entry answers the same four questions.

### 2026-07-21

* What I did: Built and ran the Q1.H1.E1 extraction stage end to end. First a measured time estimate: a reference-mirroring benchmark on the pod (RTX PRO 6000 Blackwell) gave 196 tok/s and exposed that the model load from the contended network volume (33.4 min) cost more than the extraction itself — fixed by copying the 59 GB HF cache to local NVMe, cutting loads to 46 s. Then the pipeline proper: reference-faithful math (same tokenizer call, hook capture, TOKEN_OFFSET=50 pooling mask as sinievanderben/emotion_experiment) with deliberate deviations recorded in the module docstring — bf16 (fp32 does not fit 96 GB), torch.inference_mode(), and per-story shard persistence recombined as a token-weighted mean (identical to the reference's mean, but resumable and enabling within-emotion variance analyses). Smoke run (3 emotions x 4 stories) passed including a kill-and-rerun resume check; full run extracted 1539/1539 stories, 0 errors, 171 emotions x 20 layers in 7.8 min, published as the private HF dataset abotresol/emotion-vectors-gemma-4-31b. Code promoted along the way into an installed src/emotion_vectors package split along the dependency boundary (corpus/story_store/hf_publish laptop-safe; extraction GPU-only) — first named after the org codename, renamed the same day to say what it does — ruff now enforcing top-level imports (PLC0415) with one documented exception, zero lint warnings on project code.
* What I expected vs what happened: Expected the full run to take ~27.5 min of compute per the benchmark's 196 tok/s; it took 7.8 min — the benchmark's random-sample batches carry more padding than the real per-emotion in-order batching, so the estimate was conservative by ~3.5x. Also expected writes to the slow network volume to need an async local-write-then-sync design; measured overhead was 2.5%, so the simple synchronous write stayed. One incident: a traced `source .env` leaked HF_TOKEN into a pod log (redacted; launcher now sources env with xtrace off; token rotation recommended to the user).
* What this changes about my thinking: Measure before engineering — both the async-writer idea and the estimate itself were corrected by cheap measurements. The estimator now importing the production run_batch (instead of duplicating the loop) means future estimates measure the code that actually runs. And the token-weighted-mean subtlety (per-story means cannot be averaged directly; weights are post-mask token counts) is exactly the kind of silent correctness bug the reference-first reading caught before it happened.
* What I will do next: The NRC valence/arousal correlation analysis per layer (the remaining half of Q1.H1.E1, laptop-safe against the published vectors), then Q1.H2.E1's control-prompt battery at the peak layer(s). Literature pass with citation pinpoints still owed.

### 2026-07-20

* What I did: Day one went to hardening the machinery this project runs on: audited the harness skills against the agentskills.io spec, then built and ran the harness's skill evals end to end — trigger sweeps, three-arm behaviour probes, falsification gate — all of which now lives in the harness repo's own TREE.md and RESEARCH_LOG.md (separated from this project today; this tree holds only the sprint's research). Late in the day, scoped the sprint's substantive project from the partner's planning brief: Q1 replicates the emotion-vector geometry on Gemma-4-31B via the open replication codebase, including the control-prompt cosine validation the replication skipped; Q2 (injection detection plus circumplex-graded misidentification analysis) is GATED on Q1 — no validated vectors, no injection experiment. Distilled the brief into notes/emotion-vectors-brief.md, keeping sources and dropping everything off the two-part path (logistics, staffing, the brief's own priority inversion).
* What I expected vs what happened: Expected to pick a question in the morning and start the literature pass; instead the harness evals produced findings (and incidents) worth a full day, and the sprint question arrived in the evening via the partner's brief, already well-sourced. The brief claims verification of its sources, but nothing in it has been verified in-session here yet.
* What this changes about my thinking: The day-one detour bought real protection for days two and three — fenced eval workspaces, collaboration rules, and direct evidence that the validate-claims skill measurably changes provenance behaviour. For this project it also set the standard: every number from the planning brief gets verified against the primary source (with page-level pinpoints) before it enters any deliverable.
* What I will do next: Q1.H1.E1 — clone sinievanderben/emotion_experiment, adapt it to Gemma-4-31B (transformers >= 4.51), pull the published Gemma story corpus from HF, and start the layer sweep; in parallel, the timeboxed literature pass reading the two primary papers properly (adding citation pinpoints to the brief as they are read).

Evening pivot: the second part changed. Instead of the introspection battery (Q2, now abandoned in the tree, nodes kept), the post-gate question is temporal dynamics — Q3: in synthetic stories that move through 2-3 emotions, does the per-token cosine similarity between the residual stream and each emotion vector show gradual ramp-and-crossover dynamics (the signature of evidence accumulation, as in drift-diffusion models of decision-making) or abrupt steps at lexical cues? Design notes: per-token trajectories (extraction's mean-pooling won't do), token-aligned transitions, matched lengths, random-direction and shuffled-story controls, subtle-vs-explicit evidence strength, near-vs-far circumplex pairs. Framing discipline: claims stay at "dynamics consistent with integration vs switching" — the drift-diffusion analogy generates predictions, it is not itself the claim. Gate unchanged: Q3 waits on Q1's validated vectors. Literature pass tomorrow must add: prior work on temporal dynamics of concept directions over context.

### 2026-07-19

* What I did: Set up the project scaffolding: MCP servers (arxiv, paper-search), research skills (research, falsify, validate-claims, derive-from-sources, alphaxiv-paper-lookup, research-log), the research tree (TREE.md), this log, and the mechanical validator (scripts/validate_research.py).
* What I expected vs what happened: Expected to copy configs verbatim from the eval-awareness and thesis repos; that worked, and the convergent-validity repo additionally contributed the append-only decision-record model that shaped the tree/log split.
* What this changes about my thinking: The infrastructure for honest research (falsify, validate, log, tree) is reusable across projects and cheap to port; the scarce resource is choosing a question narrow enough for 2-3 days.
* What I will do next: Pick the sprint research question, write it as Q1 in TREE.md, and do the timeboxed literature pass.
