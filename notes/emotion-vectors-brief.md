# Emotion vectors: replication + introspection stretch — distilled brief

Distilled 2026-07-20 from the planning brief compiled 2026-07-19. Numbers and
claims below come from that brief's reading of the sources; none have been
independently verified in-session yet — verify against the linked primaries
before any of them enters a deliverable.

## The core (scope to protect)

The plan is BOTH parts, with the first as a **gate** on the second.

**Part 1, the gate (~2 days): replicate Anthropic's "Emotion Concepts and
their Function in a Large Language Model"** on Gemma-4-31B (instruct, dense,
60 layers): difference-of-means emotion vectors from the residual stream, then
validate that cosine similarity of these vectors responds predictably on
control prompts. Original code was never released; adapt the open-source
replication code from "Where Do Models Find Happiness?". **Go/no-go:** if
Gemma-4-31B yields no recoverable geometry or the control-prompt validation
fails, there are no vectors worth injecting — record the null honestly and
stop; Q3 closes as infeasible-with-evidence.

**Part 2, gated on Part 1: introspection.** Inject the validated emotion
vectors and test whether the model can detect the emotional direction it is
being steered toward (per "Emergent Introspective Awareness"); if it
misidentifies, analyse the failure patterns — do wrong guesses cluster near
the injected emotion in valence-arousal (circumplex) space, or scatter?
Either answer is presentable (graded content access vs an extension of the
content-agnostic finding).

Scope rule: anything in the source planning brief that is not on the path to
these two parts (hackathon logistics, staffing splits, cut negotiations,
speculative extensions) was deliberately NOT adopted here. Its extended
related-work list is kept only as one-line context below — background
reading, not commitments.

## Method core

Per emotion e and layer l: mean-pool residual-stream activations over that
emotion's stories → u_e; contrast vector v_e = u_e minus its projection onto
the top-K PCs of neutral-story activations (K chosen to explain 50% of
variance). Stack over emotions, PCA, correlate PC1/PC2 with NRC
valence/arousal. Anthropic reference (Claude Sonnet 4.5): PC1-valence r=0.81,
PC2-arousal r=0.66. The replication reached r=0.83 (Gemma-4-E4B, Gemma
stories, layer 16). The cosine-similarity control-prompt validation is the
part the replication skipped — our primary goal fills it.

## Assets

| Asset | Location |
|---|---|
| Replication code | github.com/sinievanderben/emotion_experiment |
| Gemma story corpus (513 prompts / 1,539 stories) | HF dataset `snae/emotion_stories_gemma_4_4B` |
| Emotion list (171, Anthropic's), prompts, topics | in the repo under `prompts/` |
| NRC valence-arousal ratings | repo `emotion_valence_arousal_nrc.csv` (Mohammad 2018) |
| Subject model | HF `google/gemma-4-31B` instruct (needs transformers >= 4.51) |
| Fallback subject | Gemma-4-12B (same code path; break-glass only) |
| Steering + detection battery prior art | github.com/agastyasridharan/emotional-probes |

## Decisions carried over from planning (the ones that matter)

- Subject: Gemma-4-31B dense — not the MoE variant (routing muddies the
  residual stream); same family as the replication's E4B, and a June 2026
  LessWrong post documents introspection-adjacent signal on exactly this
  model, so a null is informative.
- Skip story generation; reuse the published Gemma corpus (corpus choice
  measurably leaks into vectors, especially arousal — keep it fixed across
  all conditions).
- Sweep layers for extraction (every 3rd of 60, refine near the peak);
  do not assume the ~2/3-depth heuristic transfers — Apertus and Gemma-E4B
  showed opposite depth trajectories.
- Thinking mode off everywhere, for comparability with prior work.
- Never cache full token-level activations across layers; mean-pool per story
  on the fly.
- Do not quantise the 31B — injection on quantised activations puts an
  asterisk on the headline result.
- Compute: single A100 80GB (~$1.20-1.90/hr), ~20-30 GPU-hours ≈ $30-80;
  weights ~62GB bf16 fit one card.

## Known pitfalls (design around from the start)

- Identification-via-steering confound: steering makes the model talk about
  the concept anyway — gate on detection as a boolean, and keep a minimal
  steering-on vs steering-off-during-response contrast.
- Always report false-positive rate and control-question shifts next to any
  detection rate (yes-bias).
- Fix a small set of prompt framings from prior work; do not invent framings
  mid-project (prompt sensitivity is enormous in open models).
- Phrase carefully: "emotion vectors" = directions correlating with emotion
  concepts under this recipe; causal/functional claims need steering evidence.

## Primary sources

- Anthropic emotions paper: transformer-circuits.pub/2026/emotions | arXiv 2604.07729
- Replication ("Where Do Models Find Happiness?"): arXiv 2606.26987
- Anthropic introspection: transformer-circuits.pub/2025/introspection
- Latent Introspection (code + samples): arXiv 2602.20031
- Content-agnostic introspection (the claim our circumplex analysis engages): arXiv 2603.05414
- Mechanisms of introspective awareness: arXiv 2603.21396
- Gemma 4 technical report: arXiv 2607.02770
- Concurrent affect-geometry work: arXiv 2604.03147, arXiv 2604.07382
- Full extended link list: see the original planning brief (2026-07-19)
