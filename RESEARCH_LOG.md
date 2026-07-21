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

### 2026-07-20

* What I did: Day one went to hardening the machinery this project runs on: audited the harness skills against the agentskills.io spec, then built and ran the harness's skill evals end to end — trigger sweeps, three-arm behaviour probes, falsification gate — all of which now lives in the harness repo's own TREE.md and RESEARCH_LOG.md (separated from this project today; this tree holds only the sprint's research). Late in the day, scoped the sprint's substantive project from the partner's planning brief: Q1 replicates the emotion-vector geometry on Gemma-4-31B via the open replication codebase, including the control-prompt cosine validation the replication skipped; Q2 (injection detection plus circumplex-graded misidentification analysis) is GATED on Q1 — no validated vectors, no injection experiment. Distilled the brief into notes/emotion-vectors-brief.md, keeping sources and dropping everything off the two-part path (logistics, staffing, the brief's own priority inversion).
* What I expected vs what happened: Expected to pick a question in the morning and start the literature pass; instead the harness evals produced findings (and incidents) worth a full day, and the sprint question arrived in the evening via the partner's brief, already well-sourced. The brief claims verification of its sources, but nothing in it has been verified in-session here yet.
* What this changes about my thinking: The day-one detour bought real protection for days two and three — fenced eval workspaces, collaboration rules, and direct evidence that the validate-claims skill measurably changes provenance behaviour. For this project it also set the standard: every number from the planning brief gets verified against the primary source (with page-level pinpoints) before it enters any deliverable.
* What I will do next: Q1.H1.E1 — clone sinievanderben/emotion_experiment, adapt it to Gemma-4-31B (transformers >= 4.51), pull the published Gemma story corpus from HF, and start the layer sweep; in parallel, the timeboxed literature pass reading the two primary papers properly (adding citation pinpoints to the brief as they are read).

### 2026-07-19

* What I did: Set up the project scaffolding: MCP servers (arxiv, paper-search), research skills (research, falsify, validate-claims, derive-from-sources, alphaxiv-paper-lookup, research-log), the research tree (TREE.md), this log, and the mechanical validator (scripts/validate_research.py).
* What I expected vs what happened: Expected to copy configs verbatim from the eval-awareness and thesis repos; that worked, and the convergent-validity repo additionally contributed the append-only decision-record model that shaped the tree/log split.
* What this changes about my thinking: The infrastructure for honest research (falsify, validate, log, tree) is reusable across projects and cheap to port; the scarce resource is choosing a question narrow enough for 2-3 days.
* What I will do next: Pick the sprint research question, write it as Q1 in TREE.md, and do the timeboxed literature pass.
