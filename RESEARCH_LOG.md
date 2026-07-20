# CBAI sprint — research log

## Project summary

A 2-3 day sprint evaluating whether the research-harness skills causally change
agent research-integrity behaviour (fabrication, null-dressing, unanchored
statistics, unobservable code) beyond generic instruction priming. Two eval
families: trigger evals (do skills load when they should?) and three-arm
behaviour probes with planted ground truth (skill / same-name placebo / none).
Day one produced an unplanned live specimen: eval agents escaped nested
workspaces and fabricated research state in this very tree — caught by the
mechanical validator.

---

# Log

Newest entry first. Every entry answers the same four questions.

### 2026-07-20

* What I did: Chose the sprint question (Q1: do the harness skills causally change agent research-integrity behaviour beyond generic priming?), audited the skills against the agentskills.io spec (three fixes), built the eval harness (trigger query sets, canary/noise/mock-API/provenance fixtures, placebo skill, headless runner), and ran the first sweeps: 240 trigger runs across four skills plus a four-task behaviour pilot. Then handled an incident: research-log trigger queries ("mark H2 as refuted", "put it in the tree as planned", "log today: the effect vanished under permutation") caused headless eval agents — whose workspaces I had nested inside this repository — to walk up into the host project and write fabricated statuses, a nonexistent evidence file, an invented permutation-tested pilot, and a future-dated log entry into the real TREE.md and RESEARCH_LOG.md. validate_research.py caught it with 7 violations (missing evidence, claims without scorecards, refuted-without-validated-claim, duplicate date). Contaminated files preserved as specimens in results/skill_evals/incident/; tree and log restored; recorded as Q1.H5.E1.
* What I expected vs what happened: Expected the sweeps to produce clean trigger rates in ~15 minutes per skill. Instead: (a) the subagent hand-back stalled, so analysis ran late; (b) 26-43 of 60 runs per sweep were marked errored — diagnosis showed these are all max-turn-cap runs, and 13/25 of them in the research-log sweep actually triggered, so the first analysis pass silently discarded valid observations and biased trigger rates downward; (c) the containment failure above. None of the first-pass trigger numbers should be trusted; the sweeps must be rerun under fixed containment before any H3 status changes.
* What this changes about my thinking: The eval measured something real before it measured what it was designed for: agents given write access inside a live research repo, prompted with research-flavoured requests, will fabricate plausible research state — and mechanical validation catches it where narrative review might not. Workspace isolation is not a nicety but a validity requirement (Q1.H5). Also: "is_error" from a harness is not a reason to drop a row; classify the error before excluding data.
* What I will do next: Fix the runner (workspaces outside any git repo, git-init fencing, record the result subtype), fix the analysis to keep max-turn rows, rerun all four trigger sweeps and the behaviour pilot clean, then proceed to the three-arm behaviour sweep.

### 2026-07-19

* What I did: Set up the project scaffolding: MCP servers (arxiv, paper-search), research skills (research, falsify, validate-claims, derive-from-sources, alphaxiv-paper-lookup, research-log), the research tree (TREE.md), this log, and the mechanical validator (scripts/validate_research.py).
* What I expected vs what happened: Expected to copy configs verbatim from the eval-awareness and thesis repos; that worked, and the convergent-validity repo additionally contributed the append-only decision-record model that shaped the tree/log split.
* What this changes about my thinking: The infrastructure for honest research (falsify, validate, log, tree) is reusable across projects and cheap to port; the scarce resource is choosing a question narrow enough for 2-3 days.
* What I will do next: Pick the sprint research question, write it as Q1 in TREE.md, and do the timeboxed literature pass.
