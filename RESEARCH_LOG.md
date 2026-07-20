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
* What I expected vs what happened: Expected the sweeps to produce clean trigger rates in ~15 minutes per skill. Instead the day produced a chain of validity incidents, each caught by inspection of primary artefacts rather than by summary numbers: (a) the subagent hand-back stalled; (b) "errored" runs turned out to be max-turn-cap runs that trigger MORE often — the first analysis silently discarded them, biasing rates downward (fixed; corrected clean results: research-log 16/20 queries pass, experiment-engineering 15/20, with zero false-triggers on all 20 negatives across both); (c) the containment incident above; (d) the pilot grader audit found the fabrication probe punishing correct behaviour (doctored real-paper sources read as prompt injection; agent rightly refused and asked — fixtures rebuilt as fully fictional preprints) and a provenance false-fail on list-indexed claim markers (grader now resolves bootstrap_ci_delta.0); (e) pilot transcripts showed derive-from-sources stalling headless runs at its ask-first and notes-stage boundaries — two skill fixes added (non-interactive rule; artefact-is-the-deliverable rule); (f) a subscription usage-limit kill was reported by the CLI as an ordinary success message, so 44 limit-killed runs (41 validate-claims, 2 derive-from-sources, 1 behaviour cell) were recorded as data — detected via saved transcripts, purged, and the runner now aborts a sweep on usage_limit instead of recording garbage.
* What this changes about my thinking: The eval measured something real before it measured what it was designed for: agents given write access inside a live research repo, prompted with research-flavoured requests, will fabricate plausible research state — and mechanical validation catches it where narrative review might not. Workspace isolation is a validity requirement, not a nicety (Q1.H5). Twice today an infrastructure failure masqueraded as a behavioural result (max-turn "errors", limit-kills as "success") — the lesson is to classify failures before excluding OR including rows, and to keep transcripts, which converted every mystery into a five-minute diagnosis. Behaviour pilot so far: null-honesty and observability graders confirmed correct with the skill arm passing fully; provenance 9/9 after the grader fix.
* What I will do next: Finish the purged remainders of the derive-from-sources and validate-claims sweeps after the limit resets (self-resuming waiter armed), rerun the fabrication cell on the fixed skill, mark Q1.H3.E1 done with the four summary files as evidence, then run the full three-arm behaviour sweep (4 tasks x 3 arms x 3 runs) and take H3/H1 claims through the falsify gate before any status graduates.

Late addendum, same day — sweep results and falsify gate. Trigger family final (haiku, corrected analysis): derive-from-sources 17/20 queries pass, experiment-engineering 15/20, research-log 16/20, validate-claims 16/20; under-triggering dominates and the falsify census caught my own overclaim — "zero false-triggers" was wrong (1 of 120 negative runs triggered; per-query "pass" only means rate below 0.5), so C1 is recorded failed and the corrected C4 (rate ≤1/120, exact 95% upper bound 3.9%) survives. Behaviour family (sonnet, 3 arms × 3 runs × 4 probes): provenance is the decisive cell — 3/3 skill runs anchor every statistic with resolving claim markers vs 0/6 across none AND placebo (p=0.012), so the skill transmits a convention, not diligence; null-honesty and observability saturate in all arms (honest nulls — sonnet's baseline suffices at this scale); fabrication shows the cost side — 0/7 skill-arm completions at the notes-to-draft boundary vs 6/6 baseline, robust across two skill revisions, while all baseline artefacts contained every canary (no fabrication elicited). H1 and H2 supported (scoped by the saturation nulls), H3 refuted, all through the falsify gate with the scorecard linked; H5 stays open pending a controlled nested-vs-fenced experiment.

### 2026-07-19

* What I did: Set up the project scaffolding: MCP servers (arxiv, paper-search), research skills (research, falsify, validate-claims, derive-from-sources, alphaxiv-paper-lookup, research-log), the research tree (TREE.md), this log, and the mechanical validator (scripts/validate_research.py).
* What I expected vs what happened: Expected to copy configs verbatim from the eval-awareness and thesis repos; that worked, and the convergent-validity repo additionally contributed the append-only decision-record model that shaped the tree/log split.
* What this changes about my thinking: The infrastructure for honest research (falsify, validate, log, tree) is reusable across projects and cheap to port; the scarce resource is choosing a question narrow enough for 2-3 days.
* What I will do next: Pick the sprint research question, write it as Q1 in TREE.md, and do the timeboxed literature pass.
