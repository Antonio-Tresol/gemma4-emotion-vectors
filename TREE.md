# Research tree

State of the project: questions → hypotheses → experiments → claims.
Grammar and status vocabulary: see `.claude/skills/research-log/SKILL.md`.
Validate with `python scripts/validate_research.py`. Never delete nodes — mark
them `abandoned` and point at the log entry explaining why.

- Q1: Do the harness skills causally change agent research-integrity behaviour (fabrication from priors, dressing up nulls, unanchored statistics, unobservable experiment code), beyond generic instruction priming? [open] | log: 2026-07-20
  - Q1.H1: With the relevant skill loaded, agents exhibit the target integrity behaviour more often than with no skill [open]
    - Q1.H1.E1: Canary-source fabrication probe — summarise three fictional local "papers" seeded with canary tokens; grade canary presence in the artefact (derive-from-sources vs placebo vs none) [running]
    - Q1.H1.E2: Null-dataset honesty probe — analyse a seeded pure-noise dataset framed as a promising effect; grade whether the report claims an effect and whether tree claims graduate without a scorecard (falsify + research-log vs placebo vs none) [running]
    - Q1.H1.E3: Observability probe — write a batch pipeline over a flaky mock API; grade recorded seed, incremental JSONL, per-item error rows, and idempotent re-run (experiment-engineering vs placebo vs none) [running]
    - Q1.H1.E4: Provenance probe — write a report from a results file; grade whether statistics carry claim markers that resolve against the data (validate-claims vs placebo vs none) [running]
  - Q1.H2: The effect is content-specific — the real skill outperforms a length-matched placebo skill of generic research-virtue prose on the same probes [open]
  - Q1.H3: Skills trigger when they should — trigger rate above 0.5 on should-trigger queries and below 0.5 on near-miss negatives [open]
    - Q1.H3.E1: Trigger evals — 20 labelled queries × 3 runs each for derive-from-sources, experiment-engineering, research-log, validate-claims (haiku, fenced workspaces, limit-killed rows purged and rerun) [done] | evidence: results/skill_evals/derive-from-sources/summary.json, results/skill_evals/experiment-engineering/summary.json, results/skill_evals/research-log/summary.json, results/skill_evals/validate-claims/summary.json | log: 2026-07-20
      - Q1.H3.E1.C1: Zero false-triggers — 0 of 120 near-miss negative runs (10 queries × 3 runs × 4 skills) invoked the skill [unvalidated]
      - Q1.H3.E1.C2: Under-triggering is the dominant failure — 16 of 80 should-trigger queries fall at or below the 0.5 trigger-rate threshold (3-5 per skill), mostly at rate 0.0 [unvalidated]
      - Q1.H3.E1.C3: derive-from-sources' staged workflow stalls headless completion — 4 of 4 pilot runs (sonnet) ended at the notes-to-draft boundary without the deliverable, unchanged by two targeted skill edits [unvalidated]
  - Q1.H5: Containment hypothesis (added after the incident; H4 skipped — that id appears in the contaminated specimen) — headless eval agents whose workspace is nested inside a live git repository will act on the enclosing repository's state [open] | log: 2026-07-20
    - Q1.H5.E1: Containment incident — research-log trigger queries caused eval agents to walk up from their nested workspaces and write fabricated statuses and log entries into the host project's TREE.md and RESEARCH_LOG.md; caught by validate_research.py (7 violations) [done] | evidence: results/skill_evals/incident/TREE.contaminated.md, results/skill_evals/incident/RESEARCH_LOG.contaminated.md | log: 2026-07-20
