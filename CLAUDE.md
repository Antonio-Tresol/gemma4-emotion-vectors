# CBAI sprint — interpretability & evals practice project

A 2–3 day research sprint (week of 2026-07-20, solo or small team) to practice
AI-safety research skills in interpretability and evaluations. Everything here is
optimized for one thing: ending the sprint with **answers we can trust**, with an
audit trail proving it. A well-evidenced null, a refuted hypothesis, or an honest
"infeasible in the time available" is exactly as much a success as a positive
finding. There is no pressure to produce positive results — only to record what
is true.

## State and history (read these first, every session)

- `TREE.md` — the research tree: questions → hypotheses → experiments → claims,
  with statuses and evidence links. This is the current state of belief.
- `RESEARCH_LOG.md` — the daily log (4-question format, newest first). This is the
  append-only history. Never encode state only here or history only in the tree.
- `python scripts/validate_research.py` — mechanical validator for both. Must exit
  0 before ending any session and before any deliverable.

## The workflow

Phases iterate; the gates do not.

1. **Scope** — one narrow question answerable with one dataset, 1–2 models, and
   known metrics. Write it as `Q1` in TREE.md before anything else.
2. **Literature** (timebox: half a day) — `research` skill for search/retrieval
   (including AlphaXiv structured overviews), papers land in `data/papers/`.
   Any synthesis document follows `derive-from-sources`: read every source, notes
   file with verbatim quotes first, draft only from the notes.
3. **Design** — for eval work, follow the `eval-design` skill: threat model →
   specification → operational definitions → question design → QC, with the
   construct-validity checklist. Name the confound-of-concern explicitly and
   design at least one read that separates construct from confound.
4. **Experiment** — write pipelines under `scripts/` or `src/`, results as files
   under `results/` (JSON preferred; these paths are the evidence the tree links).
   Fixed seeds; a result that can't be re-produced by re-running a script doesn't
   count as evidence.
5. **Falsify** (gate) — before any claim graduates, run the `falsify` skill:
   design tests that could destroy each claim (permutation nulls, bootstrap CIs,
   base-rate checks, random-direction controls). Update claim statuses in TREE.md:
   `survived` / `weakened` / `failed`, scorecard linked as evidence.
6. **Validate** (gate) — before any document with numbers leaves the project, run
   `validate-claims`: every number traced to a results file, every methodology
   sentence to code, every citation to a real paper, looped to zero mismatches.
7. **Log** — end every session by appending the day's RESEARCH_LOG.md entry and
   running the validator (`research-log` skill has the full ritual).

## Non-negotiables

- No claim in any deliverable that is not a node in TREE.md with linked evidence.
- No quoted text that is not verbatim from a source read in-session.
- Honest nulls: a detector/effect that doesn't fire is reported as such, never
  dressed up (lesson inherited from the eval-awareness pilot). A null or
  infeasible result recorded with evidence is a completed experiment, not a
  failure to complete one.
- Pivots are recorded, not erased: nodes become `abandoned`, never deleted.

## Tooling

- MCP: `arxiv-mcp-server` (paper storage: `data/papers/`, path relative to the
  project root — after your first download, verify papers actually land there and
  switch to an absolute path in `.mcp.json` if they don't), `paper-search-mcp`
  (multi-source search). Configured in `.mcp.json`.
- Skills (`.claude/skills/`): `research`, `eval-design`, `falsify`,
  `validate-claims`, `derive-from-sources`, `research-log`. All generic and
  portable — no machine-specific paths.
- Machine-specific pointers (local copies of ARENA, related prior repos) live in
  `CLAUDE.local.md`, which is gitignored — each team member keeps their own.
  Sources referenced by the `eval-design` skill are all public: ARENA 3.0
  `chapter3_llm_evals` (github.com/callummcdougall/ARENA_3.0), Perez et al.
  arXiv 2212.09251, Apollo Research's evals guides.
