# CBAI sprint — interpretability & evals practice project

Instructions for any coding agent working in this repository (Codex, Claude
Code, Cursor, Aider, and anything else reading `AGENTS.md`). `CLAUDE.md` imports
this file, so there is one source of truth rather than two that drift.

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

## Repo map (where things live, where new things go)

- `TREE.md` / `RESEARCH_LOG.md` — state and history (see above). Single
  writer per session; register experiments here BEFORE data exists.
- `DATA.md` — the data index: every HF dataset, what it holds, and how
  `fetch()` routes to it. Machine-readable twin: `ROUTES` in
  `src/emotion_vectors/artifacts.py`.
- `src/emotion_vectors/` — the installed package. Shared conventions live
  here so they cannot drift: `q3_conventions.py` (scoring), `artifacts.py`
  (data resolution), `taxonomy_report/` (notebook-11 exhibit library),
  `analysis.py`, `trajectories.py`. New notebook figure code goes in a
  module here, not inline in cells.
- `scripts/` — runnable pipelines and scorers, one file per job; generation
  recipes under `scripts/combined_story_gen/`. A NEW EXPERIMENT is: a TREE
  registration, a script here, outputs under `results/`, and an entry in
  `scripts/publish_experiment_artifacts.py` so the evidence reaches HF.
- `results/` — the evidence tree. Small JSON evidence is git-tracked; bulky
  npz/shards are gitignored and live on HF (fetch() bridges both). Never
  hand-edit anything here.
- `notebooks/` — the report, numbered in reading order with a row in
  `notebooks/README.md`; `notebooks/archive/` is the immutable bench layer.
  A new report notebook gets the next number, an index cell, key concepts,
  and a how-to-read block per figure. Report notebooks are **hand-maintained**:
  edit the notebook (and the exhibit package it imports), then re-execute with
  `.venv/bin/python -m nbconvert --to notebook --execute --inplace <notebook>`.
  Do NOT write a generator script that emits a notebook from hardcoded cell
  sources. Four such builders existed and were deleted on 2026-07-23: once the
  notebooks carried hand-written narrative and load-call-show cells, every
  builder was silently stale, and re-running one would have overwritten a
  finished notebook with its pre-extraction ancestor (`git log -- scripts/`
  has them if you ever need to look).
- `docs/` — the talk: `index.html` (interactive, self-contained) and the
  `build.py` that generates it from `docs/data/`. Numbers are transcribed from
  notebook printed records, one block per figure naming its source, so a number
  changes in the notebook first and here second. `index.html` is generated —
  edit `build.py`, never the HTML. A `slides/` directory (PDF, PPTX, a story
  clip, and their builders) was deleted on 2026-07-26 after a rewrite of the
  page left the deck stale; `git log -- docs/slides/` has it. If a deck is ever
  wanted again, regenerate from the current page rather than reviving those
  outputs.
- `data/papers/` (literature PDFs), `data/lexicons/` (third-party, manual
  download), `notes/` (drafts and registration drafts).
- `check.sh` — the repo gate (format, lint, tests, validator); the shared
  pre-commit hook runs it.

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

## Collaboration and parallelism

- **Branches and PRs between humans.** Direct commits to `main` are for solo
  work only. When more than one person is on the project, work happens on
  short-lived branches merged to `main` by PR; a PR that adds or changes
  results, claims, or documents with numbers runs `./check.sh` and the
  `validate-claims` gate before merge. `main` is always green: validator exit 0,
  all checks passing.
- **Worktrees between parallel sessions.** Two agent sessions in one clone will
  fight over TREE.md, RESEARCH_LOG.md, and `results/`. Run parallel sessions in
  separate git worktrees (`git worktree add ../<name> <branch>`; Claude Code can
  create one for a session with EnterWorktree), one branch per worktree, merged
  back like any other branch.
- **Be an orchestrator.** For work that fans out — sweeps, literature searches,
  reviews, independent experiments — delegate to subagents or an agent team and
  keep synthesis in the orchestrating session. Three rules learned the hard way
  (see the 2026-07-20 containment incident, Q1.H5): give subagents
  self-contained prompts (they do not see your conversation); give any subagent
  that executes untrusted or generated work a workspace *outside* the
  repository; and keep a **single writer** for TREE.md and RESEARCH_LOG.md —
  subagents report findings back, the orchestrator records them.

## Code and notebook conventions

Distilled from working practice (and the working-agreements style of
codeberg.org/haplesshero13/rosetta-stone); the validator does not check these,
reviewers do:

- **No machine-local absolute paths in committed code.** Path resolution lives
  in the package (`emotion_vectors.artifacts.fetch` resolves local `results/`
  then HF); a notebook must run unchanged on any clone.
- **Notebook cells are load-call-show.** Analysis and figure code is written
  as importable functions under `src/`, so it can be tested by importing and
  calling with parameters; notebooks import, call, and narrate.
- **Every figure is self-explanatory**: axis titles with units, legends for
  every color encoding, titled colorbars, chance/zero reference lines, a layer
  slider on per-layer measures, and a collapsible "How to read" block after it.
- **Start with common words**; define each technical term before first use,
  and never state a number in prose that is not computed in the cell beside it.
- **Fail loudly.** No silent fallbacks or defaults; a missing input is an
  explicit error or an explicit printed degradation, never a quiet skip.
- **Readable code.** Descriptive names (no one-letter or cryptic
  abbreviations), a short intention comment above every non-obvious block,
  docstrings naming inputs/outputs, and named tensor axes (einops/jaxtyping)
  everywhere an array changes shape.
- **Interpretation lives in the exhibit, not in chat.** Every report section
  ends with what the result means, the live hypotheses (labeled as
  hypotheses) each paired with the experiment that would decide it, and the
  open questions. If an explanation was good enough to give a collaborator
  in conversation, it belongs in the notebook before they have to ask.
- **Every figure carries its own grading scale.** A reader must be able to
  judge the result from the figure alone: labeled anchors for what FAILURE
  looks like (chance level, zero effect, the noise floor) and what a STRONG
  result would look like (the registered bar where one exists; otherwise a
  meaningful comparator such as the other arm or bank), drawn as reference
  lines/bands in the plot where possible. The how-to-read block then says it
  in words: "a good result here would be X, a bad one Y, and the observed
  pattern sits at Z between them."
- **HTML is welcome where it reads better.** Notebook markdown may use HTML
  freely when it is clearer than plain markdown: collapsible
  `<details>` blocks, styled tables, side-by-side layouts. Clarity decides,
  not purity.
- **Link everything that has a URL.** Every dataset, paper, model card,
  external tool, or repo mentioned in a notebook is a hyperlink at first
  mention: HF datasets link to their dataset page, papers to arXiv,
  the lexicon to its source page. A reader reaches any referenced artifact
  in one click, without hunting through DATA.md (DATA.md still holds the
  full index).
- **The acceptance test for every figure and section:** a teammate who did
  NOT run the experiment can say, from the exhibit alone, (1) what was
  measured and how, (2) what the figure claims, (3) which evidence file the
  numbers come from, and (4) what would change their mind - without asking
  the author anything. The point of the notebooks is that the team can
  verify the analyses; a plot that needs its author present has failed.

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
- **Shared session configuration** (checked in, applies to everyone on clone):
  - `.claude/settings.json` — compaction forced at 50% context
    (`CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`; effective on Opus 4.8 and most models,
    documented as having *no effect on Sonnet 5*, unverified on Fable), agent
    teams enabled (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`), and Opus as the
    advisor model (Fable is the intended advisor once its advisor rollout
    completes — currently unselectable per docs; revisit cost before flipping,
    since advisor calls bill at the advisor model's rates). Teams and
    advisor are experimental; advisor needs the
    Anthropic API. Personal opt-outs go in the gitignored
    `.claude/settings.local.json`, which overrides project settings.
  - `.codex/config.toml` — Codex auto-compaction at ~50% of a 400k window
    (`model_auto_compact_token_limit = 200000`; Codex takes tokens, not
    percentages — adjust if the default model's window differs). Loads only
    after you mark the project trusted, and note it *outranks* your personal
    `~/.codex/config.toml`. Codex subagents are on by default; shareable custom
    agent roles can be added under `.codex/agents/` if the project needs them.
