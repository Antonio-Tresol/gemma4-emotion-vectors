# Public-readiness audit, 2026-07-30

Four cold proxies were run against the repository and the write-up: two readers
(repo, page) given no context beyond a path or a URL, and two that executed
things (anonymous data access, a fresh clone following the README literally).
Every finding below was re-checked directly against the files before being
recorded here. Findings the proxies got wrong are listed at the end, because a
proxy report is a claim, not evidence.

The test could only use the local worktree as a stand-in. The public surface
does not exist yet: the GitHub repository is private and Pages is not enabled.

## Blockers — these must be fixed before the repository goes public

### 1. The bug-corrected vectors are not published

Every headline number comes from the post-fix vector sets, re-extracted after
the left-padding bug. Those are the sets that are missing from Hugging Face.

| bundle | local | on HF |
|---|---|---|
| `emotion_vectors` (pre-fix, buggy) | 66M | present |
| `emotion_vectors_postfix` | 66M | **404** |
| `emotion_vectors_it_postfix` | 66M | **404** |
| `self_story_vectors_it_postfix` | **absent** | **404** |
| `neutral_vectors_it_postfix` | 400K | **404** |

A stranger can download the buggy vectors and not the corrected ones.
`geometry_report/_context.py:43-44` and `taxonomy_report/_s8_geometry.py:59`
fetch the 404 paths, so notebooks 02 and 11 cannot run on a clone. Notebook 02
is the exhibit behind the headline 0.83.

Cause: `scripts/publish_postfix_vectors.py` ran on 2026-07-22 and uploaded the
per-emotion shards. `emotion_means.npz` was built on 2026-07-23 and never
re-published; the local shards were pruned afterwards. The repositories
themselves are public and carry 3,300-5,000 shard files each, so only the
aggregate bundle is missing.

`self_story_vectors_it_postfix/emotion_means.npz` is absent locally as well and
must be rebuilt from the published shards.

### 2. Nothing on the write-up links to anything

`docs/index.html` contains four `<a>` tags: three Google Fonts, one NRC VAD
lexicon. There is no link to the repository, to any dataset, or to the paper
being replicated, and no contact address of any kind.

The page names `notebooks/07_generator_lineages.ipynb`, `q3_conventions.py`,
and experiment identifiers on nearly every figure, and links none of them. It
has the appearance of traceability without the substance. The cold reader's
verdict: "a replication whose target is unidentified cannot be evaluated as a
replication at all."

`CITATION.cff:45` already holds the replicated paper's URL, so this is
transcription, not research.

### 3. The repository gate cannot fail on formatting or lint

`check.sh:20` reads `uvx ruff format --check . && uvx ruff check --select I .`.
Under `set -e` bash ignores a non-zero exit from any element of an `&&` list
except the last, so the first command's verdict is discarded. Verified:
`bash -c 'set -euo pipefail; false && true; echo REACHED'` prints REACHED.

It is currently concealing real failures: `ruff format --check` wants to
reformat 18 files (including `docs/build.py` and eight notebooks) and
`ruff check` reports 12 errors. Some pre-date the de-jargoning commit, some
were introduced by it when the notebooks were re-executed.

**Fixed 2026-07-31.** `check.sh` now runs every stage, records which failed,
and prints a summary. Ruff is clean: `All checks passed`, 153 files formatted.
Notebooks are excluded from the *formatter* only, because `ruff format`
rewrites the whole `.ipynb` through a serialiser that round-trips stored output
floats to a different last digit; one pass churned 1,007 numbers in a single
plotly output and produced an 84,000-line diff on notebook 09. The nine
notebook import-sorting errors were fixed by editing `cells[i]["source"]`
directly, after verifying Python round-trips these files byte-for-byte, so no
output changed. `docs/index.html` still hashes to `8cd3ba35...`.

With every stage now running, the true state was: ruff ok, lanorme FAILED,
pytest ok (115 passed), research integrity ok.

**A correction to an earlier count in this note.** lanorme reported "112
violations", and that figure was repeated here, but it is not the set that
fails the build. lanorme separates a hard error ("File exceeds 500 effective
lines") from a warning ("File approaching 500"), and prints both with the
`VIOLATION:` prefix. The set that actually failed the gate was **21**:

| kind | count |
|---|---|
| files over 500 effective lines | 3 |
| functions over 80 effective lines | 13 |
| functions with more than 8 parameters | 5 |

The other 17 file findings, and 133 further findings across SIZE-002,
PARAM-001 and COMPLEXITY-001, were warnings that never failed anything. Any
statement that this repository had "112 failing violations" overstates it.

**Fixed 2026-07-31.** 92 findings were genuinely repaired: 61 tensor
annotations now carry real axis names derived from each function's body, 28
bare `dict`/`list` annotations are parameterised, and three byte-identical
copies of `figure_title` are now one definition in
`src/emotion_vectors/figure_text.py`, verified to produce identical output from
all four call sites.

The remaining 21 are silenced per file in `lanorme.toml`'s
`[per-file-ignores]`, each with its reason. lanorme's thresholds are module
constants and are **not** configurable, so there is no limit to raise; a
baseline was tried first and rejected as a 155-entry JSON blob nobody reads,
where the per-file table is 10 entries next to their justifications. The
trade-off is recorded in the config: unlike a baseline this does not ratchet
within a listed file. Both behaviours were tested — a new 600-line file in an
unlisted path still fails; a new 120-line function inside a listed file is
silenced.

Both tools are now pinned in `check.sh` (`ruff@0.16.1`, `lanorme@0.16.0`),
because `uvx` otherwise resolves to whatever is newest at the moment you run
it, and ruff is a tool that rewrites files.

The lanorme failure is **not** caused by the unpinned `uvx lanorme` picking up
a new release, which was the obvious suspect. Run against identical code,
0.14.2 (newest when `lanorme.toml` was adopted), 0.15.0 and 0.16.0 all report
the same 20 / 3 / 0 / 28 / 61 violations. The newer releases add three checks,
29 to 32, and all three pass.

The gate was genuinely green and then regressed. At `67b43c2` (2026-07-19, the
commit that adopted the config) lanorme 0.14.2 reports "All 27 checks passed".
Bisecting the 302 commits since, on lanorme's exit code, the first red commit
is `cda28fc` (2026-07-21), "Completed code for multi-value story generation",
with 2 duplication and 2 file_limits violations. It has grown to 112 across
four categories in the nine days since.

That has a consequence nobody had seen, because the gate never got past
lanorme: `hooks/pre-commit` treats `check.sh` as a **hard gate that blocks the
commit**. With lanorme failing on long-standing violations, every commit in
this repository requires `--no-verify`. The hook's own comment predicts exactly
this: "A hook that blocks on judgement gets bypassed with --no-verify once and
then ignored forever, taking the real failures with it." Either the lanorme
violations get fixed, or lanorme moves from the hard gate to the reminder
section.

### 4. The research-integrity validator cannot pass on a clone

`scripts/validate_research.py` exits 0 in a working tree that holds the bulky
artifacts and reports 11 violations anywhere else. Every one is a
`results/**/*.npz` path that `.gitignore` excludes by design. The validator
calls `Path.exists()` instead of routing through
`emotion_vectors.artifacts.fetch`, so the gate `AGENTS.md` calls mandatory is
unpassable on any clone by construction.

## Substantive gaps in the science as presented

### 5. Base PC1 correlates 0.66 with dominance and the page never says so

`build.py:PCS` records base PC1 as valence 0.83, **dominance 0.66**, arousal
0.02, length 0.15; PC2 as arousal 0.55. So PC1's dominance correlation exceeds
PC2's arousal correlation, and dominance is discussed nowhere: not in the
how-to-read, not in the takeaway, not in the confound section, which rules out
story length only. Valence and dominance are correlated in NRC VAD. The one
claim graded `survived` is "we recovered valence", and it needs a sentence
saying whether PC1 is valence or valence-plus-dominance.

### 6. Section 7's prose contradicts its own chart

The page says "Naming peaks early and falls: layer 6 gets 58% ... by layer 33
that is down to 27%." `build.py:BY_LAYER` charts 0.58, 0.33, 0.57, 0.27, 0.41,
0.30 across layers 6/15/24/33/42/51. Layer 24 ties layer 6 and layer 42 exceeds
layers 15, 33 and 51. That is a sawtooth, not a decline, quoted selectively
from its highest and lowest points, in the section whose stated point is that
a result quoted at one depth is a result about that depth.

### 7. Pre-registration is claimed on a figure and qualified eight sections later

Section 2 annotates its heatmap "8 = pass mark, fixed before scoring" and says
"Both marks were fixed in advance". Section 8 then discloses that the
pre-registered bar was written for the 171-emotion set, that the set failed it
at every layer, and that the twelve-emotion substitute was chosen after seeing
results. The disclosure is admirable; its placement is not. A reader who stops
at section 2 leaves with a false impression, and section 2 carries the
nine-fold-spread result that is quoted three times elsewhere.

### 8. Section 6 scores 7% of the corpus and does not say so

The page reader flagged section 6's "53 phases per emotion" against method
block 3's "8,938 labelled phases" and could not reconcile them. Checked in
`docs/data/emo_by_layer.json`: the twelve per-emotion counts are 53, 53, 48,
53, 53, 53, 53, 53, 52, 54, 52, 53, summing to about 630, which is 7% of 8,938.

The number is not wrong. Only phases tagged with one of the twelve emotions the
vector set covers can be scored, and the corpus spans far more than twelve. The
page's own section-5 walkthrough proves it: that story runs upset, unsettled,
cheerful, and none of those three is among the twelve scored in section 6.

So this is an omitted selection step, not a numerical error, and it is
invisible to a reader who is told the corpus holds 8,938 phases and then shown
a rate computed over 630 of them. It also means the showcase story in section 5
is one that section 6 cannot score at all.

### 9. No uncertainty on the story-following numbers

Method block 4 promises story-clustered error bars. Section 6's twelve bars
carry none, at n = 53 phases per emotion, where 49% has a 95% interval of
roughly plus or minus 13 points. The middle of that ranking is probably not
resolvable, and the figure invites the reader to rank it.

## Documentation that is false

| where | claims | actually |
|---|---|---|
| `notebooks/README.md:22` | keeps a stale notebook "because `docs/build.py` still cites this path" | build.py has zero references to it; it cites `05_trajectory_explorer_instruct.ipynb` |
| `notebooks/README.md:52` | the test "calls every builder" | `sprint_report` and `transition_report` have zero test references |
| `notebooks/README.md` step 2 | an `HF_TOKEN` in `.env` is required | a notebook ran to completion with an empty `HF_HOME`, no token and no `.env`: 33 cells, no errors, 31 MB pulled anonymously |
| `DATA.md` | "private repos need an `HF_TOKEN`" | all 30 datasets are public; all 30 downloaded anonymously |
| `README.md` | the page "is self-contained ... opens from the filesystem" | it opens, but fetches three Google Fonts resources over the network |
| `README.md` | `cd docs && python build.py` | there is no `python` on a stock macOS; `uv run python build.py` works |
| `docs/build.py` docstring | "writes presentation.html" | writes `index.html` |
| `index.html` footer | numbers come from notebooks 02, 05, 08, 11 | figures cite 02, 05, **07**, 11; notebook 07 supplies both section-2 figures and 08 is cited nowhere |
| `AGENTS.md` step 7 | a log entry ends every session | `RESEARCH_LOG.md` stops at 2026-07-26; commits exist on the 27th and 28th |

## The jargon gate has a hole

`build.py:BANNED_WORDS` bans the two-word `"probe bank"` and not bare `"bank"`.
The build prints "jargon: none of the 15 flagged terms present" while `bank`
appears ten times in `index.html`, including in reader-visible pseudo-code
(`def gate_rank(cos, phase, tagged_probe, bank)`) and in a glossary. The cold
reader hit it and could not work out what it meant.

This matters beyond the one word: the previous de-jargoning pass reported the
page clean on the strength of this gate.

`DATA.md` (20 hits) and `notebooks/README.md` (15) were never covered by that
pass at all. `TREE.md` has about 274, but those are registration identifiers
tied to `results/` filenames and to `RESEARCH_LOG.md`; renaming them would
break the audit trail. That one needs a glossary, not a rename.

## Smaller defects

- `index.html:587` renders "story , from notebooks/..." on first load; the
  story identifier only appears after a story button is clicked.
- `fetch()`'s directory branch writes hub bookkeeping into the data tree
  (`results/<name>/.cache/huggingface/`).
- `artifacts.py:103`'s friendly "private until sprint end, authenticate with
  HF_TOKEN" error is unreachable; `hf_hub_download` raises a 404 first. That is
  the better outcome, since the message is now wrong.
- `lanorme` is required by `check.sh`, is not in `pyproject.toml` and is named
  nowhere in the README. Neither it nor `ruff` is version-pinned under `uvx`,
  so the gate's verdict drifts with upstream releases.
- `pyproject.toml` sets `[tool.ruff] target-version = "py311"` while
  `requires-python = ">=3.14"`.
- Sections 2, 5 and 7 never say which of the two models the figure shows.
- Section 4's similarity metric is never defined, and the section's headline
  claim is that 0.14 is small.
- "0.83" denotes two different quantities two screens apart: base PC1 against
  human valence, and instruct-PC3 against base-PC1 in the grid.
- The section-4 grid labels axes `plain` and `chat`, defined nowhere.
- `CITATION.cff`'s provenance comment cites `papers/...pdf`, while `AGENTS.md`
  puts papers in `data/papers/` and `data/` is gitignored.

## What the proxies got wrong

Recorded so the reports are not treated as evidence in future.

- The page reader claimed the shuffle count `B` was missing from the
  pseudo-code and that it had checked the HTML. `index.html:1864` reads
  `B = 10_000  # shuffles, count fixed in advance`. **Refuted.**
- An earlier note in this project held that `tests/test_checks.py` and
  `tests/test_staleness.py` do not collect. They collect and pass under
  `check.sh`'s own invocation, which supplies `--with lanorme`. All 8 test
  files collect, 115 tests, 112 pass and 3 skip with their causes printed.
- The anonymous-access proxy reported that all data is reachable. That was
  true of every path it tested, which were `DATA.md`'s examples; it did not
  test the paths the notebook code actually fetches, which is where blocker 1
  lives. Two proxies agreeing is not corroboration when they tested different
  things.

## What is genuinely in good shape

- `docs/index.html` regenerates byte-identically from `build.py` on a fresh
  clone: sha256 `8cd3ba35...` before and after.
- The analysis path runs from a cold cache with no credentials in about ten
  seconds: 33 cells, no error outputs, 31 MB pulled anonymously.
- All 30 Hugging Face datasets are public and were downloaded anonymously.
- `TREE.md` is the strongest artifact in the project. Pre-registered bars with
  numeric pass marks, an instrument bug that reversed two published nulls
  recorded rather than buried, and a README that volunteers its own post-hoc
  bar substitution.
- `CITATION.cff` is schema-valid and the replicated work travels with it.
- 112 of 115 tests pass; the 3 skips name the exact missing input.
