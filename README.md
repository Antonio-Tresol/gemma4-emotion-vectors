# Emotion vectors in Gemma 4 31B

**[Read the interactive write-up →](https://antonio-tresol.github.io/gemma4-emotion-vectors/)**

Anthropic reported that a language model keeps a separate internal direction for
each emotion. Those directions arrange themselves the way psychologists arrange
emotions, on a *circumplex*. Its two axes are valence (how pleasant the emotion
is) and arousal (how worked-up it is). We rebuilt that result on an open-weights
model, in two versions of it. We then asked something the paper did not: can the
model follow an emotion that *changes* partway through a story?

A 2–3 day research sprint. Not a paper, not peer reviewed. We report null and
failed results as such, and the write-up grades each finding by how far it was
tested.

## What we found

**The base model reproduces the published result.** An emotion vector averages the
model's internal state over stories written to evoke one emotion. Sort 171 of
them by what separates them most, a calculation that never sees a human rating.
Scores on the biggest axis correlate with published human valence ratings at an
absolute Pearson correlation of 0.83. The circumplex falls out unprompted.
*(Tested and survived a falsification pass: a deliberate attempt to destroy the
result. We shuffled the emotion labels, put bootstrap error bars on the
correlation, and dropped the five most extreme emotions on valence.)*

**The instruction-tuned model buries it.** Take the same model after instruction
tuning and valence drops from first place to third. A larger axis takes over. It
matches nothing in the base model's top five: its largest absolute correlation
against any of those five axes is 0.14. It is not valence, arousal, dominance, or
story length, and we checked all four. We can measure it and cannot say what it
encodes. *(Measured; its falsification pass is still owed.)*

**Who writes the stories decides whether any of this works.** We build each emotion
vector by averaging the model's internal state over stories written to evoke that
emotion, and nobody reports who writes them. A working layer is one where the
vectors clear a detection bar we fixed before scoring. Holding everything else
fixed, story source moves the number of working layers from 1 to 9 out of 20.
Asked for 3,072 emotional stories, the model named its character Elias in 98.2%
of them.

**Emotion tracking through a story is real, uneven, and small.** In stories written
to move through three emotions, the model's internal state hands over from one to
the next, often *before* the written turn. But nothing reaches 50%, the share of
story phases where the correct emotion's own vector ranks first out of twelve.
Chance is 1 in 12, about 8%. The ranking changes with depth, and a large share of
the unevenness belongs to the vectors rather than the model. *(Exploratory:
registered as hypothesis-generating work. The detection bar we pre-registered was
written for the 171-emotion vector set, which failed it at every layer. What
passes is the 12-emotion sets, and we substituted those after seeing the results.
The write-up says so on the page.)*

## How the project is organised

| Path | What it holds |
|---|---|
| `TREE.md` | the research tree: questions → hypotheses → experiments → claims, with status and evidence links. The current state of belief. |
| `RESEARCH_LOG.md` | the daily log, newest first. The append-only history. |
| `DATA.md` | the data index: every Hugging Face dataset and how `fetch()` routes to it. |
| `src/emotion_vectors/` | the installed package: scoring conventions, data resolution, analysis and figure code. |
| `scripts/` | runnable pipelines and scorers, one file per job. |
| `results/` | the evidence tree. Small JSON is tracked; bulky arrays live on Hugging Face and `fetch()` bridges both. |
| `notebooks/` | the report, numbered in reading order, with an index in `notebooks/README.md`. |
| `docs/` | the interactive write-up (`index.html`) and the `build.py` that generates it. |
| `check.sh` | the repo gate: format, lint, tests, research-integrity validator. |

Two conventions worth knowing before editing anything:

- **`docs/index.html` is generated.** Edit `docs/build.py`; the HTML is a build
  artifact. `build.py` is a transcription layer, not a computation. A number
  changes in the notebook first and here second.
- **Notebooks are hand-maintained**, then re-executed in place. There is no
  generator script that emits a notebook, deliberately. Four such builders
  existed, and we deleted them once the notebooks carried hand-written narrative,
  because every one was silently stale.

## Reproducing

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The analysis and figures run from committed evidence files; the heavy extraction
does not, and needs a GPU box:

```bash
uv sync --extra gpu
```

Rebuild the write-up (reads `docs/data/`, writes `docs/index.html`):

```bash
cd docs && uv run python build.py
```

`docs/` is also the GitHub Pages source (Settings → Pages → Deploy from a branch,
`main`, folder `/docs`), so committing a rebuilt `index.html` publishes it. A
reader needs no build step and no server: the file opens straight from the
filesystem. Its only external requests are for three Google Fonts resources,
so offline it renders in fallback typefaces.

One command runs the formatter, the linter, the tests, and the research-integrity
validator, which checks TREE.md and RESEARCH_LOG.md against the files they cite:

```bash
./check.sh
```

Datasets and the bulky activation arrays are published on Hugging Face and
resolved automatically by `emotion_vectors.artifacts.fetch`, which tries local
`results/` first. `DATA.md` is the human-readable index; `ROUTES` in
`src/emotion_vectors/artifacts.py` is its machine-readable twin.

## AI research automation

We built this project with heavy use of Claude Code, and we checked in the
working agreements that made that safe rather than leaving them implicit.
`AGENTS.md` is the single source of truth for any coding agent, and `CLAUDE.md`
imports it. `.claude/skills/` holds the workflow skills the sprint used:
literature search, eval design, falsification, claim validation, and research
logging.

The discipline that mattered most is mechanical, not cultural. We write
predictions and their pass marks into `TREE.md` *before* the data exists, and a
claim only graduates after a falsification pass. `scripts/validate_research.py`
then checks that every claim resolves to a file that exists. We record failed
results as failed.

## Citation

`CITATION.cff` carries the machine-readable citation, including a reference to
the work being replicated.

## Licence

MIT, for the code and the write-up in this repository. See `LICENSE`.

This does not extend to third parties' work. We do not redistribute the papers
this project reads. The model weights, the NRC VAD lexicon, and any external
corpora carry their own licences. Whoever uses them must follow those licences.

## Disclaimer

A short sprint by a small team, written up honestly rather than confidently. No
external review. Effect sizes on the story-tracking results are above chance and
below anywhere you would want to be before relying on them. We cannot yet
separate "a weak measurement of a real thing" from "an accurate measurement of a
weak thing". One caveat we take seriously and repeat on the page: we measured the
model *reading* emotions in a story, not having them. Those are different claims.

We used Claude Code throughout, including in writing this README. Numbers are
traced to evidence files, but treat the prose with the scepticism you would apply
to any unreviewed work.

## Feedback

Corrections are welcome, particularly on claims that outrun their evidence,
citations, and methodology. Open an issue.
