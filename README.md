# Emotion vectors in Gemma 4 31B

**[Read the interactive write-up →](https://antonio-tresol.github.io/cbai-cambria-project/)**

Anthropic reported that a language model keeps a separate internal direction for
each emotion, and that those directions arrange themselves the way psychologists
arrange emotions — a *circumplex*, with pleasantness and arousal as its two axes.
We rebuilt that result on an open-weights model, in two versions of it, and then
asked something the paper did not: can the model follow an emotion that *changes*
partway through a story?

A 2–3 day research sprint. Not a paper, not peer reviewed. Null and failed
results are reported as such, and the write-up grades each finding by how far it
was actually tested.

## What we found

**The plain model reproduces the published result.** Sort 171 emotion vectors by
what separates them most — a calculation that never sees a human rating — and the
biggest axis matches human *pleasantness* ratings at 0.83. The circumplex falls
out unprompted. *(Tested and survived a falsification pass.)*

**The chat-tuned model buries it.** Take the same model after instruction tuning
and pleasantness drops from first place to third. A larger axis takes over, it
matches nothing in the plain model's top five (best pairing: 0.14), and it is not
pleasantness, arousal, dominance, or story length — we checked all four. We can
measure it and cannot say what it encodes. *(Measured; its falsification pass is
still owed.)*

**Who writes the stories decides whether any of this works.** An emotion vector is
built by averaging the model's internal state over stories written to evoke that
emotion, and nobody reports who writes them. Holding everything else fixed, story
source moves the number of working layers from 1 to 9 out of 20. Asked for 3,072
emotional stories, the model named its character Elias in 98.2% of them.

**Emotion tracking through a story is real, uneven, and small.** In stories written
to move through three emotions, the model's internal state hands over from one to
the next — often *before* the written turn. But nothing reaches 50%, the ranking
changes with depth, and a large share of the unevenness belongs to the vectors
rather than the model. *(Exploratory: registered as hypothesis-generating work. The
detection bar we pre-registered was written for the 171-emotion vector set, which
failed it at every layer; what passes is the 12-emotion sets, a substitution made
after seeing results. The write-up says so on the page.)*

## How the project is organised

| Path | What it holds |
|---|---|
| `TREE.md` | the research tree: questions → hypotheses → experiments → claims, with status and evidence links. The current state of belief. |
| `RESEARCH_LOG.md` | the daily log, newest first. The append-only history. |
| `DATA.md` | the data index: every Hugging Face dataset and how `fetch()` routes to it. |
| `src/emotion_vectors/` | the installed package — scoring conventions, data resolution, analysis and figure code. |
| `scripts/` | runnable pipelines and scorers, one file per job. |
| `results/` | the evidence tree. Small JSON is tracked; bulky arrays live on Hugging Face and `fetch()` bridges both. |
| `notebooks/` | the report, numbered in reading order, with an index in `notebooks/README.md`. |
| `docs/` | the interactive write-up (`index.html`) and the `build.py` that generates it. |
| `check.sh` | the repo gate: format, lint, tests, research-integrity validator. |

Two conventions worth knowing before editing anything:

- **`docs/index.html` is generated.** Edit `docs/build.py`; the HTML is a build
  artifact. `build.py` is a transcription layer, not a computation — a number
  changes in the notebook first and here second.
- **Notebooks are hand-maintained**, then re-executed in place. There is no
  generator script that emits a notebook, deliberately: four such builders existed
  and were deleted once the notebooks carried hand-written narrative, because
  every one of them was silently stale.

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
cd docs && python build.py
```

Run every mechanical check — formatter, linter, tests, and the research-integrity
validator that checks TREE.md and RESEARCH_LOG.md against the files they cite:

```bash
./check.sh
```

Datasets and the bulky activation arrays are published on Hugging Face and
resolved automatically by `emotion_vectors.artifacts.fetch`, which tries local
`results/` first. `DATA.md` is the human-readable index; `ROUTES` in
`src/emotion_vectors/artifacts.py` is its machine-readable twin.

## AI research automation

This project was built with heavy use of Claude Code, and the working agreements
that made that safe are checked in rather than left implicit: `AGENTS.md` is the
single source of truth for any coding agent (`CLAUDE.md` just imports it), and
`.claude/skills/` holds the workflow skills the sprint actually used — literature
search, eval design, falsification, claim validation, research logging.

The discipline that mattered most is mechanical, not cultural: predictions and
their pass marks are written into `TREE.md` *before* the data exists, claims only
graduate after a falsification pass, and `scripts/validate_research.py` checks
that every claim resolves to a file that exists. Results that failed are recorded
as failed.

## Citation

`CITATION.cff` carries the machine-readable citation, including a reference to
the work being replicated.

## Licence

MIT, for the code and the write-up in this repository — see `LICENSE`.

This does not extend to third parties' work. The papers this project reads are
not redistributed here; the model weights, the NRC VAD lexicon, and any external
corpora carry their own licences and are the responsibility of whoever uses them.

## Disclaimer

A short sprint by a small team, written up honestly rather than confidently. No
external review. Effect sizes on the story-tracking results are above chance and
below anywhere you would want to be before relying on them, and we cannot yet
separate "a weak measurement of a real thing" from "an accurate measurement of a
weak thing". One caveat we take seriously and repeat on the page: we measured the
model *reading* emotions in a story, not having them. Those are different claims.

Claude Code was used throughout, including in writing this README. Numbers are
traced to evidence files, but treat the prose with the scepticism you would apply
to any unreviewed work.

## Feedback

Corrections are welcome, particularly on claims that outrun their evidence,
citations, and methodology. Open an issue.
