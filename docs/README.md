# docs/ — the talk, and the GitHub Pages source

The sprint's findings as an interactive page. Everything here is generated from
the notebooks and `results/`; nothing is typed by hand.

**This directory is what GitHub Pages serves** (Settings → Pages → Deploy from a
branch, `main`, folder `/docs`), so `index.html` must stay committed and must
stay here — it is the published page, not a build artifact you can gitignore.
It is self-contained: no build step, no server, opens from the filesystem.

| File | What it is |
|---|---|
| `index.html` | the talk, self-contained (no build step, no server) |
| `build.py` | rebuilds `index.html`; every figure names the notebook it came from |
| `data/` | the extracted figure inputs, so the build runs on any clone |

A PDF/PPTX deck and a silent story-trajectory clip used to live in `slides/`,
built by screenshotting this page. They were deleted on 2026-07-26: the page had
been rewritten and the deck had not, so the two disagreed, and a stale deck is
worse than no deck. `git log -- docs/slides/` has the builders if a deck is ever
wanted again — but regenerate from the current `index.html`, not from those
outputs.

## How the numbers get here

`build.py` holds one block per figure, and each block names the notebook whose
printed record it was copied from.

One consequence worth knowing when you edit: **changing a number means changing
the notebook first.** `build.py` is a transcription layer, not a computation. If
a notebook's printed record moves, update the matching block here and say so in
the commit.

## Rebuilding

```
python build.py                    # -> index.html  (reads data/)
```

## Status

A talk, not a paper: short sprint, no external review, nulls included and
labelled as such. The `Method` and `Q&A` sections carry the maths, the
pseudo-code and a symbol glossary for every measurement, so a claim can be
traced to code without leaving the page.

The three headline results are not equally settled, and section 8 says so on the
page: the plain-model circumplex has survived a falsification pass, the
chat-tuning result is measured but its gate is still owed, and the
story-following work is registered as exploratory.
