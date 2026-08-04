# docs/: the talk, and the GitHub Pages source

The sprint's findings as an interactive page. Everything here is generated from
the notebooks and `results/`; nothing is typed by hand.

**This directory is what GitHub Pages serves** (Settings → Pages → Deploy from a
branch, `main`, folder `/docs`). So `index.html` must stay committed, and it must
stay here. It is the published page, not a build artifact you can gitignore.
A reader needs no build step and no server: it opens from the filesystem.
Its only external requests are three Google Fonts resources.

| File | What it is |
|---|---|
| `index.html` | the talk; opens from the filesystem, no build step, no server |
| `build.py` | rebuilds `index.html`; every figure names the notebook it came from |
| `data/` | the extracted figure inputs, so the build runs on any clone |

A PDF/PPTX deck and a silent story-trajectory clip used to live in `slides/`,
built by screenshotting this page. They were deleted on 2026-07-26: the page had
been rewritten and the deck had not, so the two disagreed, and a stale deck is
worse than no deck. `git log -- docs/slides/` has the builders if a deck is ever
wanted again. Regenerate from the current `index.html`, not from those
outputs.

## The components, and when to use which

Six devices set text apart on this page. Each has one job. `build.py` enforces
the parts of this that a script can judge; the rest is on whoever edits.

| Device | Job | Where it may appear |
|---|---|---|
| `.card` | holds **one figure and its four supports**: the generated title, the key to its marks, the how-to block, the source line | around every chart, and nothing else except the three key-result summaries in the header |
| `details.howto` | how to **judge** the figure: what a failure looks like, what a strong result looks like, where the observed value sits | inside every result figure's card |
| `.legend` | how to **decode** the figure: what each colour and mark means | inside the card, unless the key is drawn in the plot itself |
| `.src` | which notebook or script the numbers came from | inside every result figure's card |
| `.callout` | a caveat that **cuts against** the surrounding text | anywhere, sparingly |
| `.takeaway` | the section's verdict | once per section, at its end |

Everything else is prose. Parallel items are a list: `<ul class="reasons">` when
the order does not matter, `<ol class="steps">` when the text claims one.

## Headings

Measured against the two Transformer Circuits papers this one replicates and
cites: 99 headings in the first, 81 in the second, mean length 6 and 5 words,
and between them three questions. The form is a **declarative claim** ("Emotion
vectors activate in expected contexts") or a **noun phrase naming a step or an
object** ("The geometry of emotion space", "Dataset construction").

So, for both section headings and sub-headings here:

- **State the finding, or name the thing.** Three to nine words. Sentence case.
- **A question only where the section opens one** that the page then answers
  with evidence, which is what section 4 does. They allow themselves about one
  in fifty.
- **Never an imperative**, and never the second person. A heading does not tell
  a reader what to do or ask them anything.
- **Never about the presentation.** "The setup, in one picture" named the
  format; "Building an emotion vector" names the content.
- **A sub-heading never restates the section heading above it.** Section 8's
  said the same thing as its own h2, so it went.

Every chart is an SVG with a fixed `viewBox` scaled to the width of its card,
so **its aspect ratio alone decides its height**. One CSS rule caps that height
at `min(72vh, 620px)`: a 1:1 chart at full width would otherwise be as tall as
the column is wide, and a figure a reader cannot see all of has lost the
comparison it exists to make. Nothing is sized in pixels per chart, and nothing
needs a media query: the ratio and the cap do it.

Three rules that are easy to get wrong:

- **The legend is not the how-to.** A legend is needed to see the figure at all,
  so it is never collapsed. A how-to is needed to judge it, and a reader who
  knows the field will skip it, so it is behind a fold. If you cannot tell which
  a sentence is, ask whether such a reader would skip it.
- **A figure gets the full width.** Commentary goes below it, as prose, not
  beside it in a second card. Three figures used to sit in a two-column grid
  with their reading next to them, which halved the chart and left the expanded
  view with no interpretation at all, because the expand control looks inside
  the card.
- **A schematic is not a result.** `data-schematic` marks a figure that carries
  no measurement, and it is the only thing exempt from needing a how-to and a
  source line. There is exactly one: the method diagram in section 1.

## How the numbers get here

`build.py` holds one block per figure, and each block names the notebook whose
printed record it was copied from.

One consequence worth knowing when you edit: **changing a number means changing
the notebook first.** `build.py` is a transcription layer, not a computation. If
a notebook's printed record moves, update the matching block here and say so in
the commit.

## Rebuilding

```
uv run python build.py             # -> index.html  (reads data/)
```

## Status

A talk, not a paper: short sprint, no external review, nulls included and
labelled as such. The `Method` and `Q&A` sections carry the maths, the
pseudo-code and a symbol glossary for every measurement, so a claim can be
traced to code without leaving the page.

The three headline results are not equally settled, and section 8 says so on the
page. The base-model circumplex has survived a falsification pass. The
instruction-tuning result is measured, but its gate is still owed. The
story-following work is registered as exploratory.
