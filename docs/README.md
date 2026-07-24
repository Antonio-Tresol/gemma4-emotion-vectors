# docs/ — the talk

The sprint's findings as a presentation, in three formats. Everything here is
generated from the notebooks and `results/`; nothing is typed by hand.

| File | What it is |
|---|---|
| `index.html` | the interactive talk, self-contained (no build step, no server) |
| `slides/emotion-vectors-talk.pdf` | 13 slides, 16:9, fonts embedded. Present from this one. |
| `slides/emotion-vectors-talk.pptx` | the same slides, editable |
| `build.py` | rebuilds `index.html`; every figure names the notebook it came from |
| `data/` | the extracted figure inputs, so the build runs on any clone |
| `slides/story0-tracking.mp4` | 19.5s silent clip, 1080p: one story's trajectory, token by token |
| `slides/render_charts.py` | screenshots each chart from `index.html`'s own code |
| `slides/render_story_video.py` | steps the token cursor frame by frame and encodes the clip |
| `slides/build_pptx.py`, `slides/build_pdf.py` | assemble the slides from those images |

Also published (search-engine blocked, link-only):
<https://antonio-tresol.github.io/emotion-vectors-presentation/>

## How the numbers get here

`build.py` holds one block per figure, and each block names the notebook whose
printed record it was copied from. The charts in the slides are screenshots of
`index.html` running its own JavaScript, so a figure cannot say one thing on the
web and another on a slide.

Two consequences worth knowing when you edit:

- **Changing a number means changing the notebook first.** `build.py` is a
  transcription layer, not a computation. If a notebook's printed record moves,
  update the matching block here and say so in the commit.
- **Regenerate in order**: `build.py`, then `render_charts.py`, then the two
  slide builders. Skipping the middle step leaves slides showing the old figure.

## Rebuilding

```
python build.py                    # -> index.html  (reads data/)
cd slides
python render_charts.py            # -> slides/png/*.png  (needs Chrome)
python build_pptx.py               # -> emotion-vectors-talk.pptx
python build_pdf.py                # -> emotion-vectors-talk.pdf
python render_story_video.py       # -> slides/story0-tracking.mp4  (needs ffmpeg)
```

`render_story_video.py --probe` renders five spanning frames instead of all 236,
which is the cheap way to check composition before committing to a full run.
`--story 1|2|3` animates the tracked-well / partly / badly stories instead of the
walkthrough, and `--layer 6` (or 15, 24, 42, 51) animates a different layer.

`render_charts.py` and `build_pdf.py` drive headless Chrome and expect it at the
standard macOS path; change `CHROME` at the top of each if yours differs.
`build_pptx.py` needs `python-pptx` and `pillow`.

## Status

A talk, not a paper: short sprint, no external review, nulls included and
labelled as such. The `Methods` and `Q&A` sections of the deck carry the
maths, the pseudo-code and a symbol glossary for every measurement, so a claim
can be traced to code without leaving the page.
