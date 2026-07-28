# Notebooks

These notebooks tell the whole story, in reading order. Each opens with its
purpose, the key concepts you need, and an index; every figure carries its
verdict and has a collapsible "How to read" underneath. Start at 01 if you want
the data first, or at 10 if you want the argument first.

| # | Notebook | The question it answers |
|---|---|---|
| 01 | `01_corpora_and_extraction.ipynb` | What data exists, who generated it, and is it clean? |
| 02 | `02_circumplex_geometry.ipynb` | Does the emotion circumplex replicate? (Yes, strongly, on the base model; instruction tuning demotes it without destroying it.) |
| 03 | `03_detection_probe_campaign.ipynb` | Do the vectors detect implicit emotion in scenarios? (No configuration passed the pre-registered bar across seven experiments; what survives is coarse valence.) |
| 04 | `04_paper_plot_parity.ipynb` | Where does every figure from the two source papers stand? |
| 05 | `05_trajectory_explorer_instruct.ipynb` | Q3 substrate: per-token emotion trajectories over three-emotion stories (exhibits only; reads pending registration) |
| 06 | `06_trajectory_explorer_base.ipynb` | The base-model arm of 05: same corpus, same figures, same controls, base-lineage probes |
| 07 | `07_generator_lineages.ipynb` | Whose stories make the best emotion probes? E11 (quality vs identity) and E12 (scale and diversity dose-response), every number computed in-cell |
| 08 | `08_transition_tracking_first_reads.ipynb` | Q3 first tranche: does the model track emotion transitions while reading? Full probe-bank x story-substrate grid, nulls, control arm, combined reading |
| 09 | `09_trajectory_explorer_deepseek_arm.ipynb` | The DeepSeek-written-stories arm of 05: same figures and levers, three probe banks including E11's DeepSeek contrasts |
| 10 | `10_the_sprint_story.ipynb` | The review synthesis: every claim with its method in plain language, its numbers computed in-cell, and its tree status, in seven acts |
| 11 | `11_tracking_taxonomy.ipynb` | Q3.H1.E2 + E3 (exploratory): what the model tracks well vs poorly (emotion, family, VAD distance, confusions, position), then the named confusion tables, boundary-lag vs stable-relabeling timing, and probe-geometry-predicts-difficulty reads |
| — | `explore_layers.ipynb` | Off the reading order, an exploration surface: the figures that 02 and 04 pin to one layer (the circumplex, the logit lens) made scrubbable over all 20 swept layers. It settles nothing and changes no claim; `scripts/logit_lens.py --all-layers` writes the sweep that fills in its second figure |
| — | `05_trajectories.ipynb` | Off the reading order, superseded: a stale copy of notebook 05 at the filename it had before the rename, kept only because `docs/build.py` still cites this path as a figure source. Read `05_trajectory_explorer_instruct.ipynb` instead — this copy predates the linked plot-and-text view and still reports the Q3.H1.E1 reads as unregistered |

## Running them (laptop or the shared pod)

Any clone works the same way — every input resolves through
`emotion_vectors.artifacts.fetch()`: local `results/` first, the public HF
datasets otherwise.

1. `uv sync` (the venv already includes plotly, ipykernel, jupyterlab).
2. Put an `HF_TOKEN=...` line in the repo-root `.env` — `fetch()` loads it
   automatically, including inside notebook kernels, so there is nothing to
   export in your shell first.
3. In VS Code / Jupyter, select the **project `.venv`** as the kernel — any
   other kernel (a system Python, a conda base environment) does not have this
   package installed, and every import in the first cell fails.
4. Headless check: `.venv/bin/python -m nbconvert --to notebook --execute
   notebooks/05_trajectory_explorer_instruct.ipynb --output /tmp/check.ipynb`.

## Editing them

These notebooks are **hand-maintained**. There is no generator: you edit the
notebook and the exhibit package it imports, then re-execute in place so the
stored outputs match the code that produced them:

```
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
```

Analysis and figure code lives in an importable package under
`src/emotion_vectors/`, one per notebook, so cells stay load-call-show and the
figures can be tested without a kernel (`tests/test_report_packages.py` calls
every builder and enforces the house contract):

| Notebook | Exhibit package |
|---|---|
| 01 | `emotion_vectors.corpora_report` |
| 02 | `emotion_vectors.geometry_report` |
| 03 | `emotion_vectors.detection_report` |
| 07 | `emotion_vectors.lineage_report` |
| 10 | `emotion_vectors.sprint_report` |
| 11 | `emotion_vectors.taxonomy_report` |

Notebooks 04, 05, 06 and 09 still hold their analysis code inline; 08's
extraction is in progress. They are edited and re-executed the same way.

Four `scripts/build_*_notebook.py` generators used to emit notebooks 07, 08, 10
and 11 from hardcoded cell sources. They were deleted on 2026-07-23: the
notebooks now carry hand-written narrative and load-call-show cells that no
generator round-trips, so every builder had gone silently stale and re-running
one would have replaced a finished notebook with its pre-extraction ancestor.

Known exception: notebook 02 needs the NRC VAD lexicon under
`data/lexicons/`, which is deliberately NOT fetchable (third-party license) —
download it from saifmohammad.com first.

The three layers of the project, so nothing here is mistaken for something it
is not: **notebooks are the report** (readable, story-ordered),
**`results/` is the evidence** (every number traces to a file),
**`TREE.md` is the record** (questions, hypotheses, experiments, claims, with
registered predictions and statuses). When a notebook and the tree disagree,
the tree wins and the notebook has a bug.

## archive/

The bench layer: one notebook per experiment, in the order the work actually
happened, kept unchanged because TREE.md evidence links point into them.
Numbering reflects session history, not reading order.

| Bench notebook | What it was |
|---|---|
| `01_corpus_exploration` | first look at the published story corpus |
| `02_vector_geometry` | base-model geometry (superseded by report 02) |
| `03_probe_validation` | first battery run, base model (E1) |
| `04_probe_sweep` | layer/readout sweep, base model (E2) |
| `05_dialogue_probes` | dialogue-transfer test, base arm (E3) |
| `06_probe_sweep_it` | instruct-model sweep with chat template (E4) |
| `07_selfgen_probes_it` | self-generated probe pilot (E5) |
| `08_scale_and_projection` | scale curve and neutral projection (E6, E7) |
| `09_paper_parity` | first parity pass (superseded by report 04) |
