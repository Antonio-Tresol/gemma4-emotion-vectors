# Notebooks

Four report notebooks tell the whole story, in reading order. Each opens with
its purpose, the key concepts you need, and an index; every figure carries its
verdict and has a collapsible "How to read" underneath.

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

## Running them (laptop or the shared pod)

Any clone works the same way — every input resolves through
`emotion_vectors.artifacts.fetch()`: local `results/` first, the private HF
datasets otherwise (they flip public at sprint end).

1. `uv sync` (the venv already includes plotly, ipykernel, jupyterlab).
2. Put an `HF_TOKEN=...` line in the repo-root `.env` — `fetch()` loads it
   automatically, including inside notebook kernels (both pod clones already
   have one).
3. In VS Code / Jupyter, select the **project `.venv`** as the kernel — the
   conda `arena-env` kernel does not have this package.
4. Headless check: `.venv/bin/python -m nbconvert --to notebook --execute
   notebooks/05_trajectory_explorer_instruct.ipynb --output /tmp/check.ipynb`.

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
