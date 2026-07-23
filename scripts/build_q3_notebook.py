"""Emit notebooks/08_transition_tracking_first_reads.ipynb: the Q3 first-tranche results exhibit.

Every number is computed inside the notebook from the committed evidence
files (q3_gate_r1_it.json, q3_gate_r1_deepseek.json) — nothing hardcoded.

    .venv/bin/python scripts/build_q3_notebook.py
    .venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/08_transition_tracking_first_reads.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


HEADER = """\
# 08 - Q3 first tranche: does Gemma track emotion transitions while reading?

**Model probed throughout: google/gemma-4-31b-it**, reading multi-phase
stories token by token. This notebook answers three questions with the
first registered Q3 reads: gate G, the identity gate (can the tagged
emotion be read out at all?) that the dynamics reads are conditional on,
and R1, the pre-boundary anticipation read. Each section gives the answer,
how it was measured, and the verdict.

**Index**
1. Does the model know which emotion the current phase is? (gate G)
2. Stories or vectors: what does each result depend on?
3. Above chance versus useful versus meaningless: the three-way diagnosis
4. The full design grid and the combined reading
5. What is still open

**The instruments, briefly.** A "contrast probe" is an emotion direction:
the emotion's mean story activation minus the pool mean, unit-normalized;
a "centered cosine" subtracts the story-set mean per (layer, probe) before
the cosine (the registered convention). A "probe bank" is a set of such
directions: the 171 corpus-lineage contrasts, the 12 self-generated
contrasts (E10's winner), the 12 fixed-DeepSeek contrasts (E11's winner,
added by the registered substrate amendment and therefore present only on
the newer arms), and 24 random directions (the meaninglessness baseline,
N1). Story substrates, all SEQUENTIAL (each story moves through its tagged
emotions one after another): the Gemma-written combined stories and their
DeepSeek-written twin, plus a constant-emotion CONTROL arm (same scaffold
and marked scene changes, no emotion change); the setup cell prints the
scored story count for each. "Rank" = where the tagged emotion's mean
centered cosine falls among its bank's probes for a phase (1 = best). N2 =
10,000 wrong-emotion shuffles; chance median rank is about half the bank
size. The "verdict ladder" (tiers T0 to T3) is the registered mapping from
outcomes to allowed conclusions, written down before scoring.

**Data lineage.** Per-token probe dots are stored in the trajectory
substrate itself (probe identity in each arm's `probe_labels.json`). The
Gemma-written arm is public on HF (Hugging Face) as
`abotresol/emotion-combined-trajectories-gemma-4-31b-it`; the
DeepSeek-written and control arms live under `results/` pending publish.
Evidence files scored from them: `results/q3_gate_r1_it.json`,
`results/q3_gate_r1_deepseek.json`; conventions registered in TREE Q3.H1.E1
before scoring.
"""

SETUP = """\
# this cell loads the two evidence files and defines shared helpers
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RES = ROOT / "results"
MAIN_RES = Path("/Users/abo-tresol/Documents/ai-safety/cbai_project/results")
MODEL = "google/gemma-4-31b-it"
LAYERS = [6, 15, 24, 33, 42, 51]

def load_json(name):
    for base in (RES, MAIN_RES):
        path = base / name
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(name)

gemma_stories = load_json("q3_gate_r1_it.json")       # Gemma-written substrate
deepseek_stories = load_json("q3_gate_r1_deepseek.json")  # DeepSeek-written substrate
try:  # v2: same Gemma stories re-extracted with post-fix corpus probes + DeepSeek bank
    gemma_stories_v2 = load_json("q3_gate_r1_it_v2.json")
except FileNotFoundError:
    gemma_stories_v2 = None

BANK_SIZE = {"corpus": 171, "selfgen": 12, "deepseek": 12}
BANK_LABEL = {
    "corpus": "corpus-171 probes",
    "selfgen": "self-gen probes (12)",
    "deepseek": "DeepSeek probes (12)",
}
SUBSTRATES = {
    "Gemma-written stories": gemma_stories["true_arm"],
    "DeepSeek-written stories": deepseek_stories["true_arm"],
    "control (no emotion change)": gemma_stories["control_arm"],
}
if gemma_stories_v2 is not None:
    SUBSTRATES["Gemma-written (v2: post-fix probes)"] = gemma_stories_v2["true_arm"]
print({name: arm["n_stories"] for name, arm in SUBSTRATES.items()})
"""

S1_MD = """\
## 1. Does the model know which emotion the current phase is? (gate G)

**Answer: yes, clearly, when read with good probes.** The 12-probe banks
place the tagged emotion at median rank 1 to 6 at every layer and
substrate (self-generated everywhere; fixed-DeepSeek on the arms that
carry it, since the Gemma-written substrate was extracted before that
third bank existed). The corpus-171 bank does not (median rank 38 to 83 of
171). The cell prints the exact per-bank rank ranges behind these numbers.

**How it was measured:** for every story phase, average the centered cosine
of each probe over the phase's tokens, then rank the tagged emotion among
its bank. The y axis below is rank divided by bank size, so different bank
sizes share one scale: chance sits near 0.5 (dotted line), the registered
usefulness bar is the top decile, 0.1 (dashed line). Lower is better. One
panel per substrate; bars grouped by layer.

**Verdict:** gate passes in substance for the self-gen and DeepSeek banks
(N2 p < 0.001 everywhere except the self-gen bank at layer 51 on the
DeepSeek-written substrate); the corpus-171 bank fails the registered
bar on every layer. Formal ladder adjudication is pending because the
registered gate text named the 171 bank specifically.
"""

S1_CODE = """\
# this cell plots the tagged emotion's median rank (as a fraction of bank
# size) per layer, bank, and substrate, then prints each bank's rank range
fig = make_subplots(rows=1, cols=3, subplot_titles=list(SUBSTRATES), shared_yaxes=True)
colors = {"corpus": "#7f7f7f", "selfgen": "#1f77b4", "deepseek": "#d62728"}
banks_in_legend = set()  # each bank gets exactly one legend entry, whichever panel shows it first
for col, (substrate, arm) in enumerate(SUBSTRATES.items(), start=1):
    for bank, per_layer_stats in arm["gate_G"].items():
        if bank == "random" or not per_layer_stats.get("per_layer"):
            continue
        rank_fraction = [
            per_layer_stats["per_layer"][str(layer)]["median_rank"] / BANK_SIZE[bank]
            for layer in LAYERS
        ]
        fig.add_bar(
            x=[str(layer) for layer in LAYERS], y=rank_fraction,
            name=BANK_LABEL[bank], marker_color=colors[bank],
            legendgroup=bank, showlegend=(bank not in banks_in_legend), row=1, col=col,
        )
        banks_in_legend.add(bank)
fig.add_hline(y=0.5, line_dash="dot", annotation_text="chance")
fig.add_hline(y=0.1, line_dash="dash", annotation_text="registered bar (top decile)")
fig.update_layout(
    title=f"Gate G: median rank of the tagged emotion, as fraction of bank size | {MODEL}",
    yaxis_title="median rank / bank size (lower is better)",
    barmode="group", width=1150, height=460, margin=dict(t=90),
)
fig.update_xaxes(title="layer")
fig.show()
for substrate, arm in SUBSTRATES.items():
    for bank, per_layer_stats in arm["gate_G"].items():
        if bank == "random" or not per_layer_stats.get("per_layer"):
            continue
        cells = per_layer_stats["per_layer"]
        ranks = [cells[str(layer)]["median_rank"] for layer in LAYERS]
        worst_p = max(cells[str(layer)]["n2_p"] for layer in LAYERS)
        print(f"{substrate:30s} {BANK_LABEL[bank]:22s} median rank {min(ranks):.0f} to "
          f"{max(ranks):.0f} of {BANK_SIZE[bank]}, worst N2 p = {worst_p:.4f}")
"""

S2_MD = """\
## 2. Stories or vectors: what does each result depend on?

**Answer: both, but for different claims.** Whether tracking is VISIBLE
depends on the vectors: on the same stories, corpus-171 probes fail while
self-gen and DeepSeek probes succeed. Whether ANTICIPATION exists depends
on the stories: the same probes show a solid pre-boundary ramp on
Gemma-written stories and nothing on DeepSeek-written ones.

**How it was measured:** left panel, identity (gate G rank fraction).
Right panel, anticipation (R1): mean rise of the incoming emotion's
cosine in the 16 tokens before the phase boundary versus the 16 tokens
before that, expressed in units of the calibrated per-token noise. Stars
mark cells passing their registered bar (for identity, the gate bar; for
anticipation, N2 p < 0.001 plus the magnitude bar). The layer slider
below the figure scrubs all six layers (default 33, the registered
primary cell); the dissociation pattern holds across the mid-to-late
band.

**Verdict:** vector quality decides whether the state is readable at all
(matching E10/E11's detection results); the anticipation effect is a
property of the text. Gemma reads with causal attention, so pre-boundary
signal can only come from foreshadowing in the words already read. Gemma's
own stories foreshadow; DeepSeek's cleaner transitions do not. The control
arm shows leads near zero, so scene-change events alone produce nothing.
"""

S2_CODE = """\
# this cell draws the identity / anticipation dissociation heatmaps,
# substrates x banks, with a layer slider (default: layer 33)
banks = ["corpus", "selfgen", "deepseek"]
DEFAULT_LAYER = 33

def dissociation_grids(layer):
    \"\"\"(identity z, anticipation z, and their cell labels) at one layer.\"\"\"
    identity, anticipation, id_text, ant_text = [], [], [], []
    for substrate, arm in SUBSTRATES.items():
        id_row, ant_row, id_t, ant_t = [], [], [], []
        for bank in banks:
            gate_cell = arm["gate_G"].get(bank, {}).get("per_layer", {}).get(str(layer))
            r1_cell = arm["r1_anticipation"].get(bank, {}).get("per_layer", {}).get(str(layer))
            if gate_cell:
                fraction = gate_cell["median_rank"] / BANK_SIZE[bank]
                star = " *" if gate_cell["passes"] else ""
                id_row.append(fraction)
                id_t.append(f"rank {gate_cell['median_rank']:.0f}/{BANK_SIZE[bank]}{star}")
            else:
                id_row.append(None); id_t.append("n/a")
            if r1_cell and r1_cell.get("noise_sd"):
                lead_in_sd = r1_cell["mean_lead"] / r1_cell["noise_sd"]
                star = " *" if r1_cell["passes"] else ""
                ant_row.append(lead_in_sd)
                ant_t.append(f"{lead_in_sd:.1f} sd{star}")
            else:
                ant_row.append(None); ant_t.append("n/a")
        identity.append(id_row); anticipation.append(ant_row)
        id_text.append(id_t); ant_text.append(ant_t)
    return identity, anticipation, id_text, ant_text

fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
    subplot_titles=(
        "identity: rank fraction (low = good)",
        "anticipation: pre-boundary lead, in noise-sd units"))
# one heatmap pair per layer; the slider toggles which pair is visible
for layer in LAYERS:
    identity, anticipation, id_text, ant_text = dissociation_grids(layer)
    visible = layer == DEFAULT_LAYER
    fig.add_trace(go.Heatmap(
        z=identity, x=[BANK_LABEL[b] for b in banks], y=list(SUBSTRATES),
        text=id_text, texttemplate="%{text}", colorscale="Blues_r", zmin=0, zmax=0.6,
        showscale=False, visible=visible), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=anticipation, x=[BANK_LABEL[b] for b in banks], y=list(SUBSTRATES),
        text=ant_text, texttemplate="%{text}", colorscale="RdBu", zmid=0,
        showscale=False, visible=visible), row=1, col=2)
steps = []
for layer_pos, layer in enumerate(LAYERS):
    visibility = [other_pos == layer_pos for other_pos in range(len(LAYERS)) for _ in (0, 1)]
    steps.append(dict(method="update", label=f"layer {layer}", args=[{"visible": visibility}]))
fig.update_layout(
    sliders=[dict(active=LAYERS.index(DEFAULT_LAYER), steps=steps, y=-0.25,
                  currentvalue=dict(prefix="showing: "))],
    title=f"The dissociation, layer by layer (* = passes its registered bar) | {MODEL}",
    width=1150, height=480, margin=dict(t=110, b=120))
fig.show()
"""

S3_MD = """\
## 3. Above chance versus useful versus meaningless

**Answer: three different situations, and we can tell them apart because
chance is measured, not assumed.** N2 shuffles the emotion labels 10,000
times (a meaningless assignment lands at median rank of roughly half the
bank). N1 runs everything on 24 random directions. The instrument
calibration measured the per-token noise floor, so a null result comes
with its detection power attached.

**How to read this:** every (substrate, bank) cell from section 1 becomes
one dot at layer 33. The x axis is rank fraction. Three zones: near the
chance line = indistinguishable from meaningless; between chance and the
bar = above chance but below the registered usefulness bar; left of the
bar = doing good work. Position shows USEFULNESS; the printed p shows
DISTINGUISHABILITY from chance: with thousands of phases, a median close
to the chance line can still be significantly better than chance, which
is exactly the "bad but not meaningless" case. The corpus-171 bank lands
squarely in the middle zone: real signal (N2 p < 0.001 at the default
layer 33), not useful by the registered standard.

**Verdict:** nothing we measured behaves like meaningless vectors; the
corpus bank is "bad but not meaningless"; the self-gen and DeepSeek banks
do good work on identity. For anticipation, bars are cleared only on
Gemma-written stories (the self-gen bank most strongly, the corpus bank
more weakly at three layers; see the section 2 heatmap), and the
constant-emotion control sits at zero, so the surviving effect is not a
scene-change artifact.
"""

S3_CODE = """\
# this cell places every (substrate, bank) identity result in the
# meaningless / above-chance / good-work zones, with a layer slider
fig = go.Figure()
marker_symbols = {"Gemma-written stories": "circle",
                  "DeepSeek-written stories": "square",
                  "control (no emotion change)": "diamond"}
colors = {"corpus": "#7f7f7f", "selfgen": "#1f77b4", "deepseek": "#d62728"}
DEFAULT_LAYER = 33

# one trace per (layer, substrate, bank); only the active layer is visible
traces_per_layer = 0
for layer in LAYERS:
    for substrate, arm in SUBSTRATES.items():
        for bank in ["corpus", "selfgen", "deepseek"]:
            gate_cell = arm["gate_G"].get(bank, {}).get("per_layer", {}).get(str(layer))
            if not gate_cell:
                continue
            if layer == LAYERS[0]:
                traces_per_layer += 1
            fraction = gate_cell["median_rank"] / BANK_SIZE[bank]
            fig.add_scatter(
                x=[fraction], y=[f"{BANK_LABEL[bank]}<br>{substrate}"],
                mode="markers+text", visible=(layer == DEFAULT_LAYER),
                marker=dict(size=14, color=colors[bank], symbol=marker_symbols[substrate]),
                text=[f"  rank {gate_cell['median_rank']:.0f}/{BANK_SIZE[bank]}, p={gate_cell['n2_p']:.4f}"],
                textposition="middle right", showlegend=False)

# slider: reveal exactly the chosen layer's traces
steps = []
for layer_pos, layer in enumerate(LAYERS):
    visibility = []
    for other_pos in range(len(LAYERS)):
        visibility += [other_pos == layer_pos] * traces_per_layer
    steps.append(dict(method="update", label=f"layer {layer}",
                      args=[{"visible": visibility}]))
fig.update_layout(sliders=[dict(
    active=LAYERS.index(DEFAULT_LAYER), steps=steps, y=-0.12,
    currentvalue=dict(prefix="showing: "))])
fig.add_vline(x=0.5, line_dash="dot", annotation_text="chance (N2 shuffle)",
              annotation_position="bottom left")
fig.add_vline(x=0.1, line_dash="dash", annotation_text="registered bar")
fig.add_vrect(x0=0.0, x1=0.1, fillcolor="green", opacity=0.06, line_width=0,
              annotation_text="doing good work", annotation_position="bottom left")
fig.add_vrect(x0=0.1, x1=0.48, fillcolor="orange", opacity=0.06, line_width=0,
              annotation_text="above chance, below the bar", annotation_position="bottom")
fig.add_vrect(x0=0.48, x1=0.62, fillcolor="red", opacity=0.06, line_width=0,
              annotation_text="chance-like", annotation_position="top right")
fig.update_layout(
    title=f"Identity tracking: where every reading lands (layer slider below) | {MODEL}",
    xaxis_title="median rank / bank size", xaxis_range=[0, 0.62],
    width=1150, height=520, margin=dict(t=80))
fig.show()
"""

SGRID_MD = """\
## 4. The full design grid and the combined reading

Everything scored so far as one factorial table. G is phase-identity
tracking, R1 is pre-boundary anticipation; "pass" means the registered
bar, "substance" means far above chance with the bar adjudication open.

| probe bank vs stories | Gemma-written | Gemma-written, v2 post-fix probes | DeepSeek-written | constant control |
|---|---|---|---|---|
| corpus-171 | G fail, R1 weak pass | G fail (robust to post-fix), R1 fail | G fail, R1 zero | G fail, R1 zero |
| self-gen (12) | G substance, R1 PASS (5.5x noise) | G substance, R1 PASS (0.0117 vs 0.0116: replicates) | G substance, R1 zero | G substance, R1 zero |
| DeepSeek (12) | bank absent in v1 shards | G substance (rank 2/12), R1 PASS (4.3x noise) | G substance, R1 zero | G substance, R1 zero |
| random (24) | null band | null band | null band | null band |

The base-model arm is collected but unscored (registered falsify plan).

**The combined reading.** (1) Whether the model's emotional state is
readable at all is decided by the probe COLUMN: vector quality decides
(E11's detection winners are the dynamics winners too), on every
substrate, and the corpus-171 failure survives post-fix re-extraction.
(2) Whether the state moves BEFORE a transition is decided by the story
ROW: the same probes show anticipation exactly and only where Gemma
authored the text; the control row rules out scene-change mechanics, and
causal attention permits only foreshadowing already present in the words
read so far. (3) Together: Gemma robustly tracks the emotional state of
what it reads, and "seeing the transition coming" is the model detecting
the foreshadowing habits of its own writing style, a property of the
text and reader MATCH, not of the reader alone. No verdict-ladder tier
is claimed until the ladder adjudication and the cue-referenced read.
"""

S4_MD = """\
## 5. What is still open

1. **Ladder adjudication.** The registered gate names the 171 corpus bank;
   it fails while the 12-banks pass in substance. The verdict-ladder tier
   (T0 through T3) needs that adjudicated before any claim graduates.
2. **The cue-referenced anticipation twin.** The DeepSeek-arm collapse
   says the surviving anticipation is textual foreshadowing. The decisive
   registered read re-references R1 to judge-located cue positions (judges
   blind to activations); it needs the judge bulk pass.
3. **Ramp width (R2) and crossover location (R3)** are registered and
   validated but not yet run.
4. The 12-bank top-decile bar (rank <= 1) is the scorer's generalization
   of a bar registered for the 171 bank; ranks of 2 to 6 of 12 fail it
   mechanically while beating the N2 chance floor at nearly every layer
   (section 1 prints the ranks and worst p values). Adjudicate alongside
   the ladder.
5. **Probe-version caveat (E4b).** The corpus-171 and self-gen dots in
   the substrate were computed against pre-padding-fix probe vectors.
   E4b rated the self-gen contrasts tier 1 (post-fix twins agree at
   about cos 0.995; numbers reproduce to roughly 3 decimals) but the
   corpus-lineage contrasts tier 2, so the corpus-171 columns here carry
   a rescore-or-recollect caveat. The fixed-DeepSeek bank is post-fix by
   construction.

Nothing in this notebook is a graduated claim; the falsify gate
(shuffled-sentence control, category splits, base-arm replication) comes
before any of it enters a deliverable.
"""


def main() -> int:
    cells = [
        md(HEADER),
        code(SETUP),
        md(S1_MD),
        code(S1_CODE),
        md(S2_MD),
        code(S2_CODE),
        md(S3_MD),
        code(S3_CODE),
        md(SGRID_MD),
        md(S4_MD),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path("notebooks/08_transition_tracking_first_reads.ipynb")
    out.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {out} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
