"""Emit notebooks/07_generator_lineages.ipynb: the E11/E12 comparison exhibit.

Every figure reads committed results JSONs (no recomputation of frozen
numbers); corpus panels read the story files. Rebuild any time with:

    .venv/bin/python scripts/build_lineage_notebook.py
    .venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/07_generator_lineages.ipynb
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
# 07 - Generator lineages: whose stories make the best emotion probes?

**Model probed throughout: google/gemma-4-31b-it** (the instruct model). The
generators only write the stories; every vector is extracted from Gemma
reading them.

**The question.** Emotion vectors are built from story corpora. E5/E6 showed
they are corpus-dependent objects, and E10 showed self-generated stories beat
an external corpus written by a much weaker model (Gemma-4-4B). That left two
explanations tangled: does the probed model need its OWN stories
(distribution match), or just GOOD stories (generator quality)? E11 and E12
untangle them with third-party corpora from a strong external generator
(DeepSeek-v4-pro via OpenRouter).

**Index**
1. The lineages, side by side (what each corpus looks like)
2. Lexical diversity: the scaffold-degeneracy covariate
3. Detection accuracy per layer (E11's registered R1 read)
4. Dose-response: how many stories do probes actually need? (E12)
5. Direction geometry across lineages (E11's R2 read)
6. The preference read: where corpus quality still pays (R3)
7. What we are saying, exactly (verdict and caveats)

**Definitions used everywhere.** "Battery": our 12-emotion probe set (happy,
inspired, loving, proud, calm, desperate, angry, guilty, sad, afraid,
nervous, surprised). "Contrast probe": an emotion's mean story activation
minus the mean over all 12, unit-normalized. "Dual-battery bar": a layer
passes if the probes place the target emotion in the top 3 by centered
cosine for at least 8 of 12 scenarios on BOTH the paper battery and a
held-out battery. "RSA" (representational similarity analysis): correlate
the 12x12 emotion-similarity matrices of two probe sets. Evidence files:
`results/e11_lineage.json`, `results/e12_scale_curve_fixed_arm.json`,
`results/e12_diverse_fullcorpus_grid.json`, `results/e12_scale_curve.json`
(all committed; corpora and vectors on HF, see `emotion_vectors.artifacts`).
"""

SETUP = """\
# this cell loads the frozen evidence JSONs and the three story corpora
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RES = ROOT / "results"
MAIN_RES = Path("/Users/abo-tresol/Documents/ai-safety/cbai_project/results")
MODEL = "google/gemma-4-31b-it"

def load_json(name):
    for base in (RES, MAIN_RES):
        path = base / name
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(name)

e11 = load_json("e11_lineage.json")
fixed_curve = load_json("e12_scale_curve_fixed_arm.json")
full_grid = load_json("e12_diverse_fullcorpus_grid.json")
try:  # both-arm dose-response; written by scripts/score_e12_scale.py
    both_curves = load_json("e12_scale_curve.json")
except FileNotFoundError:
    both_curves = None
    print("e12_scale_curve.json not present yet: section 4 shows the fixed arm only")

def load_grouped(name):
    for base in (RES, MAIN_RES):
        path = base / name / "stories_grouped.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            return {row["emotion"]: row["stories"] for row in rows}
    return None

corpora = {
    "fixed DeepSeek (E11)": load_grouped("openrouter_stories"),
    "diverse DeepSeek (E12)": load_grouped("openrouter_stories_diverse"),
}
# total story count per corpus (None where the corpus files are absent)
story_counts = {}
for name, corpus in corpora.items():
    story_counts[name] = sum(len(stories) for stories in corpus.values()) if corpus else None
print(story_counts)
"""

S1_MD = """\
## 1. The lineages, side by side

Three corpora, one instruction core ("write a short third-person story,
around 150 words, about a person experiencing X, without naming X"):

| lineage | generator | n per emotion | prompt recipe | HF dataset |
|---|---|---|---|---|
| self-generated | gemma-4-31b-it (the probed model itself) | 256 | fixed instruction | emotion-stories-gemma-4-31b-it |
| fixed DeepSeek | deepseek-v4-pro | 256 | identical fixed instruction | emotion-stories-deepseek-v4-pro |
| diverse DeepSeek | deepseek-v4-pro | 1024 | instruction plus a pinned persona x setting from a deterministic 8x8 grid | emotion-stories-deepseek-v4-pro-diverse |

**How to read the samples below:** all three stories answer the same
assignment ("sad"). Notice the self-generated corpus's house style (it
famously names 98% of its protagonists Elias), versus DeepSeek's varied
scenes, versus the diverse arm where the persona and setting were pinned by
us. The samples are the first story of the emotion in each corpus, not
cherry-picked.
"""

S1_CODE = """\
# this cell prints one "sad" story from each available corpus
for name, corp in corpora.items():
    if corp:
        print(f"=== {name} | assigned emotion: sad ===")
        print(corp["sad"][0][:500])
        print()
"""

S2_MD = """\
## 2. Lexical diversity: the scaffold-degeneracy covariate

E11's exploratory R4 read: mean pairwise 5-gram Jaccard overlap within each
corpus (higher = stories share more exact 5-word phrases). **What this
buys:** it separates "the corpus matches the model's distribution" from "the
corpus collapsed onto one scaffold". The self-generated corpus is 53x more
self-similar than fixed DeepSeek, yet its probes work, so scaffold
degeneracy is not what makes probes function. The diverse arm pushes overlap
still lower.
"""

S2_CODE = """\
# this cell plots within-corpus lexical overlap (R4) plus unique-opening counts
r4 = e11["r4_diversity_EXPLORATORY"]
names = ["selfgen", "weak_external", "strong_external"]
labels = ["self-generated\\n(gemma-it)", "weak external\\n(gemma-4-4B corpus)", "fixed DeepSeek"]
# per-lineage Jaccard overlap, None where the lineage was not measured
jaccard_values = []
for name in names:
    lineage_stats = r4.get(name)
    jaccard_values.append(lineage_stats["mean_pairwise_jaccard"] if lineage_stats else None)
bar_labels = [f"{value:.4f}" if value else "n/a" for value in jaccard_values]
fig = go.Figure(go.Bar(x=labels, y=jaccard_values, text=bar_labels, textposition="outside"))
fig.update_layout(
    title=f"Within-corpus 5-gram Jaccard overlap (higher = more repetitive) | probes read from {MODEL}",
    yaxis_title="mean pairwise Jaccard", width=850, height=420, margin=dict(t=80))
fig.show()
"""

S3_MD = """\
## 3. Detection accuracy per layer (E11's registered R1 read)

**How to read this:** each row is a probe lineage, each column a layer of
gemma-4-31b-it. Cell text is "paper battery / held-out battery" correct
counts out of 12 (target emotion in top 3 under centered cosine). A cell is
outlined when it clears the dual-battery bar (both counts at or above 8).
**What this buys:** the full picture behind the passing-layer headline, so
you can see WHERE each lineage works, not just how often. The diverse arm's
full-corpus grid comes from `e12_diverse_fullcorpus_grid.json`; the other
three rows are E11's frozen grid.
"""

S3_CODE = """\
# this cell draws the per-layer dual-battery heatmap for all four probe sets
rows = {
    "self-generated n=256": e11["r1_dual_battery"]["selfgen"]["per_layer"],
    "weak external (4B corpus)": e11["r1_dual_battery"]["weak_external"]["per_layer"],
    "fixed DeepSeek n=256": e11["r1_dual_battery"]["strong_external"]["per_layer"],
    "diverse DeepSeek n=1024": full_grid["r1_dual_battery"]["diverse_deepseek_n1024"]["per_layer"],
}
first_lineage = next(iter(rows.values()))
layers = sorted(int(layer_key) for layer_key in first_lineage)
z, text = [], []
for per_layer in rows.values():
    # (paper, held-out) count pair per layer; keys may be str or int per source JSON
    counts = []
    for layer in layers:
        key = str(layer) if str(layer) in per_layer else layer
        counts.append(per_layer[key])
    z.append([min(paper, held_out) for paper, held_out in counts])
    text.append([
        f"{paper}/{held_out}" + (" *" if paper >= 8 and held_out >= 8 else "")
        for paper, held_out in counts
    ])
fig = go.Figure(go.Heatmap(
    z=z, x=[str(L) for L in layers], y=list(rows), text=text, texttemplate="%{text}",
    colorscale="Blues", zmin=0, zmax=12, colorbar_title="min(paper, held-out)"))
fig.update_layout(
    title=f"Dual-battery counts per layer (paper/held-out, * = passes the 8/12 bar) | {MODEL}",
    xaxis_title="layer", width=1150, height=430, margin=dict(t=80))
fig.show()
for name, r1_src in [("self-generated", e11["r1_dual_battery"]["selfgen"]),
                     ("weak external", e11["r1_dual_battery"]["weak_external"]),
                     ("fixed DeepSeek", e11["r1_dual_battery"]["strong_external"]),
                     ("diverse DeepSeek", full_grid["r1_dual_battery"]["diverse_deepseek_n1024"])]:
    print(f"{name:16s} passing layers: {r1_src['passing_layers']}")
"""

S4_MD = """\
## 4. Dose-response: how many stories do probes actually need? (E12)

**How to read this:** x is stories per emotion (log scale), y is how many of
the 20 layers pass the dual-battery bar. Markers are 5 seeded subsamples per
n; the line tracks the seed mean. **What this buys:** the answer to "would
more stories help?". The fixed arm saturates near n=64; horizontal reference
lines mark self-generated probes (5 passing layers at n=256) and the
fixed-corpus ceiling (9). The diverse arm sits BELOW the fixed arm at every
matched n (at n=256: mean 6.2 vs 9.0) and never reaches the fixed ceiling
even at n=1018: pinning personas and settings adds non-emotional variance,
so each emotion mean is noisier per story. The right panel shows the
contrast cosine of each subsample's probes to the self-generated probes at
layer 33: probe FUNCTION saturates before probe GEOMETRY stops moving.
"""

S4_CODE = """\
# this cell plots passing-layer count and probe-direction convergence versus n
src = both_curves["arms"] if both_curves else {"fixed": fixed_curve["arms"]["fixed"]}
fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "dual-battery passing layers vs corpus size",
    "probe direction cosine to self-generated probes (layer 33)"))
colors = {"fixed": "#1f77b4", "diverse": "#d62728"}
for arm, curve in src.items():
    points = curve["points"]
    corpus_sizes = sorted({point["n"] for point in points})
    for col, metric in ((1, "n_passing_layers"), (2, "mean_contrast_cos_to_selfgen_L33")):
        # one marker per seeded subsample, plus the seed mean at each corpus size
        seed_sizes = [point["n"] for point in points]
        seed_values = [point[metric] for point in points]
        seed_means = []
        for size in corpus_sizes:
            values_at_size = [point[metric] for point in points if point["n"] == size]
            seed_means.append(np.mean(values_at_size))
        fig.add_scatter(x=seed_sizes, y=seed_values, mode="markers",
                        marker=dict(color=colors.get(arm, "gray"), opacity=0.4),
                        name=f"{arm} seeds", legendgroup=arm, showlegend=(col == 1), row=1, col=col)
        fig.add_scatter(x=corpus_sizes, y=seed_means, mode="lines+markers",
                        line=dict(color=colors.get(arm, "gray")),
                        name=f"{arm} mean", legendgroup=arm, showlegend=(col == 1), row=1, col=col)
fig.add_hline(y=9, line_dash="dot", annotation_text="fixed-corpus ceiling (9)", row=1, col=1)
fig.add_hline(y=5, line_dash="dot", annotation_text="self-generated n=256 (5)", row=1, col=1)
fig.update_xaxes(type="log", title="stories per emotion")
fig.update_yaxes(title="passing layers (of 20)", row=1, col=1)
fig.update_yaxes(title="mean contrast cosine", row=1, col=2)
fig.update_layout(title=f"E12 dose-response, 5 seeds per point | probes read from {MODEL}",
                  width=1150, height=470, margin=dict(t=90))
fig.show()
"""

S5_MD = """\
## 5. Direction geometry across lineages (E11's R2 read)

**How to read this:** each cell compares two probe sets at layer 33
(geometry peak). Top number: mean per-emotion contrast cosine (do the
DIRECTIONS agree?). Bottom number: RSA over the 12x12 emotion-similarity
matrices (does the SHAPE of emotion space agree?). **What this buys:** the
mechanism picture. A stronger generator converges toward the probed model's
own directions (0.57 versus the weak corpus's 0.22), and the diverse arm
keeps the shape (RSA 0.70) while rotating individual directions (cos 0.46).
"""

S5_CODE = """\
# this cell draws the pairwise cosine and RSA matrix across the four lineages
pairs = {**e11["r2_cross_lineage_layer33"], **full_grid["r2_cross_lineage_layer33"]}
short = {"selfgen": "self-gen", "selfgen_postfix": "self-gen",
         "weak_external": "weak ext", "fixed_deepseek_n256": "fixed DS",
         "strong_external": "fixed DS", "diverse_deepseek_n1024": "diverse DS"}
names = ["self-gen", "weak ext", "fixed DS", "diverse DS"]
cos = np.full((4, 4), np.nan)
rsa = np.full((4, 4), np.nan)
# fill both symmetric matrices from the "A_vs_B" pair keys
for pair_key, pair_stats in pairs.items():
    lineage_a, lineage_b = pair_key.split("_vs_")
    if short.get(lineage_a) in names and short.get(lineage_b) in names:
        row = names.index(short[lineage_a])
        col = names.index(short[lineage_b])
        cos[row, col] = cos[col, row] = pair_stats["mean_contrast_cos"]
        rsa[row, col] = rsa[col, row] = pair_stats["rsa_12x12"]
# cell labels: cosine + RSA where the pair was measured, blank otherwise
text = []
for row in range(4):
    row_labels = []
    for col in range(4):
        if np.isnan(cos[row][col]):
            row_labels.append("")
        else:
            row_labels.append(f"cos {cos[row][col]:.2f}<br>RSA {rsa[row][col]:.2f}")
    text.append(row_labels)
fig = go.Figure(go.Heatmap(z=cos, x=names, y=names, text=text, texttemplate="%{text}",
                           colorscale="RdBu", zmid=0, colorbar_title="contrast cos"))
fig.update_layout(title=f"Cross-lineage probe agreement at layer 33 | {MODEL}",
                  width=750, height=560, margin=dict(t=80))
fig.show()
"""

S6_MD = """\
## 6. The preference read: where corpus quality still pays (R3)

**How to read this:** for 64 activities with measured preference Elo scores
(fixed chat instrument, post-fix), the max absolute correlation between any
probe's cosine and Elo, per lineage. **What this buys:** a finer-grained
functional read than top-3 detection. Detection saturates (section 4), but
this read keeps improving with corpus quality and diversity, self-generated
0.62 to fixed DeepSeek 0.71 to diverse DeepSeek 0.77. An exploratory
matched-n check (`e12_pref_matched_n_exploratory.json`) shows the diverse
arm subsampled to n=256 still scores 0.760 (5 seeds, 0.746 to 0.772), so
the gain is the diversity itself, not the 4x corpus size. Caveat: max over
12 emotions x 4 layers per bar, so treat levels, not tiny gaps, as the
signal.
"""

S6_CODE = """\
# this cell plots the preference probe-Elo correlation per lineage
r3 = {"self-generated": e11["r3_preference_probe_elo"]["selfgen"],
      "weak external": e11["r3_preference_probe_elo"]["weak_external"],
      "fixed DeepSeek": full_grid["r3_preference_probe_elo"]["fixed_deepseek_n256"],
      "diverse DeepSeek": full_grid["r3_preference_probe_elo"]["diverse_deepseek_n1024"]}
fig = go.Figure(go.Bar(
    x=list(r3), y=[stats["abs_r"] for stats in r3.values()],
    text=[f"{stats['abs_r']:.3f}<br>L{stats['layer']} {stats['emotion']}" for stats in r3.values()],
    textposition="outside"))
fig.update_layout(
    title=f"Max |r| between probe cosine and preference Elo (64 activities) | {MODEL}",
    yaxis_title="max |Pearson r|", yaxis_range=[0, 0.9], width=850, height=440, margin=dict(t=80))
fig.show()
"""

S7_MD = """\
## 7. What we are saying, exactly

1. **Generator quality, not generator identity, drives probe function**
   (E11, gated by C4's falsify pass): fixed DeepSeek probes pass at 9 layers
   versus self-generated 5 and weak-external 1.
2. **Detection probes saturate early** (E12): the fixed arm reaches its
   ceiling near 64 stories per emotion; 64 strong-generator stories already
   beat 256 self-generated ones.
3. **Prompt diversity is a COST for detection at matched n** (E12): diverse
   at n=256 passes at 6.2 layers (seed mean) versus fixed's 9.0, and even
   n=1018 diverse (7 layers) never reaches the fixed ceiling. The registered
   growth branch did not fire; the diversity read fired in reverse.
4. **The same diversity HELPS the preference read, at matched n**: diverse
   n=256 scores 0.760 versus fixed's 0.706 (exploratory matched-n check),
   with the full ordering 0.62 (self-gen) to 0.71 (fixed) to 0.77 (diverse
   full). Coarse detection wants a clean, low-variance corpus; the
   fine-grained behavioral correlate wants coverage.
5. **Practical recipe**: for detection probes, roughly 64-256 fixed-prompt
   stories per emotion from any strong generator; add prompt diversity only
   when the downstream read is fine-grained, and expect a detection tax.

**Caveats.** "The emotion direction" is a three-convention-qualified object:
corpus lineage (this notebook), readout centering (E9), and extraction
format — the E4b raw-text-versus-chat-template audit found contrast
directions agreeing at only mean cos 0.52-0.75 (min 0.18, angry at layer
33) across formats, while raw means stay at 0.92-0.99; every number here is
raw-text extraction (`results/raw_vs_chat_extraction_cosine.json`). One
strong external generator tested (DeepSeek-v4-pro);
"quality" is operationalized by that single point plus the weak 4B corpus.
The diverse arm changed the prompt, not just diversity (persona and setting
content are shared across emotions and removed by 12-pool centering, but
second-order interactions are not controlled). The preference ordering is a
three-point trend on a max-statistic, not a gated claim; matched-n reads
come from `e12_scale_curve.json` when both arms are present.
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
        md(S4_MD),
        code(S4_CODE),
        md(S5_MD),
        code(S5_CODE),
        md(S6_MD),
        code(S6_CODE),
        md(S7_MD),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path("notebooks/07_generator_lineages.ipynb")
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
