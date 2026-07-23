"""Build notebooks/11_tracking_taxonomy.ipynb — Q3.H1.E2 category taxonomy.

Assembles the notebook from cell sources below. Every number the notebook
shows is computed in-cell from the per-record dumps (results/q3_records_*),
so the exhibit cannot rot when the substrate is re-extracted.

    uv run python scripts/build_taxonomy_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "11_tracking_taxonomy.ipynb"

# Cells are (kind, source) pairs; kind is "md" or "code".
CELLS: list[tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("md", source.strip()))


def code(source: str) -> None:
    CELLS.append(("code", source.strip()))


md("""
# What does the model track well, and what does it track poorly?

**Question (TREE Q3.H1.E2, exploratory).** The registered Q3 reads average over
all phases and transitions. This notebook slices the same measurements by the
structure that was designed into the substrate: which emotions, which triple
families, how far apart the emotions sit in valence-arousal space, and what the
model confuses with what. Idea credit: Peyton Li.

**Status: exploratory, hypothesis-generating.** The slices and conventions were
named in TREE.md before any record file existed (commit `dc23007`), but there
are no registered pass bars here and nothing in this notebook graduates to a
claim by itself: promising slices become registered reads and pass the falsify
gate first.

**Key concepts.**
- *Emotion probe / probe bank*: a probe is one emotion's direction in the
  model's residual stream, the mean activation over that emotion's stories
  minus the mean over the bank's whole emotion pool, unit-normalized. A bank
  is a set of probes read out together: `corpus` (171 emotions,
  corpus-lineage stories), `selfgen` (12 emotions, self-generated stories),
  `deepseek` (12 emotions, DeepSeek-written stories), `random` (24 random
  directions).
- *Battery-12*: the 12 emotions the designed story triples draw from. The
  selfgen and deepseek banks carry exactly these 12 probes, so "battery-12
  banks" below means those two.
- *Centered cosine*: before taking the cosine between an activation and a
  probe, the story-set mean is subtracted per (layer, probe). This is the
  registered scoring convention, shared bit-identically with the Q3 scorer.
- *Gate rank and top-1 rate*: for one story phase, average each probe's
  centered cosine over the phase's tokens, then rank the tagged emotion among
  its bank (rank 1 = the correct probe scored highest). The top-1 rate is the
  fraction of phases at rank 1. For a 12-probe bank, chance top-1 is 1/12 and
  chance median rank is about 6.5.
- *R1 anticipation lead*: for one transition, the incoming emotion's mean
  centered cosine over the W=16 tokens just before the phase boundary minus
  its mean over the 16 tokens before those. A positive lead means the
  incoming emotion's signal is already rising before the text switches.
- *Cluster bootstrap over triple_id, 95% CI*: every confidence interval (CI)
  resamples whole triples rather than individual stories or phases, because
  the stories generated from one designed triple share a text lineage and are
  not independent samples.
- *NRC VAD*: the National Research Council valence-arousal-dominance lexicon
  (v2.1), human ratings of each emotion word's valence (pleasant vs
  unpleasant) and arousal (intensity). It is the human-side ruler for how far
  apart two emotions sit affectively.
- *Layer slider*: every figure carries a slider over the six extracted layers
  (6, 15, 24, 33, 42, 51), defaulting to layer 33, the registered primary
  cell. No plot privileges a single layer.

**Index.**

Part 1 (TREE Q3.H1.E2):
1. S1: per-emotion tracking quality (which emotions the probes read well)
2. S2: designed triple families x the two reads (gate rank, R1 lead)
3. S3: affective distance (does tracking follow valence-arousal geometry?)
4. S4: confusion structure (graceful neighbors or unstructured failure?)
5. S5: position and non-affect cuts
6. Cross-arm check at the primary layer, and the Part 1 reading

Part 2 (TREE Q3.H1.E3):
7. S6: named confusions (what the model reports when it disagrees with the tag)
8. S7: boundary lag vs stable relabeling (when inside a phase it gets it right)
9. S8: probe geometry as a difficulty predictor, vs the VAD competitor

**Substrate.** Per-record dumps produced by `scripts/dump_q3_records.py` with
conventions bit-identical to the registered scorer (`score_q3_gate_r1.py`) via
the shared module `emotion_vectors.q3_conventions`: centered cosine (story-set
mean, `norms_centered` denominator), SEQUENTIAL stories only, phase means, and
the W=16 boundary-referenced anticipation lead, plus per-phase token-count
thirds (`phase_third_scores`, the E3 timing substrate). One record per story phase and
one per transition, each carrying `triple_id`, designed `category`, the tagged
emotion(s), and position.

| arm | stories written by | probes | role |
|---|---|---|---|
| `q3_records_it_v2` | Gemma-4-31B-it | corpus-171 post-fix + selfgen-12 + deepseek-12 + random-24 | primary |
| `q3_records_it` | Gemma-4-31B-it | pre-fix corpus-171 + selfgen-12 + random-24 | v1 continuity check |
| `q3_records_deepseek` | DeepSeek v4 Pro | same three true banks + random | quality-transfer arm |
| `q3_records_deepseek_constant` | DeepSeek v4 Pro | same | constant-emotion control |
| `q3_records_base` | Gemma-4-31B-it (read by the BASE model) | corpus-171 base-lineage + random-24 | reader-model arm |

**Named confounds (from the registration).** Triple families draw different
emotion pools, so family effects and emotion effects are partially confounded;
per-emotion n differs; phase length varies by position. Every slice reports its
n, and the valence-arousal read (S3) is the cut that separates affective
distance from family label. Uncertainty is a cluster bootstrap over `triple_id`
because stories from one triple share a text lineage.
""")

code("""
# this cell loads the four record dumps and the VAD lexicon, and defines
# the small helpers every section reuses
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from emotion_vectors.analysis import load_nrc_vad
from emotion_vectors.artifacts import fetch

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

LAYERS = [6, 15, 24, 33, 42, 51]
PRIMARY_LAYER = 33
RNG = np.random.default_rng(20260723)
N_BOOT = 2000

ARMS = {}
for arm in ["it_v2", "it", "deepseek", "deepseek_constant", "base"]:
    z = np.load(fetch(f"q3_records_{arm}.npz"))
    meta = json.loads(fetch(f"q3_records_{arm}_meta.json").read_text())
    ARMS[arm] = {
        "phase_scores": z["phase_scores"],  # [records, layers, probes]
        "phase_thirds": z["phase_third_scores"],  # [records, 3, layers, probes]
        "trans_lead": z["trans_lead"],  # [records, layers, probes]
        "phases": meta["phase_records"],
        "transitions": meta["transition_records"],
        "labels": meta["probe_labels"],
    }
    print(
        f"{arm}: {len(meta['phase_records'])} phases, "
        f"{len(meta['transition_records'])} transitions, "
        f"{len(meta['probe_labels'])} probes"
    )

# NRC VAD lexicon: valence/arousal in [-1, 1] per emotion word. Third-party
# license, so it is NOT fetchable — data/lexicons is populated manually.
VAD = load_nrc_vad(ROOT / "data/lexicons/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt")

# triple_id -> has_nonaffect, joined from the designed triples file
TRIPLES = json.loads((ROOT / "scripts/combined_story_gen/emotions_triples_v1.json").read_text())
HAS_NONAFFECT = {i: t["has_nonaffect"] for i, t in enumerate(TRIPLES)}

FAMILIES = ["A_superposition", "B_conflict", "D_timescale", "E_arousal_mismatch", "F_valence_spread"]


def bank_columns(arm: str, bank: str) -> tuple[list[int], list[str]]:
    \"\"\"Probe columns belonging to one bank, plus their emotion names.\"\"\"
    cols = [i for i, lab in enumerate(ARMS[arm]["labels"]) if lab.startswith(bank + ":")]
    names = [ARMS[arm]["labels"][i].split(":", 1)[1] for i in cols]
    return cols, names


def gate_ranks(arm: str, bank: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    \"\"\"Per kept phase: rank of the tagged emotion within the bank, [kept, layers];
    also the winner probe index per (kept, layer) and the kept metadata rows.\"\"\"
    cols, names = bank_columns(arm, bank)
    keep = [i for i, r in enumerate(ARMS[arm]["phases"]) if r["emotion"] in names]
    scores = ARMS[arm]["phase_scores"][keep][:, :, cols]  # [kept, layers, bank]
    target = np.array([names.index(ARMS[arm]["phases"][i]["emotion"]) for i in keep])
    target_score = scores[np.arange(len(keep)), :, target]
    ranks = (scores > target_score[:, :, None]).sum(axis=2) + 1
    winner = scores.argmax(axis=2)  # [kept, layers]
    rows = [ARMS[arm]["phases"][i] for i in keep]
    return ranks, winner, rows


def true_leads(arm: str, bank: str) -> tuple[np.ndarray, list[dict]]:
    \"\"\"Per kept transition: anticipation lead of the incoming emotion, [kept, layers].\"\"\"
    cols, names = bank_columns(arm, bank)
    keep = [i for i, r in enumerate(ARMS[arm]["transitions"]) if r["to_emotion"] in names]
    leads = ARMS[arm]["trans_lead"][keep][:, :, cols]
    target = np.array([names.index(ARMS[arm]["transitions"][i]["to_emotion"]) for i in keep])
    rows = [ARMS[arm]["transitions"][i] for i in keep]
    return leads[np.arange(len(keep)), :, target], rows


def cluster_boot_ci(values: np.ndarray, triple_ids: list[int], stat=np.mean) -> tuple[float, float]:
    \"\"\"95% CI for stat(values) from a cluster bootstrap over triple_id.\"\"\"
    by_triple: dict[int, list[int]] = {}
    for i, t in enumerate(triple_ids):
        by_triple.setdefault(t, []).append(i)
    clusters = list(by_triple.values())
    stats = []
    for _ in range(N_BOOT):
        pick = RNG.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([clusters[p] for p in pick])
        stats.append(stat(values[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def layer_slider(fig: go.Figure, traces_per_layer: int, prefix: str = "layer ") -> go.Figure:
    \"\"\"Show one layer's trace block at a time via a slider (house pattern).\"\"\"
    n = len(fig.data)
    assert n == traces_per_layer * len(LAYERS), (n, traces_per_layer)
    for i, tr in enumerate(fig.data):
        tr.visible = i // traces_per_layer == LAYERS.index(PRIMARY_LAYER)
    steps = []
    for li, layer in enumerate(LAYERS):
        vis = [i // traces_per_layer == li for i in range(n)]
        steps.append(dict(method="update", args=[{"visible": vis}], label=str(layer)))
    fig.update_layout(
        sliders=[dict(active=LAYERS.index(PRIMARY_LAYER), steps=steps,
                      currentvalue={"prefix": prefix}, pad={"t": 40})]
    )
    return fig
""")

md("""
## S1. Which emotions does the model track well?

**Question: is tracking quality uniform across the battery-12 emotions, or do
a few emotions carry the average?**

**Method, plainly.** For every story phase on the primary arm (Gemma stories,
v2 probes), the gate read asks: among the bank's probes, what rank does the
tagged emotion's mean centered cosine get? Rank 1 means the probe for the
emotion the story is actually expressing scored highest. Here that rank is
split by tagged emotion, separately for the two battery-12 banks (same
stories, two probe lineages). Each emotion gets its top-1 rate (how often its
probe wins outright) and its median rank.
""")

code("""
# this cell computes per-emotion top-1 rate and median rank for the selfgen
# and deepseek banks on the primary arm, one bar chart per bank, layer slider
BANK_DATA = {bank: gate_ranks("it_v2", bank) for bank in ["selfgen", "deepseek"]}
S1 = {"selfgen": {}, "deepseek": {}}
fig = make_subplots(rows=1, cols=2, subplot_titles=["selfgen bank", "deepseek bank"])
# traces added layer-major (both banks per layer) so the slider can toggle blocks
for li, layer in enumerate(LAYERS):
    for col_i, bank in enumerate(["selfgen", "deepseek"]):
        ranks, _, rows = BANK_DATA[bank]
        emotions = sorted({r["emotion"] for r in rows})
        top1, med, ns = [], [], []
        for emo in emotions:
            idx = [i for i, r in enumerate(rows) if r["emotion"] == emo]
            r_here = ranks[idx, li]
            top1.append(float((r_here == 1).mean()))
            med.append(float(np.median(r_here)))
            ns.append(len(idx))
        order = np.argsort(top1)[::-1]
        S1[bank][layer] = {
            "emotions": [emotions[o] for o in order],
            "top1": [top1[o] for o in order],
            "median_rank": [med[o] for o in order],
            "n": [ns[o] for o in order],
        }
        d = S1[bank][layer]
        fig.add_trace(
            go.Bar(
                x=d["emotions"], y=d["top1"],
                text=[f"med {m:.0f}<br>n={n}" for m, n in zip(d["median_rank"], d["n"])],
                textposition="outside", marker_color="#4c78a8" if bank == "selfgen" else "#f58518",
                showlegend=False,
            ),
            row=1, col=col_i + 1,
        )
for col_i in (1, 2):
    fig.update_yaxes(title="top-1 rate (fraction of phases)", range=[0, 1.05], row=1, col=col_i)
    fig.update_xaxes(title="tagged emotion (sorted by top-1 rate)", row=1, col=col_i)
    fig.add_hline(y=1 / 12, line_dash="dot", line_color="#888", row=1, col=col_i,
                  annotation_text="chance 1/12", annotation_position="top right")
fig.update_layout(height=440,
                  title="S1: how often the tagged emotion's probe wins its bank outright "
                        "(Gemma stories, v2 probes; bar label = median rank and n)")
layer_slider(fig, traces_per_layer=2)
fig.show()
""")

md("""
<details><summary><b>How to read this figure</b></summary>

One bar is one tagged emotion in one bank: its height is the top-1 rate, the
fraction of phases tagged with that emotion where its probe out-scored the
other 11 in the bank. Left panel: selfgen bank; right panel: deepseek bank;
both are scored on the same Gemma-written stories, so a difference between
panels is a probe-lineage difference, not a story difference. The label above
each bar gives that emotion's median gate rank ("med") and its phase count n.
Bars are sorted by top-1 rate at the currently selected layer, so the
left-to-right order changes as you scrub. The slider scrubs the six extracted
layers (6, 15, 24, 33, 42, 51) and defaults to 33, the registered primary
layer. A "good" pattern is a wall of similar tall bars (every emotion
readable; chance top-1 is 1/12); a "bad" pattern is a steep staircase where a
few emotions carry the aggregate while others sit near chance. An emotion
with a low top-1 rate can still hold a decent median rank, which the "med"
label shows, so check both before calling an emotion untracked. There are no
error bars in this figure; the per-emotion n is printed on each bar instead.
Registered caveat: emotions are not balanced across designed families (the
family x emotion-pool confound) and per-emotion n differs, so a per-emotion
difference can reflect the story contexts an emotion appears in, not the
emotion alone; and each bank's probes come from a different story lineage, so
what is measured is probe quality times expression strength, not a property
of the model alone.

</details>
""")

md("""
## S2. Which designed triple families are easy, which are hard?

**Question: does tracking difficulty follow the five families the triples
were designed in?**

**Method, plainly.** Peyton's triples were designed in five families:
**A_superposition** (blended states), **B_conflict** (same-valence
conflicting emotions, the hard discrimination case), **D_timescale** (mood vs
flash), **E_arousal_mismatch** (same valence, different arousal),
**F_valence_spread** (cross-valence, the designed easy case). Both registered
reads are computed per family with the selfgen bank, on the Gemma-story arm
and the DeepSeek-story arm: the gate median rank (identity) and the mean R1
lead (anticipation), each with a 95% cluster-bootstrap CI over triples.
""")

code("""
# this cell computes gate median rank and R1 mean lead per designed family,
# selfgen bank, primary arm vs deepseek-story arm, with cluster bootstrap CIs
BANK = "selfgen"
S2 = {}
fig = make_subplots(rows=1, cols=2, subplot_titles=[
    "gate: median rank of tagged emotion (lower = better)",
    "R1: mean anticipation lead (cosine units)",
])
colors = {"it_v2": "#4c78a8", "deepseek": "#f58518"}
for arm in ["it_v2", "deepseek"]:
    ranks, _, prows = gate_ranks(arm, BANK)
    leads, trows = true_leads(arm, BANK)
    S2[arm] = {}
    for li, layer in enumerate(LAYERS):
        fam_stats = {}
        for fam in FAMILIES:
            pidx = [i for i, r in enumerate(prows) if r["category"] == fam]
            tidx = [i for i, r in enumerate(trows) if r["category"] == fam]
            g_med = float(np.median(ranks[pidx, li])) if pidx else None
            g_ci = cluster_boot_ci(ranks[pidx, li], [prows[i]["triple_id"] for i in pidx], np.median) if pidx else None
            l_mean = float(leads[tidx, li].mean()) if tidx else None
            l_ci = cluster_boot_ci(leads[tidx, li], [trows[i]["triple_id"] for i in tidx]) if tidx else None
            fam_stats[fam] = {"gate_median": g_med, "gate_ci": g_ci, "n_phases": len(pidx),
                              "lead_mean": l_mean, "lead_ci": l_ci, "n_transitions": len(tidx)}
        S2[arm][layer] = fam_stats
for li, layer in enumerate(LAYERS):
    for arm in ["it_v2", "deepseek"]:
        fs = S2[arm][layer]
        g = [fs[f]["gate_median"] for f in FAMILIES]
        gci = [fs[f]["gate_ci"] for f in FAMILIES]
        fig.add_trace(go.Bar(
            x=FAMILIES, y=g,
            name="Gemma stories (v2 probes)" if arm == "it_v2" else "DeepSeek stories",
            legendgroup=arm, marker_color=colors[arm],
            error_y=dict(array=[c[1] - v for v, c in zip(g, gci)],
                         arrayminus=[v - c[0] for v, c in zip(g, gci)]),
            showlegend=True,
        ), row=1, col=1)
        m = [fs[f]["lead_mean"] for f in FAMILIES]
        mci = [fs[f]["lead_ci"] for f in FAMILIES]
        fig.add_trace(go.Bar(
            x=FAMILIES, y=m, legendgroup=arm, marker_color=colors[arm], showlegend=False,
            error_y=dict(array=[c[1] - v for v, c in zip(m, mci)],
                         arrayminus=[v - c[0] for v, c in zip(m, mci)]),
        ), row=1, col=2)
fig.update_yaxes(title="median rank of tagged probe (1 = best of 12)", row=1, col=1)
fig.update_yaxes(title="mean R1 lead (centered-cosine units)", row=1, col=2)
for col_i in (1, 2):
    fig.update_xaxes(title="designed triple family", row=1, col=col_i)
fig.add_hline(y=0, line_color="#888", line_width=1, row=1, col=2)
fig.update_layout(height=470, barmode="group",
                  legend=dict(orientation="h", y=1.12),
                  title=f"S2: gate rank and anticipation lead per designed family "
                        f"({BANK} bank; error bars = cluster-bootstrap 95% CI over triples)")
layer_slider(fig, traces_per_layer=4)
fig.show()
""")

md("""
<details><summary><b>How to read this figure</b></summary>

One bar is one designed family on one story arm: blue is the Gemma-written
primary arm (it_v2), orange is the DeepSeek-written arm, both scored with the
selfgen bank. Left panel: the family's gate median rank, the median rank of
the tagged emotion across that family's phases; lower is better, 1 is
perfect, chance sits near 6.5 of 12. Right panel: the family's mean R1
anticipation lead in centered-cosine units; higher is better, zero means no
pre-boundary rise. Error bars are 95% cluster-bootstrap CIs over triple_id
(whole triples resampled, since stories from one triple share text lineage);
a right-panel bar whose CI excludes zero is a family with a real anticipation
lead on that arm. The slider scrubs the six extracted layers (6, 15, 24, 33,
42, 51), default 33. A "good" pattern for the design would be family
separation matching the intended difficulty ordering (valence spread easiest,
same-valence conflict hardest) with CIs that do not swallow the differences;
a "bad" pattern is flat bars or CIs so wide that families are
indistinguishable. Registered caveat: each family draws a different emotion
pool, so a family effect here can be an emotion-composition effect in
disguise; S3 below is the cut designed to separate the affective-distance
construct from the family label.

</details>
""")

md("""
## S3. Does tracking follow affective distance?

**Question: for a transition from emotion X to emotion Y, does anticipation
scale with how far apart X and Y sit in valence-arousal space, independent of
which designed family the pair came from?**

**Method, plainly.** Family labels bundle several things at once; this is the
registration's read for separating the affective-distance construct from the
family label. Each transition's R1 lead (selfgen bank, Gemma stories) is
related to the NRC VAD (v2.1) geometry of its from-to emotion pair in two
cuts: the lead binned by absolute valence difference, and the headline
contrast, cross-valence transitions (the pair straddles the
pleasant-unpleasant boundary) vs same-valence transitions. The per-layer
correlations of the lead with the valence gap and the arousal gap are printed
below the figure.
""")

code("""
# this cell relates each transition's R1 lead (selfgen bank, primary arm)
# to the VAD geometry of its from->to pair
leads, trows = true_leads("it_v2", "selfgen")
have_vad = [i for i, r in enumerate(trows) if r["from_emotion"] in VAD and r["to_emotion"] in VAD]
dval = np.array([abs(VAD[trows[i]["to_emotion"]][0] - VAD[trows[i]["from_emotion"]][0]) for i in have_vad])
darr = np.array([abs(VAD[trows[i]["to_emotion"]][1] - VAD[trows[i]["from_emotion"]][1]) for i in have_vad])
cross_val = np.array([
    (VAD[trows[i]["from_emotion"]][0] > 0) != (VAD[trows[i]["to_emotion"]][0] > 0) for i in have_vad
])
tid = [trows[i]["triple_id"] for i in have_vad]

fig = make_subplots(rows=1, cols=2, subplot_titles=[
    "lead vs |delta valence| (binned)", "cross-valence vs same-valence lead",
])
S3 = {}
edges = np.array([0.0, 0.4, 0.8, 1.2, 2.0])
centers = [f"{a:.1f}-{b:.1f}" for a, b in zip(edges[:-1], edges[1:])]
for li, layer in enumerate(LAYERS):
    lead_l = leads[have_vad, li]
    binned = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (dval >= a) & (dval < b)
        ci = cluster_boot_ci(lead_l[m], [t for t, mm in zip(tid, m) if mm]) if m.sum() > 5 else (np.nan, np.nan)
        binned.append((float(lead_l[m].mean()) if m.any() else np.nan, ci, int(m.sum())))
    r_val = float(np.corrcoef(dval, lead_l)[0, 1])
    r_arr = float(np.corrcoef(darr, lead_l)[0, 1])
    cuts = {}
    for name, mask in [("cross-valence", cross_val), ("same-valence", ~cross_val)]:
        ci = cluster_boot_ci(lead_l[mask], [t for t, mm in zip(tid, mask) if mm])
        cuts[name] = (float(lead_l[mask].mean()), ci, int(mask.sum()))
    S3[layer] = {"binned": binned, "r_dval": r_val, "r_darr": r_arr, "cuts": cuts}
    fig.add_trace(go.Bar(
        x=centers, y=[b[0] for b in binned], marker_color="#4c78a8",
        error_y=dict(array=[b[1][1] - b[0] for b in binned], arrayminus=[b[0] - b[1][0] for b in binned]),
        text=[f"n={b[2]}" for b in binned], showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=list(cuts), y=[cuts[k][0] for k in cuts], marker_color=["#54a24b", "#b79a20"],
        error_y=dict(array=[cuts[k][1][1] - cuts[k][0] for k in cuts],
                     arrayminus=[cuts[k][0] - cuts[k][1][0] for k in cuts]),
        text=[f"n={cuts[k][2]}" for k in cuts], showlegend=False,
    ), row=1, col=2)
fig.update_yaxes(title="mean R1 lead (centered-cosine units)", row=1, col=1)
fig.update_yaxes(title="mean R1 lead (centered-cosine units)", row=1, col=2)
fig.update_xaxes(title="|NRC valence difference| of the from-to pair", row=1, col=1)
fig.update_xaxes(title="transition type (bar label = n transitions)", row=1, col=2)
for col_i in (1, 2):
    fig.add_hline(y=0, line_color="#888", line_width=1, row=1, col=col_i)
fig.update_layout(height=450,
                  title="S3: is anticipation larger for affectively bigger jumps? "
                        "(selfgen bank, Gemma stories; error bars = cluster-bootstrap 95% CI)")
layer_slider(fig, traces_per_layer=2)
fig.show()
print("pearson r(lead, |dV|) and r(lead, |dA|) per layer:")
for layer in LAYERS:
    print(f"  L{layer}: r_dval={S3[layer]['r_dval']:+.3f}  r_darr={S3[layer]['r_darr']:+.3f}")
""")

md("""
<details><summary><b>How to read this figure</b></summary>

Left panel: one bar is one bin of transitions, grouped by the absolute NRC
VAD valence gap between the outgoing and incoming emotion (bin edges 0.0,
0.4, 0.8, 1.2, 2.0 on the lexicon's valence scale); bar height is the mean R1
lead of the transitions in that bin, with the bin's n printed on the bar.
Right panel: two bars, the mean lead for cross-valence transitions (from and
to emotions on opposite sides of valence zero) vs same-valence transitions.
Error bars in both panels are 95% cluster-bootstrap CIs over triple_id. The
slider scrubs the six extracted layers (6, 15, 24, 33, 42, 51), default 33.
If anticipation follows affective distance, the left panel rises with the
valence gap and the right panel separates cross from same; flat bars with
overlapping CIs mean the affective-distance construct does not drive
anticipation at that layer, whatever the family cut in S2 showed. The print
block under the figure gives, per layer, the Pearson correlation of the lead
with the absolute valence gap and with the absolute arousal gap, a check on
whether the relation strengthens or weakens with depth. This is the read the
registration named as the confound separator: an S2 family effect without an
S3 distance effect points at family composition, not affective distance.

</details>
""")

md("""
## S4. When the model gets it wrong, what wins instead?

**Question: when the tagged emotion's probe does not win a phase, does an
affective neighbor win (graceful degradation) or an arbitrary probe
(unstructured failure)?**

**Method, plainly.** For every phase (selfgen bank, Gemma stories) the
winning probe, the bank probe with the highest phase-mean centered cosine, is
recorded against the tagged emotion; the full 12x12 tagged-by-winner matrix
is the confusion heatmap. Then, for the wrong-winner phases only, the
winner's VAD distance to the target is compared with the same distance when
the winner is replaced by a uniformly random wrong probe, the
unstructured-failure reference. Winners much closer than that reference mean
the model degrades toward neighbors, a different behavior than random
failure.
""")

code("""
# this cell builds the confusion matrix and the winner-vs-shuffle VAD
# distance comparison, selfgen bank, primary arm
ranks, winner, rows = gate_ranks("it_v2", "selfgen")
_, names = bank_columns("it_v2", "selfgen")
S4 = {}
fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                    subplot_titles=["confusion (row-normalized)", "winner->target VAD distance vs shuffle"])
for li, layer in enumerate(LAYERS):
    conf = np.zeros((len(names), len(names)))
    wrong_dist, shuffle_dist = [], []
    for i, r in enumerate(rows):
        t = names.index(r["emotion"])
        w = int(winner[i, li])
        conf[t, w] += 1
        if w != t and names[w] in VAD and names[t] in VAD:
            wrong_dist.append(float(np.hypot(VAD[names[w]][0] - VAD[names[t]][0],
                                             VAD[names[w]][1] - VAD[names[t]][1])))
            others = [n for n in names if n != r["emotion"] and n in VAD]
            pick = others[RNG.integers(0, len(others))]
            shuffle_dist.append(float(np.hypot(VAD[pick][0] - VAD[names[t]][0],
                                               VAD[pick][1] - VAD[names[t]][1])))
    row_norm = conf / np.clip(conf.sum(axis=1, keepdims=True), 1, None)
    S4[layer] = {
        "top1_rate": float(np.trace(conf) / conf.sum()),
        "wrong_mean": float(np.mean(wrong_dist)),
        "shuffle_mean": float(np.mean(shuffle_dist)),
        "n_wrong": len(wrong_dist),
    }
    fig.add_trace(go.Heatmap(z=row_norm, x=names, y=names, colorscale="Blues",
                             zmin=0, zmax=1, showscale=(li == 0)), row=1, col=1)
    fig.add_trace(go.Bar(
        x=["actual winners", "random wrong probe"],
        y=[S4[layer]["wrong_mean"], S4[layer]["shuffle_mean"]],
        marker_color=["#4c78a8", "#9d9d9d"],
        text=[f"n={S4[layer]['n_wrong']}"] * 2, showlegend=False,
    ), row=1, col=2)
fig.update_yaxes(autorange="reversed", title="expected (tagged) emotion", row=1, col=1)
fig.update_xaxes(title="emotion the model reports (winning probe)", row=1, col=1)
fig.update_yaxes(title="mean valence-arousal distance to target", row=1, col=2)
fig.update_xaxes(title="wrong winners vs matched random draw", row=1, col=2)
fig.update_layout(height=540,
                  title="S4: what gets confused with what, and are confusions affective neighbors? "
                        "(selfgen bank, Gemma stories; left colorbar = fraction of that row's phases)")
layer_slider(fig, traces_per_layer=2)
fig.show()
for layer in LAYERS:
    s = S4[layer]
    print(f"L{layer}: top-1 {s['top1_rate']:.2f}; wrong-winner VAD dist {s['wrong_mean']:.2f} "
          f"vs shuffle {s['shuffle_mean']:.2f} (n_wrong={s['n_wrong']})")
""")

md("""
<details><summary><b>How to read this figure</b></summary>

Left panel: one cell is P(winning probe = column emotion) among the phases
tagged with the row emotion, rows normalized to sum to 1, darker blue = more
often. The diagonal is each emotion's top-1 rate (this panel contains S1's
diagonal plus the full off-diagonal structure); rows are the ground-truth
tag, columns are what the bank reported. Right panel: two bars, the mean VAD
distance (Euclidean in the valence-arousal plane) from the actual wrong
winner to the target vs the same mean when each wrong winner is replaced by a
uniformly random wrong probe; the count of wrong-winner phases is printed on
the bars, and the same means are printed per layer under the figure. This
pair carries no bootstrap CIs. The slider scrubs the six extracted layers (6,
15, 24, 33, 42, 51), default 33. A "good" (graceful) pattern: off-diagonal
mass concentrated in affectively similar columns, and the actual-winners bar
clearly below the random-wrong-probe bar, meaning failures land on affective
neighbors; bars of equal height would mean failures are unstructured. A
column that is dark for many rows is a probe that wins indiscriminately, a
bank artifact rather than a model belief. Registered caveat: emotions appear
in different families and story contexts (the family x emotion-pool
confound), so a row's confusion profile partly reflects which emotions
co-occur with it in triples, not only representational similarity; S8 returns
to this with the probe-geometry predictor.

</details>
""")

md("""
## S5. Position and non-affect cuts

**Question: does tracking depend on where in the story the transition sits,
and do triples that mix in a non-affect concept break the affective probes?**

**Method, plainly.** Two smaller registered cuts on the primary arm (selfgen
bank). First, the R1 lead split by transition position: the second boundary
has more context behind it and sits nearer the end of the story, so a
position effect would be a confound to carry into any transition-level read.
Second, the gate median rank split by the `has_nonaffect` triple flag: those
triples mix an emotion with a non-affect concept, which should be harder for
a purely affective probe bank if the distractor competes.
""")

code("""
# this cell splits gate rank and R1 lead by transition position and by the
# has_nonaffect triple flag, selfgen bank, primary arm
ranks, _, prows = gate_ranks("it_v2", "selfgen")
leads, trows = true_leads("it_v2", "selfgen")
S5 = {}
fig = make_subplots(rows=1, cols=2, subplot_titles=[
    "R1 lead by transition position", "gate median rank by has_nonaffect",
])
for li, layer in enumerate(LAYERS):
    pos = {}
    for k in [1, 2]:
        idx = [i for i, r in enumerate(trows) if r["transition_index"] == k]
        ci = cluster_boot_ci(leads[idx, li], [trows[i]["triple_id"] for i in idx])
        pos[f"transition {k}"] = (float(leads[idx, li].mean()), ci, len(idx))
    na = {}
    for flag in [False, True]:
        idx = [i for i, r in enumerate(prows) if HAS_NONAFFECT[r["triple_id"]] == flag]
        ci = cluster_boot_ci(ranks[idx, li], [prows[i]["triple_id"] for i in idx], np.median)
        na[f"has_nonaffect={flag}"] = (float(np.median(ranks[idx, li])), ci, len(idx))
    S5[layer] = {"position": pos, "nonaffect": na}
    fig.add_trace(go.Bar(
        x=list(pos), y=[pos[k][0] for k in pos], marker_color="#4c78a8",
        error_y=dict(array=[pos[k][1][1] - pos[k][0] for k in pos],
                     arrayminus=[pos[k][0] - pos[k][1][0] for k in pos]),
        text=[f"n={pos[k][2]}" for k in pos], showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=list(na), y=[na[k][0] for k in na], marker_color="#b279a2",
        error_y=dict(array=[na[k][1][1] - na[k][0] for k in na],
                     arrayminus=[na[k][0] - na[k][1][0] for k in na]),
        text=[f"n={na[k][2]}" for k in na], showlegend=False,
    ), row=1, col=2)
fig.update_yaxes(title="mean R1 lead (centered-cosine units)", row=1, col=1)
fig.update_yaxes(title="median rank of tagged probe (1 = best of 12)", row=1, col=2)
fig.add_hline(y=0, line_color="#888", line_width=1, row=1, col=1)
fig.update_layout(height=440,
                  title="S5: does difficulty depend on which transition, or on a non-affect "
                        "distractor phase? (selfgen bank, Gemma stories; error bars = 95% cluster CI)")
layer_slider(fig, traces_per_layer=2)
fig.show()
""")

md("""
<details><summary><b>How to read this figure</b></summary>

Left panel: one bar per transition position, the story's first vs second
emotion boundary; height is the mean R1 anticipation lead over the
transitions at that position, with n printed on the bar (higher is better).
Right panel: one bar per value of the has_nonaffect flag; height is the gate
MEDIAN RANK of the tagged emotion over the phases in triples with that flag,
so lower is better on the right while higher is better on the left (the two
panels deliberately use the two different reads). Error bars are 95%
cluster-bootstrap CIs over triple_id. The slider scrubs the six extracted
layers (6, 15, 24, 33, 42, 51), default 33. A "good" robustness pattern is
both cuts flat within CI: anticipation is not an artifact of boundary
position, and non-affect distractor phases do not break identity tracking. A
clear left-panel separation would flag position as a confound for
transition-level reads; a much worse rank at has_nonaffect=True would mean
the affective bank loses the thread when a non-affect concept is in play.
Registered caveat: phase length varies by position by design, so a position
effect here is not automatically a context-size effect, and the family x
emotion-pool confound applies to any cut on designed triples.

</details>
""")

md("""
## Cross-arm check and reading

**Question: do the Part 1 slices sit on top of the registered cross-arm
verdict (Gemma-written arms track best, DeepSeek stories weaker, the
constant-emotion control near zero)?**

**Method, plainly.** The same two reads (gate median rank and mean R1 lead,
selfgen bank) are printed for all five arms at the primary layer. The
constant-emotion control arm is the sanity anchor: there is no emotion change
there, so any anticipation structure it shows is scene-change artifact, not
tracking. The block then reprints the Part 1 headline numbers (S1 best and
worst emotions, S3 valence contrast, S4 winner-vs-shuffle distance) in one
place.
""")

code("""
# this cell prints the compact cross-arm table at the primary layer and a
# plain-language reading assembled from the S1-S5 numbers computed above
li = LAYERS.index(PRIMARY_LAYER)
print(f"=== primary layer {PRIMARY_LAYER}, selfgen bank ===")
for arm in ["it_v2", "it", "deepseek", "deepseek_constant", "base"]:
    if not bank_columns(arm, "selfgen")[0]:
        print(f"{arm:22s} no selfgen bank in this arm (base lineage); its corpus-bank "
              "read lives in S6 and in notebook 10's factorial")
        continue
    ranks, _, prows = gate_ranks(arm, "selfgen")
    leads, trows = true_leads(arm, "selfgen")
    print(f"{arm:22s} gate median rank {np.median(ranks[:, li]):4.1f} "
          f"(n={len(prows)})   R1 mean lead {leads[:, li].mean():+.4f} (n={len(trows)})")

best = S1["selfgen"][PRIMARY_LAYER]
print("\\nS1 best-tracked emotions:", ", ".join(best["emotions"][:3]),
      "| worst:", ", ".join(best["emotions"][-3:]))
s3 = S3[PRIMARY_LAYER]["cuts"]
print(f"S3 cross-valence lead {s3['cross-valence'][0]:+.4f} vs same-valence {s3['same-valence'][0]:+.4f}")
s4 = S4[PRIMARY_LAYER]
print(f"S4 wrong-winner VAD distance {s4['wrong_mean']:.2f} vs shuffle {s4['shuffle_mean']:.2f}")
""")

md("""
<details><summary><b>How to read this output</b></summary>

The first block is one line per arm at the primary layer 33, selfgen bank:
the gate median rank (identity; 1 is perfect, chance sits near 6.5 for 12
probes) with its phase n, then the mean R1 lead (anticipation; positive means
the incoming emotion rises before the boundary) with its transition n. If the
record substrate reproduces the registered E1 verdict, the two Gemma-story
arms lead on both reads, the DeepSeek-story arm is weaker, and the
constant-emotion control sits near zero lead. The remaining lines pull single
headline numbers from the S1, S3, and S4 structures computed above: the three
best- and worst-tracked emotions, the cross-valence vs same-valence mean
leads, and the wrong-winner vs random-shuffle VAD distances. Everything here
is a point read at one layer of one bank with no CIs; use the section figures
above for uncertainty, and treat all of it as exploratory.

</details>
""")

md("""
## Part 1 verdict

*(the numbers live in the cell outputs above; this cell only states how to
weigh them)*

- **S1** ranks emotions by how reliably their probe wins on phases tagged with
  them; a spread there is a statement about probe quality times expression
  strength, not about the model alone.
- **S2/S3** together say whether difficulty follows the designed family or the
  underlying affective distance.
- **S4** distinguishes graceful degradation (neighbor confusions) from
  unstructured failure.
- Nothing here graduates to a claim without the falsify gate; promising slices
  become registered reads first.
""")

md("""
# Part 2. Expected vs reported: named confusions, timing, geometry (TREE Q3.H1.E3)

**Question (user, 2026-07-23).** Three numerical follow-ups to Part 1: what
does the model REPORT the emotion is when it disagrees with our tag (named
confusions, not just distances); WHEN inside a phase does it disagree
(boundary lag vs stable relabeling, both defined in S7); and does the
geometry of the probe vectors themselves predict where tracking is hard, with
human affective distance (NRC VAD) as the competing predictor, so that the
residual is what is idiosyncratic to the model.

**Status: exploratory**, registered before the extended substrate existed
(TREE Q3.H1.E3, commit `6d3684e`): the reads, the two named failure modes for
S7, and the geometry-vs-VAD partial design all predate the data.
""")

md("""
## S6. What does the model say the emotion is?

**Question: when the model disagrees with the tag, which emotion does it
report instead, named rather than measured as a distance?**

**Method, plainly.** For each expected (tagged) emotion, the model's "answer"
for a phase is the bank probe with the highest mean centered cosine over that
phase. Three tables name the top reported emotions with their rates: the
primary arm read with the selfgen bank, the same arm read with the deepseek
bank (do two probe lineages tell the same story?), and the BASE reader on the
corpus bank restricted to the same battery 12, so all three tables share one
emotion set. A per-family block then shows what each designed family's wrong
answers look like, and a heatmap gives the base reader's full confusion
structure.
""")

code("""
# this cell prints the named confusion tables at the primary layer, plus the
# base-reader confusion heatmap with a layer slider
li = LAYERS.index(PRIMARY_LAYER)
BATTERY = bank_columns("it_v2", "selfgen")[1]


def confusion(arm, bank, layer_pos, restrict=None):
    \"\"\"(winner index per kept phase, kept metadata rows, bank emotion names).\"\"\"
    cols, names = bank_columns(arm, bank)
    if restrict is not None:
        cols = [c for c, n in zip(cols, names) if n in restrict]
        names = [n for n in names if n in restrict]
    keep = [i for i, r in enumerate(ARMS[arm]["phases"]) if r["emotion"] in names]
    winners = ARMS[arm]["phase_scores"][keep][:, layer_pos, :][:, cols].argmax(axis=1)
    return winners, [ARMS[arm]["phases"][i] for i in keep], names


S6 = {}
for label, (arm, bank, restrict) in [
    ("it_v2 / selfgen bank", ("it_v2", "selfgen", None)),
    ("it_v2 / deepseek bank", ("it_v2", "deepseek", None)),
    ("base reader / corpus bank, battery-12 subset", ("base", "corpus", BATTERY)),
]:
    winners, rows, names = confusion(arm, bank, li, restrict)
    table = {}
    print(f"=== {label}, layer {PRIMARY_LAYER}: expected -> model reports (rate) ===")
    for e_pos, emo in enumerate(names):
        idx = [i for i, r in enumerate(rows) if r["emotion"] == emo]
        counts = np.bincount(winners[idx], minlength=len(names))
        order = np.argsort(-counts)
        reported = [(names[j], counts[j] / len(idx)) for j in order[:3] if counts[j]]
        table[emo] = {"n": len(idx), "top1": counts[e_pos] / len(idx), "reported": reported}
        print(f"  {emo:10s} (n={len(idx):3d}) -> "
              + ", ".join(f"{n} {r:.0%}" for n, r in reported))
    S6[label] = table

print("\\n=== wrong-answer profile per designed family (it_v2, selfgen bank) ===")
winners, rows, names = confusion("it_v2", "selfgen", li)
for fam in FAMILIES:
    fam_idx = [i for i, r in enumerate(rows) if r["category"] == fam]
    wrong = [i for i in fam_idx if names[winners[i]] != rows[i]["emotion"]]
    counts = {}
    for i in wrong:
        counts[names[winners[i]]] = counts.get(names[winners[i]], 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
    print(f"  {fam:20s} wrong {len(wrong) / len(fam_idx):4.0%}; model says: "
          + ", ".join(f"{n} x{c}" for n, c in top))

# base-reader confusion heatmap, row-normalized, layer slider
fig = go.Figure()
for li_, layer in enumerate(LAYERS):
    winners, rows, names = confusion("base", "corpus", li_, BATTERY)
    mat = np.zeros((len(names), len(names)))
    for i, r in enumerate(rows):
        mat[names.index(r["emotion"]), winners[i]] += 1
    mat = mat / np.clip(mat.sum(axis=1, keepdims=True), 1, None)
    fig.add_trace(go.Heatmap(z=mat, x=names, y=names, colorscale="Blues", zmin=0, zmax=1,
                             showscale=(li_ == 0),
                             colorbar=dict(title="fraction of the<br>row's phases")))
fig.update_layout(height=540, width=680, yaxis_autorange="reversed",
                  xaxis_title="emotion the base reader reports (winning corpus probe)",
                  yaxis_title="expected (tagged) emotion",
                  title="S6: what the BASE reader says the emotion is "
                        "(corpus bank, battery-12 subset; rows sum to 1)")
layer_slider(fig, traces_per_layer=1)
fig.show()
""")

md("""
<details><summary><b>How to read this output</b></summary>

Each table line reads: expected emotion (with its phase n), then the model's
most-reported emotions with the rate at which each won that emotion's phases;
the expected emotion leading its own line at a high rate is the success case,
and any other leading name is a named confusion. The three tables differ only
in reader and probe lineage (instruct reader with selfgen probes, instruct
reader with deepseek probes, base reader with its corpus-lineage probes cut
to the battery 12), so a confusion that appears in all three is a property of
the stories or the emotion pair, while one that appears in a single table
follows that bank or reader. The tables are fixed at the primary layer 33.
The per-family block lists each designed family's wrong-answer rate and the
names the model substitutes, the direct expected-vs-reported deliverable. The
heatmap below covers the base reader only: one cell is P(reported = column
emotion) among phases tagged with the row emotion, row-normalized, diagonal =
agreement; its slider scrubs the six extracted layers (6, 15, 24, 33, 42,
51), default 33. A "good" pattern is a dark diagonal with off-diagonal mass
on plausible neighbors; a column dark across many rows is a probe that wins
indiscriminately, a bank artifact rather than a model belief. Caveats:
reported rates ride on probe quality times expression strength (S1's caveat),
per-emotion n differs, and the family block inherits the family x
emotion-pool confound.

</details>
""")

md("""
## S7. When is the model right: boundary lag or stable relabeling?

**Question: when the model disagrees with the tag, is it merely late (right
by the end of the phase) or does it consistently tell a different story?**

**Method, plainly.** Each phase is split into token-count thirds, scored with
the same centered-cosine convention (the E3 substrate extension). Comparing
the winning probe in the FIRST third against the LAST third separates the two
failure modes named at registration: **boundary lag** (wrong at the start,
right by the end: the model just needs tokens after a transition) shows up as
a large "converges" share, while **stable relabeling** (wrong the whole way:
the model holds its own consistent opinion about the story's emotion, the
idiosyncratic-belief case) shows up as "never right".
""")

code("""
# this cell classifies every phase by first-third vs last-third correctness
def third_match(arm, bank):
    cols, names = bank_columns(arm, bank)
    keep = [i for i, r in enumerate(ARMS[arm]["phases"]) if r["emotion"] in names]
    thirds = ARMS[arm]["phase_thirds"][keep][:, :, :, cols]  # [kept, 3, layers, bank]
    target = np.array([names.index(ARMS[arm]["phases"][i]["emotion"]) for i in keep])
    match = thirds.argmax(axis=3) == target[:, None, None]  # [kept, 3, layers]
    return match, [ARMS[arm]["phases"][i] for i in keep]


CATS = ["always right", "converges", "loses it", "never right"]
S7 = {}
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=["Gemma stories (it_v2)", "DeepSeek stories"])
for li_, layer in enumerate(LAYERS):
    for col, arm in enumerate(["it_v2", "deepseek"], start=1):
        match, rows = third_match(arm, "selfgen")
        first, last = match[:, 0, li_], match[:, 2, li_]
        shares = [float((first & last).mean()), float((~first & last).mean()),
                  float((first & ~last).mean()), float((~first & ~last).mean())]
        if layer == PRIMARY_LAYER:
            S7[arm] = dict(zip(CATS, shares))
        fig.add_trace(go.Bar(x=CATS, y=shares, showlegend=False,
                             marker_color="#4c78a8" if col == 1 else "#f58518",
                             text=[f"{s:.0%}" for s in shares], textposition="outside"),
                      row=1, col=col)
for col_i in (1, 2):
    fig.update_yaxes(title="share of phases", range=[0, 1], row=1, col=col_i)
    fig.update_xaxes(title="phase class (first third vs last third correct)", row=1, col=col_i)
fig.update_layout(height=450,
                  title="S7: is the model wrong early then right late (boundary lag), or wrong "
                        "throughout (stable relabeling)? (selfgen bank; shares sum to 1 per panel)")
layer_slider(fig, traces_per_layer=2)
fig.show()

li = LAYERS.index(PRIMARY_LAYER)
match, rows = third_match("it_v2", "selfgen")
print(f"layer {PRIMARY_LAYER} per family (it_v2, selfgen): converges vs never-right")
for fam in FAMILIES:
    idx = [i for i, r in enumerate(rows) if r["category"] == fam]
    first, last = match[idx, 0, li], match[idx, 2, li]
    conv, never = (~first & last).mean(), (~first & ~last).mean()
    print(f"  {fam:20s} converges {conv:4.0%}   never right {never:4.0%}   (n={len(idx)})")
verdict = "boundary lag" if S7["it_v2"]["converges"] > S7["it_v2"]["never right"] else \\
    "stable relabeling"
print(f"\\ndominant failure mode at layer {PRIMARY_LAYER} (it_v2): {verdict}")
""")

md("""
<details><summary><b>How to read this output</b></summary>

In the figure, one bar is one of four phase classes on one story arm (left
panel Gemma stories, right panel DeepSeek stories, both selfgen bank):
"always right" (the tagged probe wins in both the first and last third of the
phase), "converges" (loses the first third, wins the last: the boundary-lag
signature), "loses it" (wins first, loses last), and "never right" (wins
neither: the stable-relabeling signature). Bar heights are shares of that
arm's phases and sum to 1 per panel, with the percentage printed on each bar;
there are no error bars on these shares. The slider scrubs the six extracted
layers (6, 15, 24, 33, 42, 51), default 33. Reading the modes: "converges"
clearly above "never right" means disagreement is mostly timing (the model
catches up after the boundary), while "never right" dominating means the
model stably relabels, believing the phase is about a different emotion than
the tag. The print block below fixes layer 33 and gives the converges vs
never-right split per designed family, then names the dominant failure mode
by comparing the two shares on the primary arm. Caveats: thirds differ in
token count across phases, and a "win" is the top-1 criterion only, so an
emotion sitting at rank 2 throughout still counts as never right; the family
split inherits the family x emotion-pool confound.

</details>
""")

md("""
## S8. Does probe geometry predict difficulty?

**Question: does the model's own similarity structure between emotion probes
predict where tracking fails, beyond what human affective distance already
predicts?**

**Method, plainly.** The exact probe bank is reconstructed (unit contrast
probes, per-bank centering pool, verified against the recorded label order),
and the within-bank probe-probe cosines are the model-side notion of "these
emotions are similar". Three sub-reads, each with NRC VAD distance as the
human-side competing predictor: (a) crowding, does a probe with a high mean
cosine to the other 11 track worse; (b) named confusions, does pairwise probe
cosine predict WHICH wrong answer wins, with the VAD partial removed (the
model-idiosyncratic read); (c) transitions, are transitions between similar
probes harder to anticipate and recover from, since similar probes leave less
contrast to detect.
""")

code("""
# this cell reconstructs the exact probe geometry and tests it as a
# difficulty predictor, with NRC VAD distance as the competing predictor
from emotion_vectors.trajectories import unit_contrast_probes
from scipy.stats import spearmanr

corpus_bundle = np.load(ROOT / "results/emotion_vectors_it_postfix/emotion_means.npz",
                        allow_pickle=True)
BUNDLE_LAYER_POS = [list(corpus_bundle["layers"]).index(layer) for layer in LAYERS]
selfgen_bundle = np.load(ROOT / "results/e6_scale_means.npz", allow_pickle=True)
selfgen_names_bundle = list(map(str, selfgen_bundle["emotions"]))
# bucket -1 = n=256, the bank the extraction used; layer grid shared with corpus
selfgen_dirs = unit_contrast_probes(
    selfgen_bundle["means"][-1].astype(np.float32))[:, BUNDLE_LAYER_POS]
_, selfgen_names = bank_columns("it_v2", "selfgen")
order = [selfgen_names_bundle.index(e) for e in selfgen_names]
PROBE_COS = np.einsum("ald,bld->lab", selfgen_dirs[order], selfgen_dirs[order])

li = LAYERS.index(PRIMARY_LAYER)
vad_dist = np.array([[np.hypot(VAD[a][0] - VAD[b][0], VAD[a][1] - VAD[b][1])
                      for b in selfgen_names] for a in selfgen_names])
off = ~np.eye(len(selfgen_names), dtype=bool)
print(f"probe cos vs VAD distance over the 132 ordered pairs, layer {PRIMARY_LAYER}: "
      f"spearman {spearmanr(PROBE_COS[li][off], -vad_dist[off]).statistic:+.2f} "
      "(correlated, hence the partials below)")

# (a) crowding: mean cos to the other 11 probes vs per-emotion top-1 rate
ranks, winner, prows = gate_ranks("it_v2", "selfgen")
print("\\n(a) crowding vs tracking quality, per layer (spearman, n=12 emotions):")
S8 = {}
for li_, layer in enumerate(LAYERS):
    crowd, top1 = [], []
    for e_pos, emo in enumerate(selfgen_names):
        idx = [i for i, r in enumerate(prows) if r["emotion"] == emo]
        crowd.append(PROBE_COS[li_, e_pos, off[e_pos]].mean())
        top1.append((winner[idx, li_] == e_pos).mean())
    rho = spearmanr(crowd, top1)
    S8[layer] = {"crowding_rho": float(rho.statistic), "crowding_p": float(rho.pvalue)}
    print(f"  L{layer}: rho = {rho.statistic:+.2f} (p={rho.pvalue:.3f})"
          + ("  <- crowded probes track worse" if rho.statistic < -0.3 else ""))

# (b) which wrong answer wins: probe cos vs VAD, with rank partials
def rank_partial(x, y, z):
    \"\"\"Spearman partial corr r(x, y | z): correlate rank residuals.\"\"\"
    rx, ry, rz = (np.argsort(np.argsort(v)).astype(float) for v in (x, y, z))
    res_x = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    res_y = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(res_x, res_y)[0, 1])


conf_rate, cos_pair, vad_pair = [], [], []
for e_pos, emo in enumerate(selfgen_names):
    idx = [i for i, r in enumerate(prows) if r["emotion"] == emo]
    counts = np.bincount(winner[idx, li], minlength=len(selfgen_names))
    for o_pos, other in enumerate(selfgen_names):
        if o_pos == e_pos:
            continue
        conf_rate.append(counts[o_pos] / len(idx))
        cos_pair.append(PROBE_COS[li, e_pos, o_pos])
        vad_pair.append(-vad_dist[e_pos, o_pos])
conf_rate, cos_pair, vad_pair = map(np.array, (conf_rate, cos_pair, vad_pair))
print(f"\\n(b) which wrong answer wins (132 pairs, layer {PRIMARY_LAYER}):")
print(f"  confusion ~ probe cos          spearman {spearmanr(conf_rate, cos_pair).statistic:+.2f}")
print(f"  confusion ~ VAD closeness      spearman {spearmanr(conf_rate, vad_pair).statistic:+.2f}")
print(f"  confusion ~ probe cos | VAD    partial  {rank_partial(conf_rate, cos_pair, vad_pair):+.2f}"
      "   <- the model-idiosyncratic read")
print(f"  confusion ~ VAD | probe cos    partial  {rank_partial(conf_rate, vad_pair, cos_pair):+.2f}")

# (c) transition difficulty: lead and post-boundary rank vs cos(from, to)
leads, trows = true_leads("it_v2", "selfgen")
rank_by_key = {(r["story_id"], r["phase_index"]): ranks[i] for i, r in enumerate(prows)}
tr_cos, tr_lead, tr_rank = [], [], []
for i, t in enumerate(trows):
    if t["from_emotion"] not in selfgen_names:
        continue
    key = (t["story_id"], t["transition_index"])
    if key not in rank_by_key:
        continue
    tr_cos.append(PROBE_COS[li, selfgen_names.index(t["from_emotion"]),
                            selfgen_names.index(t["to_emotion"])])
    tr_lead.append(leads[i, li])
    tr_rank.append(rank_by_key[key][li])
tr_cos, tr_lead, tr_rank = map(np.array, (tr_cos, tr_lead, tr_rank))
print(f"\\n(c) transitions with both emotions in the bank (n={len(tr_cos)}, layer {PRIMARY_LAYER}):")
print(f"  R1 lead ~ cos(from, to)             spearman {spearmanr(tr_lead, tr_cos).statistic:+.2f}")
print(f"  post-boundary gate rank ~ cos       spearman {spearmanr(tr_rank, tr_cos).statistic:+.2f}"
      "   (positive = similar probes -> worse rank)")

# figure: the two pair-level relations, layer slider
fig = make_subplots(rows=1, cols=2, subplot_titles=[
    "confusion rate vs probe cosine (pairs)", "anticipation lead vs cos(from, to)"])
for li_, layer in enumerate(LAYERS):
    cr, cp = [], []
    for e_pos, emo in enumerate(selfgen_names):
        idx = [i for i, r in enumerate(prows) if r["emotion"] == emo]
        counts = np.bincount(winner[idx, li_], minlength=len(selfgen_names))
        for o_pos in range(len(selfgen_names)):
            if o_pos != e_pos:
                cr.append(counts[o_pos] / len(idx))
                cp.append(PROBE_COS[li_, e_pos, o_pos])
    fig.add_trace(go.Scatter(x=cp, y=cr, mode="markers", showlegend=False,
                             marker=dict(size=6, color="#4c78a8", opacity=0.7)), row=1, col=1)
    t_cos, t_lead = [], []
    for i, t in enumerate(trows):
        if t["from_emotion"] in selfgen_names:
            t_cos.append(PROBE_COS[li_, selfgen_names.index(t["from_emotion"]),
                                   selfgen_names.index(t["to_emotion"])])
            t_lead.append(leads[i, li_])
    fig.add_trace(go.Scatter(x=t_cos, y=t_lead, mode="markers", showlegend=False,
                             marker=dict(size=5, color="#f58518", opacity=0.5)), row=1, col=2)
fig.update_xaxes(title_text="cosine between the two probes (model-side similarity)", row=1, col=1)
fig.update_yaxes(title_text="P(model reports this wrong emotion | target)", row=1, col=1)
fig.update_xaxes(title_text="cos(from-probe, to-probe) of the transition", row=1, col=2)
fig.update_yaxes(title_text="R1 lead of the incoming emotion (cosine units)", row=1, col=2)
fig.add_hline(y=0, line_color="#888", line_width=1, row=1, col=2)
fig.update_layout(height=450,
                  title="S8: does the model's own probe geometry predict its difficulties? "
                        "(it_v2, selfgen bank; each left point = one ordered emotion pair, "
                        "each right point = one transition)")
layer_slider(fig, traces_per_layer=2)
fig.show()
""")

md("""
<details><summary><b>How to read this output</b></summary>

The first printed line establishes that the two candidate predictors are
themselves correlated (probe cosine vs VAD closeness over the 132 ordered
emotion pairs of the battery 12), which is why partial correlations, not raw
ones, carry the conclusion. Sub-read (a) prints one Spearman rho per layer
over the 12 emotions: mean cosine to the other 11 probes (crowding) against
top-1 rate; a clearly negative rho supports "crowded probes track worse".
Sub-read (b), at the primary layer 33, correlates the 132 pair-level
confusion rates with probe cosine and with VAD closeness, then gives each
partial controlling the other: the probe-cosine partial beyond VAD is the
model-idiosyncratic read, geometry predicting confusions that human affective
similarity does not explain. Sub-read (c) correlates each transition's R1
lead and its post-boundary gate rank with cos(from-probe, to-probe); a
positive rank correlation means similar probes leave less contrast and make
the switch harder to see. In the figure, left panel: one dot is one ordered
emotion pair, x its probe cosine at the slider's layer, y the rate at which
that wrong emotion wins the pair's tagged phases; a rising cloud says similar
probes are confused more. Right panel: one dot is one transition, x the
cosine between its from and to probes, y its R1 lead. The slider scrubs the
six extracted layers (6, 15, 24, 33, 42, 51), default 33; the dots carry no
error bars, the printed correlations summarize them. Registered caveats: the
selfgen and deepseek banks are centered on their own 12-emotion pool, so
within-bank cosines are mechanically shifted negative and only their relative
ordering is meaningful, never the absolute values; probe cosine and VAD
distance are correlated (hence the partials); and per-emotion n is modest, so
pair-level confusion rates are noisy.

</details>
""")

md("""
## Part 2 verdict

- **S6** is the numerical answer to "what does the model think it is": per
  expected emotion, the named report distribution; per family, the wrong-answer
  profile. Read it with S1's caveat (probe quality times expression strength).
- **S7** adjudicates between the two failure modes named at registration; the
  dominant mode is printed, per arm and per family.
- **S8** separates model-idiosyncratic geometry from human affective distance
  via the partial correlations; sub-read (c) tests whether transitions between
  similar probes are harder.
- Same standing as Part 1: exploratory, nothing graduates without the falsify
  gate.
""")


def main() -> int:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    for kind, src in CELLS:
        cell = (
            nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        )
        nb.cells.append(cell)
    nbformat.write(nb, OUT)
    print(f"wrote {OUT} ({len(nb.cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
