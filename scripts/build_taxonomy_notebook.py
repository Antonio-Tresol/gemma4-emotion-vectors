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
claim by itself.

**Substrate.** Per-record dumps produced by `scripts/dump_q3_records.py` with
conventions bit-identical to the registered scorer (`score_q3_gate_r1.py`) via
the shared module `emotion_vectors.q3_conventions`: centered cosine (story-set
mean, `norms_centered` denominator), SEQUENTIAL stories only, phase means, and
the W=16 boundary-referenced anticipation lead. One record per story phase and
one per transition, each carrying `triple_id`, designed `category`, the tagged
emotion(s), and position.

| arm | stories written by | probes | role |
|---|---|---|---|
| `q3_records_it_v2` | Gemma-4-31B-it | corpus-171 post-fix + selfgen-12 + deepseek-12 + random-24 | primary |
| `q3_records_it` | Gemma-4-31B-it | pre-fix corpus-171 + selfgen-12 + random-24 | v1 continuity check |
| `q3_records_deepseek` | DeepSeek v4 Pro | same three true banks + random | quality-transfer arm |
| `q3_records_deepseek_constant` | DeepSeek v4 Pro | same | constant-emotion control |

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
for arm in ["it_v2", "it", "deepseek", "deepseek_constant"]:
    z = np.load(fetch(f"q3_records_{arm}.npz"))
    meta = json.loads(fetch(f"q3_records_{arm}_meta.json").read_text())
    ARMS[arm] = {
        "phase_scores": z["phase_scores"],  # [records, layers, probes]
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
## S1 — Which emotions does the model track well?

For every story phase, the gate read asks: among the bank's probes, what rank
does the tagged emotion's mean centered cosine get? Rank 1 means the probe for
the emotion the story is actually expressing scored highest. Here that rank is
split **by tagged emotion** on the primary arm (Gemma stories, v2 probes),
battery-12 banks. Bars show the top-1 rate (how often the right probe wins
outright); the label under each bar gives the median rank and n.
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
fig.update_yaxes(title="top-1 rate", range=[0, 1.05], row=1, col=1)
fig.update_layout(height=420, title="S1: per-emotion top-1 rate (primary arm, battery-12 banks)")
layer_slider(fig, traces_per_layer=2)
fig.show()
""")

md("""
## S2 — Which designed triple families are easy, which are hard?

Peyton's triples were designed in five families: **A_superposition** (blended
states), **B_conflict** (same-valence conflicting emotions, the hard
discrimination case), **D_timescale** (mood vs flash), **E_arousal_mismatch**
(same valence, different arousal), **F_valence_spread** (cross-valence, the
easy case). Both reads are shown per family with cluster-bootstrap 95% CIs.
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
            x=FAMILIES, y=g, name=f"{arm} stories", legendgroup=arm, marker_color=colors[arm],
            error_y=dict(array=[c[1] - v for v, c in zip(g, gci)],
                         arrayminus=[v - c[0] for v, c in zip(g, gci)]),
            showlegend=(li == 0),
        ), row=1, col=1)
        m = [fs[f]["lead_mean"] for f in FAMILIES]
        mci = [fs[f]["lead_ci"] for f in FAMILIES]
        fig.add_trace(go.Bar(
            x=FAMILIES, y=m, legendgroup=arm, marker_color=colors[arm], showlegend=False,
            error_y=dict(array=[c[1] - v for v, c in zip(m, mci)],
                         arrayminus=[v - c[0] for v, c in zip(m, mci)]),
        ), row=1, col=2)
fig.update_layout(height=450, barmode="group",
                  title=f"S2: designed family x read ({BANK} bank; error bars = cluster bootstrap 95% CI)")
layer_slider(fig, traces_per_layer=4)
fig.show()
""")

md("""
## S3 — Does tracking follow affective distance?

Family labels bundle several things at once. The cleaner question: for a
transition from emotion X to emotion Y, does detection scale with **how far
apart X and Y sit in valence-arousal space** (NRC VAD v2.1)? Two cuts: the
anticipation lead binned by VAD distance, and the headline contrast
cross-valence vs same-valence transitions.
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
fig.update_layout(height=430, title="S3: anticipation lead vs affective distance (selfgen bank, Gemma stories)")
layer_slider(fig, traces_per_layer=2)
fig.show()
print("pearson r(lead, |dV|) and r(lead, |dA|) per layer:")
for layer in LAYERS:
    print(f"  L{layer}: r_dval={S3[layer]['r_dval']:+.3f}  r_darr={S3[layer]['r_darr']:+.3f}")
""")

md("""
## S4 — When the model gets it wrong, what wins instead?

Failure has structure or it does not. For every phase where the tagged
emotion's probe is **not** rank 1, the winning probe is recorded. Left: the
12x12 confusion heatmap (rows = tagged emotion, columns = winner). Right: the
VAD distance from winner to target, against the same distance when the winner
is replaced by a uniformly random wrong probe (the unstructured-failure
reference). Winners much closer than the shuffle mean the model degrades
toward neighbors, which is a different behavior than random failure.
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
fig.update_yaxes(autorange="reversed", row=1, col=1)
fig.update_layout(height=520, title="S4: confusion structure (selfgen bank, Gemma stories)")
layer_slider(fig, traces_per_layer=2)
fig.show()
for layer in LAYERS:
    s = S4[layer]
    print(f"L{layer}: top-1 {s['top1_rate']:.2f}; wrong-winner VAD dist {s['wrong_mean']:.2f} "
          f"vs shuffle {s['shuffle_mean']:.2f} (n_wrong={s['n_wrong']})")
""")

md("""
## S5 — Position and non-affect triples

Two smaller cuts. First transition vs second transition: the second boundary
has more context behind it and sits nearer the end of the story. And triples
flagged `has_nonaffect` mix an emotion with a non-affect concept, which should
be harder for a purely affective probe bank.
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
fig.update_layout(height=420, title="S5: position and non-affect cuts (selfgen bank, Gemma stories)")
layer_slider(fig, traces_per_layer=2)
fig.show()
""")

md("""
## Cross-arm check and reading

The same S2 family cut on the constant-emotion control arm is the sanity
anchor: there is no emotion change there, so any family structure it shows is
scene-change artifact, not tracking.
""")

code("""
# this cell prints the compact cross-arm table at the primary layer and a
# plain-language reading assembled from the S1-S5 numbers computed above
li = LAYERS.index(PRIMARY_LAYER)
print(f"=== primary layer {PRIMARY_LAYER}, selfgen bank ===")
for arm in ["it_v2", "it", "deepseek", "deepseek_constant"]:
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
## Verdict

*(filled by the run below — the markdown here states how to read it, the
numbers live in the cell outputs above)*

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
