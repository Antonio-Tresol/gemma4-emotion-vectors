"""Emit notebooks/10_the_sprint_story.ipynb: the narrative capstone.

The whole sprint as a reviewed story: each act states the claim, explains
the method in plain language, computes its numbers from the evidence files
in the adjacent cell, and shows the best example we found. Built for a
supervisor walkthrough; statuses quote TREE.md verbatim.

Exemplar story ids are module constants (EXEMPLAR_*) so the
best-example search can update them without touching the narrative.

    .venv/bin/python scripts/build_journey_notebook.py
    .venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/10_the_sprint_story.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

# best-example story ids, filled by the exemplar search (see docstring)
EXEMPLAR_GEMMA_STORY = "t053_seq_p5_118712f3"  # strong pre-boundary ramp, readable foreshadowing
EXEMPLAR_DEEPSEEK_STORY = "t051_seq_p3_fe29b1c4"  # clean step at the boundary, no lead
EXEMPLAR_CONTROL_STORY = "t002_seq_p5_e37ea766"  # constant emotion, flat trace


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
# 10 - The sprint story: what we asked, how we measured, what held up

**For review.** A 2-3 day replication sprint on Anthropic's emotion-concepts
result, run on `google/gemma-4-31b` (base) and `google/gemma-4-31b-it`
(instruct; the model of interest). Every claim below appears with its method
in plain language, its numbers computed from the committed evidence file in
the cell beside it, and its current status in the research tree (TREE.md).
Nothing here is asserted from memory.

**The one-paragraph summary.** Emotion directions extracted from stories
replicate the paper's geometry on the base model, and they detect, predict
preferences, and steer on the instruct model once the readout convention is
right. Along the way we caught our own instrument bugs and kept the audit
trail. The sprint's own contributions: which stories make the best probes
(generator quality, not identity; about 64 good stories suffice; diversity
taxes detection but helps fine-grained reads), and a factorial result on
dynamics (the model tracks the emotional state of what it reads; apparent
anticipation of transitions follows the author of the text, not the reader).

**Index**
1. Act I: the vectors and the circumplex (geometry replicates)
2. Act II: seven honest nulls, then the readout lesson (detection works centered)
3. Act III: the instrument bugs we caught, and what survived them
4. Act IV: behavior: preferences and causal steering
5. Act V: whose stories make the best probes (E10, E11, E12)
6. Act VI: dynamics: reading a story token by token (Q3 first tranche)
7. Act VII: exemplar stories, chosen to teach the finding
8. What we claim, what we do not, and what is next

**Terms used throughout.** An "emotion vector" is a difference of means: the
average residual-stream activation while the model reads stories about one
emotion, minus the average over all emotions, one vector per (emotion,
layer). A "probe" is that vector unit-normalized, read out by cosine
similarity. "Centered cosine" subtracts the evaluation set's own mean
activation first. "Battery" is our 12-emotion test set. Statuses in
brackets, like [survived], quote the tree.
"""

SETUP = """\
# this cell loads every evidence file the acts cite
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RES_CANDIDATES = [ROOT / "results", Path("/Users/abo-tresol/Documents/ai-safety/cbai_project/results")]
MODEL_IT = "google/gemma-4-31b-it"
MODEL_BASE = "google/gemma-4-31b"

def load_json(name):
    for base in RES_CANDIDATES:
        path = base / name
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(name)

geometry = load_json("emotion_geometry_correlations.json")
geometry_it = load_json("emotion_geometry_correlations_it.json")
geometry_demotion = load_json("geometry_pc_demotion.json")
c4_scorecard = load_json("falsify_c4_scorecard.json")
steering_a2 = load_json("steering_it_fixed/scores.json")
steering_a8 = load_json("steering_it_a8_fixed/scores.json")
preferences = load_json("preferences_it_chat_fixed/scores_postfix_probes.json")
e11 = load_json("e11_lineage.json")
e12_curve = load_json("e12_scale_curve.json")
e12_pref = load_json("e12_pref_matched_n_exploratory.json")
extraction_audit = load_json("e4b_extraction_impact.json")
raw_vs_chat = load_json("raw_vs_chat_extraction_cosine.json")
q3_gemma = load_json("q3_gate_r1_it.json")
q3_deepseek = load_json("q3_gate_r1_deepseek.json")
q3_v2 = load_json("q3_gate_r1_it_v2.json")
calibration = load_json("trajectory_instrument_calibration_it.json")
print("all evidence files loaded")
"""

ACT1_MD = """\
## Act I: the vectors and the circumplex

**Claim (C1, [survived]).** Difference-of-means emotion vectors from the
base model's residual stream reproduce the paper's circumplex geometry:
the first principal component of the 171 emotion vectors correlates with
human valence ratings at about 0.83, and instruction tuning demotes (but
does not destroy) this structure.

**Method, plainly.** Have the model read ~1,500 stories labeled with 171
emotions; average its layer activations per emotion; subtract the grand
mean. Run principal component analysis (PCA) on the 171 vectors per layer
and correlate each component with published valence and arousal norms for
the emotion words. No training, no supervision: if emotional structure is
there, PCA finds it.

**Why believe it.** The correlation survived the extraction-bug
re-collection (Act III) with a change of 0.004, and it sits inside its own
bootstrap confidence interval. The figure below shows every measured
layer: the base model holds a 0.78-0.83 plateau across the whole late
band (layers 33-57), and the instruct demotion spans that same band while
early layers keep valence on the first component. The deep exhibit is
notebook 02.
"""

ACT1_CODE = """\
# this cell prints Act I's numbers from the evidence files
base_layer, base_row = max(
    geometry["per_layer"].items(), key=lambda kv: abs(kv[1]["pc1_valence"]["pearson_r"])
)
it_pc1_33 = geometry_it["per_layer"]["33"]["pc1_valence"]["pearson_r"]
print(f"base model  | peak PC1-valence |r| = {abs(base_row['pc1_valence']['pearson_r']):.3f} "
      f"at layer {base_layer}")
print(f"instruct    | PC1-valence |r| at layer 33 = {abs(it_pc1_33):.3f} (the demotion); "
      f"valence resurfaces as PC{geometry_demotion['it']['best_pc']} "
      f"at |r| = {geometry_demotion['it']['best_r']:.3f}")
geo = extraction_audit["lineages"]["corpus_base"]["geometry"]
print(f"post-fix robustness check (E4b, base geometry): peak |r| "
      f"{geo['peak_abs_r_before']:.4f} -> {geo['peak_abs_r_after']:.4f} "
      f"(delta {geo['peak_abs_r_delta']:.4f})")

# per-layer view (the layer rule: no measure privileges a single layer)
base_curve = {int(k): abs(v["pc1_valence"]["pearson_r"]) for k, v in geometry["per_layer"].items()}
it_curve = {int(k): abs(v["pc1_valence"]["pearson_r"]) for k, v in geometry_it["per_layer"].items()}
fig = go.Figure()
fig.add_scatter(x=sorted(base_curve), y=[base_curve[L] for L in sorted(base_curve)],
                mode="lines+markers", name=f"{MODEL_BASE} (base)")
fig.add_scatter(x=sorted(it_curve), y=[it_curve[L] for L in sorted(it_curve)],
                mode="lines+markers", name=f"{MODEL_IT} (instruct)")
fig.update_layout(
    title="PC1-valence correlation at every measured layer (Act I's claim, all layers)",
    xaxis_title="layer", yaxis_title="|Pearson r| between PC1 and valence norms",
    width=950, height=420, margin=dict(t=70))
fig.show()
"""

ACT2_MD = """\
## Act II: seven honest nulls, then the readout lesson

**Claim (C4, [survived] its falsify gate).** The vectors detect implicit
emotion in scenario texts, but only under a centered readout: subtract the
scenario set's own mean activation before taking cosines. Under the naive
uncentered readout, seven pre-registered experiments returned honest nulls.

**Method, plainly.** Write 12 scenarios that imply an emotion without
naming it. Ask: does the target emotion's probe rank in the top 3 by
cosine? The registered bar: at least 8 of 12 correct on two independent
scenario sets at the same layer. The centering insight came from an audit
against a neighboring methodology (persona vectors): shared
"story-reading" structure buries the emotional signal unless removed.

**Why believe it.** The falsify gate attacked it four ways: a
selection-adjusted permutation null (never passes, p <= 0.0002), random
probe sets (their best dual-battery count caps at 6 of 12 at the 95th
percentile vs 10 of 12 observed), scenario-bootstrap band stability, and an
independent probe lineage converging on the same layers. All four survived.
The nulls before it are reported as nulls, conditional on their readout
family: that is what pre-registration bought us.
"""

ACT2_CODE = """\
# this cell prints Act II's gate numbers from the falsify scorecard
for arm in ("selfgen_postfix", "strong_deepseek"):
    a = c4_scorecard[arm]
    t1, t2, t3 = (a["t1_selection_adjusted_perm"], a["t2_scenario_bootstrap"],
                  a["t3_random_probe_sets"])
    print(f"{arm}: {len(a['passing_layers'])} passing layers {a['passing_layers']}, "
          f"observed dual-battery count {a['observed_max_min_count']}/12 -> {a['verdict']}")
    print(f"  t1 selection-adjusted permutation: p = {t1['p_null_any_passing_layer']:.4f} "
          f"(bar {t1['bar']}) pass={t1['pass']}")
    print(f"  t2 scenario bootstrap: best-layer stability "
          f"{max(t2['pass_stability_per_layer'].values()):.2f} (bar {t2['bar']}) pass={t2['pass']}")
    print(f"  t3 random probe sets: random p95 {t3['random_max_p95']:.0f}/12 "
          f"vs observed {a['observed_max_min_count']}/12 pass={t3['pass']}")
t4 = c4_scorecard["t4_dialogue_crosscheck"]
print(f"t4 dialogue cross-check: p = {t4['p_vs_own_null']:.4f} pass={t4['pass']} "
      f"(caveat: {t4['caveat']})")
"""

ACT3_MD = """\
## Act III: the instrument bugs we caught, and what survived

Two extraction bugs were found by our own audits, not by luck, and both
are documented with before/after measurements rather than silently fixed.

**The padding bug.** Gemma tokenizers pad left by default; two hot paths
assumed right. Consequence: some readout positions landed mid-prompt.
Every affected experiment was re-collected; the tree records per-claim
blast radius. Geometry moved 0.004 (robust); the preference correlation
moved 0.70 to 0.64 (bar still cleared); the weakest corpus contrasts
rotated hardest (their tier-2 verdict is quoted below).

**The extraction-format lesson.** Raw-text versus chat-template extraction
produces probe directions agreeing at only cosine 0.5-0.75 while raw
means stay above 0.9. "The emotion direction" is therefore a
three-convention-qualified object: corpus lineage, readout centering, and
extraction format. The whole sprint used raw text consistently.

**Why this act is in a review notebook.** Because the honest version of a
replication includes its instrument story; both bugs would have been
invisible without pre-registered bars to notice against.
"""

ACT3_CODE = """\
# this cell prints the audit's per-lineage verdicts and the format lesson
for lineage, verdict in extraction_audit["verdicts"].items():
    print(f"{lineage:24s} {verdict}")
per_layer = raw_vs_chat["per_layer"]
means = [v["contrast_cos_mean"] for v in per_layer.values()]
raw_means = [v["raw_cos_mean"] for v in per_layer.values()]
worst_layer, worst = min(per_layer.items(), key=lambda kv: kv[1]["contrast_cos_min"])
print(f"raw-vs-chat probe contrast: mean cosine {min(means):.2f}-{max(means):.2f} across layers; "
      f"worst single emotion {worst['contrast_cos_min']:.3f} "
      f"({worst['contrast_cos_argmin']}, layer {worst_layer}); "
      f"raw means stay >= {min(raw_means):.2f}")
"""

ACT4_MD = """\
## Act IV: behavior: preferences and causal steering

**Claim (H3, [supported]).** The probes predict the model's revealed
preferences: correlating "how does this activity make you feel" probe
readings with a 64-activity Elo tournament of the model's choices clears
the registered 0.5 bar at 0.64 with the post-fix probes, below the
paper's 0.71-0.74 band (the pre-fix probe read was 0.70).

**Claim (C3, [survived]).** The directions are causal: adding an emotion
vector to the residual stream while the model writes about its day shifts
the emotional valence of what it writes, in the vector's direction, at
both tested strengths, without breaking coherence.

**Method, plainly.** For preferences: run a round-robin tournament where
the model picks between activities; fit Elo scores; correlate with probe
cosines. For steering: add alpha times the unit vector at layer 33 during
generation; a blinded read scores the output's valence; the registered
bar is a sign test (at least 9 of 12 emotions shift the right way).
"""

ACT4_CODE = """\
# this cell prints Act IV's numbers from the fixed-instrument evidence
print(f"preference probe-Elo max |r| (post-fix probes): {preferences['p2_max_abs_r']:.4f} "
      f"at layer {preferences['p2_best_layer']} (registered bar {preferences['p2_registered_bar']})")
print(f"preference Elo split: positive-category mean {preferences['p1_positive_mean']:.0f} "
      f"vs negative {preferences['p1_negative_mean']:.0f} (P1 pass={preferences['p1_pass']})")
for label, scores in (("alpha=2", steering_a2), ("alpha=8", steering_a8)):
    print(f"steering {label}: valence sign test {scores['p3_sign_agreements']}/12 "
          f"(bar {scores['p3_registered_bar']}), pass={scores['p3_pass']}")

# visual: both registered bars drawn where the numbers land
fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "steering: valence sign test (bar = 9 of 12)",
    "preferences: probe-Elo max |r| (bar = 0.5)"))
sign_counts = [steering_a2["p3_sign_agreements"], steering_a8["p3_sign_agreements"]]
fig.add_bar(x=["alpha=2", "alpha=8"], y=sign_counts, text=sign_counts,
            textposition="outside", showlegend=False, row=1, col=1)
fig.add_hline(y=9, line_dash="dash", annotation_text="registered bar", row=1, col=1)
fig.add_bar(x=["post-fix probes"], y=[preferences["p2_max_abs_r"]],
            text=[f"{preferences['p2_max_abs_r']:.3f}"], textposition="outside",
            showlegend=False, row=1, col=2)
fig.add_hline(y=0.5, line_dash="dash", annotation_text="registered bar", row=1, col=2)
fig.update_layout(title=f"Act IV at a glance | {MODEL_IT}, steering at layer 33 (the only steered layer)",
                  width=950, height=400, margin=dict(t=90))
fig.show()
"""

ACT5_MD = """\
## Act V: whose stories make the best probes

The sprint's main methodological contribution, in three registered steps.

**E10.** Probes built from the model's OWN stories beat probes from an
external corpus written by a much weaker model, decisively.

**E11 (the deconfound).** But was that self-knowledge or just better
stories? A third corpus by a strong external generator (DeepSeek-v4-pro),
same instruction verbatim, answered it: the quality branch fired. Strong
external probes pass detection at 9 layers versus self-generated 5 and
weak-external 1. Generator QUALITY, not identity, drives probe function.

**E12 (the dose-response).** Per-story vectors let us re-average any
subset without touching a GPU: detection saturates near 64 stories per
emotion, and a 4x larger prompt-diversified corpus LOSES detection at
every matched size (heterogeneous settings add non-emotional variance)
while WINNING the fine-grained preference read at matched size. "Better
data" is read-dependent, not a scalar.

**Method, plainly.** Same extraction, same battery bar, same layers for
every corpus; only the author of the stories changes. The subsampling
curves use 5 random seeds per size. The full exhibit is notebook 07.
"""

ACT5_CODE = """\
# this cell prints the passing-layer ladder and the E12 dissociation
for arm, label in (("selfgen", "self-generated n=256"),
                   ("weak_external", "weak external (4B)"),
                   ("strong_external", "fixed DeepSeek n=256")):
    r1 = e11["r1_dual_battery"][arm]
    print(f"{label:24s} passing layers: {len(r1['passing_layers'])}  {r1['passing_layers']}")
fixed_points = e12_curve["arms"]["fixed"]["points"]
diverse_points = e12_curve["arms"]["diverse"]["points"]
def seed_mean(points, n):
    vals = [p["n_passing_layers"] for p in points if p["n"] == n]
    return sum(vals) / len(vals)
print(f"E12 at matched n=256 (fixed arm caps at 255 kept stories): "
      f"fixed {seed_mean(fixed_points, 255):.1f} vs diverse {seed_mean(diverse_points, 256):.1f} "
      f"passing layers (5-seed means)")
print(f"E12 preference at matched n=256: diverse {e12_pref['diverse_n256_mean']:.3f} vs fixed {e12_pref['references']['fixed_n256']:.3f}")

# visual: the passing-layer ladder and the dose-response curves
fig = make_subplots(rows=1, cols=2, subplot_titles=(
    "detection: passing layers per corpus (E10/E11)",
    "dose-response: passing layers vs corpus size (E12, seed means)"))
ladder = [("self-gen n=256", "selfgen"), ("weak external (4B)", "weak_external"),
          ("fixed DeepSeek n=256", "strong_external")]
names = [name for name, _ in ladder]
counts = [len(e11["r1_dual_battery"][arm]["passing_layers"]) for _, arm in ladder]
fig.add_bar(x=names, y=counts, text=counts, textposition="outside",
            showlegend=False, row=1, col=1)
for arm_name, color in (("fixed", "#1f77b4"), ("diverse", "#d62728")):
    points = e12_curve["arms"][arm_name]["points"]
    sizes = sorted({point["n"] for point in points})
    means = [sum(pt["n_passing_layers"] for pt in points if pt["n"] == n)
             / len([pt for pt in points if pt["n"] == n]) for n in sizes]
    fig.add_scatter(x=sizes, y=means, mode="lines+markers", name=f"{arm_name} prompt",
                    line=dict(color=color), row=1, col=2)
fig.update_xaxes(type="log", title="stories per emotion", row=1, col=2)
fig.update_yaxes(title="passing layers (of 20)")
fig.update_layout(title=f"Act V at a glance | probes read from {MODEL_IT}; deep exhibit: notebook 07",
                  width=1000, height=420, margin=dict(t=90))
fig.show()
"""

ACT6_MD = """\
## Act VI: dynamics: reading a story token by token

**The question (Q3, first tranche scored, [running]).** In stories that
move through three emotions, does the model's internal state track the
current phase, and does it move before a transition?

**Method, plainly.** For every token of 5,000+ stories, store the residual
stream's dot product onto every probe. Gate read G: within a phase, does
the tagged emotion's probe rank first among its bank? Anticipation read
R1: does the incoming emotion's cosine rise in the 16 tokens before the
boundary versus the 16 before that? Three nulls: 24 random directions,
10,000 wrong-emotion shuffles, and a constant-emotion CONTROL corpus with
the same scene-change events but no emotion change. A calibration run
measured the per-token noise floor first, so magnitudes have units.

**The factorial answer.** Identity tracking follows the probe COLUMN:
good banks (self-generated, DeepSeek) track at median rank 1-6 of 12 on
every substrate; the corpus-171 bank fails everywhere, and a full
re-extraction with post-fix probes proved that failure is not a bug
artifact. Anticipation follows the story ROW: it exists only where Gemma
authored the text, vanishes on DeepSeek-written stories, and is absent in
the control. Since the model reads causally, the surviving effect is
foreshadowing detection: Gemma-the-writer plants cues that
Gemma-the-reader picks up. The full grid is notebook 08, section 4.
"""

ACT6_CODE = """\
# this cell prints the key cells of the Q3 factorial grid at layer 33
def cell(source, arm, read, bank):
    node = source[arm][read].get(bank, {}).get("per_layer", {}).get("33")
    return node

for label, source, arm in (("Gemma stories", q3_gemma, "true_arm"),
                           ("DeepSeek stories", q3_deepseek, "true_arm"),
                           ("control", q3_gemma, "control_arm"),
                           ("Gemma stories v2 (post-fix)", q3_v2, "true_arm")):
    parts = []
    for bank in ("corpus", "selfgen", "deepseek"):
        g = cell(source, arm, "gate_G", bank)
        r = cell(source, arm, "r1_anticipation", bank)
        if g:
            lead_sd = (r["mean_lead"] / r["noise_sd"]) if r and r.get("noise_sd") else float("nan")
            parts.append(f"{bank}: rank {g['median_rank']:g}, lead {lead_sd:+.1f} sd")
    print(f"{label:28s} " + " | ".join(parts))
print(f"noise floor at layer 33 (calibration): {calibration['random_probe']['33']['cos_sd']:.4f} cosine units")

# visual: the factorial dissociation, identity and anticipation, layer slider
LAYERS_Q3 = [6, 15, 24, 33, 42, 51]
BANKS = ["corpus", "selfgen", "deepseek"]
BANK_SIZE = {"corpus": 171, "selfgen": 12, "deepseek": 12}
grid_sources = {"Gemma stories": (q3_gemma, "true_arm"),
                "DeepSeek stories": (q3_deepseek, "true_arm"),
                "control": (q3_gemma, "control_arm"),
                "Gemma v2 (post-fix)": (q3_v2, "true_arm")}
fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                    subplot_titles=("identity: rank / bank size (low = good)",
                                    "anticipation: lead in noise-sd units"))
for layer in LAYERS_Q3:
    id_z, ant_z, id_txt, ant_txt = [], [], [], []
    for label, (source, arm) in grid_sources.items():
        id_row, ant_row, id_t, ant_t = [], [], [], []
        for bank in BANKS:
            g = source[arm]["gate_G"].get(bank, {}).get("per_layer", {}).get(str(layer))
            r = source[arm]["r1_anticipation"].get(bank, {}).get("per_layer", {}).get(str(layer))
            id_row.append(g["median_rank"] / BANK_SIZE[bank] if g else None)
            id_t.append(f"{g['median_rank']:g}/{BANK_SIZE[bank]}" if g else "n/a")
            lead_sd = (r["mean_lead"] / r["noise_sd"]) if r and r.get("noise_sd") else None
            ant_row.append(lead_sd)
            ant_t.append(f"{lead_sd:+.1f}" if lead_sd is not None else "n/a")
        id_z.append(id_row); ant_z.append(ant_row); id_txt.append(id_t); ant_txt.append(ant_t)
    visible = layer == 33
    fig.add_trace(go.Heatmap(z=id_z, x=BANKS, y=list(grid_sources), text=id_txt,
                             texttemplate="%{text}", colorscale="Blues_r", zmin=0, zmax=0.6,
                             showscale=False, visible=visible), row=1, col=1)
    fig.add_trace(go.Heatmap(z=ant_z, x=BANKS, y=list(grid_sources), text=ant_txt,
                             texttemplate="%{text}", colorscale="RdBu", zmid=0,
                             showscale=False, visible=visible), row=1, col=2)
steps = []
for layer_pos, layer in enumerate(LAYERS_Q3):
    visibility = [pos == layer_pos for pos in range(len(LAYERS_Q3)) for _ in (0, 1)]
    steps.append(dict(method="update", label=f"layer {layer}", args=[{"visible": visibility}]))
fig.update_layout(sliders=[dict(active=LAYERS_Q3.index(33), steps=steps, y=-0.28,
                                currentvalue=dict(prefix="showing: "))],
                  title=f"Act VI at a glance | {MODEL_IT} reading; deep exhibit: notebook 08",
                  width=1000, height=460, margin=dict(t=90, b=130))
fig.show()
"""

ACT7_MD = """\
## Act VII: three stories that teach the finding

The statistics above are population claims; these three stories are the
clearest single examples the search found (chosen by the statistic, shown
with their text so the mechanism is visible; single stories illustrate,
they never prove).

1. **A Gemma-written story with visible foreshadowing**: watch the
   incoming emotion's trace rise BEFORE the marked boundary, and read the
   sentence that caused it.
2. **A DeepSeek-written story with a clean step**: the same probes, flat
   before the boundary, jumping after it. The transition arrives with the
   event, not before.
3. **A control story (constant emotion)**: scene changes happen, the
   trace stays put. Scene mechanics alone move nothing.

In the Gemma story (`t053_seq_p5_118712f3`, tired to joyful to loving),
the loving trace rises through the pre-boundary sentence "He stepped
forward and pulled her into his arms, burying his face in the crook of
her neck.", written before the tagged loving phase begins. In the
DeepSeek story (`t051_seq_p3_fe29b1c4`, heartbroken to proud to afraid),
the afraid phase opens with the transition event "The ferry lurched.",
and the afraid trace moves only after it. The control story
(`t002_seq_p5_e37ea766`, loving throughout) keeps its trace flat across
both marked scene changes.

Each panel: token-level trace of the relevant probes at layer 33 (layer
slider available), phase boundaries marked, story text below with the
probe heat over words. Model: gemma-4-31b-it reading in every panel.
"""

ACT7_CODE_TEMPLATE = """\
# this cell renders the three exemplar stories as linked text-trajectory panels
import numpy as np
from IPython.display import HTML, display
from transformers import AutoTokenizer

from emotion_vectors import trajectory_plots as tp
from emotion_vectors.artifacts import fetch
from emotion_vectors.linked_trajectory import linked_trajectory_html

_tok = AutoTokenizer.from_pretrained(MODEL_IT)

EXEMPLARS = [
    ("Gemma-written, foreshadowed transition", "combined_trajectories", "{gemma_id}"),
    ("DeepSeek-written, clean step", "combined_trajectories_deepseek", "{deepseek_id}"),
    ("Constant-emotion control, flat", "combined_trajectories_deepseek_constant", "{control_id}"),
]


def probe_col(labels, emotion):
    # selfgen-bank probe when the emotion is in the 12-emotion set, corpus bank otherwise
    for bank in ("selfgen", "corpus"):
        if f"{{bank}}:{{emotion}}" in labels:
            return labels.index(f"{{bank}}:{{emotion}}"), bank
    raise KeyError(emotion)


for panel_title, route, sid in EXEMPLARS:
    manifest_rows = [json.loads(l) for l in fetch(f"{{route}}/manifest.jsonl").read_text().splitlines()]
    labels = json.loads(fetch(f"{{route}}/probe_labels.json").read_text())
    layers = json.loads(fetch(f"{{route}}/run_config.json").read_text())["layers"]
    row = next(r for r in manifest_rows if r["story_id"] == sid)
    shard = np.load(fetch(f"{{route}}/shards/{{sid}}.npz"))
    emotions, starts = row["phase_emotions"], row["phase_token_starts"]
    cols = [probe_col(labels, e) for e in emotions]
    # per-token centered cosines from shard dots/norms_centered, the notebook 05/09
    # scrubber convention: story-level dot centering, smoothing window 8
    cos_by_layer = {{}}
    for i, layer in enumerate(layers):
        d = shard["dots"].astype(np.float32)[:, i]
        d = d - d.mean(0, keepdims=True)
        nc = np.clip(shard["norms_centered"].astype(np.float32)[:, i], 1e-6, None)
        cos_by_layer[layer] = tp.smooth(d[:, [c for c, _ in cols]] / nc[:, None], window=8)
    tokens = [_tok.decode([int(t)]) for t in shard["token_ids"]]
    phases = " -> ".join(f"{{e}} (t={{s}})" for e, s in zip(emotions, starts))
    banks = ", ".join(f"{{e}}:{{bank}}" for e, (_, bank) in zip(emotions, cols))
    print(f"=== {{panel_title}} | {{sid}} | expected: {{phases}} | probes: {{banks}} ===")
    display(HTML(linked_trajectory_html(
        cos_by_layer, tokens, emotions, starts, default_layer=33,
        title=f"{{MODEL_IT}} reading | {{sid}} | expected: {{phases}}",
    )))
"""

ACT8_MD = """\
## What we claim, what we do not, and what is next

**Claimed, with gates passed:** circumplex geometry on base (C1);
centered-readout detection on both models (C4); preference correlation
(H3); causal steering (C3); generator quality over identity (E11, gated
by C4's falsify pass).

**Recorded but not yet claimed:** everything in Act VI. The Q3 verdict
ladder needs its adjudication (the registered gate names the 171 bank,
which fails while the 12-banks pass in substance) and the cue-referenced
anticipation read (judges locate the foreshadowing words, blind to
activations; the boundary-referenced result predicts the lead survives
cue-referencing on Gemma stories and nothing appears on DeepSeek ones).

**Honest limitations.** One probed model family; one strong external
generator; "quality" operationalized by two points; the preference
dissociation is exploratory; probe objects are three-convention-qualified
(lineage, centering, extraction format).

**Where everything lives.** TREE.md is the claim graph with evidence
links; RESEARCH_LOG.md the day-by-day record; notebooks 01-09 the
exhibits; all corpora, vectors, and per-token trajectory substrates are
public on Hugging Face under `abotresol` so every number here reproduces
from public data without GPU access.
"""


def main() -> int:
    act7_code = ACT7_CODE_TEMPLATE.format(
        gemma_id=EXEMPLAR_GEMMA_STORY,
        deepseek_id=EXEMPLAR_DEEPSEEK_STORY,
        control_id=EXEMPLAR_CONTROL_STORY,
    )
    cells = [
        md(HEADER),
        code(SETUP),
        md(ACT1_MD),
        code(ACT1_CODE),
        md(ACT2_MD),
        code(ACT2_CODE),
        md(ACT3_MD),
        code(ACT3_CODE),
        md(ACT4_MD),
        code(ACT4_CODE),
        md(ACT5_MD),
        code(ACT5_CODE),
        md(ACT6_MD),
        code(ACT6_CODE),
        md(ACT7_MD),
        code(act7_code),
        md(ACT8_MD),
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
    out = Path("notebooks/10_the_sprint_story.ipynb")
    out.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {out} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
