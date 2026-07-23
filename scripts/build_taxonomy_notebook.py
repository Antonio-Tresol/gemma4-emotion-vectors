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
# this cell loads the record dumps, the VAD lexicon, and the designed-triples
# flag through the notebook-11 exhibit library; every section below is
# load-call-show over emotion_vectors.taxonomy_report
import numpy as np

from emotion_vectors import taxonomy_report as tr

LAYERS, PRIMARY_LAYER = tr.LAYERS, tr.PRIMARY_LAYER

# one shared generator, threaded through the section builders in notebook
# order (S2, S3, S4, S5) so every bootstrap CI reproduces exactly
RNG = np.random.default_rng(20260723)

ARMS = tr.load_arms(["it_v2", "it", "deepseek", "deepseek_constant", "base"])
for arm, records in ARMS.items():
    print(
        f"{arm}: {len(records['phases'])} phases, "
        f"{len(records['transitions'])} transitions, "
        f"{len(records['labels'])} probes"
    )

# NRC VAD lexicon (third-party license, populated manually) and the
# triple_id -> has_nonaffect map from the designed triples file
VAD = tr.load_vad()
HAS_NONAFFECT = tr.load_has_nonaffect()
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
fig, S1 = tr.s1_top1_figure(ARMS)
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
fig, S2 = tr.s2_family_figure(ARMS, RNG)
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
fig, S3 = tr.s3_affective_distance_figure(ARMS, VAD, RNG)
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
fig, S4 = tr.s4_confusion_figure(ARMS, VAD, RNG)
fig.show()
for layer in LAYERS:
    layer_stats = S4[layer]
    print(f"L{layer}: top-1 {layer_stats['top1_rate']:.2f}; "
          f"wrong-winner VAD dist {layer_stats['wrong_mean']:.2f} "
          f"vs shuffle {layer_stats['shuffle_mean']:.2f} (n_wrong={layer_stats['n_wrong']})")
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
fig, S5 = tr.s5_position_nonaffect_figure(ARMS, HAS_NONAFFECT, RNG)
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
print("\\n".join(tr.cross_arm_lines(ARMS, S1, S3, S4)))
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
fig, S6 = tr.s6_named_confusions(ARMS)
print("\\n".join(S6["lines"]))
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
fig, S7 = tr.s7_thirds_figure(ARMS)
fig.show()
print("\\n".join(S7["lines"]))
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
fig, S8 = tr.s8_geometry_figure(ARMS, VAD)
print("\\n".join(S8["lines"]))
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
