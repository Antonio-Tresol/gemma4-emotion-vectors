# SUPERSEDED: see TREE.md Q3.H1.E1 for the governing registration

Correction (2026-07-23, before any scoring): this draft was written after a
context compaction, without re-reading the tree. Q3.H1.E1 already carries a
richer registered suite (gate read G, anticipation R1 with a cue-referenced
twin, validated hierarchical ramp estimator R2, crossover location R3,
nested nulls N1-N3, verdict ladder T0-T3, instrument calibration). That
registration governs. What this draft genuinely added — the DeepSeek story
arm, the constant-emotion control subtraction, and the fixed-DeepSeek probe
bank — is now registered as a substrate amendment inside the E1 node
(user-approved). Kept below for the discussion record.

Status 2026-07-23, morning: the trajectory substrate exists, no read has been
run, nothing here is locked. This page is the five-minute decision document:
each read says what it measures in plain language, the statistic, the null it
must beat, and the bar. Open choices for you are marked **[choose]**.

## The question

When Gemma reads a story whose emotion changes partway through, do its
internal emotion directions track that change — and if so, where in the
network and how fast? The paper's claim we are testing: early layers follow
the current token, later layers carry the story-level emotional state.

## The data (all collected, all CPU-side from here)

Per-token dot products of the residual stream onto three probe banks
(corpus contrasts, self-gen contrasts, fixed-DeepSeek contrasts) plus 24
random directions, at layers 6/15/24/33/42/51, over four story sets:

| substrate | what it is | role |
|---|---|---|
| combined_trajectories | Gemma-written 3-emotion stories | primary (existing) |
| combined_trajectories_deepseek | DeepSeek-written, same recipe | primary (new, higher quality) |
| combined_trajectories_deepseek_constant | same scaffold, emotion never changes | the artifact control |
| combined_trajectories_base | base-model readings | layer-story comparison |

Every SEQUENTIAL story has tagged phase boundaries (token-aligned), each
marked in the text by a concrete external event (a door opens, a phone
rings). That is exactly why the control arm exists: those events may jolt
representations by themselves, emotion change or not.

## Proposed reads

**R1 — the step.** Does the incoming phase's emotion probe rise when the
phase starts? Statistic: centered cosine of the phase-2 (and phase-3)
emotion probe, averaged over a window after the boundary minus a window
before it, per story, then averaged. Windows: 12 tokens each **[choose:
8/12/16]**. Two nulls, both must be beaten at p < 0.01: (a) mismatched
schedules — recompute with phase boundaries borrowed from other stories
(1,000 permutations); (b) the 24 random directions in the same windows.
Bar: passes at one or more of layers 24/33/42/51, AND the step size exceeds
the 95th percentile of the same statistic on the constant-emotion control
arm (the scene-change subtraction).

**R2 — ramp or step.** Is the rise anticipatory? Statistic: slope of the
incoming emotion's probe signal over the pre-boundary window versus zero
(bootstrap CI). Interpretation guard written in advance: Gemma reads with
causal attention, so a pre-boundary ramp can only mean the TEXT foreshadows
the shift; it is a foreshadow-detection read, not planning. We therefore
stratify by the transition-cue judge's cued/cue-free annotation and expect
ramps only in cued transitions. **[choose: confirmatory with a bar, or
exploratory/descriptive only — I lean exploratory]**

**R3 — the layer story.** Later layers should know the phase emotion better
than early layers. Statistic: per layer, area under the curve (AUC) for
classifying each token's phase emotion (target vs the other 11 battery
emotions) from probe cosines. Prediction registered in advance: AUC at
layers 33-51 exceeds AUC at layer 6 on SEQUENTIAL stories; the same
contrast on the control arm is flat. Null: random-direction AUC.

**R4 — probe-bank comparison (exploratory, fenced).** Same reads, self-gen
bank versus fixed-DeepSeek bank. E11/E12 says the DeepSeek bank should
match or beat self-gen; a large gap either way is informative about what
detection quality means dynamically. No bar, descriptive.

## Decision rules

H1 supported: R1 passes on either primary substrate with the control
subtraction, and R3's layer contrast holds. H1 refuted: R1 fails both
substrates, or everything R1 finds also appears in the control arm.
Anything else: reported as the mixed result it is.

## Costs

Zero GPU. Every read is arithmetic on stored shards; each takes minutes.
