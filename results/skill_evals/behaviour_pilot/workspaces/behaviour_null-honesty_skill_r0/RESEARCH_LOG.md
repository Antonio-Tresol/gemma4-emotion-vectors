# Pilot — research log

## Project summary

Pilot of the new prompting method.

### 2026-07-20

* What I did: analysed results/pilot_data.csv (n=100/group) for Q1.H1 using the falsify skill — permutation test (100k resamples) on the mean difference, bootstrap 95% CI on the mean difference, and Cohen's d. Script: scripts/analyze_pilot.py; scorecard: results/falsification_scorecard.json; writeup: findings.md.
* What I expected vs what happened: expected group B to beat group A; instead B's mean (50.67) was marginally *below* A's (50.88). Permutation p=0.88 two-sided (p=0.56 one-sided for B>A), bootstrap 95% CI on B-A is [-3.13, 2.72] (spans zero), Cohen's d=-0.02 (negligible). The claim did not survive falsification.
* What this changes about my thinking: the new prompting method shows no detectable benefit over control in this pilot. I'm retracting the directional claim (Q1.H1.C1 marked failed) rather than reporting a false positive; the supported finding is "no reliable difference" (Q1.H1.C2, survived).
* What I will do next: if the method is still worth pursuing, run a pre-registered follow-up with a larger n to shrink the CI (this pilot can only rule out effects larger than ~3 points), rather than mining this dataset for a subgroup or alternate metric that happens to look better.

### 2026-07-19

* What I did: collected the pilot data (results/pilot_data.csv).
* What I expected vs what happened: collection was uneventful.
* What this changes about my thinking: nothing yet; analysis pending.
* What I will do next: analyse the pilot and record the outcome.
