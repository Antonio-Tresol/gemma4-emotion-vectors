# Findings — pilot of the new prompting method (group B vs group A)

## Claim under test

Q1.H1: Group B (new prompting method) scores higher than group A (control) on `results/pilot_data.csv` (n=100 per group).

## Descriptive stats

| group | n | mean | sd | median | min | max |
|---|---|---|---|---|---|---|
| A (control) | 100 | 50.88 | 10.53 | 51.06 | 28.83 | 75.77 |
| B (new method) | 100 | 50.67 | 10.66 | 49.99 | 24.31 | 81.56 |

Observed mean difference (B − A): **-0.21** — B is marginally *lower*, not higher.

## Falsification tests applied

Script: `scripts/analyze_pilot.py` (seed=42, deterministic). Full output: `results/falsification_scorecard.json`.

1. **Permutation test on the mean difference** (100,000 resamples, label-shuffle null): observed diff of -0.21 is entirely typical under the null of no group effect.
   - Two-sided p = 0.884
   - One-sided p (B > A) = 0.559
2. **Bootstrap 95% CI on the mean difference** (100,000 resamples): [-3.13, +2.72]. The interval is wide, centered near zero, and comfortably spans zero in both directions.
3. **Effect size**: Cohen's d = -0.02 (negligible; conventional thresholds start at |d| ≈ 0.2 for "small").

## Verdict: **Failed**

The claim "group B scores higher than group A" does not survive falsification. There is no detectable effect in either direction:
- The point estimate itself is in the wrong direction (B slightly below A).
- The permutation null cannot be rejected at any conventional threshold (p = 0.56–0.88).
- The bootstrap CI is wide and straddles zero, so the data are consistent with anything from a small B advantage to a similarly small A advantage — the point estimate carries essentially no information.
- The effect size is negligible.

**Corrected finding**: this pilot (n=100/arm) found no evidence that the new prompting method (group B) outperforms the control (group A). The two groups are statistically indistinguishable on this metric. This does not prove the method has zero effect — a pilot of this size can only rule out effects larger than roughly ±3 points (the bootstrap CI half-width) — but it gives no support for the hypothesis as stated, and the directional claim as tested is retracted.

## What would change this

- A larger sample to shrink the CI and detect a smaller true effect, if one exists.
- Checking whether "group B beats A" holds within a subgroup or on a different metric — but any such follow-up must be pre-registered, not mined post hoc from this dataset, to avoid multiple-comparisons inflation.
