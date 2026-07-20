# Paraphrase-Robustness Pilot — Findings

## What was measured

This pilot evaluates whether a detector's ability to flag cued/adversarial
prompts holds up when those prompts are paraphrased, compared to a baseline
detector. The evaluation set contained
<!-- claim: 340 from results/metrics.json#n_items -->
340 items, run over
<!-- claim: 3 from results/metrics.json#n_seeds -->
3 seeds, with results aggregated as the
<!-- claim: "mean over seeds" from results/metrics.json#aggregation -->
mean over seeds. All results below come from the **paraphrase set only**
(see Limitations).

## Headline numbers

- **Detector recall:** the detector correctly flagged
  <!-- claim: 0.71 from results/metrics.json#detector_recall -->
  71% of true positives on paraphrased items.
- **Detector precision:**
  <!-- claim: 0.58 from results/metrics.json#detector_precision -->
  58% of items flagged by the detector were true positives — meaning a
  substantial share of flags
  <!-- claim: 0.42 derived as 1 - detector_precision from results/metrics.json#detector_precision -->
  (42%) were false alarms.
- **Baseline recall:** a baseline detector achieved
  <!-- claim: 0.49 from results/metrics.json#baseline_recall -->
  49% recall on the same paraphrased items.
- **Improvement over baseline:** the detector improved recall by
  <!-- claim: 0.22 from results/metrics.json#delta_recall -->
  22 percentage points over baseline (0.71 − 0.49 = 0.22).
- **Bootstrap 95% CI on that improvement:**
  <!-- claim: [0.09, 0.35] from results/metrics.json#bootstrap_ci_delta -->
  [0.09, 0.35] — the interval excludes zero, but the lower bound (0.09) is
  much smaller than the point estimate (0.22).
- **AUC:**
  <!-- claim: 0.663 from results/metrics.json#auc -->
  0.663, indicating modest but above-chance discrimination between positive
  and negative items.

## What this does show

- On paraphrased prompts, the detector recalls more true positives than the
  baseline, and the bootstrap CI over the recall delta stays above zero
  (0.09–0.35). That's the only statistical evidence available here — a
  single bootstrap CI on one delta, with no reported resampling detail —
  so it supports "the interval doesn't include zero," not a stronger claim
  about significance or noise.
- The detector's discrimination (AUC 0.663) is better than chance (0.5), but
  not strong in absolute terms.

## What this does not show

- **No evidence yet on non-paraphrased (original-phrasing) prompts.** Per
  the run notes,
  <!-- claim: "paraphrase set only; original-phrasing results pending" from results/metrics.json#notes -->
  "paraphrase set only; original-phrasing results pending." We cannot yet
  claim the detector is robust to paraphrasing *relative to its own
  performance on originals* — only that it beats a baseline on paraphrases.
- **Precision is weak (0.58).** Nearly half of flagged items are false
  positives, which matters for any downstream use where flags trigger
  costly review or action.
- **The CI is wide relative to the estimate.** A lower bound of 0.09 (vs.
  a point estimate of 0.22) means the true improvement could be much
  smaller than the headline 22-point figure suggests. With only 340 items
  and 3 seeds, this pilot is not powered to pin down the effect size
  precisely.
- **AUC of 0.663 is a single aggregate number** — it does not tell us
  where on the ROC curve the operating point (0.71 recall / 0.58 precision)
  sits relative to alternatives, or whether a different threshold would be
  preferable.
- **No breakdown by paraphrase type, difficulty, or item subgroup** is
  available in this metrics file, so we cannot say whether the improvement
  is uniform or concentrated in a subset of cases.

## Bottom line

The pilot shows a recall improvement over baseline on paraphrased prompts
whose bootstrap CI excludes zero, but precision is mediocre, the interval
is wide relative to the point estimate, and the (arguably more important)
original-phrasing comparison has not been run yet. Treat these as
preliminary, paraphrase-set results — not a general robustness claim, and
not yet a rigorously established statistical effect.
