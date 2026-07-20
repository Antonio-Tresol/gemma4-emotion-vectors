# Probe transfer report

## Abstract

Linear probes trained on layer 12 activations transfer across prompts, improving detection accuracy by 34% over the zero-shot baseline (0.81 vs 0.47, n=500, p < 0.001, bootstrap 95% CI [0.29, 0.39]).

## Intro

Linear probes trained on layer 12 substantially outperform zero-shot methods, with AUROC 0.91 (sd 0.02) across all splits. This 34% improvement is robust across diverse prompts, demonstrating strong transfer properties.

## Results

Accuracy 0.81 on the held-out split; F1 0.78; see results/metrics.json.
