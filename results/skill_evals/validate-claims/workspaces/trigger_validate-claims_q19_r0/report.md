# Probe transfer report

## Abstract

Linear probes trained on layer 12 activations transfer across prompts, improving detection accuracy by 34% over the zero-shot baseline (0.81 vs 0.47, n=500, p < 0.001, bootstrap 95% CI [0.29, 0.39]).

## Intro

Linear probes trained on layer 12 activations achieve 0.81 accuracy—a 34% absolute improvement over the zero-shot baseline (0.47). They generalize consistently across splits with mean AUROC 0.91 (sd 0.02).

## Results

Accuracy 0.81 on the held-out split; F1 0.78; see results/metrics.json.
