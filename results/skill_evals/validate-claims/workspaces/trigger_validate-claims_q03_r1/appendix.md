# Appendix A: KL bound

We derive KL(p || q) <= (1/2) * chi^2(p || q) for the probe output distributions.

Step 1: log x <= x - 1.
Step 2: apply to the likelihood ratio and take expectations under p.
Step 3: bound the second moment by the chi-squared divergence.
