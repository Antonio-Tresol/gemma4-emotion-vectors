# Findings

- F1: probe transfer holds across 3 of 4 prompt families (AUROC 0.91, 0.89, 0.93; fails on family D at 0.55).
- F2: ablating layer 12 drops accuracy from 0.81 to 0.52.
- F3: the effect survives a permutation null (p = 0.002, 1000 shuffles).
- Open: F2 was measured with an older config; may need a rerun.
