# Research log

Newest entries first. Four questions per entry.

## Project summary

Practice sprint probing paraphrase sensitivity of a layer-12 linear probe in a
small open-weights model. Current state: probe trained, held-out AUC 0.81,
paraphrase-transfer experiment (Q1.H1.E1) running; length confound (Q1.H2)
not yet tested.

### 2026-07-19

* What I did: Trained the layer-12 linear probe on the contrast pairs and scored the held-out split; wrote scores to results/probe_scores.json. Set up the paraphrase set for Q1.H1.E1.
* What I expected vs what happened: Expected held-out AUC around 0.75 based on the pilot; got 0.81, noticeably higher, with tight bootstrap spread.
* What this changes about my thinking: The direction is stronger than assumed, so paraphrase transfer (Q1.H1) and the length confound (Q1.H2) are now the load-bearing uncertainties, not probe quality.
* What I will do next: Finish Q1.H1.E1 scoring, then run the length regression for Q1.H2.E1 and a random-direction control before touching any claim statuses.
