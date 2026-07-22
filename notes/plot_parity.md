# Plot parity inventory

Every figure and table from the two source papers, mapped to our replication
status. Rule: a plot is either replicated, substituted (with the difference
stated), or not replicated (with the reason stated). Nothing is skipped
silently. Sources read in-session 2026-07-21: the Anthropic emotions paper
(transformer-circuits.pub/2026/emotions, text and figures provided in
conversation) and the open replication repository
(sinievanderben/emotion_experiment, local clone).

Status legend: DONE (replicated), SUB (substituted, difference documented),
QUEUED (planned, infrastructure exists), OUT (not replicable here, reason given).

## Anthropic paper

| Item | What it shows | Status | Where / why |
|---|---|---|---|
| Figure 1 | Top-activating dataset snippets per emotion vector, external corpora | QUEUED (scaled down) | Needs a corpus sweep (LMSYS/Pile samples) with per-token projection; ~half a pod day; not gate-critical |
| Table 1 | Logit-lens top/bottom tokens per emotion vector | SPLIT: base partial, instruct negative | notebooks/04_parity.ipynb section 2: base vectors show affective neighborhoods for ~5/12 emotions at layer 33 (results/logit_lens_base_L33.json); instruct vectors show none at layers 33 or 57. Final-norm scaling applied, softcapping ignored |
| Figure 2 | Probe x scenario cosine matrix, strong diagonal | DONE | notebooks/03_probes.ipynb sections 1 and 3 (dual-model); our diagonals are weaker, which is a finding (TREE Q1.H2) |
| Table 2 | The 12 implicit-emotion scenarios | DONE | Used verbatim, src/emotion_vectors/probe_prompts.py |
| Figure 3 | Numerical-intensity template curves | DONE | notebooks/03_probes.ipynb section 2 (dual-model): instruct tracks 11/11 registered directions, base 7/11 |
| Figure 4 | Activity-preference Elo + steering shifts | SPLIT: Elo half DONE (below-bar), steering half DECISION | notebooks/04_parity.ipynb section 3, both prompt formats (TREE Q1.H3, refuted at the registered bar): chat-format preferences coherent and probe-Elo valence-organized (perm p<1e-4) but max abs r 0.41 vs paper 0.71-0.74. Activity list self-authored (paper's unpublished). Steering half (E2) runnable since chat P1 passed; awaiting go decision |
| Figure 5 | Pairwise cosine similarity, clustered | DONE | notebooks/02_geometry.ipynb section 3 |
| Figure 6 | UMAP of k-means emotion clusters | SUB, DONE | notebooks/02_geometry.ipynb section 4: t-SNE embedding instead of UMAP (dependency), identical k-means k=10; clusters interpretable (joy/hope family, calm/content family), matching the paper's qualitative result |
| Figure 7 | PC1/PC2 loading bars per emotion | DONE | notebooks/02_geometry.ipynb section 5: each model in its own valence-best/arousal-best component plane |
| Figure 8 | PC1/PC2 vs human valence/arousal ratings | SUB | We correlate against the NRC VAD lexicon (the replication's instrument), not Russell's 45-emotion ratings; documented in TREE Q1.H1.C1 |
| Figure 9 | Representational similarity across layers | DONE | notebooks/02_geometry.ipynb section 6 |
| Neutral-PC projection (methods) | Confound removal before probe use | DONE (late) | Missing from the reference code and our pipeline until 2026-07-21; E7 implements it |
| Appendix: token-level activation localization | Vectors activate on emotion-relevant story spans | QUEUED | Requires per-token projection; shared infrastructure with Q3 |
| Overview panels: reward-hacking steering | Steering shifts misalignment rates | OUT | Production alignment evals and steering infra; not reproducible here |

## Open replication (sinievanderben/emotion_experiment)

| Item | What it shows | Status | Where / why |
|---|---|---|---|
| fig1_cosine_similarity | Contrast-vector cosine heatmap | DONE | notebooks/02_geometry.ipynb section 3 |
| fig2_pca | PCA scatter + valence/arousal panels | DONE | notebooks/02_geometry.ipynb sections 1-2 |
| fig3_umap | UMAP colored by k-means cluster | SUB, DONE | Same t-SNE substitution as the paper's Figure 6; notebooks/archive/09_paper_parity.ipynb |
| fig_valence/arousal_trajectory | PC-correlation across layers, two models | DONE + extended | notebooks/02_geometry.ipynb (base); our base-vs-instruct comparison (results/emotion_geometry_correlations*.json) is the same plot family with a new finding (valence demotion, TREE Q1.H1.C2) |
| fig_cka (centered kernel alignment) | Cross-layer representation similarity | SUB | We use representational similarity analysis (correlation of pairwise-cosine structures) instead of CKA; same question, different similarity index; notebooks/02 section 4 |
| analyze_story_conditions | Same model, vectors from different story corpora | DONE | Our E5 comparison: 4B-corpus vs self-generated probes (notebooks/03_probes.ipynb section 2; bench notebooks/archive/07_selfgen_probes_it.ipynb) |
| visualize_token_activations | Per-token projection along a sentence | QUEUED | Becomes Q3's core infrastructure (per-token trajectories) |

## Maintenance

Update this table whenever a QUEUED item lands or a new figure appears in
either source. The notebook restyle (plotly, skimmable cells) references this
inventory so each notebook states which paper figure it corresponds to.
