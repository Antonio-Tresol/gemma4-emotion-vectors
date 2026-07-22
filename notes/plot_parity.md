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
| Figure 4 | Activity-preference Elo + steering shifts | REPRODUCES (after instrument fix) | notebooks/04_parity.ipynb sections 3-4 (TREE Q1.H3 [supported]): probe-Elo max abs r 0.70 layer 33 (paper 0.71-0.74), Elo split 1727 vs -578; steering valence-signed 11/12 and 10/12, dose-responsive. Earlier negatives were padding-bug artifacts (Q1.H3.E4). Plain format dead; delta-vs-r coupling 0.32 below the paper's 0.85. Activity list self-authored |
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

## Complete census: every numbered item in the Anthropic paper (audited 2026-07-22)

The paper contains **86 numbered figures and 16 numbered tables** (main text:
Figures 1-39, Tables 1-5; appendix: Figures 40-86, Tables 6-16). The tables
above cover the main results; this census classifies everything else so no
gap is silent. Classification rules: benign per-token illustrations get a
Gemma-it analogue (QUEUED); anything needing blackmail/reward-hacking/
sycophancy rollouts or large on-policy Claude transcript corpora is OUT;
post-training (base vs RLHF) panels become base-vs-instruct proxies
(QUEUED-SUB, same substitution class as Figure 8).

### Main text, beyond the table above (Figures 10-39, Tables 3-5)

| Item | What it shows | Status | Why / cost |
|---|---|---|---|
| Figure 10 + Table 3 | User-token vs assistant-token probe activations differ on mismatch prompts | QUEUED | Author Table 3-style prompts + dual-token readout; pod hours |
| Figure 11 + Table 4 | Assistant token predicts response emotion better than user token | QUEUED | Generate responses (vLLM) + score both tokens; ~pod day |
| Figures 12-15 | Per-token/layer dynamics: prefix carry-over, dosage, negation, entity binding | QUEUED | All ride on the per-token projection infrastructure (Q3) |
| Table 5 + Figure 16 | Mixed logistic-regression probe, 15-way accuracy + max-activating snippets | QUEUED | Train mixed probe on dialogue data; ~1 pod day |
| Figures 17-20 | Self- vs other-speaker probe structure | QUEUED | Present/other-speaker extraction; dialogue infra half-built; ~1 pod day |
| Figures 21-25 | Per-token illustrations on assistant transcripts (surprise, happy, anger, agentic) | QUEUED (Gemma analogue) | Constructed analogous prompts; rides on per-token infra |
| Figures 26-35 | Blackmail / reward-hacking / sycophancy transcripts and steering rates | OUT | Production alignment evals and rollout behavior; the known blocker |
| Figures 36-39 | Post-training preserves probe structure; per-prompt activation shifts | QUEUED-SUB | Base-vs-instruct proxy for RLHF; directly validates our C2 finding; ~1 pod day; high value |

### Appendix (Figures 40-86, Tables 6-16)

| Item | What it shows | Status | Why / cost |
|---|---|---|---|
| Figures 40-51 | Per-token activation of 12 vectors on their own training stories | QUEUED (one infra) | All variants of one per-token sweep (the inventory's token-level localization item) |
| Figures 52-53 + Tables 6-8 | Steering delta-log-prob of emotion words + steered completions | QUEUED | The causal test we have never run; vectors + vLLM ready; ~1 pod day; highest value |
| Table 9 | 64 activities sorted by Elo + probe activations | DONE (ours) | results/preferences_it*/scores.json, self-authored activity substitute |
| Figure 54 + Tables 10-11 | Preference change vs steering + steered completions | Figure 54 DONE (negative, A/B-logit variant); Tables 10-11 QUEUED | Q1.H3.E2 ran at alphas 2/8: dose-robust causal null (notebooks/04 section 4). Steered free completions (the tables) not generated |
| Figure 55 | Preference correlation + steering across layers | QUEUED | Rides on Q1.H3 + E2 |
| Figure 56, Figure 58 | LLM-judged valence/arousal validation (vs preference r, vs human norms) | SUB | We use NRC VAD directly as the instrument; documented in Q1.H1.C1 |
| Table 12, Figure 57 | k-means cluster membership; PC1/PC2 circumplex projection | DONE | notebooks/02 sections 4-5 |
| Tables 13-14, Figure 59 | Present- vs other-speaker steering and vector similarity | QUEUED | Rides on Figures 17-20 extraction |
| Figures 60-65, 69-74, Table 15 | Deflection vectors: extraction, orthogonalization, antagonistic prompts | QUEUED (new vector class) | Needs a new deflection-vector extraction; ~2 pod days |
| Figures 66-68 | Deflection vectors on blackmail/reward-hacking transcripts | OUT | Same production-eval blocker |
| Figure 75 | Story vs present-speaker probes on implicit scenarios | QUEUED | Rides on present-speaker extraction |
| Figures 76-79 | Probe comparisons over large on-policy Claude transcript corpora | OUT | Corpora not public |
| Figures 80-83 | Per-token illustrations (sad user, self-aware monologue, danger, suicide care) | QUEUED (Gemma analogue) | Rides on per-token infra |
| Table 16, Figure 84 | Post-training per-emotion probe deltas; per-layer training-diff heatmap | QUEUED-SUB | Base-vs-instruct proxy; complements C2; high value |
| Figures 85-86 | Preference experiment on the base model + base-vs-post-trained consistency | QUEUED | Variant of Figure 4 (base arm of Q1.H3) |

### Census summary

~11 DONE or substituted-done, ~74 QUEUED (including appendix variants,
Gemma analogues, and post-training proxies), ~15 OUT (all on one blocker
class: blackmail/reward-hacking/sycophancy rollouts and on-policy Claude
corpora), 2 in progress. Top three queued items by scientific value:
1. Figures 52-53 (steering delta-log-prob) — the never-run causal leg.
2. Figures 36-39/84 + Table 16 (post-training proxy) — the paper's own
   quantities for our C2 instruction-tuning finding.
3. Figures 17-19/59 (self- vs other-speaker structure) — untouched Part-2
   claim with infra half-built.
