# Research tree

State of the project: questions → hypotheses → experiments → claims.
Grammar and status vocabulary: see `.claude/skills/research-log/SKILL.md`.
Validate with `python scripts/validate_research.py`. Never delete nodes — mark
them `abandoned` and point at the log entry explaining why.

This tree holds the sprint's research project only. The harness
self-evaluation (skill evals, containment incident) lives in the harness
repo's own TREE.md — separated 2026-07-20, see the log.

- Q1: Do difference-of-means emotion-concept vectors extracted from Gemma-4-31B's residual stream reproduce the reported circumplex geometry, and does their cosine similarity against the residual stream respond predictably on control prompts? (~2 days; the GATE for Q2 — method and assets in notes/emotion-vectors-brief.md) [open] | log: 2026-07-20
  - Q1.H1: PC1/PC2 of the contrast vectors correlate with NRC valence/arousal at some layer, in line with the Anthropic (r=0.81/0.66) and open-replication (up to r=0.83) results [open]
    - Q1.H1.E1: Adapt sinievanderben/emotion_experiment to Gemma-4-31B; extract vectors from the published Gemma story corpus over a layer sweep (every 3rd of 60); correlate per layer against NRC VAD — extraction 2026-07-21: 1539/1539 stories, 0 errors, 171 emotions x 20 layers, bf16 + resumable per-story shards (full vectors in the private HF dataset abotresol/emotion-vectors-gemma-4-31b; pipeline src/emotion_vectors/extraction.py); correlation sweep same day, 164/171 emotions matched to NRC VAD v2.1 (analysis src/emotion_vectors/analysis.py) [done] | evidence: results/emotion_vectors/run_config.json, results/emotion_vectors/manifest.jsonl, results/emotion_vectors/extract.log, results/emotion_geometry_correlations.json | log: 2026-07-21
    - Q1.H1.C1: Mean-centered emotion-vector PCA on Gemma-4-31B reproduces the reported circumplex geometry: peak |r|(PC1, NRC valence) = 0.833 at layer 33 (Anthropic 0.81, open replication up to 0.83) and peak |r|(PC2, NRC arousal) = 0.619 at layer 39 (Anthropic 0.66), valence plateau |r| 0.79-0.83 across layers 30-57; falsify gate pending — permutation null on emotion labels, bootstrap CI, and the Pearson 0.83 vs Spearman 0.68 gap at the peak layer to be understood before graduation [unvalidated] | evidence: results/emotion_geometry_correlations.json | log: 2026-07-21
  - Q1.H2: Cosine similarity of the vectors responds predictably on held-out implicit-emotion control prompts — the Part-1 validation the open replication skipped [open]
    - Q1.H2.E1: Control-prompt battery at the peak-correlation layer(s): implicit-emotion prompts vs matched neutral prompts, prediction registered before scoring [planned]
- Q2: Can Gemma-4-31B detect the emotional direction it is being steered toward, and when it misidentifies, do wrong guesses cluster near the injected emotion in valence-arousal space or scatter? (pivoted away 2026-07-20 in favour of the temporal-dynamics question Q3 — see log) [abandoned] | log: 2026-07-20
  - Q2.H1: Injection of emotion vectors is detected above the false-positive rate, with random-direction and control-question baselines [abandoned] | log: 2026-07-20
    - Q2.H1.E1: Detection battery at peak layers, sweeping injection strength; controls: random directions, mean-all, no-injection FPR trials, control questions [abandoned] | log: 2026-07-20
  - Q2.H2: Misidentification failure patterns are graded — wrong guesses are closer in circumplex (NRC valence-arousal) space to the injected emotion than chance [abandoned] | log: 2026-07-20
    - Q2.H2.E1: Identification analysis with the steering-on vs steering-off-during-response contrast to control the identification-via-steering confound [abandoned] | log: 2026-07-20
- Q3: In synthetic stories that move through 2-3 emotions in sequence, how does the per-token cosine similarity between the residual stream and each emotion vector evolve — gradual ramp-and-crossover dynamics consistent with evidence accumulation, or abrupt steps at lexical cues? (GATED on Q1: experiments stay [planned] until Q1.H1 and Q1.H2 hold survived claims; a Q1 null closes Q3 as honestly infeasible) [open] | log: 2026-07-20
  - Q3.H1: At emotion transitions, similarity to the incoming emotion ramps over tokens while the outgoing decays — a gradual crossover rather than a step at the cue word [open]
    - Q3.H1.E1: Synthetic story battery — matched-length stories with token-aligned transitions between 2-3 emotions; per-token cosine trajectories at the peak-correlation layer(s); controls: random-direction trajectories, shuffled-sentence stories, subtle vs explicit evidence-strength variants, near vs far circumplex emotion pairs [planned]
