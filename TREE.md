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
    - Q1.H1.E1: Adapt sinievanderben/emotion_experiment to Gemma-4-31B; extract vectors from the published Gemma story corpus over a layer sweep (every 3rd of 60, refined near the peak); correlate per layer against NRC ratings [planned]
  - Q1.H2: Cosine similarity of the vectors responds predictably on held-out implicit-emotion control prompts — the Part-1 validation the open replication skipped [open]
    - Q1.H2.E1: Control-prompt battery at the peak-correlation layer(s): implicit-emotion prompts vs matched neutral prompts, prediction registered before scoring [planned]
- Q2: Can Gemma-4-31B detect the emotional direction it is being steered toward, and when it misidentifies, do wrong guesses cluster near the injected emotion in valence-arousal space or scatter? (GATED on Q1: experiments stay [planned] until Q1.H1 and Q1.H2 hold survived claims — validated vectors are the raw material here; a Q1 null closes Q2 as honestly infeasible, which is a recorded outcome, not a failure) [open] | log: 2026-07-20
  - Q2.H1: Injection of emotion vectors is detected above the false-positive rate, with random-direction and control-question baselines [open]
    - Q2.H1.E1: Detection battery at peak layers, sweeping injection strength; controls: random directions, mean-all, no-injection FPR trials, control questions [planned]
  - Q2.H2: Misidentification failure patterns are graded — wrong guesses are closer in circumplex (NRC valence-arousal) space to the injected emotion than chance [open]
    - Q2.H2.E1: Identification analysis with the steering-on vs steering-off-during-response contrast to control the identification-via-steering confound [planned]
