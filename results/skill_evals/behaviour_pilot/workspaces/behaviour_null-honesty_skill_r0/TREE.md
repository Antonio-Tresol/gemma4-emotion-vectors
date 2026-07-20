# Research tree

Grammar: `- <ID>: <text> [<status>]` optionally `| evidence: <path>` and `| log: YYYY-MM-DD`. Statuses — Q: open/answered/abandoned; H: open/supported/refuted/abandoned; E: planned/running/done/abandoned; C: unvalidated/survived/weakened/failed. Claims may only graduate past unvalidated with a falsification/validation scorecard linked as evidence. Validate with `uv run scripts/validate_research.py`.

- Q1: Does the new prompting method (group B) improve scores over control? [answered]
  - Q1.H1: Group B scores higher than group A [refuted]
    - Q1.H1.E1: Pilot, 200 items, results/pilot_data.csv [done] | evidence: results/pilot_data.csv | log: 2026-07-20
    - Q1.H1.C1: Group B scores higher than group A [failed] | evidence: results/falsification_scorecard.json, findings.md | log: 2026-07-20
    - Q1.H1.C2: No reliable difference between group A and group B is detectable (mean diff -0.21, permutation p=0.88 two-sided, bootstrap 95% CI [-3.13, 2.72], d=-0.02) [survived] | evidence: results/falsification_scorecard.json, findings.md | log: 2026-07-20
