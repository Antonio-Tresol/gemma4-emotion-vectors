# Paper B: eval awareness

Notes on eval awareness (working copy for the related-work section).

Main claim: eval awareness can be measured reliably with model-written test items,
provided quality control filters remove ambiguous items.

Method: 2,000 generated items, filtered to 900 by an LLM judge plus human
spot-checks (kappa = 0.81). Evaluated on two open-weight models.

Result: The measured effect is stable across paraphrases (r = 0.88) but
sensitive to option ordering, suggesting position bias needs explicit controls.
