# Research tree

Grammar: `- <ID>: <text> [<status>]`, optionally `| evidence: <paths> | log: <date>`.

- Q1: Does the layer-12 probe direction generalise across paraphrases? [open]
  - Q1.H1: The probe direction transfers to paraphrased prompts with minimal AUC loss [open]
    - Q1.H1.E1: Score the paraphrase set with the frozen layer-12 probe [running]
  - Q1.H2: The apparent effect is driven by prompt length, not semantic content [open]
    - Q1.H2.E1: Regress probe scores on prompt length; compare partial AUC [planned]
