# CBAI sprint — research log

## Project summary

A 2-3 day sprint replicating Anthropic's emotion-concepts finding on
Gemma-4-31B — difference-of-means emotion vectors from the residual stream,
circumplex geometry check, and the cosine-similarity control-prompt validation
the open replication skipped (Q1, the gate) — then, only if the gate passes,
testing whether the model can detect the emotional direction it is being
steered toward and whether misidentifications are graded in valence-arousal
space (Q2). Method, assets, and pitfalls distilled in
notes/emotion-vectors-brief.md. The harness self-evaluation that occupied day
one lives in the harness repo's own tree and log.

---

# Log

Newest entry first. Every entry answers the same four questions.

### 2026-07-22

* What I did: Continuation of the 2026-07-21 session past local midnight — same working session, split across entries by the log's date convention; everything below happened on the 22nd, everything before (the falsify gate, the notebook overhaul, preferences, steering, the first E8 diagnostics) is recorded in yesterday's entry and its addenda. An infrastructure-and-audit stretch that ended up moving the science more than any planned experiment. (1) E8 thinking-mode arm: gemma-4-it's tokenizer defaults to thinking-OFF, whose prompt ends by closing an empty thought channel — re-collecting the numerical templates thinking-ON doubles most excursions, produces the first panels to beat the random-direction null, and keeps 10/11 registered signs; every prior -it measurement used the (slightly worse) OFF read position. (2) The vLLM investigation the user requested: docs read (official extract_hidden_states API exists, >= 0.18, we run 0.22.1), then a definitive three-arm bench — HF loop 1,014 tok/s at 20 layers; vLLM hooks 3,400 tok/s at 1 layer but 528 at 20 (per-layer copies invert the advantage); vLLM official API 977 tok/s while persisting full per-token safetensors (~25 GB) — knowledge written to the research-harness experiment-engineering skill as two separate sections (generation vs extraction, independent capabilities, sources linked) plus runnable references, and the notebook communication contract ported to the communicate-results skill. (3) Terminology rule with a correction absorbed: define our own coinages ("battery"), leave canonical mech-interp vocabulary alone. (4) The user asked for a sanity check of our extraction against ARENA's persona-vectors material — a 16-item audit that produced the day's two biggest events.
* What I expected vs what happened: Expected the audit to bless the pipeline with minor notes. Instead: (a) INSTRUMENT BUG — Gemma-4 tokenizers pad LEFT by default and two hot paths assumed right, so E1/E3/E2's A/B logits were mostly read mid-prompt (every non-longest row per batch of 16); evidence compromised, C1/C2(H3) downgraded, H3 reopened, fixes committed (forced right padding + padding-agnostic indexing), corrected re-collections running as this is written. The TOKEN_OFFSET skip was also only fully applied to the longest story per extraction batch (impact check queued). Single-prompt collections unaffected. (b) REVERSAL — ARENA centers both sides of the cosine; our battery_matrix centered only the probe. Registered as E9 with the exploratory peek disclosed, scored on the full grid under the dual-battery rule: base passes at SEVEN configurations (up to 10/12+8/12, and 9/9 at geometry-peak layer 33), instruct passes at a layer-42 band (8/12+9/12, permutation p 0.0033/0.0002, 10,000 shuffles), and the fixed-baseline control does nothing — the effect is specifically removal of the scenario set's own shared component. H2 reopened; C4 registered unvalidated pending its own gate.
* What this changes about my thinking: Readout convention is not plumbing — it is part of the hypothesis. Seven pre-registered experiments honestly failed under the uncentered readout, and one audit-prompted convention change flipped the verdict on both models; the E1-E7 nulls stand as recorded for their readout family, which is exactly why registering the family matters. Also: silent tokenizer defaults are instrument state (padding_side joins the chat-template and thinking-mode lessons — verify, never assume), audits against a neighboring methodology (persona vectors) are cheap relative to what they catch, and every negative-result headline should name the conventions it is conditional on.
Rerun verdict addendum (same day, after the corrected collections landed): the padding bug was the
whole H3 story. Chat-format preferences under the fixed instrument: Elo split 1727 vs -578
(paper-class), max probe-Elo |r|=0.7013 at layer 33 (paper 0.71-0.74), organization perm p<1e-4 —
E3's P1 and P2 both PASS. Steering: valence-sign test passes at both doses (11/12 and 10/12, bar 9,
binomial p 0.003), dose-responsive (loving +38 -> +296, desperate -13 -> -79), coherence intact;
only the fine-grained delta-vs-r coupling stays below bar (0.32 vs the paper's 0.85, narrow
12-emotion range). Plain format remains a dead instrument. C1/C2 graduated [failed] as instrument
artifacts, C3 [survived] with permutation and binomial nulls, H3 flipped refuted -> SUPPORTED: the
paper's Figure 4 reproduces in substance. Also completed E5's registered cross-lineage comparison at
converged scale: self-generated vs external-corpus contrast directions still cos 0.219 at stability
0.997 — emotion vectors on this model are corpus-dependent objects (results/e5_cross_lineage_n256.json).

Lineage addendum (user-prompted, same day): registered head-to-head E10 under the working readouts
answers "own stories or others' stories?" decisively for detection — self-generated n=256 probes pass
the dual-battery bar across a band of layers (11/12+9/12 at layer 33, the sprint's best battery
result) while corpus probes under matched conventions pass nowhere; preference correlation is a tie
(~0.6 both). Generator = probed model, the reference's convention, is the better extraction recipe;
C4's gate should run on the self-generated lineage.

Extraction-audit addendum (user-prompted, same day: "how sure are we the emotion-vector computation
is correct? p(?)"): assessed at p ~ 0.85, uncertainty concentrated in the queued E4b impact check —
then closed most of it without a GPU. The realization: the TOKEN_OFFSET deviation is pure masking,
so its token-level blast radius is computable from tokenization alone. Registered and ran the laptop
half of E4b (scripts/audit_extraction_offset.py): the manifests' recorded n_tokens adjudicate the
mechanics story-by-story — every discriminating story across four extraction sets matches the
left-pad closed form exactly (1019+134+120+141 vs 0 intended, 0 unexplained), and the leaked-framing
share of pooled token mass is 12.1% [2.1%, 20.8%] for the corpus-171 vectors, 13.2%/7.1% for the
dialogue sets, and a predicted 4.1% [3.4%, 4.6%] for the self-gen n=256 lineage (uniform lengths =
little padding = least affected — fortunate, since provenance checks proved e6_scale_means.npz is
byte-identical to its pre-fix committed blob, so E10's winning probes and Q3's SELFGEN_PROBES all
predate the fix; nothing shipped so far was extracted post-fix). Two queued audit leftovers closed
analytically against the reference clone: recombination is token-weighted on both sides (identical
algebra; locked in tests/test_extraction_math.py with the mask-semantics closed forms). Still open:
the GPU cosine half (re-extract battery emotions post-fix, both lineages — needs the pod; no
credentials in this session) and the raw-text-vs-chat-template extraction cosine. Lesson: when a bug
is in the mask, not the model, its blast radius is a tokenizer-only computation — quantify before
spending GPU time.

E4b GPU-half addendum (session continuing past midnight into 07-23; user-directed: "go do that and
check the before and after"): all four live lineages re-extracted post-fix on cambria-longfellow
(6,278 stories, 0 errors; runner died once on a PATH-less tmux shell, relaunched), scored against
bars registered in scripts/compare_extraction_postfix.py before any post-fix vector existed. Split
verdict: raw means untouched (>=0.9975 everywhere); C1 geometry robust (peak |r| delta 0.0043 — the
headline claim survives the instrument fix); neutral subspace unchanged (E7 unaffected); self-gen
contrasts move least (min 0.994998, tier1 by 2e-6 — the scaffold-cancellation prediction from the
e11 session's Elias finding held: least-diverse corpus, least rotation); corpus contrasts fire the
tier2 bar in the low-n tail (worst 0.62 at -it L57; battery-12 subset 0.82-0.98), within C2's known
bootstrap noise floor but recorded as follow-ups, not excused: H3 probe-Elo and E9/E10 corpus
battery legs to be rescored on postfix bundles, Q3 corpus-lineage columns carry a probe-version
caveat pending rescore-or-recollect. Lineage discipline shipped: -postfix HF repos with LINEAGE.md
cards (publish launched detached on the pod), ROUTES updated, predecessors frozen. Cross-session
coordination worked and bit once: the e11 session stood down a colliding GPU launch on the first
heads-up, then swept my uncommitted artifacts.py edit into its commit from the shared clone —
exactly the AGENTS.md worktree failure mode; it moved to a worktree (branch e11) and this session
committed its work as the release. One instrument lesson for the harness: fp16 bundles overflow
float16 norms in numpy — cast before cosine (caught because a 0.0 story-cosine contradicted a 0.99
subspace agreement in the same read).

Closure addendum (late evening, user-directed "rerun everything"): the three registered E4b
follow-ups on this session's slate are done and every touched claim kept its status with amended
numbers. (1) H3 probe-Elo rescored on post-fix corpus-it probes: max |r| 0.7013 -> 0.6448 at layer
33, organization perm p<1e-4 — bar (0.5) still cleared, C3 amended from "paper strength" to "below
paper strength"; the pre-fix rotation inflated the fine-grained read. (2) C4's falsify gate ran on
the two E11 winners (self-gen postfix + DeepSeek fixed-prompt), bars and seeds registered in the
script header before running, observed grids asserted identical to E11's: SURVIVED everything —
sweep-wide selection-adjusted null never passes (p<=0.0002), 1,000 random probe sets cap at 6 vs
observed 10, band cores bootstrap-stable (0.79-0.82 at L33/39, edges fragile and scoped out),
dialogue probes converge independently (p~0). C4 graduated survived; the sprint headline is now a
gated claim. (3) Steering re-run with post-fix directions at both doses: P3 valence-sign 10/12 at
alpha-2 AND alpha-8 (bar 9), coherence intact, P2 coupling below bar (0.399/0.299, improved from
0.32/0.15) — C3's causal leg confirmed on the corrected instrument. Cross-session division held all
day: the e11 session ran E11 (quality branch fired: DeepSeek probes best, convergence-with-strength
mechanism), registered E12, and covered the E9/E10 battery rescores inside its scoring; this session
kept H3/E7/C4 and the GPU sequencing. Remaining on the board: notebook 01-04 rebuild on postfix
bundles, the Q3 corpus-probe-version decision (caveat vs re-collect), the raw-vs-chat extraction
cosine, and the origin push.

Publication addendum (user decision, same day, superseding the entry below): all project
datasets flipped PUBLIC today, ahead of the validate-claims gate — the published material is
substrate (stories, activations, shards, configs) with lineage-documented cards, not claims;
claim documents still wait for their gates. Also published today: the Q3 combined-story corpus
(5,888 stories) and the per-token trajectory substrate (5,888 shards, 6 layers x 207 probes),
with the fetch() resolver routing both.

E11 addendum (user-proposed, same day, scored just past local midnight): the third-lineage test
answered its question in one evening. A DeepSeek-v4-pro corpus (3,072 stories, identical
instruction, $0.60, leakage 0.2%) was generated, QC'd, extracted post-fix, and scored against
the E4b session's post-fix self-gen and corpus bundles with a scorer pre-validated to reproduce
E10's grid digit-for-digit. The QUALITY branch fired: strong-external probes pass the
dual-battery bar at nine layers vs self-gen's five and weak-external's one, win the preference
correlation (0.706), and sit at contrast cos 0.57 to self-gen's directions (weak: 0.22) — a
stronger generator converges toward the probed model's own directions. The exploratory diversity
covariate (E4b's suggestion) killed scaffold-degeneracy as a necessary condition: the winning
DeepSeek corpus is 53x LESS self-similar (and the Gemma self-gen corpus turns out to be almost
degenerate — 98.2% of its stories star one character). E10's recipe softens from a lineage rule
to an economy rule; E6's corpus-dependence narrows to weak generators. Full grids in
results/e11_lineage.json; corpus at abotresol/emotion-stories-deepseek-v4-pro. Coordination with
the E4b session throughout: shared GPU serialized, post-fix convention agreed pre-scoring, their
two battery-leg rescores covered by this run, one working-tree collision disclosed and repaired
(1e08190/7bc6518), this session moved to a worktree after.

E12 registration addendum (user-prompted, same day): the MORE/DIVERSE half of the E11 question,
registered as Q1.H2.E12 before the diverse corpus exists. Two arms: E11's fixed-prompt DeepSeek
corpus (n=256, per-story residuals already on the pod — the n-curve below 256 is free, CPU-only
subsampling) and a new n=1024/emotion diverse-prompt corpus (deterministic 8-personas x
8-settings grid pinned per slot; the grid is identical across emotions so 12-pool contrast
centering removes setting-content confounds by construction). Reads locked before scoring:
dual-battery pass-layer count vs n (5 seeds per n), contrast-cos-to-selfgen vs n at layer 33, and
the matched-n=256 diversity read. Smoke passed (24 stories: slot->variant verified, 0 leaks, 0
truncations, v4-pro resolved). Generation is laptop-side OpenRouter (~$2.3 of the $40); extraction
queues behind the E4b session's steering card.

Parked question addendum (user-prompted, same day): recorded Q4 in the tree — do emotion-concept
vectors causally modulate assistant behaviors, sycophancy first? Sparked by the paper's Figure 10
observation that "loving" activation rises at the Assistant colon regardless of user emotion (quote
verified verbatim against the paper in-session). Literature check (arXiv abstracts read in-session):
sycophancy steering exists via CAA and via off-the-shelf persona vectors (2605.21006 — and the
persona direction is largely independent of the sycophancy direction), emotion-vector steering is
validated for expression only (2604.04064); emotion vectors as the steering family for sycophancy
appears untested. No experiments scheduled this sprint; assets (layer-33 vectors, calibrated
steering harness, self-gen probes) make it a cheap future pickup.

Teammate note (recorded by the orchestrating session, single-writer rule): Peyton Li's Q3 prep landed
in main yesterday evening (68b7d42) — 173 emotion triples with category structure and non-affect
controls (scripts/emotions_triples_v1.json) plus an OpenRouter pilot notebook for three-emotion
stories that never name their emotions. Nothing further of Peyton's found on the pod (searched);
their empty placeholder scripts (generate_combined_stories.py, extract_emotion_vector.py) await the
Q3 pipeline build.

* What I will do next: When the corrected re-collections land: rescore E1/E3/E2 with the original registered reads and record H3's true verdict; run C4's falsify gate (bootstrap over scenarios, selection-adjusted null, dialogue-probe cross-check); quantify the extraction offset impact (re-extract battery emotions, cosine-compare); the two remaining audit items (equal-weight recombination check, raw-text vs chat-template extraction cosine); then rebuild the notebook 03/04 sections and the sprint headline around whatever survives. Q3 scope decision and census queue unchanged, now with C4's outcome as an input. Publication decision (2026-07-22): all 13 Hugging Face datasets (corpora, vectors, and the new experiment-artifacts repo covering every activation/prompt/score so replication needs no re-inference) STAY PRIVATE until sprint end; they flip public together once the final headline passes the validate-claims gate.

### 2026-07-21

* What I did: Built and ran the Q1.H1.E1 extraction stage end to end. First a measured time estimate: a reference-mirroring benchmark on the pod (RTX PRO 6000 Blackwell) gave 196 tok/s and exposed that the model load from the contended network volume (33.4 min) cost more than the extraction itself — fixed by copying the 59 GB HF cache to local NVMe, cutting loads to 46 s. Then the pipeline proper: reference-faithful math (same tokenizer call, hook capture, TOKEN_OFFSET=50 pooling mask as sinievanderben/emotion_experiment) with deliberate deviations recorded in the module docstring — bf16 (fp32 does not fit 96 GB), torch.inference_mode(), and per-story shard persistence recombined as a token-weighted mean (identical to the reference's mean, but resumable and enabling within-emotion variance analyses). Smoke run (3 emotions x 4 stories) passed including a kill-and-rerun resume check; full run extracted 1539/1539 stories, 0 errors, 171 emotions x 20 layers in 7.8 min, published as the private HF dataset abotresol/emotion-vectors-gemma-4-31b. Code promoted along the way into an installed src/emotion_vectors package split along the dependency boundary (corpus/story_store/hf_publish laptop-safe; extraction GPU-only) — first named after the org codename, renamed the same day to say what it does — ruff now enforcing top-level imports (PLC0415) with one documented exception, zero lint warnings on project code.
* What I expected vs what happened: Expected the full run to take ~27.5 min of compute per the benchmark's 196 tok/s; it took 7.8 min — the benchmark's random-sample batches carry more padding than the real per-emotion in-order batching, so the estimate was conservative by ~3.5x. Also expected writes to the slow network volume to need an async local-write-then-sync design; measured overhead was 2.5%, so the simple synchronous write stayed. One incident: a traced `source .env` leaked HF_TOKEN into a pod log (redacted; launcher now sources env with xtrace off; token rotation recommended to the user).
* What this changes about my thinking: Measure before engineering — both the async-writer idea and the estimate itself were corrected by cheap measurements. The estimator now importing the production run_batch (instead of duplicating the loop) means future estimates measure the code that actually runs. And the token-weighted-mean subtlety (per-story means cannot be averaged directly; weights are post-mask token counts) is exactly the kind of silent correctness bug the reference-first reading caught before it happened.
* What I will do next: The NRC valence/arousal correlation analysis per layer (the remaining half of Q1.H1.E1, laptop-safe against the published vectors), then Q1.H2.E1's control-prompt battery at the peak layer(s). Literature pass with citation pinpoints still owed.

Midnight addendum — the faithful-methodology arm ran end to end (E4, E5 pilot), and the story inverted twice. E4 (gemma-4-31b-it, chat template, -it probes from the 4B corpus): battery verdict NONE again (best chat/layer-57 6/12 paper, 4/12 held-out) — assistant-state alone is not the failure mode. But the -it geometry re-check produced the day's most interesting finding: valence collapses out of PC1 from layer 9 (0.833 base -> 0.090 at 33) while PC1's variance share doubles — and the diagnostic shows the axis is demoted, not destroyed (PC3, |r|=0.762; the BASE model's valence direction still reads out at |r|=0.788 in -it activations). The paper reports valence AS PC1 on its RLHF'd model, so this is a structural difference, recorded as Q1.H1.C2. E5 pilot (self-generated -it stories and dialogues, leakage 1%/0% vs the base model's 44%): NONE passes the registered rule, but self-story probes give the campaign's best config (7/12 + 6/12), and the probe sets barely agree with each other directionally (cos 0.22/0.09 vs corpus probes) while all partially working — C2's noise-ceiling signature. Scale is now the leading explanation and is directly testable: E6 registered (256 stories/emotion, score battery at n in {16,64,128,256}) with predictions committed before scoring, generation launched overnight. If E6's curve plateaus below threshold, the honest headline becomes: geometry replicates, probe-behaviour does not, and the difference is the model, not our method.

Second addendum — gate day. The plateau branch fired and the campaign closed. E6 (vLLM on Blackwell, after a three-failure cascade solved and documented: NVMe venv for the stale-file-handle corruption, TRITON_ATTN for the SM 12.x capability error, flashinfer uninstalled for the JIT sampler — 2,424 stories in ~9 min vs the ~2.5 h HF loop): probe directions converge (cos 0.944 -> 0.997) while battery scores stay flat at 4-6/12 across n in {16,64,128,256} — scale exonerated, C2 failed as registered. E7 (neutral-PC projection, the paper's confound-removal step missing from the reference and our pipeline until caught): moves valence PC3 -> PC2 and buys ~1 scenario, does not rescue. The falsify gate then ran with pre-declared thresholds (scripts/falsify_gate.py): C1 SURVIVED (perm p<1e-4, bootstrap CI [0.777, 0.879], drop-5 robustness 0.825) -> H1 supported; C2 FAILED; C3 WEAKENED (the null stands and design power is 0.99, but recomputed best top-3 counts don't clear the binomial rank-null) -> H2 refuted under the registered criteria. Two new arms extended the pattern instead of breaking it. Logit lens (Q1.H1.E2): base vectors show affective vocabulary neighborhoods for 5/12 emotions at layer 33; instruct vectors show junk at both layers — a split verdict. The preference experiment (Q1.H3, the paper's Figure 4 left half, 64 self-authored activities in the paper's 8 categories committed before measurement, Bradley-Terry Elo module with recovery tests, reads declared in the scorer before collection landed): on the paper's exact plain format the model reveals NO coherent preferences (E1, both predictions fail, spread 994 vs 999) — but the chat-template arm (E3) shows E1's flatness was format artefact: preferences become coherent (all six lowest-Elo activities unsafe/misaligned; P1 pass) while the probe read stays below the registered bar (max |r|=0.405 vs 0.5; paper 0.71-0.74) with significant valence organization (r=0.447, perm p<1e-4). C1(H3) graduated weakened; H3 refuted with survivals documented. That makes four independent representation-present-readout-weak signatures: battery, demotion, lens, preferences. Report layer rebuilt to match: four notebooks (data/geometry/probes/parity), dual-model everywhere with -it as protagonist, every figure carrying an explicit "maps to: Anthropic Figure N / open replication / our extension" line, and a complete census of all 86 figures + 16 tables (notes/plot_parity.md) — ~11 done/sub, ~74 queued with costs, ~15 out on one blocker class. Incidents recorded: a pod launch failed silently twice (stale checkout the pod can't git-pull, then a wrong HF_HOME re-downloading 31B onto a 94%-full disk) — fixed with rsync deploys, HF_HUB_OFFLINE=1, and monitors that alert on dead processes, not just success lines. Late-night continuation (user pre-authorized E2, prompted E8): the causal arm ran and the campaign's fifth and strongest signature landed. E8 (numerical-template confounds, registered reads): the templates' rho magnitudes never beat a random-direction null (with 6 x-values the activation drifts near-monotonically with N, norm included), our excursions are 10-50x smaller than the paper's, mean-pooling readouts shrink them further (the paper itself reads a single pre-response token — verified in-session) — what survives is the 11/11 sign pattern (p ~ 2^-11): sign, not magnitude. E2 (steering, both halves of paper Figure 4 now measured): at alphas 2 and 8 x residual RMS — the calibrated coherence-preserving range — steering redistributes at most +/-7 Elo (~2% of the paper's +212/-303), uncorrelated with the probe-Elo profile (r=-0.03/+0.15 vs the paper's 0.85), signs at chance, while choice coherence stays fully intact. C2(H3) graduated [survived]: no detectable causal preference content in the explored range. Two instrument lessons recorded on the way: the alpha-calibration statistic was sparse-fit noise (caught because the full-run deltas contradicted it; escalation arm run before recording the verdict), and the registered P2 global-mean statistic was identically zero under mean-anchored Elo (pairwise choices see only redistribution; amended pre-scoring, documented in the tree). Open decisions for tomorrow: the Q3 scope call (H2/H3 refuted but H1's valence axis gate-survived — a narrowed valence-only Q3 is defensible), the census queue (post-training proxies Figures 36-39/84 now the top value item, then steering delta-log-prob Figures 52-53 — note E2's null makes the log-prob variant MORE interesting, not less: it tests expression rather than preference), and the response-token readout extension flagged in E8. Pod idle after 05:10 UTC.

Night addendum — the H2 follow-ups ran. E2 (layer x readout x format sweep, 61 prompts x 20 layers x 3 readouts, held-out confirmation battery committed before scoring): NO combination passed the registered rule — the best cell (layer 57, last token) hit 7/12 on the paper's scenarios and only 4/12 on our held-out set, exactly the selection mirage the confirmation battery exists to catch. So E1's null is not a position/layer artefact. The collection also surfaced the day's most reframing fact: Gemma-4-31B has no chat template — it is a base model — so the paper's pre-response assistant-state measurement has no direct analog here, and the cleanest test of that explanation is the instruct variant (noted as candidate E4; needs disk planning, local NVMe is now ~30 GB free). E3 (dialogue-derived probes, 12 emotions x 16 model-generated dialogues, resumable) is generating in a background tmux; extraction and re-scoring follow when it lands.

Late addendum — Q1.H2.E1 ran end to end with predictions registered and committed before scoring: the paper's own Table 2 scenarios and Figure 3 numerical templates, activations collected at the pre-response token (49 prompts, layers 33/39, 28 s of inference), scored laptop-side in notebooks/03_probe_validation.ipynb. Honest outcome: **H2 weakened**. P1 (scenario diagonal) failed at both registered layers — target emotion in the top 3 of 12 probes for only 3/12 scenarios at each layer; what survives is a small coarse-valence effect (diag>off-diag, and layer 39's best-ranked scenarios are exactly the positive-valence ones). P2 (numerical semantics) scored 7/11 named directions, one under threshold — three templates track near-perfectly (tylenol, runway, exam at |rho|~1), two invert (fasting, dog-missing). So the geometry replicates (H1) while the behavioural probe validation does not (H2, on this model/format) — which is precisely the situation the Q3 gate exists to catch. Candidate explanations queued as testable follow-ups in Q1.H2.C1: RLHF-specific assistant-state affect, single-token reading, story-to-dialogue probe transfer (the paper's appendix tests exactly this variant), layer choice. Notebooks reorganised into notebooks/{01_corpus_exploration, 02_vector_geometry, 03_probe_validation}.ipynb.

Evening addendum — the correlation sweep ran (src/emotion_vectors/analysis.py against the day's vectors, NRC VAD v2.1, 164/171 emotions matched): peak |r|(PC1, valence) = 0.833 at layer 33 with a plateau of 0.79-0.83 across layers 30-57, peak |r|(PC2, arousal) = 0.619 at layer 39. Against the literature's 0.81/0.66 (Anthropic) and 0.83 (open replication's best), the valence geometry replicates at the top of the reported range and arousal lands slightly under, weaker than valence exactly as reported. Q1.H1.E1 is [done]; the finding is recorded as claim Q1.H1.C1 [unvalidated] — it does not graduate until the falsify gate runs (permutation null on emotion labels, bootstrap CI, and an explanation for the Pearson 0.83 vs Spearman 0.68 gap at the peak layer, which smells of a few influential emotions doing outsized work). Next session: falsify C1, then Q1.H2's control-prompt battery at layers 33/39.

### 2026-07-20

* What I did: Day one went to hardening the machinery this project runs on: audited the harness skills against the agentskills.io spec, then built and ran the harness's skill evals end to end — trigger sweeps, three-arm behaviour probes, falsification gate — all of which now lives in the harness repo's own TREE.md and RESEARCH_LOG.md (separated from this project today; this tree holds only the sprint's research). Late in the day, scoped the sprint's substantive project from the partner's planning brief: Q1 replicates the emotion-vector geometry on Gemma-4-31B via the open replication codebase, including the control-prompt cosine validation the replication skipped; Q2 (injection detection plus circumplex-graded misidentification analysis) is GATED on Q1 — no validated vectors, no injection experiment. Distilled the brief into notes/emotion-vectors-brief.md, keeping sources and dropping everything off the two-part path (logistics, staffing, the brief's own priority inversion).
* What I expected vs what happened: Expected to pick a question in the morning and start the literature pass; instead the harness evals produced findings (and incidents) worth a full day, and the sprint question arrived in the evening via the partner's brief, already well-sourced. The brief claims verification of its sources, but nothing in it has been verified in-session here yet.
* What this changes about my thinking: The day-one detour bought real protection for days two and three — fenced eval workspaces, collaboration rules, and direct evidence that the validate-claims skill measurably changes provenance behaviour. For this project it also set the standard: every number from the planning brief gets verified against the primary source (with page-level pinpoints) before it enters any deliverable.
* What I will do next: Q1.H1.E1 — clone sinievanderben/emotion_experiment, adapt it to Gemma-4-31B (transformers >= 4.51), pull the published Gemma story corpus from HF, and start the layer sweep; in parallel, the timeboxed literature pass reading the two primary papers properly (adding citation pinpoints to the brief as they are read).

Evening pivot: the second part changed. Instead of the introspection battery (Q2, now abandoned in the tree, nodes kept), the post-gate question is temporal dynamics — Q3: in synthetic stories that move through 2-3 emotions, does the per-token cosine similarity between the residual stream and each emotion vector show gradual ramp-and-crossover dynamics (the signature of evidence accumulation, as in drift-diffusion models of decision-making) or abrupt steps at lexical cues? Design notes: per-token trajectories (extraction's mean-pooling won't do), token-aligned transitions, matched lengths, random-direction and shuffled-story controls, subtle-vs-explicit evidence strength, near-vs-far circumplex pairs. Framing discipline: claims stay at "dynamics consistent with integration vs switching" — the drift-diffusion analogy generates predictions, it is not itself the claim. Gate unchanged: Q3 waits on Q1's validated vectors. Literature pass tomorrow must add: prior work on temporal dynamics of concept directions over context.

### 2026-07-19

* What I did: Set up the project scaffolding: MCP servers (arxiv, paper-search), research skills (research, falsify, validate-claims, derive-from-sources, alphaxiv-paper-lookup, research-log), the research tree (TREE.md), this log, and the mechanical validator (scripts/validate_research.py).
* What I expected vs what happened: Expected to copy configs verbatim from the eval-awareness and thesis repos; that worked, and the convergent-validity repo additionally contributed the append-only decision-record model that shaped the tree/log split.
* What this changes about my thinking: The infrastructure for honest research (falsify, validate, log, tree) is reusable across projects and cheap to port; the scarce resource is choosing a question narrow enough for 2-3 days.
* What I will do next: Pick the sprint research question, write it as Q1 in TREE.md, and do the timeboxed literature pass.
