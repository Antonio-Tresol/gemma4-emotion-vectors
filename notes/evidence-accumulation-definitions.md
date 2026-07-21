# Evidence accumulation in humans — working definitions

**Purpose.** Ground the Q3 framing. Q3 asks whether per-token cosine similarity
between Gemma's residual stream and each emotion vector shows *gradual
ramp-and-crossover* dynamics ("consistent with evidence accumulation, as in
drift-diffusion models") or *abrupt steps at lexical cues*. That analogy only
does honest work if "evidence accumulation" is defined precisely and its
species/level boundaries are respected. This file is the definitional backbone
for that. It is **not** a claim that the LLM accumulates evidence — the
drift-diffusion analogy generates predictions; it is not itself the claim
(framing discipline, RESEARCH_LOG 2026-07-20).

**Provenance rules applied here.** Every *quoted* sentence in Part A is verbatim
from a source **read in-session on 2026-07-21** (the raw abstract text returned
by the paper-search MCP / PubMed, or — where flagged — a double-confirmed
sentence). Definitions in Part B that are not directly quotable from an abstract
are marked **[paraphrase]** and attributed; the full texts of the paywalled
classics (Ratcliff & McKoon 2008; Gold & Shadlen 2007; Bogacz et al. 2006) were
**not** read in-session, so no page/section pinpoint is claimed for them and no
sentence is quoted from their bodies. One review full text (Forstmann et al.
2016, PMC author manuscript) was read via a summarizing fetch, which can
paraphrase; sentences taken only from that fetch are therefore treated as
**[paraphrase — verify against PDF]**, not as quotations.

---

## Part A — Verbatim source quotes (read in-session, 2026-07-21)

**[S1] Forstmann, B.U., Ratcliff, R., & Wagenmakers, E.-J. (2016).** Sequential
Sampling Models in Cognitive Neuroscience: Advantages, Applications, and
Extensions. *Annual Review of Psychology* 67:641–666. PMID 26393872 / PMCID
PMC5112760. — *abstract, verbatim:*
> "Sequential sampling models assume that people make speeded decisions by
> gradually accumulating noisy information until a threshold of evidence is
> reached. In cognitive science, one such model—the diffusion decision model—is
> now regularly used to decompose task performance into underlying processes
> such as the quality of information processing, response caution, and a priori
> bias."

**[S2] Ratcliff, R., & McKoon, G. (2008).** The diffusion decision model: theory
and data for two-choice decision tasks. *Neural Computation* 20(4):873–922. PMID
18085991. — *abstract, verbatim:*
> "The diffusion decision model allows detailed explanations of behavior in
> two-choice discrimination tasks. In this article, the model is reviewed to
> show how it translates behavioral data—accuracy, mean response times, and
> response time distributions—into components of cognitive processing. Three
> experiments are used to illustrate experimental manipulations of three
> components: stimulus difficulty affects the quality of information on which a
> decision is based; instructions emphasizing either speed or accuracy affect
> the criterial amounts of information that a subject requires before initiating
> a response; and the relative proportions of the two stimuli affect biases in
> drift rate and starting point."

**[S3] Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J.D. (2006).**
The physics of optimal decision making: a formal analysis of models of
performance in two-alternative forced-choice tasks. *Psychological Review*
113(4):700–765. PMID 17014301. — *abstract, verbatim:*
> "...they begin by analyzing 6 models of TAFC decision making and show that all
> but one can be reduced to the drift diffusion model, implementing the
> statistically optimal algorithm (most accurate for a given speed or fastest
> for a given accuracy). They prove further that there is always an optimal
> trade-off between speed and accuracy that maximizes various reward functions,
> including reward rate (percentage of correct responses per unit time)..."

**[S4] Gold, J.I., & Shadlen, M.N. (2007).** The neural basis of decision making.
*Annual Review of Neuroscience* 30:535–574. PMID 17600525. — *abstract,
verbatim:*
> "...most decisions share common elements including deliberation and
> commitment. Here we evaluate recent progress in understanding how these basic
> elements of decision formation are implemented in the brain."

**[S5] Usher, M., & McClelland, J.L. (2001).** The time course of perceptual
choice: the leaky, competing accumulator model. *Psychological Review*
108(3):550–592. PMID 11488378. — *abstract, verbatim:*
> "The time course of perceptual choice is discussed in a model of gradual,
> leaky, stochastic, and competitive information accumulation in nonlinear
> decision units. Special cases of the model match a classical diffusion
> process, but leakage and competition work together to address several
> challenges to existing diffusion, random walk, and accumulator models."

**[S6] Brown, S.D., & Heathcote, A. (2008).** The simplest complete model of
choice response time: linear ballistic accumulation. *Cognitive Psychology*
57(3):153–178. PMID 18243170. — *abstract, verbatim:*
> "We propose a linear ballistic accumulator (LBA) model of decision making and
> reaction time. The LBA is simpler than other models of choice response time,
> with independent accumulators that race towards a common response threshold.
> Activity in the accumulators increases in a linear and deterministic manner."

**[S7] O'Connell, R.G., Dockree, P.M., & Kelly, S.P. (2012).** A supramodal
accumulation-to-bound signal that determines perceptual decisions in humans.
*Nature Neuroscience* 15(12):1729–1735. PMID 23103963. — *abstract, verbatim:*
> "In theoretical accounts of perceptual decision-making, a decision variable
> integrates noisy sensory evidence and determines action through a
> boundary-crossing criterion. Signals bearing these very properties have been
> characterized in single neurons in monkeys, but have yet to be directly
> identified in humans. Using a gradual target detection task, we isolated a
> freely evolving decision variable signal in human subjects that exhibited
> every aspect of the dynamics observed in its single-neuron counterparts... we
> found that the signal was completely domain general..."

**[S8] Latimer, K.W., Yates, J.L., Meister, M.L.R., Huk, A.C., & Pillow, J.W.
(2015).** Single-trial spike trains in parietal cortex reveal discrete steps
during decision-making. *Science* 349(6244):184–187. PMID 26160947 / PMCID
PMC4799998. — *abstract, verbatim:*
> "Neurons in the macaque lateral intraparietal (LIP) area exhibit firing rates
> that appear to ramp upward or downward during decision-making. These ramps are
> commonly assumed to reflect the gradual accumulation of evidence toward a
> decision threshold. However, the ramping in trial-averaged responses could
> instead arise from instantaneous jumps at different times on different trials.
> ... We compared models with latent spike rates governed by either continuous
> diffusion-to-bound dynamics or discrete 'stepping' dynamics. Roughly
> three-quarters of the choice-selective neurons we recorded were better
> described by the stepping model."

**[S9] Latimer, K.W., et al. (2016).** Response to Comment on "Single-trial spike
trains...". *Science* 351(6280):1406. PMID 27013724. — *abstract, verbatim:*
> "...that stepping dynamics provided a better statistical description of lateral
> intraparietal area spike trains than diffusion-to-bound dynamics for a
> majority of neurons."

---

## Part B — Definitions (derived from Part A)

### 1. The core construct: evidence accumulation (sequential sampling)

When a person makes a quick decision between alternatives, they do not decide in
a single instant on a single sample. Instead they take in noisy information over
a short interval, add it up, and commit once the running total is convincing
enough. That process — *integrate noisy evidence over time until a criterion is
met* — is **evidence accumulation**; the family of formal models built on it is
called **sequential sampling models** [S1]. Two ingredients define it and
separate it from alternatives:

- **Gradual integration**, not a single readout: evidence is summed across
  successive samples ([S1] "gradually accumulating noisy information"; [S5]
  "gradual... information accumulation").
- **A commitment threshold** (a *bound*): accumulation continues until the total
  crosses a criterion, which both selects the choice and ends the timing ([S1]
  "until a threshold of evidence is reached"; [S4] "deliberation and
  commitment"; [S7] "boundary-crossing criterion").

The quantity that crosses the bound is the **decision variable**: a running
integral of noisy evidence that determines the action ([S7], verbatim: "a
decision variable integrates noisy sensory evidence and determines action
through a boundary-crossing criterion").

### 2. What is "evidence"? (the accumulated quantity)

Informally, *evidence* is momentary information favoring one alternative over the
other. **[paraphrase]** In the normative reading, the optimal quantity to
accumulate is the **log-likelihood ratio** of the alternatives given the sensory
sample, summed over time; accumulating it to a fixed bound implements the
**sequential probability ratio test (SPRT)**, the test that reaches a target
accuracy in the fewest samples. This is the standard normative interpretation
attributed to the perceptual-decision literature (Gold & Shadlen 2007 [S4];
Bogacz et al. 2006 [S3]) — I did **not** read either body text in-session, so
this is attribution, not a quotation. What *is* directly quotable is the
consequence Bogacz et al. prove: the drift-diffusion model "implement[s] the
statistically optimal algorithm (most accurate for a given speed or fastest for
a given accuracy)" [S3].

### 3. The reference model and its parameters (the diffusion decision model)

The **diffusion decision model (DDM)** is the canonical instance for two-choice
decisions: a single decision variable starts between two bounds and drifts
toward one of them under constant push plus moment-to-moment noise; the first
bound it hits is the choice, and the time taken is the decision time [S1, S2].
Its power is that it maps observable behavior — choice proportions *and* the full
shape of the response-time distributions, correct and error — onto a small set of
psychologically interpretable parameters ([S2] "translates behavioral
data—accuracy, mean response times, and response time distributions—into
components of cognitive processing"; [S1] "decompose task performance into
underlying processes"). Glossary:

- **Drift rate** — the *rate/quality* of evidence: how fast, on average, the
  decision variable moves toward the correct bound. It indexes stimulus
  difficulty or subject ability. Directly grounded: [S2] "stimulus difficulty
  affects the quality of information on which a decision is based"; [S1] "quality
  of information processing." **[paraphrase — verify against PDF]** Forstmann et
  al. gloss it as the average amount of evidence accumulated per unit time.

- **Boundary separation** (threshold / **response caution**) — the *distance*
  the variable must travel before committing. Wider bounds → more evidence
  required → slower but more accurate; this parameter *is* the speed–accuracy
  setting. Grounded: [S2] "speed or accuracy affect the criterial amounts of
  information that a subject requires before initiating a response"; [S1]
  "response caution." **[paraphrase — verify against PDF]** Forstmann et al.
  describe boundary separation as implementing the speed–accuracy trade-off.

- **Starting point** (*a priori* bias) — where accumulation begins between the
  bounds. Off-center = a prior lean toward one choice before any evidence
  arrives. Grounded: [S2] "the relative proportions of the two stimuli affect
  biases in drift rate and starting point"; [S1] "a priori bias."

- **Non-decision time** — **[paraphrase — verify against PDF]** an additive
  offset for everything that is *not* the accumulation itself: sensory encoding
  before the decision starts and motor execution after it ends. It appears in no
  abstract read in-session; attributed to Forstmann et al. 2016 [S1] / Ratcliff &
  McKoon 2008 [S2], wording to be confirmed against the PDFs.

- **Within-trial noise** — the moment-to-moment randomness in the accumulation
  path. It is why a fixed drift still yields a *distribution* of choices and
  times rather than one deterministic outcome, and why boundary separation trades
  errors against speed ([S1] the model is of "noisy" accumulation).

### 4. The model *family* — and which member Q3 actually invokes

"Evidence accumulation" is a family, not one equation. The distinction matters
for Q3: **the 1-D DDM has a single accumulator between two bounds, but Q3.H1's
"ramp-and-crossover" (incoming emotion ramps *while* outgoing decays) is a
two-accumulator / competing-accumulator picture, not a single diffusion.**

- **DDM** — one variable, two bounds; optimal for 2-choice [S2, S3].
- **Leaky competing accumulator (LCA)** — separate accumulators per alternative
  that *leak* (forget) and *inhibit each other*; "special cases... match a
  classical diffusion process" but leakage + competition add realism [S5]. This
  is the natural formal home for **crossover** dynamics (one channel rising as a
  competing channel falls).
- **Linear ballistic accumulator (LBA)** — "independent accumulators that race
  towards a common response threshold," rising linearly and deterministically
  within a trial [S6]; a deliberately simplified race model.

**Transfer to Q3.** If Q3.H1's per-emotion cosine trajectories are read as
"accumulators," the right analogue is a **competing-accumulator** model (LCA-like
[S5]), and the claim vocabulary should be "dynamics consistent with *competitive
integration* vs. *switching*," not "the DDM." The DDM supplies the *bound* and
*noise* intuitions; the crossover shape comes from competition.

### 5. Normative reading: optimality and the speed–accuracy trade-off

Accumulation-to-bound is not just descriptive — for the two-alternative case it
is *optimal*: the DDM "implement[s] the statistically optimal algorithm (most
accurate for a given speed or fastest for a given accuracy)," and there is "an
optimal trade-off between speed and accuracy that maximizes various reward
functions, including reward rate" [S3]. Behaviorally, the trade-off is set by
boundary separation (§3): the same evidence, integrated to a higher bound, buys
accuracy with time.

### 6. Neural signatures — and the species/level boundary ("in humans")

This is where "**in humans**" has to be stated carefully, because the most famous
evidence is **not** human:

- **Macaque, single-neuron.** In monkeys, neurons in the lateral intraparietal
  area (LIP) show firing rates that *ramp* up or down during a decision, long
  read as the neural correlate of accumulation toward threshold [S8]. This is
  single-unit electrophysiology in *macaques*, not humans.
- **Human, macroscopic.** The direct human analogue is an EEG signal — the
  **centro-parietal positivity (CPP)** — identified by O'Connell, Dockree & Kelly
  (2012) as "a freely evolving decision variable signal in human subjects that
  exhibited every aspect of the dynamics observed in its single-neuron
  counterparts," and notably **domain-general** (same dynamics across sensory
  modalities) [S7]. The same paper states the boundary plainly: decision-variable
  signals had "been characterized in single neurons in monkeys, but have yet to
  be directly identified in humans" until their work [S7].

So: the *computational construct* (DDM, LCA, LBA; §1–5) is fit to **human**
behavior; the *cellular* ramping picture is **macaque**; the validated **human**
neural signature of accumulation-to-bound is the **CPP** [S7]. Any "in humans"
statement about neural evidence should cite [S7], not the LIP work.

### 7. The ramping-vs-stepping controversy — the direct Q3.H1 analogue

Q3.H1 is, almost exactly, a controversy that already exists in this field.
Latimer et al. (2015) asked whether LIP's apparent *ramp* is real gradual
accumulation or an artifact of averaging, and found the latter for most neurons:
"the ramping in trial-averaged responses could instead arise from instantaneous
jumps at different times on different trials," and comparing "continuous
diffusion-to-bound dynamics" vs. discrete "stepping" dynamics, "[r]oughly
three-quarters of the choice-selective neurons... were better described by the
stepping model" [S8]; they reaffirmed this against a published Comment [S9]. The
claim remains contested — but the *method* is the transferable part.

**The trial-averaging confound (direct methodological warning for Q3.H1.E1).**
A smooth group-average ramp is consistent with *every individual trial stepping
abruptly at a variable latency* [S8]. Applied to Q3: a per-token cosine
trajectory **averaged over stories** could look like a gradual emotional ramp
even if every single story steps at its cue word. The design consequence — a
sharpening of Q3.H1.E1, which currently specifies averaged trajectories plus
controls — is to add **single-story (single-trial) trajectories** and an explicit
**ramp-vs-step model comparison** per story (mirroring Latimer's approach),
rather than reading gradualness off the average alone.

### 8. How this constrains the Q3 framing (summary)

1. The drift-diffusion analogy is a **prediction generator, not the claim**
   (framing discipline). Claim vocabulary stays at "dynamics consistent with
   *integration* vs. *switching*."
2. The crossover Q3.H1 predicts is a **competing-accumulator** signature (LCA
   [S5]), not a 1-D DDM one — say "competitive integration," reserve "DDM" for
   the bound/noise intuitions.
3. "**In humans**" for the *neural* signature means the **CPP** [S7], not macaque
   LIP [S8]; keep the species/level boundary explicit.
4. Latimer's **trial-averaging confound** [S8] transfers directly: analyze
   **single-story** trajectories and do a **ramp-vs-step comparison**, don't infer
   gradualness from the average.

---

## Sources (full list, in-session reads 2026-07-21)

- [S1] Forstmann, Ratcliff & Wagenmakers (2016), *Annu. Rev. Psychol.* 67:641–666. PMID 26393872 / PMC5112760.
- [S2] Ratcliff & McKoon (2008), *Neural Comput.* 20(4):873–922. PMID 18085991.
- [S3] Bogacz, Brown, Moehlis, Holmes & Cohen (2006), *Psychol. Rev.* 113(4):700–765. PMID 17014301.
- [S4] Gold & Shadlen (2007), *Annu. Rev. Neurosci.* 30:535–574. PMID 17600525.
- [S5] Usher & McClelland (2001), *Psychol. Rev.* 108(3):550–592. PMID 11488378.
- [S6] Brown & Heathcote (2008), *Cogn. Psychol.* 57(3):153–178. PMID 18243170.
- [S7] O'Connell, Dockree & Kelly (2012), *Nat. Neurosci.* 15(12):1729–1735. PMID 23103963.
- [S8] Latimer, Yates, Meister, Huk & Pillow (2015), *Science* 349(6244):184–187. PMID 26160947 / PMC4799998.
- [S9] Latimer et al. (2016), *Science* 351(6280):1406. PMID 27013724.

**Owed follow-ups (not blocking):** verify the four DDM-parameter wordings and the
non-decision-time definition against the Ratcliff & McKoon (2008) / Forstmann et
al. (2016) PDFs before any of these quotes enter an external deliverable; obtain a
verbatim source for the "evidence = accumulated log-likelihood ratio" definition
(Gold & Shadlen 2007 body, or a Bogacz et al. 2006 full-text PDF) rather than
carrying it as paraphrase.
