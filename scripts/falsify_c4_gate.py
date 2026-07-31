"""C4 falsify gate — can the centered-cosine detection claim be destroyed?

CLAIM UNDER TEST (Q1.H2.C4, restated on the post-fix instruments): under the
centered-cosine readout (scenario-set mean removed from activations, probes
12-pool-centered and unit-normalized, chat_last), story-derived emotion
probes detect implicit scenario emotion on gemma-4-31b-it, passing the
dual-battery bar (target-in-top-3 >= 8/12 on BOTH batteries) across a band
of layers. Gated lineages (user-directed 2026-07-23): self-gen n=256
post-fix and strong-external DeepSeek fixed-prompt (E11's two winners).
Scoring is byte-compatible with scripts/score_e11_lineage.py (the three
shared helpers are reproduced verbatim; the observed grid is asserted equal
to results/e11_lineage.json before any test runs).

TESTS, registered before running (seeds fixed, bars fixed):

  T1 Selection-adjusted permutation null (the sweep's multiple comparisons):
     permute the scenario->target assignment within each battery
     independently (10,000 draws, seed 20260723); recompute the FULL
     20-layer sweep per draw; null statistic = max over layers of
     min(paper_count, heldout_count). p1 = P(null max >= observed max).
     Also reported: P(null draw yields >= 1 passing layer). BAR: p1 < 0.01.

  T2 Scenario bootstrap (fragility to battery composition): resample the 12
     scenarios per battery with replacement (10,000 draws, seed 20260724),
     re-centering activations within each draw (the readout's own convention
     applied to the resampled set); pass-stability per observed passing
     layer = fraction of draws still >= 8/12 on both. BAR: at least one
     passing layer with pass-stability >= 0.5; band-level fragility below
     that WEAKENS (qualified claim), does not fail.

  T3 Random-probe-set control (is it the probes or the pipeline?): 1,000
     sets of 12 Gaussian directions (seed 20260725) pushed through the
     identical pipeline (12-pool centering, unit norm); statistic as T1.
     BAR: observed max > 95th percentile of random maxima.

  T4 Dialogue-probe cross-check (convergent instrument): dialogue-derived
     probes for the same 12 emotions (pre-fix extraction, documented 7.1%
     leak — the only dialogue set in existence), same readout, same T1 null
     at reduced N=2,000. Read: dialogue probes' own max-statistic beats its
     null at p < 0.05. Failure WEAKENS (construct breadth), does not fail
     the story-probe claim.

  T5 Base-rate reference (reported, not a verdict input): Binomial(12, 3/12)
     tail P(X >= 8) — the naive per-layer chance of a single battery pass.

VERDICT RULES (per lineage): SURVIVES = T1, T2, T3 bars all met. WEAKENED =
T1 and T3 met but T2 or T4 missed (qualified: detection real, band fragile
or story-specific). FAILED = T1 or T3 missed (the sweep or the pipeline
explains the result).

    uv run python scripts/falsify_c4_gate.py   # writes results/falsify_c4_scorecard.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.stats import binom

from emotion_vectors.probe_prompts import HELDOUT_SCENARIOS, SCENARIOS

BATTERY = (
    "happy inspired loving proud calm desperate angry guilty sad afraid nervous surprised".split()
)
SWEEP_FILE = Path("results/probe_sweep_it/activations.npz")
E11_FILE = Path("results/e11_lineage.json")
BUNDLES = {
    "selfgen_postfix": Path("results/self_story_vectors_it_postfix/emotion_means.npz"),
    "strong_deepseek": Path("results/openrouter_vectors_it_means.npz"),
}
DIALOGUE_BUNDLE = Path("results/dialogue_vectors_it_means.npz")  # pre-fix; T4 caveat
OUT_FILE = Path("results/falsify_c4_scorecard.json")
N_PERM, N_BOOT, N_RANDOM, N_PERM_T4 = 10_000, 10_000, 1_000, 2_000
PASS_BAR = 8


# --- identical to scripts/score_e11_lineage.py (kept in lockstep; the grid
# equality assertion below catches drift) ---------------------------------
def load_battery_means(path: Path) -> Float[np.ndarray, "emotions layers d_model"]:
    z = np.load(path, allow_pickle=True)
    means = z["means"].astype(np.float32)
    if means.ndim == 4:
        means = means[-1]
    emotions = [str(e) for e in z["emotions"]]
    return means[[emotions.index(e) for e in BATTERY]]


def contrast_probes_12pool(
    means: Float[np.ndarray, "emotions d_model"],
) -> Float[np.ndarray, "emotions d_model"]:
    contrast = means - means.mean(axis=0, keepdims=True)
    return contrast / np.clip(np.linalg.norm(contrast, axis=-1, keepdims=True), 1e-8, None)


def centered_cos(
    acts: Float[np.ndarray, "scenarios d_model"], probes: Float[np.ndarray, "emotions d_model"]
) -> Float[np.ndarray, "scenarios emotions"]:
    centered = acts - acts.mean(axis=0, keepdims=True)
    cos = centered @ probes.T
    return cos / np.clip(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8, None)


# --------------------------------------------------------------------------


def top3_membership(
    cos: Float[np.ndarray, "scenarios emotions"],
) -> Bool[np.ndarray, "scenarios emotions"]:
    """[scenarios, emotions] bool: emotion j in scenario i's top 3."""
    rank = np.argsort(cos, axis=1)[:, ::-1][:, :3]
    member = np.zeros(cos.shape, dtype=bool)
    for i, row in enumerate(rank):
        member[i, row] = True
    return member


def counts_all_layers(
    acts: Float[np.ndarray, "scenarios layers d_model"],
    means: Float[np.ndarray, "emotions layers d_model"],
    target_idx: Int[np.ndarray, "scenarios"],
    rows: list[int],
) -> tuple[Int[np.ndarray, "layers"], list[Bool[np.ndarray, "scenarios emotions"]]]:
    """(counts[layer], membership matrices per layer) for one battery."""
    n_layers = acts.shape[1]
    counts, members = np.zeros(n_layers, dtype=int), []
    for lp in range(n_layers):
        cos = centered_cos(acts[rows, lp], contrast_probes_12pool(means[:, lp]))
        member = top3_membership(cos)
        members.append(member)
        counts[lp] = int(member[np.arange(len(rows)), target_idx].sum())
    return counts, members


def perm_null_max(
    members_paper: list[Bool[np.ndarray, "scenarios emotions"]],
    members_held: list[Bool[np.ndarray, "scenarios emotions"]],
    t_paper: Int[np.ndarray, "scenarios"],
    t_held: Int[np.ndarray, "scenarios"],
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[Float[np.ndarray, "draws"], Int[np.ndarray, "draws"]]:
    """Null distribution of max-over-layers min(paper, held) under independent
    within-battery target permutation, plus per-draw passing-layer counts."""
    n_layers = len(members_paper)
    maxes, n_passing = np.empty(n_perm), np.empty(n_perm, dtype=int)
    for d in range(n_perm):
        pp, ph = rng.permutation(t_paper), rng.permutation(t_held)
        mins = np.array(
            [
                min(
                    members_paper[lp][np.arange(len(pp)), pp].sum(),
                    members_held[lp][np.arange(len(ph)), ph].sum(),
                )
                for lp in range(n_layers)
            ]
        )
        maxes[d] = mins.max()
        n_passing[d] = int(
            sum(
                members_paper[lp][np.arange(len(pp)), pp].sum() >= PASS_BAR
                and members_held[lp][np.arange(len(ph)), ph].sum() >= PASS_BAR
                for lp in range(n_layers)
            )
        )
    return maxes, n_passing


def bootstrap_stability(
    acts: Float[np.ndarray, "scenarios layers d_model"],
    means: Float[np.ndarray, "emotions layers d_model"],
    rows_paper: list[int],
    rows_held: list[int],
    t_paper: Int[np.ndarray, "scenarios"],
    t_held: Int[np.ndarray, "scenarios"],
    layer_positions: list[int],
    rng: np.random.Generator,
) -> dict[int, float]:
    """Pass-stability per layer position: fraction of scenario-bootstrap draws
    (independent per battery, re-centered per draw) still >= bar on both."""
    out = {}
    for lp in layer_positions:
        probes = contrast_probes_12pool(means[:, lp])
        ok = 0
        for _ in range(N_BOOT):
            sp = rng.integers(0, len(rows_paper), len(rows_paper))
            sh = rng.integers(0, len(rows_held), len(rows_held))
            cp = centered_cos(acts[[rows_paper[i] for i in sp], lp], probes)
            ch = centered_cos(acts[[rows_held[i] for i in sh], lp], probes)
            n_p = int(top3_membership(cp)[np.arange(len(sp)), t_paper[sp]].sum())
            n_h = int(top3_membership(ch)[np.arange(len(sh)), t_held[sh]].sum())
            ok += n_p >= PASS_BAR and n_h >= PASS_BAR
        out[lp] = ok / N_BOOT
    return out


def random_probe_maxes(
    acts: Float[np.ndarray, "scenarios layers d_model"],
    rows_paper: list[int],
    rows_held: list[int],
    t_paper: Int[np.ndarray, "scenarios"],
    t_held: Int[np.ndarray, "scenarios"],
    rng: np.random.Generator,
) -> Float[np.ndarray, "draws"]:
    """T3: max-statistic for random 12-probe sets through the same pipeline."""
    n_layers, d_model = acts.shape[1], acts.shape[2]
    maxes = np.empty(N_RANDOM)
    for s in range(N_RANDOM):
        raw = rng.standard_normal((12, n_layers, d_model)).astype(np.float32)
        mins = []
        for lp in range(n_layers):
            probes = contrast_probes_12pool(raw[:, lp])
            cp = centered_cos(acts[rows_paper, lp], probes)
            ch = centered_cos(acts[rows_held, lp], probes)
            mins.append(
                min(
                    int(top3_membership(cp)[np.arange(len(rows_paper)), t_paper].sum()),
                    int(top3_membership(ch)[np.arange(len(rows_held)), t_held].sum()),
                )
            )
        maxes[s] = max(mins)
    return maxes


def gate_lineage(
    name: str,
    means: Float[np.ndarray, "emotions layers d_model"],
    acts: Float[np.ndarray, "scenarios layers d_model"],
    rows: dict[str, list[int]],
    targets: dict[str, Int[np.ndarray, "scenarios"]],
    layers: list[int],
) -> dict[str, object]:
    c_paper, m_paper = counts_all_layers(acts, means, targets["paper"], rows["paper"])
    c_held, m_held = counts_all_layers(acts, means, targets["held"], rows["held"])
    mins = np.minimum(c_paper, c_held)
    observed_max = int(mins.max())
    passing_pos = [lp for lp in range(len(layers)) if mins[lp] >= PASS_BAR]

    rng1 = np.random.default_rng(20260723)
    null_max, null_npass = perm_null_max(
        m_paper, m_held, targets["paper"], targets["held"], N_PERM, rng1
    )
    p1 = float((null_max >= observed_max).mean())
    p_any_pass = float((null_npass >= 1).mean())

    rng2 = np.random.default_rng(20260724)
    stability = bootstrap_stability(
        acts,
        means,
        rows["paper"],
        rows["held"],
        targets["paper"],
        targets["held"],
        passing_pos,
        rng2,
    )

    rng3 = np.random.default_rng(20260725)
    rand_max = random_probe_maxes(
        acts, rows["paper"], rows["held"], targets["paper"], targets["held"], rng3
    )
    rand_p95 = float(np.quantile(rand_max, 0.95))

    t1_pass = p1 < 0.01
    t2_pass = any(v >= 0.5 for v in stability.values())
    t3_pass = observed_max > rand_p95
    verdict = (
        "survives"
        if (t1_pass and t2_pass and t3_pass)
        else ("weakened" if (t1_pass and t3_pass) else "failed")
    )
    return {
        "per_layer_counts": {
            str(layers[lp]): [int(c_paper[lp]), int(c_held[lp])] for lp in range(len(layers))
        },
        "passing_layers": [layers[lp] for lp in passing_pos],
        "observed_max_min_count": observed_max,
        "t1_selection_adjusted_perm": {
            "n_perm": N_PERM,
            "p_max_geq_observed": p1,
            "p_null_any_passing_layer": p_any_pass,
            "null_max_p95": float(np.quantile(null_max, 0.95)),
            "bar": "p < 0.01",
            "pass": t1_pass,
        },
        "t2_scenario_bootstrap": {
            "n_boot": N_BOOT,
            "pass_stability_per_layer": {
                str(layers[lp]): round(v, 4) for lp, v in stability.items()
            },
            "bar": ">= 1 layer with stability >= 0.5",
            "pass": t2_pass,
        },
        "t3_random_probe_sets": {
            "n_sets": N_RANDOM,
            "random_max_p95": rand_p95,
            "random_max_max": float(rand_max.max()),
            "bar": "observed max > random p95",
            "pass": t3_pass,
        },
        "verdict": verdict,
    }


def main() -> int:
    sweep = np.load(SWEEP_FILE, allow_pickle=True)
    layers = [int(x) for x in sweep["layers"]]
    acts = sweep["chat_last"].astype(np.float64)
    order = [
        (kind, name, target)
        for kind, battery in (("scenario", SCENARIOS), ("heldout", HELDOUT_SCENARIOS))
        for name, target, _ in battery
    ]
    rows = {
        "paper": [i for i, (k, _, _) in enumerate(order) if k == "scenario"],
        "held": [i for i, (k, _, _) in enumerate(order) if k == "heldout"],
    }
    targets = {
        "paper": np.array([BATTERY.index(order[i][2]) for i in rows["paper"]]),
        "held": np.array([BATTERY.index(order[i][2]) for i in rows["held"]]),
    }

    e11 = json.loads(E11_FILE.read_text())
    result: dict[str, object] = {"claim": "Q1.H2.C4", "doc": __doc__.split("TESTS")[0].strip()}
    lineage_key = {"selfgen_postfix": "selfgen", "strong_deepseek": "strong_external"}
    for name, bundle in BUNDLES.items():
        means = load_battery_means(bundle)
        report = gate_lineage(name, means, acts, rows, targets, layers)
        expected = e11["r1_dual_battery"][lineage_key[name]]["passing_layers"]
        assert report["passing_layers"] == expected, (
            f"{name}: grid drift vs e11_lineage.json: {report['passing_layers']} != {expected}"
        )
        result[name] = report

    # T4: dialogue-probe cross-check (pre-fix dialogue extraction; documented caveat)
    if DIALOGUE_BUNDLE.exists():
        means_d = load_battery_means(DIALOGUE_BUNDLE)
        c_p, m_p = counts_all_layers(acts, means_d, targets["paper"], rows["paper"])
        c_h, m_h = counts_all_layers(acts, means_d, targets["held"], rows["held"])
        obs = int(np.minimum(c_p, c_h).max())
        null_max, _ = perm_null_max(
            m_p, m_h, targets["paper"], targets["held"], N_PERM_T4, np.random.default_rng(20260726)
        )
        result["t4_dialogue_crosscheck"] = {
            "bundle": str(DIALOGUE_BUNDLE),
            "caveat": "pre-fix extraction (7.1% leak share); only dialogue set in existence",
            "observed_max_min_count": obs,
            "p_vs_own_null": float((null_max >= obs).mean()),
            "bar": "p < 0.05 (informative only: weakens, never fails)",
            "pass": bool(float((null_max >= obs).mean()) < 0.05),
        }
    else:
        result["t4_dialogue_crosscheck"] = {"skipped": f"{DIALOGUE_BUNDLE} not found"}

    result["t5_base_rate_reference"] = {
        "binomial_p_single_battery_geq_8_of_12_at_chance_3_of_12": float(
            1 - binom.cdf(PASS_BAR - 1, 12, 0.25)
        )
    }
    OUT_FILE.write_text(json.dumps(result, indent=2) + "\n")
    for name in BUNDLES:
        r = result[name]
        print(
            f"{name}: max={r['observed_max_min_count']} passing={r['passing_layers']} | "
            f"T1 p={r['t1_selection_adjusted_perm']['p_max_geq_observed']:.4f} "
            f"T2 {r['t2_scenario_bootstrap']['pass']} "
            f"T3 p95={r['t3_random_probe_sets']['random_max_p95']:.1f} -> {r['verdict'].upper()}"
        )
    print(f"T4: {result['t4_dialogue_crosscheck']}")
    print(f"-> {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
