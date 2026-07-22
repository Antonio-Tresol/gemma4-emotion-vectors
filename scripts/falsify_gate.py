"""Falsify gate for the sprint's three claims — produces the scorecards.

Thresholds declared here, before running (the falsify skill's rule):

C1 (geometry replication, base model): SURVIVES iff the layer-33 PC1-valence
    correlation has (a) permutation p < 0.001 against 10,000 label shuffles,
    (b) bootstrap 95% confidence interval lower bound > 0.6 over emotions,
    (c) robustness: dropping the 5 highest-|valence| emotions keeps |r| > 0.6
    (the Pearson-vs-Spearman gap concern).

C2 (noise-ceiling explanation): the claim's registered consequence was that
    battery scores rise with corpus size. E6 measured the consequence; this
    scorecard formalizes prediction vs outcome. FAILED iff best scores do not
    increase from n=16 to n=256 while probe-direction stability does.

C3 (probes do not function on this model): SURVIVES iff (a) no configuration
    reaches the registered 8/12 on both batteries (recomputed, not quoted),
    (b) the best observed counts exceed the binomial rank-null (showing the
    battery had signal sensitivity, so the null is not a dead instrument),
    (c) a power check: probes with paper-like per-scenario top-3 probability
    0.9 would pass the registered bar with probability > 0.95.

    uv run python scripts/falsify_gate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import binom, pearsonr, spearmanr
from sklearn.decomposition import PCA

from emotion_vectors.analysis import load_nrc_vad
from emotion_vectors.scoring import score_battery

if TYPE_CHECKING:
    from jaxtyping import Float

RNG = np.random.default_rng(20260722)
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 5_000


def geometry_inputs() -> tuple["Float[np.ndarray, 'matched']", "Float[np.ndarray, 'matched']"]:
    """(PC1 scores, valence) for the base model at layer 33, matched emotions."""
    bundle = np.load("results/emotion_vectors/emotion_means.npz", allow_pickle=True)
    emotions = list(map(str, bundle["emotions"]))
    layer_pos = list(bundle["layers"]).index(33)
    means = bundle["means"][:, layer_pos, :].astype(np.float64)
    vad = load_nrc_vad(Path("data/lexicons/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"))
    matched = [i for i, e in enumerate(emotions) if e.lower() in vad]
    valence = np.array([vad[emotions[i].lower()][0] for i in matched])
    scores = PCA(n_components=1).fit_transform(means - means.mean(axis=0))[:, 0]
    return scores[matched], valence


def falsify_c1() -> dict[str, object]:
    pc1, valence = geometry_inputs()
    observed = abs(float(pearsonr(pc1, valence).statistic))
    perm = np.array(
        [
            abs(float(pearsonr(pc1, RNG.permutation(valence)).statistic))
            for _ in range(N_PERMUTATIONS)
        ]
    )
    p_perm = float((perm >= observed).mean())
    idx = np.arange(len(pc1))
    boots = [
        abs(float(pearsonr(pc1[s], valence[s]).statistic))
        for s in (RNG.choice(idx, len(idx)) for _ in range(N_BOOTSTRAP))
    ]
    ci_low, ci_high = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    keep = np.argsort(-np.abs(valence))[5:]
    r_dropped = abs(float(pearsonr(pc1[keep], valence[keep]).statistic))
    verdict = "survived" if (p_perm < 0.001 and ci_low > 0.6 and r_dropped > 0.6) else "weakened"
    return {
        "claim": "Q1.H1.C1",
        "observed_abs_r": round(observed, 4),
        "permutation_p": p_perm,
        "n_permutations": N_PERMUTATIONS,
        "bootstrap_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "spearman_r": round(abs(float(spearmanr(pc1, valence).statistic)), 4),
        "drop5_extreme_valence_abs_r": round(r_dropped, 4),
        "thresholds": "p<0.001, ci_low>0.6, drop5>0.6",
        "verdict": verdict,
    }


def battery_best(n_bucket_pos: int) -> tuple[int, int]:
    """(best paper, best heldout) over formats x layers x readouts at one corpus size."""
    e6 = np.load("results/e6_scale_means.npz", allow_pickle=True)
    means = e6["means"].astype(np.float32)[n_bucket_pos]
    sweep = np.load("results/probe_sweep_it/activations.npz", allow_pickle=True)
    prompts = [json.loads(line) for line in open("results/probe_sweep_it/prompts.jsonl")]
    batteries = {
        k: [i for i, p in enumerate(prompts) if p["kind"] == k] for k in ("scenario", "heldout")
    }
    best = {"scenario": 0, "heldout": 0}
    for fmt in map(str, sweep["formats"]):
        for readout in ("last", "mean_all", "mean_content"):
            acts = sweep[f"{fmt}_{readout}"].astype(np.float32)
            for lp in range(means.shape[1]):
                for kind in best:
                    activations = np.stack([acts[i, lp] for i in batteries[kind]])
                    count, _ = score_battery(means[:, lp, :], activations)
                    best[kind] = max(best[kind], count)
    return best["scenario"], best["heldout"]


def falsify_c2() -> dict[str, object]:
    e6 = np.load("results/e6_scale_means.npz", allow_pickle=True)
    buckets = list(map(int, e6["n_buckets"]))
    curve = {n: battery_best(i) for i, n in enumerate(buckets)}
    rose = curve[buckets[-1]][0] > curve[buckets[0]][0] + 1  # material rise on paper battery
    return {
        "claim": "Q1.H2.C2",
        "prediction": "battery best scores rise with stories-per-emotion if noise binds",
        "best_scores_by_n": {str(n): list(v) for n, v in curve.items()},
        "probe_convergence_note": "directions converge to cos 0.997 by n=128 (notebook 08)",
        "verdict": "failed" if not rose else "survived",
    }


def falsify_c3() -> dict[str, object]:
    buckets = np.load("results/e6_scale_means.npz", allow_pickle=True)["n_buckets"]
    best_paper, best_heldout = battery_best(len(buckets) - 1)
    p_chance_best = float(1 - binom.cdf(best_paper - 1, 12, 0.25))
    p_chance_bar = float(1 - binom.cdf(7, 12, 0.25))
    power = float((1 - binom.cdf(7, 12, 0.9)) ** 2)
    no_pass = not (best_paper >= 8 and best_heldout >= 8)
    verdict = "survived" if (no_pass and p_chance_best < 0.05 and power > 0.95) else "weakened"
    return {
        "claim": "Q1.H2.C3",
        "recomputed_best_at_n256": [int(best_paper), int(best_heldout)],
        "registered_bar": [8, 8],
        "no_configuration_passes": no_pass,
        "binomial_p_best_vs_rank_null": round(p_chance_best, 5),
        "binomial_p_bar_vs_rank_null": round(p_chance_bar, 5),
        "power_if_paper_like_probes": round(power, 4),
        "thresholds": "no pass AND best above chance (p<0.05) AND power>0.95",
        "verdict": verdict,
    }


def main() -> int:
    for name, fn in (("c1", falsify_c1), ("c2", falsify_c2), ("c3", falsify_c3)):
        card = fn()
        out = Path(f"results/falsify_{name}_scorecard.json")
        out.write_text(json.dumps(card, indent=2) + "\n")
        print(f"{card['claim']}: {card['verdict']}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
