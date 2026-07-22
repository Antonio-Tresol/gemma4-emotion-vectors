"""Q1.H3.E2 scoring — steered Elo shifts vs the E3 correlation profile.

Laptop-side. Reads the steered A/B logits (results/steering_it), refits
Bradley-Terry Elo per steered emotion, and scores the three predictions
registered in TREE.md Q1.H3.E2 before collection:

  P1 (sanity): steered runs stay on-distribution — mean |A-B logit gap|
      within 3x the unsteered chat run's.
  P2 (causal, the paper's bottom scatter r=0.85; amended pre-scoring, see
      TREE Q1.H3.E2): across the 12 steered emotions, the mean Elo delta of
      the POSITIVE-CATEGORY activities (vs the unsteered chat baseline)
      correlates with that emotion's E3 probe-Elo r at Pearson r >= 0.5,
      positive sign. The global-mean delta is identically zero under the
      mean-anchored Elo, so redistribution is the identifiable analog.
  P3 (direction sign test): for each steered emotion, the mean Elo delta of
      positive-category activities (helpful/engaging/social/self_curiosity)
      has the sign its NRC valence predicts; pass at >= 9/12 agreements.

Writes results/steering_it/scores.json.

    uv run python scripts/score_steering.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import pearsonr

from emotion_vectors.analysis import load_nrc_vad
from emotion_vectors.preferences import bradley_terry, elo, soft_wins, win_matrix

if TYPE_CHECKING:
    from jaxtyping import Float, Int

STEER_DIR = Path("results/steering_it")
BASELINE_DIR = Path("results/preferences_it_chat")
LEXICON = Path("data/lexicons/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt")
POSITIVE = ("helpful", "engaging", "social", "self_curiosity")


def elo_from_logits(
    ab_logits: "Float[np.ndarray, 'pairs two']",
    pair_i: "Int[np.ndarray, 'pairs']",
    pair_j: "Int[np.ndarray, 'pairs']",
    n_items: int,
) -> "Float[np.ndarray, 'items']":
    """Bradley-Terry Elo from one collection's ordered-pair logits."""
    p_first = soft_wins(ab_logits.astype(np.float64))
    return elo(bradley_terry(win_matrix(pair_i, pair_j, p_first, n_items)))


def collect_rows(
    pair_i: "Int[np.ndarray, 'pairs']",
    pair_j: "Int[np.ndarray, 'pairs']",
    n_activities: int,
    pos_idx: list[int],
    baseline_elo: "Float[np.ndarray, 'items']",
    baseline_gap: float,
    probe_r: dict[str, float],
) -> list[dict[str, object]]:
    """One scored row per steered emotion file."""
    vad = load_nrc_vad(LEXICON)
    rows = []
    for f in sorted(STEER_DIR.glob("steered_*.npy")):
        emotion = f.stem.removeprefix("steered_")
        logits = np.load(f)
        steered_elo = elo_from_logits(logits, pair_i, pair_j, n_activities)
        delta = steered_elo - baseline_elo
        gap = float(np.abs(logits[:, 0] - logits[:, 1]).mean())
        rows.append(
            {
                "emotion": emotion,
                "probe_elo_r": probe_r.get(emotion),
                "valence": vad.get(emotion, (None,))[0],
                "mean_abs_gap": round(gap, 3),
                "gap_ratio_vs_baseline": round(gap / baseline_gap, 3),
                "mean_delta_elo": round(float(delta.mean()), 2),
                "mean_delta_elo_positive_categories": round(float(delta[pos_idx].mean()), 2),
                "delta_elo_per_activity": [round(float(d), 1) for d in delta],
            }
        )
    return rows


def main() -> int:
    meta = json.loads((STEER_DIR / "run_meta.json").read_text())
    pair_i = np.array(meta["pair_i"])
    pair_j = np.array(meta["pair_j"])
    activities = [json.loads(line) for line in open(BASELINE_DIR / "activities.jsonl")]
    pos_idx = [k for k, a in enumerate(activities) if a["category"] in POSITIVE]

    baseline_bundle = np.load(BASELINE_DIR / "preferences.npz")
    baseline_elo = elo_from_logits(
        baseline_bundle["ab_logits"],
        baseline_bundle["pair_i"],
        baseline_bundle["pair_j"],
        len(activities),
    )
    baseline_gap = float(
        np.abs(baseline_bundle["ab_logits"][:, 0] - baseline_bundle["ab_logits"][:, 1]).mean()
    )

    e3 = json.loads((BASELINE_DIR / "scores.json").read_text())
    best_block = next(b for b in e3["probe_elo_by_layer"] if b["layer"] == e3["p2_best_layer"])
    probe_r = dict(zip(e3["probe_emotions_matched"], best_block["per_probe_r"]))
    rows = collect_rows(
        pair_i, pair_j, len(activities), pos_idx, baseline_elo, baseline_gap, probe_r
    )

    p1_pass = all(r["gap_ratio_vs_baseline"] <= 3.0 for r in rows)
    scored = [r for r in rows if r["probe_elo_r"] is not None]
    xs = np.array([r["probe_elo_r"] for r in scored])
    ys = np.array([r["mean_delta_elo_positive_categories"] for r in scored])
    p2 = pearsonr(xs, ys)
    p2_pass = bool(p2.statistic >= 0.5)
    sign_hits = [
        r
        for r in rows
        if r["valence"] is not None
        and np.sign(r["mean_delta_elo_positive_categories"]) == np.sign(r["valence"])
    ]
    p3_pass = len(sign_hits) >= 9

    out = {
        "experiment": "Q1.H3.E2",
        "alpha": meta["alpha"],
        "layer": meta["layer"],
        "baseline_mean_abs_gap": round(baseline_gap, 3),
        "per_emotion": rows,
        "p1_pass": bool(p1_pass),
        "p2_pearson_r": round(float(p2.statistic), 4),
        "p2_p_value": float(p2.pvalue),
        "p2_registered_bar": 0.5,
        "p2_pass": p2_pass,
        "p3_sign_agreements": len(sign_hits),
        "p3_registered_bar": 9,
        "p3_pass": bool(p3_pass),
        "paper_reference": "bottom-scatter r=0.85; mean deltas +212 (blissful) / -303 (hostile)",
    }
    (STEER_DIR / "scores.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"P1 {'PASS' if p1_pass else 'FAIL'} (max gap ratio {max(r['gap_ratio_vs_baseline'] for r in rows):.2f})"
    )
    print(
        f"P2 {'PASS' if p2_pass else 'FAIL'}: delta-Elo vs probe-r Pearson r={p2.statistic:.3f} (p={p2.pvalue:.4f}, bar 0.5)"
    )
    print(
        f"P3 {'PASS' if p3_pass else 'FAIL'}: {len(sign_hits)}/12 valence-sign agreements (bar 9)"
    )
    for r in sorted(rows, key=lambda r: r["mean_delta_elo_positive_categories"]):
        print(
            f"  {r['emotion']:10s} dpos={r['mean_delta_elo_positive_categories']:+7.1f}  probe_r={r['probe_elo_r']}  gapx={r['gap_ratio_vs_baseline']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
