"""Export the preference and steering evidence for the write-up's section 5.

The other blocks in docs/build.py are numbers transcribed by hand from notebook
printed records. This one is not: the arrays are per-emotion and per-layer, too
long to copy without error, and the file they come from has a near-twin whose
numbers are wrong to publish. So it is extracted, and the choice of source file
is made here once, in the open.

    uv run python scripts/export_deck_preferences.py

WHICH FILES, AND WHY THESE ONES.

`preferences_it_chat_fixed/scores.json` reports max |r| = 0.7013, which is all
but identical to the paper's 0.71-0.74. That number was measured with pre-fix
probes. `scores_postfix_probes.json` rescores the same read with the corrected
probes and reports 0.6448. TREE.md records the amendment and the honest
phrasing: clears the registered 0.5 bar with margin, below the paper. The
flattering file is the one not read here.

The same applies to steering. TREE's headline sign agreement of 11/12 and 10/12
came from arms that injected pre-fix directions. The clean re-steer with
post-fix directions gives 10/12 at both doses, and those are the runs read here.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "data" / "preferences.json"

# post-fix probes, not the pre-fix twin beside it (see the module docstring)
CORRELATIONAL = RESULTS / "preferences_it_chat_fixed" / "scores_postfix_probes.json"
STEERING = {
    "2": RESULTS / "steering_it_postfix" / "scores.json",
    "8": RESULTS / "steering_it_a8_postfix" / "scores.json",
}
PAPER_RANGE = [0.71, 0.74]  # Sofroniew et al. 2026, Figure 4 left half
SIGN_BAR = 9  # of 12, registered before scoring


def main() -> int:
    correlational = json.loads(CORRELATIONAL.read_text())
    by_layer = [
        {
            "layer": row["layer"],
            "max_abs_r": round(row["max_abs_r"], 4),
            "valence_r": round(row["valence_organization_r"], 4),
            "perm_p": row["valence_organization_perm_p"],
        }
        for row in correlational["probe_elo_by_layer"]
    ]

    doses = {}
    for alpha, path in STEERING.items():
        run = json.loads(path.read_text())
        rows = sorted(run["per_emotion"], key=lambda r: r["valence"])
        # a run "agrees" for an emotion when the steering shifted preferences
        # in the direction that emotion's valence predicts
        agreements = sum(
            1 for r in rows if (r["valence"] > 0) == (r["mean_delta_elo_positive_categories"] > 0)
        )
        doses[alpha] = {
            "layer": run["layer"],
            "coherence_ok": run["p1_pass"],
            "sign_agreements": agreements,
            "coupling_r": round(run["p2_pearson_r"], 4),
            "coupling_bar": run["p2_registered_bar"],
            "emotions": [
                {
                    "emotion": r["emotion"],
                    "valence": round(r["valence"], 3),
                    "delta": round(r["mean_delta_elo_positive_categories"], 2),
                    "probe_r": round(r["probe_elo_r"], 4),
                }
                for r in rows
            ],
        }

    payload = {
        "best_layer": correlational["p2_best_layer"],
        "best_abs_r": round(correlational["p2_max_abs_r"], 4),
        "registered_bar": correlational["p2_registered_bar"],
        "paper_range": PAPER_RANGE,
        "by_layer": by_layer,
        "sign_bar": SIGN_BAR,
        "doses": doses,
    }

    # the two arms must describe the same emotions, or the figures compare
    # different things while looking like they do not
    names = {alpha: [e["emotion"] for e in d["emotions"]] for alpha, d in doses.items()}
    if names["2"] != names["8"]:
        raise ValueError(f"steering arms disagree on emotions: {names}")

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  correlational: |r| {payload['best_abs_r']} at layer {payload['best_layer']}")
    print(f"  registered bar {payload['registered_bar']}, paper {PAPER_RANGE[0]}-{PAPER_RANGE[1]}")
    for alpha, dose in doses.items():
        print(
            f"  steering alpha {alpha}: {dose['sign_agreements']}/12 signs "
            f"(bar {SIGN_BAR}), coupling {dose['coupling_r']} vs bar {dose['coupling_bar']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
