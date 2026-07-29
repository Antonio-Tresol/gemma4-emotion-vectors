"""Section 7: recompute every number the verdict cites.

``verdict_stats`` returns no figure, only ``stats`` whose ``lines`` restate
each cited number directly from the frozen evidence files, so the verdict
markdown can be checked against the printout line by line. The
extraction-format caveat has its own exhibit in ``_caveat`` and is not
duplicated here: each number is computed in exactly one place.
"""

from __future__ import annotations

from typing import Any

from ._data import STORY_SOURCE_LABELS, StorySourceEvidence, seed_mean_by_n
from ._e11 import R1_SOURCES
from ._e12 import _matched_n_detection


def verdict_stats(evidence: StorySourceEvidence) -> dict[str, Any]:
    """Every number cited by the section-7 verdict.

    Input: ``load_evidence()`` output. Returns ``stats`` with:
      lines               -- the printed record (R1 passing-layer counts, the
                             E12 seed-mean dose-response tables, the
                             matched-n detection comparison, and the
                             preference values)
      n_passing           -- story source -> R1 passing-layer count
      matched_detection   -- the matched-n detection numbers (from
                             ``dose_response_figure``'s helper, so the
                             verdict and the figure cannot drift)
      pref_abs_r          -- story source -> preference max |r|
      pref_matched_mean   -- diverse-at-n=256 seed-mean |r|

    The extraction-format caveat's numbers live in
    ``extraction_format_figure``; they are deliberately not recomputed here.
    """
    lines: list[str] = []

    # R1: passing-layer counts per story source (verdict item 1).
    # "r1_dual_battery" is the frozen key the scorers wrote for the
    # detection read scored on both sets of scenarios.
    n_passing: dict[str, int] = {}
    for key, (attribute, name) in R1_SOURCES.items():
        arm = getattr(evidence, attribute)["r1_dual_battery"][name]
        n_passing[key] = len(arm["passing_layers"])
        lines.append(
            f"R1 {STORY_SOURCE_LABELS[key]}: {n_passing[key]} passing layers"
            f" {arm['passing_layers']}"
        )

    # E12: seed-mean dose-response tables (verdict items 2 and 3)
    for arm_name, curve in evidence.scale_curves["arms"].items():
        seed_means = seed_mean_by_n(curve["points"], "n_passing_layers")
        lines.append(
            f"E12 {arm_name} arm, seed-mean passing layers: "
            + ", ".join(f"n={size}: {mean:.1f}" for size, mean in seed_means.items())
        )
    matched_detection = _matched_n_detection(evidence)
    lines.append(
        f"matched-n detection: diverse n=256 {matched_detection['diverse_n256']:.1f} vs fixed"
        f" full corpus n={matched_detection['fixed_full_n']:.0f}"
        f" {matched_detection['fixed_full']:.1f}; diverse full"
        f" n={matched_detection['diverse_full_n']:.0f} {matched_detection['diverse_full']:.1f}"
    )

    # R3: preference ordering and the matched-n check (verdict item 4)
    pref_abs_r = {
        "selfgen": evidence.e11["r3_preference_probe_elo"]["selfgen"]["abs_r"],
        "weak_external": evidence.e11["r3_preference_probe_elo"]["weak_external"]["abs_r"],
        "fixed_deepseek": evidence.full_grid["r3_preference_probe_elo"]["fixed_deepseek_n256"][
            "abs_r"
        ],
        "diverse_deepseek": evidence.full_grid["r3_preference_probe_elo"]["diverse_deepseek_n1024"][
            "abs_r"
        ],
    }
    lines.append(
        "preference max |r|: "
        + ", ".join(f"{STORY_SOURCE_LABELS[key]} {value:.3f}" for key, value in pref_abs_r.items())
    )
    matched_seeds = evidence.matched_n["diverse_n256_seeds"]
    pref_matched_mean = evidence.matched_n["diverse_n256_mean"]
    lines.append(
        f"preference matched-n, diverse at n=256: {pref_matched_mean:.3f}"
        f" (seeds {min(matched_seeds):.3f} to {max(matched_seeds):.3f})"
    )

    return {
        "lines": lines,
        "n_passing": n_passing,
        "matched_detection": matched_detection,
        "pref_abs_r": pref_abs_r,
        "pref_matched_mean": pref_matched_mean,
    }
