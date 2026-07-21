"""Q1.H1 circumplex geometry check — entry point.

Thin CLI over emotion_vectors.analysis: PCA of the mean-centered emotion
vectors per layer, correlated against NRC VAD valence/arousal, compared to the
literature (Anthropic r=0.81/0.66; open replication up to 0.83).

Run where the extraction outputs live (pod or after downloading the HF
dataset):
    uv run python scripts/analyze_emotion_geometry.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotion_vectors.analysis import analyze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/emotion_vectors"))
    parser.add_argument(
        "--lexicon-file",
        type=Path,
        default=Path("data/lexicons/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/emotion_geometry_correlations.json")
    )
    args = parser.parse_args()

    result = analyze(args.results_dir, args.lexicon_file, args.out)
    peak_v, peak_a = result["peak_valence_layer"], result["peak_arousal_layer"]
    per_layer = result["per_layer"]
    print(
        f"matched {result['n_matched_to_nrc_vad']}/{result['n_emotions']} emotions to NRC VAD\n"
        f"peak PC1-valence: layer {peak_v}, "
        f"r={per_layer[peak_v]['pc1_valence']['pearson_r']}\n"
        f"peak PC2-arousal: layer {peak_a}, "
        f"r={per_layer[peak_a]['pc2_arousal']['pearson_r']}\n"
        f"literature: {result['literature_reference']}\n"
        f"written -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
