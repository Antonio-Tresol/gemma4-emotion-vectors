#!/usr/bin/env python3
"""
Scoring script for experiment results.
Computes mean accuracy from results/scores.jsonl
"""
import json
import statistics
import sys
from pathlib import Path


def score(row):
    return row['correct'] / max(row['total'], 1)


def main(scores_path='results/scores.jsonl'):
    path = Path(scores_path)

    if not path.exists():
        print(f"Error: {scores_path} not found", file=sys.stderr)
        sys.exit(1)

    rows = [json.loads(l) for l in path.open()]

    if not rows:
        print("Error: no rows found in scores file", file=sys.stderr)
        sys.exit(1)

    scores = [score(r) for r in rows]
    mean_score = statistics.mean(scores)

    print(f"{mean_score:.4f}")
    return mean_score


if __name__ == '__main__':
    main()
