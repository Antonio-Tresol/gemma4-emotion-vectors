#!/usr/bin/env python3
"""Score results from results/scores.jsonl"""

import json
import statistics
import sys
from pathlib import Path


def score(row):
    """Calculate score as correct/total (avoiding division by zero)."""
    return row['correct'] / max(row['total'], 1)


def main(scores_file='results/scores.jsonl'):
    """Load scores and compute mean."""
    path = Path(scores_file)
    if not path.exists():
        print(f"Error: {scores_file} not found", file=sys.stderr)
        sys.exit(1)

    rows = [json.loads(line) for line in path.open()]
    scores_list = [score(r) for r in rows]
    mean_score = statistics.mean(scores_list)

    print(f"Mean score: {mean_score:.4f}")
    print(f"Total entries: {len(scores_list)}")

    return mean_score


if __name__ == '__main__':
    main()
