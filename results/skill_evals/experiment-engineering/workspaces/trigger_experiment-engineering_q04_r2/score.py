#!/usr/bin/env python3
"""Score results from a JSONL file containing correct/total counts."""

import json
import statistics
import sys
from pathlib import Path


def score(row):
    """Calculate score as correct / total."""
    return row['correct'] / max(row['total'], 1)


def main():
    scores_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/scores.jsonl')

    if not scores_file.exists():
        print(f"Error: {scores_file} not found", file=sys.stderr)
        sys.exit(1)

    rows = [json.loads(line) for line in scores_file.open()]
    scores = [score(r) for r in rows]

    mean_score = statistics.mean(scores)
    print(f"Mean score: {mean_score:.4f}")
    print(f"Total rows: {len(scores)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
