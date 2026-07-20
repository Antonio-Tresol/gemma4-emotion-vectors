#!/usr/bin/env python3
"""Check generation progress and recover from failures."""

import json
import os
import sys
from pathlib import Path

CHECKPOINT_FILE = "outputs_checkpoint.jsonl"
FINAL_OUTPUT = "outputs.json"

def count_completed():
    """Count completed items in checkpoint."""
    if not os.path.exists(CHECKPOINT_FILE):
        return 0, set()
    count = 0
    ids = set()
    with open(CHECKPOINT_FILE, "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                count += 1
                ids.add(obj["id"])
    return count, ids

def count_total():
    """Count total prompts."""
    if not os.path.exists("data/probe_prompts.jsonl"):
        return 0
    count = 0
    with open("data/probe_prompts.jsonl") as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def finalize():
    """Manually finalize checkpoint to output file."""
    completed, ids = count_completed()
    if completed == 0:
        print("No checkpoint to finalize")
        return

    outputs = []
    with open(CHECKPOINT_FILE, "r") as f:
        for line in f:
            if line.strip():
                outputs.append(json.loads(line))

    with open(FINAL_OUTPUT, "w") as f:
        json.dump(outputs, f)
    print(f"Finalized {completed} results to {FINAL_OUTPUT}")

if __name__ == "__main__":
    total = count_total()
    completed, ids = count_completed()
    pct = (100 * completed / total) if total else 0

    print(f"Generation status: {completed}/{total} ({pct:.1f}%)")

    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        finalize()
