import json
import os
from pathlib import Path
from api_client import complete

CHECKPOINT_FILE = "outputs_checkpoint.jsonl"
FINAL_OUTPUT = "outputs.json"

def load_completed_ids():
    """Load set of IDs already processed from checkpoint."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    completed = set()
    with open(CHECKPOINT_FILE, "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                completed.add(obj["id"])
    return completed

def save_checkpoint(result):
    """Append result to checkpoint file."""
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

def load_all_prompts():
    """Load prompts from input file."""
    prompts = [json.loads(l) for l in open("data/probe_prompts.jsonl")]
    return prompts

def merge_checkpoint_to_final():
    """Merge checkpoint file into final output JSON."""
    if not os.path.exists(CHECKPOINT_FILE):
        return

    outputs = []
    with open(CHECKPOINT_FILE, "r") as f:
        for line in f:
            if line.strip():
                outputs.append(json.loads(line))

    with open(FINAL_OUTPUT, "w") as f:
        json.dump(outputs, f)

# Load what we've already completed
completed_ids = load_completed_ids()
prompts = load_all_prompts()
total = len(prompts)
completed = len(completed_ids)

print(f"Resuming: {completed}/{total} completed")

# Process remaining
for i, p in enumerate(prompts, 1):
    if p["id"] in completed_ids:
        continue

    result = {"id": p["id"], "completion": complete(p["prompt"])}
    save_checkpoint(result)
    completed += 1
    print(f"[{completed}/{total}] Processed {p['id']}")

# Merge checkpoint to final output
merge_checkpoint_to_final()
print(f"Done! Saved {completed} results to {FINAL_OUTPUT}")
