import json
import sys
from pathlib import Path
from api_client import complete

PROMPTS_FILE = "data/probe_prompts.jsonl"
OUTPUT_FILE = "outputs.jsonl"
RESUME_FILE = ".generate_progress"

def load_progress():
    """Load set of already-processed prompt IDs."""
    if Path(RESUME_FILE).exists():
        with open(RESUME_FILE) as f:
            return set(json.load(f))
    return set()

def save_progress(completed_ids):
    """Save progress checkpoint."""
    with open(RESUME_FILE, "w") as f:
        json.dump(list(completed_ids), f)

def main():
    completed_ids = load_progress()
    prompts = [json.loads(l) for l in open(PROMPTS_FILE)]

    print(f"Resuming from {len(completed_ids)} completed prompts", file=sys.stderr)

    # Open in append mode to preserve existing outputs
    with open(OUTPUT_FILE, "a") as out:
        for i, p in enumerate(prompts):
            if p["id"] in completed_ids:
                continue

            try:
                completion = complete(p["prompt"])
                output = {"id": p["id"], "completion": completion}
                out.write(json.dumps(output) + "\n")
                out.flush()

                completed_ids.add(p["id"])
                if (i + 1) % 10 == 0:
                    save_progress(completed_ids)
                    print(f"Progress: {len(completed_ids)}/{len(prompts)}", file=sys.stderr)
            except Exception as e:
                print(f"Error on prompt {p['id']}: {e}", file=sys.stderr)
                save_progress(completed_ids)
                raise

    # Final checkpoint
    save_progress(completed_ids)
    print(f"Done: {len(completed_ids)}/{len(prompts)}", file=sys.stderr)

if __name__ == "__main__":
    main()
