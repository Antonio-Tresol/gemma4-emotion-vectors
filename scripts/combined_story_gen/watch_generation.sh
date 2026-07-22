#!/bin/bash
# Progress monitor for the combined-story generation (Q3 corpus).
# Works from any clone, for any operator (human or Claude):
#
#   bash scripts/combined_story_gen/watch_generation.sh [RESULTS_DIR] [LOG_FILE]
#
# Defaults assume you run it from the repo root that launched the generation.
# One-shot status print; wrap in `watch -n 30` or a loop for continuous view.
# Known output locations on the shared pod:
#   /workspace/cambria/results/combined_stories        (workspace clone)
#   /root/cbai-cambria-project/results/combined_stories (home clone)
set -u

RESULTS_DIR="${1:-results/combined_stories}"
LOG_FILE="${2:-${RESULTS_DIR}/generate.log}"
RAW="${RESULTS_DIR}/stories_raw.jsonl"

echo "=== combined-story generation status ($(date -u +%H:%M:%SZ)) ==="

if [ -f "${RESULTS_DIR}/run_config.json" ]; then
  python3 - "$RESULTS_DIR" <<'PYEOF'
import json
import sys

cfg = json.load(open(f"{sys.argv[1]}/run_config.json"))
n_triples = 2 if cfg.get("smoke") else 173
target = n_triples * cfg["per_triple"]
print(f"config: per_triple={cfg['per_triple']} smoke={cfg.get('smoke')} target~{target} stories")
PYEOF
else
  echo "no run_config.json in ${RESULTS_DIR} (run not started or wrong dir)"
fi

if [ -f "$RAW" ]; then
  kept=$(wc -l < "$RAW")
  echo "stories written: ${kept}"
else
  echo "no stories_raw.jsonl yet"
fi

if [ -f "$LOG_FILE" ]; then
  echo "--- last log lines ---"
  grep -vE "^Loading|it/s\]" "$LOG_FILE" | tail -4
else
  echo "no log at ${LOG_FILE}"
fi

echo "--- gpu ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

echo "--- generation processes ---"
pgrep -af "generate_combined_stories" | grep -v pgrep || echo "none running"
