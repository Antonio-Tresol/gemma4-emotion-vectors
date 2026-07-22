#!/bin/bash
# Progress monitor for the Q3 trajectory extraction (per-token probe shards).
# Works from any clone, for any operator (human or Claude):
#
#   bash scripts/combined_story_gen/watch_trajectories.sh [RESULTS_DIR]
#
# One-shot status print; wrap in `watch -n 30` or a loop for continuous view.
# Known output location on the shared pod:
#   /workspace/cambria/results/combined_trajectories
# NOTE: the extraction process may linger after logging DONE (exit hang);
# trust the log line, not the process list, for completion.
set -u

RESULTS_DIR="${1:-results/combined_trajectories}"
LOG="${RESULTS_DIR}/extract.log"
TARGET=5888

echo "=== trajectory extraction status ($(date -u +%H:%M:%SZ)) ==="

if [ -f "${RESULTS_DIR}/run_config.json" ]; then
  echo "--- config:"
  head -c 400 "${RESULTS_DIR}/run_config.json"; echo
else
  echo "no run_config.json yet (imports/model load take ~5 min with no output)"
fi

if [ -f "${RESULTS_DIR}/manifest.jsonl" ]; then
  done_n=$(wc -l < "${RESULTS_DIR}/manifest.jsonl")
  echo "stories done: ${done_n}/${TARGET}"
else
  echo "no manifest yet"
fi

if [ -f "$LOG" ]; then
  echo "--- last log lines:"
  tail -3 "$LOG"
  grep -a 'DONE:' "$LOG" >/dev/null && echo ">>> RUN COMPLETE <<<"
fi

echo "--- gpu:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

echo "--- extraction processes:"
pgrep -af "extract_trajectories" | grep -v pgrep || echo "none running"
