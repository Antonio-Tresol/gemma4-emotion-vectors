#!/usr/bin/env bash
# Pod launcher for the Q1 extraction pipeline. Run it inside tmux so the job
# survives a dropped SSH session:
#
#   tmux new-session -d -s extract 'scripts/pod/run_extraction.sh full'
#   tail -f extract_full.log
#
# Modes:
#   smoke      3 emotions x 4 stories, publishes to the '-smoke' HF repo
#   full       whole corpus, publishes to the real HF repo
#   benchmark  timing run (scripts/estimate_extraction_time.py --benchmark)
#
# Assumes: repo at the current directory's project root, .env with HF_TOKEN,
# and the HF cache copied to local NVMe (HF_HOME below) for fast model loads —
# the cache on the network volume loads in ~33 min, the local copy in ~46 s.
set -euo pipefail

MODE="${1:?usage: run_extraction.sh smoke|full|benchmark}"
cd "$(dirname "$0")/../.."
LOG="extract_${MODE}.log"

# Secrets: .env is sourced with xtrace off and never echoed — a traced `source`
# once leaked HF_TOKEN into a log on this project. Keep it that way.
set +x
set -a
source ./.env
set +a

export HF_HOME="${HF_HOME:-/root/hf-local}"
export UV_LINK_MODE=copy # uv cache and venv sit on different filesystems

case "$MODE" in
smoke) CMD=(uv run python scripts/extract_emotion_vectors.py --smoke --publish) ;;
full) CMD=(uv run python scripts/extract_emotion_vectors.py --publish) ;;
benchmark) CMD=(uv run python scripts/estimate_extraction_time.py --benchmark --sample-batches 6) ;;
*) echo "unknown mode: $MODE" >&2 && exit 2 ;;
esac

echo "[run_extraction] mode=$MODE log=$LOG hf_home=$HF_HOME"
set +e
"${CMD[@]}" 2>&1 | tee "$LOG"
STATUS=$?
set -e
echo "[run_extraction] exit=$STATUS mode=$MODE" | tee -a "$LOG"
exit "$STATUS"
