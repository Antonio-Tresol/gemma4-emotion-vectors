#!/usr/bin/env bash
# One-time vLLM environment setup for the Blackwell pod (RTX PRO 6000, SM 12.x).
# Background and symptom table: notes/vllm-parallel-inference-template.md
# ("Blackwell survival guide"). Run from the repo root on the pod:
#
#   scripts/pod/setup_vllm_env.sh
#
# Produces /root/vllm-env (local NVMe — the /workspace network volume corrupts
# large wheel installs) with vLLM, no flashinfer (its SM 12.x capability probe
# and JIT sampler both break on this card), and this repo installed editable.
#
# Generation jobs then run with:
#   export VLLM_ATTENTION_BACKEND=TRITON_ATTN
#   export VLLM_USE_FLASHINFER_SAMPLER=0
#   /root/vllm-env/bin/python scripts/generate_dialogue_stories.py --backend vllm ...
# (Our launchers set these; if you write a new one, copy the exports.)
set -euo pipefail

UV=~/.local/bin/uv
VENV=/root/vllm-env

$UV venv "$VENV" --python 3.14
$UV pip install --python "$VENV/bin/python" vllm -e "$(pwd)"
$UV pip uninstall --python "$VENV/bin/python" flashinfer-python

"$VENV/bin/python" - << 'EOF'
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.version.cuda and tuple(map(int, torch.version.cuda.split("."))) >= (12, 9), (
    "torch must be built for CUDA >= 12.9 for SM 12.x; reinstall with a matching wheel"
)
try:
    import flashinfer  # noqa: F401
    raise SystemExit("flashinfer is still installed — uninstall it (breaks on SM 12.x)")
except ImportError:
    print("flashinfer absent: OK")
print("vllm env ready")
EOF
