#!/usr/bin/env bash
# Create the two Python 3.10 venvs (.venv-yolo, .venv-mowa) with CUDA 12.1 torch.
# A CUDA 12.1-capable NVIDIA GPU is required (MOWA hard-codes .cuda()).
#
# Usage:  ./scripts/setup_env.sh [python-executable]
set -euo pipefail

PYTHON="${1:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TORCH_INDEX="https://download.pytorch.org/whl/cu121"

# On Windows Git-Bash the venv interpreter is under Scripts/, on Linux under bin/.
venv_python() {
  if [ -f "$1/Scripts/python.exe" ]; then echo "$1/Scripts/python.exe"; else echo "$1/bin/python"; fi
}

make_venv() {
  local name="$1" reqs="$2"
  echo ""
  echo "=== Creating $name ==="
  "$PYTHON" -m venv "$name"
  local py; py="$(venv_python "$name")"
  "$py" -m pip install --upgrade pip
  echo "--- Installing CUDA torch (cu121) first ---"
  "$py" -m pip install torch==2.1.2 torchvision==0.16.2 --index-url "$TORCH_INDEX"
  echo "--- Installing $reqs ---"
  "$py" -m pip install -r "$reqs"
  echo "$name ready."
}

echo "Using interpreter: $("$PYTHON" --version 2>&1)"
make_venv ".venv-yolo" "requirements-yolo.txt"
make_venv ".venv-mowa" "requirements-mowa.txt"

echo ""
echo "Both environments created. Next: clone MOWA + download assets (docs/DATA_SETUP.md)."
