#!/usr/bin/env bash
# Creates .venv and installs dependencies if not already present.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
# Prefer Python 3.11+ when available
if command -v python3.11 &>/dev/null; then
  PYTHON="${PYTHON:-python3.11}"
elif command -v python3.12 &>/dev/null; then
  PYTHON="${PYTHON:-python3.12}"
else
  PYTHON="${PYTHON:-python3}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}..."
  "${PYTHON}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo "Installing dependencies..."
pip install -r "${PROJECT_ROOT}/requirements.txt" --quiet

echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=predictive-maintenance --display-name="Predictive Maintenance (venv)" 2>/dev/null || true

echo ""
echo "Virtual environment ready: ${VENV_DIR}"
echo "Activate with: source .venv/bin/activate"
