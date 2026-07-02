#!/usr/bin/env bash
# Auto-bootstrap: ensures venv exists, then activates it.
# Usage: source scripts/activate.sh
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  bash "${PROJECT_ROOT}/scripts/setup_venv.sh"
fi

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
