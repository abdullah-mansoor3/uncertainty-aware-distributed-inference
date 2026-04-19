#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PREP_SCRIPT="${REPO_ROOT}/experiments/prepare_datasets.py"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "Using Python: ${PYTHON_BIN}"
echo "Downloading raw datasets to: ${REPO_ROOT}/data/raw"

"${PYTHON_BIN}" "${PREP_SCRIPT}" \
  --download-only \
  --raw-dir "${REPO_ROOT}/data/raw" \
  --verbose

echo "Raw dataset download complete."
echo "Next step:"
echo "  ${PYTHON_BIN} ${PREP_SCRIPT} --prepare-only --raw-dir ${REPO_ROOT}/data/raw --processed-dir ${REPO_ROOT}/data/processed --sample-size 100 --seed 42 --verbose"
