#!/usr/bin/env bash
# run_audit.sh
# Run the dataset audit for all five pretrain datasets.
# Records: num_nodes, feature shape, dtype, text field availability.
#
# Prerequisites:
#   - repos/bgrl and repos/graphmae cloned
#   - datasets downloaded (or downloadable) under DATA_ROOT
#   - Python environment with torch, torch_geometric or dgl, ogb installed
#
# Usage:
#   bash scripts/run_audit.sh                    # audit all datasets
#   bash scripts/run_audit.sh arxiv              # audit one dataset
#   DATA_ROOT=/scratch/data bash scripts/run_audit.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
DATASET="${1:-all}"   # first positional arg or "all"
LOG_DIR="${REPO_ROOT}/runs/audit"

mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/audit_${DATASET}_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if [ ! -d "${REPO_ROOT}/repos/bgrl" ] || [ ! -d "${REPO_ROOT}/repos/graphmae" ]; then
    echo "[WARNING] repos/bgrl or repos/graphmae is missing."
    echo "          Some dataset loaders may not be available yet."
    echo "          Clone the model repos before running a full audit."
    echo ""
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

echo "================================================================"
echo "  GFM Safety — Dataset Audit"
echo "  Dataset:   ${DATASET}"
echo "  Data root: ${DATA_ROOT}"
echo "  Log:       ${LOG_FILE}"
echo "  $(date -u)"
echo "================================================================"

python "${REPO_ROOT}/src/audit_datasets.py" \
    --dataset "${DATASET}" \
    --data-root "${DATA_ROOT}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Audit complete.  Log saved to: ${LOG_FILE}"
echo "Fill in notes/dataset_audit.md with the results above."
