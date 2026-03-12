#!/usr/bin/env bash
# run_preprocess_sbert.sh
# Run SBERT preprocessing for all five pretrain datasets.
# Saves one .pt file per dataset under OUT_DIR.
#
# Prerequisites:
#   - audit_datasets.py run and notes/dataset_audit.md filled in
#   - src/preprocess_sbert_features.py text extraction TODOs implemented
#   - Python environment with torch, sentence-transformers, ogb installed
#   - GPU recommended (encoding 2.4M products is slow on CPU)
#
# Usage:
#   bash scripts/run_preprocess_sbert.sh                    # all datasets
#   bash scripts/run_preprocess_sbert.sh arxiv              # one dataset
#   ENCODER=all-mpnet-base-v2 bash scripts/run_preprocess_sbert.sh
#
# Output files:
#   data/arxiv_sbert.pt
#   data/products_sbert.pt
#   data/wn18rr_sbert.pt
#   data/roman-empire_sbert.pt
#   data/pcba_sbert.pt  (only once PCBA strategy is decided)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data}"
ENCODER="${ENCODER:-all-MiniLM-L6-v2}"
DATASET="${1:-all}"     # first positional arg or "all"
FORCE="${FORCE:-}"      # set FORCE=--force to overwrite existing .pt files
LOG_DIR="${REPO_ROOT}/runs/preprocess"

mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/preprocess_${DATASET}_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

echo "================================================================"
echo "  GFM Safety — SBERT Feature Preprocessing"
echo "  Dataset:   ${DATASET}"
echo "  Encoder:   ${ENCODER}"
echo "  Data root: ${DATA_ROOT}"
echo "  Out dir:   ${OUT_DIR}"
echo "  Log:       ${LOG_FILE}"
echo "  $(date -u)"
echo "================================================================"
echo ""

# Warn if text extraction TODOs are likely still unimplemented
echo "[INFO] This script will raise NotImplementedError for datasets whose"
echo "       text extraction functions still have TODO placeholders."
echo "       Fill in src/preprocess_sbert_features.py before running."
echo ""

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

python "${REPO_ROOT}/src/preprocess_sbert_features.py" \
    --dataset "${DATASET}" \
    --data-root "${DATA_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --encoder "${ENCODER}" \
    ${FORCE} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Preprocessing complete.  Log saved to: ${LOG_FILE}"
echo ""
echo "Output .pt files:"
ls -lh "${OUT_DIR}"/*_sbert.pt 2>/dev/null || echo "  (none found — check for errors above)"
echo ""
echo "Next step: record run details in notes/experiment_log.md"
