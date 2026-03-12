#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs notes

python tools/parse_logs.py > outputs/meeting_table.md

cat > notes/meeting_progress_arxiv_sbert.md <<'MD'
# Meeting Progress: arxiv_sbert

## Environment
- llm: GPT-5.3-Codex
- python env: graphmae_env

## Main Artifacts (data/)
- data/arxiv_sbert.pt
- data/ogbn_arxiv/processed/data_processed
- data/ogbn_arxiv/raw/
- data/ogbn_arxiv/split/time/
- data/ogb/bgrl_ogbn-arxiv/raw/data.pt

## Metric Notes
- BGRL values are printed as-is from logs (likely percent-style values).
- GraphMAE accuracy values are in [0,1] as printed in logs.
- warning `missing_best_metrics_used_final` means BGRL logs did not print explicit best metrics, so parser uses final evaluation as best.

## Parsed Results
MD

cat outputs/meeting_table.md >> notes/meeting_progress_arxiv_sbert.md

cat >> notes/meeting_progress_arxiv_sbert.md <<'MD'

## Next Steps
- Re-run this script after any new training logs are added.
- Review rows with non-null warning fields and fill missing metrics from future runs.
MD

printf '%s\n' "notes/meeting_progress_arxiv_sbert.md"
