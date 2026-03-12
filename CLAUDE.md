# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project: **"On the safety of graph foundation models."** Investigates whether accuracy ranking equals safety ranking by replacing raw node features in two self-supervised graph learning models (BGRL and GraphMAE) with unified 768-dim SBERT embeddings across five datasets (Arxiv, Products, WN18RR, Roman-Empire, PCBA). Model architectures are unchanged — only the feature tensor is swapped at data loading time.

## Hard Rules

1. **Never guess node/text alignment.** If mapping is not confirmed, mark as BLOCKED or EXPERIMENTAL.
2. **Never invent metrics.** All numbers must come from `logs/**/*.log`.
3. Any code change must include: diff summary, one smoke command, expected key output lines.
4. Keep patches minimal — preserve existing behavior when new flags are not provided.
5. Log every run; record command lines for reproducibility.

## Commands

### Dataset Audit
```bash
bash scripts/run_audit.sh              # all datasets
bash scripts/run_audit.sh arxiv        # single dataset
```

### SBERT Feature Preprocessing
```bash
bash scripts/run_preprocess_sbert.sh                  # all text datasets
bash scripts/run_preprocess_sbert.sh arxiv            # single dataset
ENCODER=all-mpnet-base-v2 bash scripts/run_preprocess_sbert.sh
```

### Model Training

**BGRL** (PyG, conda env: `llm`):
```bash
cd repos/bgrl && conda run -n llm python train.py \
  --name arxiv --root ../../data --epochs 100 --cache-step 10 \
  --feat-pt ../../data/arxiv_sbert.pt
```

**GraphMAE** (DGL, conda env: `graphmae_env`, CPU with `--device -1`):
```bash
cd repos/graphmae && conda run -n graphmae_env python main_transductive.py \
  --dataset ogbn-arxiv --device -1 --seeds 0 \
  --max_epoch 30 --max_epoch_f 30 \
  --feat-pt ../../data/arxiv_sbert.pt
```

### Linear Probe Evaluation
```bash
python eval/run_lp.py --model bgrl --dataset ogbn-arxiv --ckpt repos/bgrl/checkpoints/arxiv_seed0.pt
```

### Log Parsing & Reporting
```bash
python tools/parse_logs.py > outputs/meeting_table.md
python tools/make_claude_state.py > notes/claude/PROJECT_STATE.md
```

## Architecture

### 4-Layer Pipeline
1. **Data** (complete): Raw datasets → SBERT 768d `.pt` files with metadata
2. **Training + Eval** (current focus): Train BGRL/GraphMAE on SBERT features → linear probe eval with frozen encoder
3. **Safety Eval** (pending): 6 safety scenarios TBD
4. **Results** (final): Normalize, rank comparison, visualizations

### SBERT Feature Integration
Both models patched with `_load_external_features()` triggered by `--feat-pt` flag:
- **BGRL** replaces `data.x` (PyG tensor)
- **GraphMAE** replaces `graph.ndata["feat"]` (DGL tensor), skips `scale_feats()` since SBERT is L2-normalized

### SBERT `.pt` Schema
Each `data/{dataset}_sbert.pt` is a dict with keys: `x` (float32 tensor `[num_nodes, 768]`), `dataset`, `encoder`, `dim`, `num_nodes`, `created_at`, `script_hash`, `notes`.

Default encoder: `all-mpnet-base-v2` (768d). NOT `all-MiniLM-L6-v2` (384d).

### Evaluation Framework (`eval/`)
- `run_lp.py`: Frozen encoder + task-specific linear head (node/graph/link)
- `load_encoder.py`: Load BGRL/GraphMAE checkpoints, freeze params
- `heads.py`: NodeHead, GraphHead (mean pooling), LinkHead (dot product)

## Two Conda Environments

| Env | Python | Use |
|-----|--------|-----|
| `llm` | 3.11 | BGRL, preprocessing, SBERT encoding |
| `graphmae_env` | 3.10 | GraphMAE (CPU stable path, `--device -1`) |

## Dataset Status

| Dataset | SBERT | Training | Notes |
|---------|-------|----------|-------|
| **Arxiv** | done | done | Fully working end-to-end |
| **WN18RR** | done | partial | Alignment NOT confirmed; eval protocol TBD (MRR/Hits@K) |
| **PCBA** | done | partial | Graph-level embeddings exist; node-level integration TBD |
| **Roman-Empire** | blocked | — | No confirmed text source |
| **Products** | blocked | — | Node-to-text alignment unverified |

## Known Issues

- **PyTorch 2.6+ OGB cache**: Requires `weights_only=False` in `torch.load()`. Handled by retry context manager in preprocessing/audit scripts.
- **WSL2 RAM**: Default ~8GB; ogbn-products (2.4M nodes) needs 16GB+. Increase via `.wslconfig`.
- **GraphMAE convergence**: `best_epoch=29=final_epoch` means model not fully converged. Increase `--max_epoch`.
