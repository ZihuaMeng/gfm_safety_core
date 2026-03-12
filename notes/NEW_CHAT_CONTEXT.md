# NEW_CHAT_CONTEXT (for fresh ChatGPT session)

## 1) TL;DR
- Project goal: swap raw graph features with SBERT features for GFM safety experiments (BGRL + GraphMAE), starting from arXiv.
- arXiv pipeline is working end-to-end for both models with external `--feat-pt` loading.
- `data/arxiv_sbert.pt` exists and is valid (`x: [169343, 768]`, float32, encoder `all-mpnet-base-v2`).
- BGRL and GraphMAE both show log evidence of feature replacement from 128-dim to 768-dim.
- Current meeting metrics are log-derived and already materialized in `outputs/meeting_table.md` / `outputs/summary.json`.
- BGRL parser currently uses final eval as best when explicit best lines are missing (`missing_best_metrics_used_final`).
- GraphMAE baseline run recorded for `pre30/ft30/seed0` on arXiv.
- WN18RR and PCBA artifacts exist but evaluation framing is not finalized for this meeting.
- Products text alignment to `ogbn-products` node order is NOT confirmed (TSGFM file is experimental only).
- Next immediate actions: regenerate meeting summary, run GraphMAE seed1/seed2, add SBERT-only baseline.

## 2) Hard rules / constraints
- Never guess node/text alignment; if not proven, label as **BLOCKED** or **EXPERIMENTAL**.
- Never invent metrics; all numbers must come from `logs/**/*.log` (or derived outputs generated from them).
- Keep code patches minimal and behavior-preserving when new flags are absent.
- Prefer reproducibility: log exact commands and keep outputs under `logs/` + `outputs/`.
- Treat `weights_only=False` workarounds as scoped/trusted-cache exceptions only (not blanket practice).
- Do not claim completion for datasets whose alignment/evaluation definitions are unresolved.

## 3) Environment + setup (exact)
- Host: **Windows 11 + WSL2 Ubuntu 24.04**.
- Project root: `~/projects/gfm_safety`.
- Conda env usage:
  - `llm`: preprocessing + BGRL flows.
  - `graphmae_env`: GraphMAE runs (CPU path currently stable).
- Note: local RTX 5060 setup had CUDA compatibility friction for some stacks; GraphMAE is run on CPU for stability.

### Snapshot commands (already run)
```bash
conda env list
conda run -n llm python -c "import sys,torch; print(sys.version); print(torch.__version__)"
conda run -n graphmae_env python -c "import sys,numpy as np,torch,torchdata,dgl; print(sys.version); print('numpy',np.__version__); print('torch',torch.__version__); print('torchdata',getattr(torchdata,'__version__','')); print('dgl',dgl.__version__)"
```

### Snapshot outputs captured
- `conda env list`: `base`, `llm`, `graphmae_env`.
- `llm`: Python `3.11.14`, torch `2.10.0+cu128`.
- `graphmae_env`: Python `3.10.19`, numpy `1.26.4`, torch `2.1.2+cu121`, torchdata `0.7.1`, dgl `1.1.3`.

## 4) Repo structure overview
(derived from `ls` + depth-limited listing)

```text
~/projects/gfm_safety/
├── data/            # datasets + SBERT artifacts (*.pt) + OGB/PyG caches
├── repos/
│   ├── bgrl/        # BGRL code + --feat-pt integration
│   └── graphmae/    # GraphMAE code + --feat-pt integration
├── src/             # preprocessing/audit scripts
├── logs/            # model run logs (bgrl/, graphmae/)
├── outputs/         # parser outputs (meeting_table.md, summary.json)
├── notes/           # meeting progress, rules, audit notes
├── tools/           # parse_logs.py
└── scripts/         # automation wrappers (make_meeting_md.sh, etc.)
```

- `data/`: datasets and feature artifacts (`*.pt`, OGB/PyG caches).
- `repos/bgrl/`: BGRL codebase with `--feat-pt` integration and OGB cache handling.
- `repos/graphmae/`: GraphMAE codebase with `--feat-pt` integration in `main_transductive.py`.
- `src/`: project scripts (`audit_datasets.py`, `preprocess_sbert_features.py`).
- `logs/`: run logs by model (`logs/bgrl/`, `logs/graphmae/`).
- `notes/`: planning/meeting artifacts and constraints (`AGENT_RULES.md`, meeting notes).
- `tools/`: log parser (`tools/parse_logs.py`).
- `scripts/`: automation wrappers (`make_meeting_md.sh`, etc.).
- `outputs/`: generated meeting summaries (`meeting_table.md`, `summary.json`).
- `runs/`: checkpoints/output folders.

## 5) Completed milestones (specific + reproducible)

### 5.1 SBERT arXiv feature artifact exists and schema is verified
File: `data/arxiv_sbert.pt`

Verified by loading with Python (`torch.load(..., weights_only=False)`):
- type: `dict`
- keys: `['created_at', 'dataset', 'dim', 'encoder', 'notes', 'num_nodes', 'script_hash', 'x']`
- `dataset`: `ogbn-arxiv`
- `encoder`: `all-mpnet-base-v2`
- `dim`: `768`
- `x` shape/dtype: `(169343, 768)`, `torch.float32`
- notes: `title_and_abs=169343 title_only=0 empty=0 missing_mapping=0`

Other `.pt` artifacts currently present under `data/`:
- `data/amazon_products_tsgfm_sbert.pt`
- `data/arxiv_sbert.pt`
- `data/pcba_sbert_graph.pt`
- `data/wn18rr_sbert.pt`

### 5.2 BGRL patched for `--feat-pt` and verified
- CLI arg added in `repos/bgrl/utils.py` (`--feat-pt`, `dest=feat_pt`).
- Integration point in `repos/bgrl/train.py`:
  - `_load_external_features(...)` validates schema/count/dtype/finiteness.
  - Replacement occurs before layer construction so input dim updates from `data.x.shape[1]`.
- Log evidence (`logs/bgrl/*.log`):
  - `[feat_pt] Replacing data.x for 'arxiv': original shape=(169343, 128) → new shape=(169343, 768)`
  - model print shows first layer `GCNConv(768, 512)`.

### 5.3 GraphMAE patched for `--feat-pt` and verified
- CLI arg added in `repos/graphmae/graphmae/utils.py` (`--feat-pt`, `dest=feat_pt`).
- Integration point in `repos/graphmae/main_transductive.py`:
  - `_load_external_features(...)` validates dict/`x`/count/dtype/finiteness.
  - replacement happens after `load_dataset()` and before `args.num_features` / `build_model()`.
- Log evidence (`logs/graphmae/arxiv_sbert_graphmae_pre30_ft30_seed0_20260225_220541.log`):
  - `[feat_pt] Replacing graph.ndata['feat'] ... (169343, 128) → (169343, 768)`.

### 5.4 Log-driven arXiv results (copied from generated outputs)
Source of truth: `outputs/meeting_table.md` / `outputs/summary.json`.

| run | source_log | best_val | best_test | notes |
|---|---|---:|---:|---|
| BGRL baseline run | `logs/bgrl/arxiv_sbert_run1_20260225_075303.log` | 55.4923 | 49.3816 | parser warning: `missing_best_metrics_used_final` |
| GraphMAE pre30/ft30 seed0 | `logs/graphmae/arxiv_sbert_graphmae_pre30_ft30_seed0_20260225_220541.log` | 0.5809 (epoch 29) | 0.5685 | `final_acc=0.5685±0.0000` |

## 6) Known issues encountered and fixes
- GraphMAE on Colab had DGL/torchdata/numpy binary-compat friction in prior attempts (historical note); stable path moved to local CPU env (`graphmae_env`).
- BGRL Colab cache path creation issue was addressed by `utils.create_dirs(...)` behavior and verified via successful OGB cache creation log:
  - `.../data/ogb/bgrl_ogbn-arxiv/raw/data.pt` written successfully.
- PyTorch `weights_only` compatibility on OGB cache load handled safely in `repos/bgrl/data.py`:
  - detect error, retry in scoped context manager with `weights_only=False` for trusted local cache only.

## 7) Dataset status + blockers
- **arXiv**: working end-to-end for BGRL + GraphMAE with SBERT features.
- **WN18RR**: `data/wn18rr_sbert.pt` exists (`x` shape `(40943, 768)`), but metadata notes alignment is not confirmed; treat as **pipeline smoke only; eval TBD**.
- **PCBA**: `data/pcba_sbert_graph.pt` exists with `x_graph` `(437929, 768)`; graph-level embedding integration into node-level SSL is **TBD**.
- **Products (ogbn-products)**: raw text alignment not confirmed; `data/amazon_products_tsgfm_sbert.pt` metadata explicitly says it is from TSGFM texts and **NOT ogbn-products** (`EXPERIMENTAL ONLY`).
- **Roman-Empire**: no confirmed node-text source in current artifacts; mark **BLOCKED** (no fabricated alignment).

## 8) Exact next commands the new chat should run

### 8.1 Regenerate meeting summary
```bash
cd ~/projects/gfm_safety
python tools/parse_logs.py > outputs/meeting_table.md
bash scripts/make_meeting_md.sh
```

### 8.2 GraphMAE arXiv baseline (pre30/ft30, CPU, SBERT)
```bash
cd ~/projects/gfm_safety/repos/graphmae
conda run -n graphmae_env python main_transductive.py \
  --dataset ogbn-arxiv \
  --device -1 \
  --seeds 0 \
  --max_epoch 30 \
  --max_epoch_f 30 \
  --feat-pt ../../data/arxiv_sbert.pt
```

### 8.3 BGRL arXiv baseline with SBERT
```bash
cd ~/projects/gfm_safety/repos/bgrl
conda run -n llm python train.py \
  --name arxiv \
  --root ../../data \
  --epochs 100 \
  --cache-step 10 \
  --feat-pt ../../data/arxiv_sbert.pt
```

## 9) Next steps plan (meeting-aligned)
- Run GraphMAE with `seed=1` and `seed=2`, then report mean±std.
- Add SBERT-only baseline (logistic regression or small MLP directly on SBERT features).
- Define evaluation protocols for WN18RR / PCBA / Products / Roman-Empire before any claim of multi-dataset pretrain completion.
- Keep alignment-sensitive datasets explicitly labeled BLOCKED/EXPERIMENTAL until mapping proof is added.
