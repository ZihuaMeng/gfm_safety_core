# Task Breakdown: SBERT-based Features for GFM Safety Project

## Project theme
On the safety of graph foundation models.

## Current assigned tasks (from 2020-02-20)
1. Use SBERT to convert five datasets into 768d embeddings and save them as five `.pt`
   files using a unified preprocessing script.
2. Modify the data loading interface in BGRL and GraphMAE to load SBERT-based features.

## Datasets (pretrain set)
- Arxiv        (OGB: ogbn-arxiv)
- Products     (OGB: ogbn-products)
- WN18RR       (knowledge graph; source TBD — must confirm loader after repo clone)
- Roman-Empire (heterophilous node classification; source TBD)
- PCBA         (OGB: ogbg-molpcba, graph-level, molecule; no natural text)

## Text availability (initial assumption — must confirm in audit)
- Text available: Arxiv, Products, WN18RR, Roman-Empire
- No text:        PCBA (molecule fingerprints, no per-node NL description)

---

## Integration points confirmed from upstream source code
(Confirmed against Namkyeong/BGRL_Pytorch and THUDM/GraphMAE on 2026-02-23.
 CANNOT confirm for the actual project forks until repos/ is populated.)

### BGRL (PyTorch Geometric)
- Feature field:    `data.x`  (PyG Data object)
- Input dim read:   `train.py` → `layers = [self._dataset.x.shape[1]] + hidden_layers`
- Load point:       `train.py` → `self._dataset = data.Dataset(root, name)[0]`
- Augmentation:     `utils.py Augmentation._feature_masking()` — clones `data.x`;
                    reads `data.x.shape[1]` for mask size; transparent to upstream swap
- Default datasets: Planetoid (cora/citeseer/pubmed), Coauthor (cs/physics),
                    Amazon (photo/computers), WikiCS
- NOT included:     ogbn-arxiv, ogbn-products, WN18RR, Roman-Empire, PCBA
                    → new loader entries needed in `utils.decide_config()`

### GraphMAE (DGL)
- Feature field:    `graph.ndata["feat"]`
- Input dim read:   `data_util.py` → `num_features = graph.ndata["feat"].shape[1]`
- Load point:       `data_util.py load_dataset()` — returns `(graph, (num_features, num_classes))`
- Scaling:          `scale_feats()` (StandardScaler) applied to raw feat before storage;
                    SBERT embeddings are already unit-normalised — must skip or gate this
- Entry point:      `main_transductive.py` → `graph, (num_features, num_classes) = load_dataset(dataset_name)`
                    then `feat = graph.ndata["feat"]` passed to `pretrain()`
- Default datasets: cora, citeseer, pubmed, ogbn-arxiv (basic), PPI
- NOT confirmed:    ogbn-products, WN18RR, Roman-Empire, PCBA in project fork

---

## Critical uncertainties (must resolve before Phase B/C)
1. Which exact fork/commit of BGRL and GraphMAE the project uses.
2. Whether the project fork already adds ogbn-products, WN18RR, Roman-Empire, PCBA loaders.
3. Exact text field and file path for each dataset (e.g., ogbn-arxiv needs separate
   `titleabs.tsv` from OGB's nodeidx2paperid mapping).
4. Node ordering alignment guarantee: is `data.x[i]` / `graph.ndata["feat"][i]` always
   the same node as `texts[i]`?  Must confirm per dataset.
5. PCBA strategy (decision needed):
   - raw passthrough (keep original atom features, skip SBERT)
   - zero 768-dim placeholder
   - SMILES → SBERT (if SMILES accessible and approved by PI)
6. Whether GraphMAE's `scale_feats()` must be disabled for SBERT features (likely yes).

---

## Implementation phases

### Phase A — Dataset audit  (→ src/audit_datasets.py)
- Load each dataset via its native loader.
- Print: num_nodes, num_edges, x/feat shape, dtype, text field availability.
- Confirm node ordering (how? document the mechanism).
- Record findings in notes/dataset_audit.md.
- Status: TODO — requires repos/ to be populated.

### Phase B — Unified SBERT preprocessing  (→ src/preprocess_sbert_features.py)
- For text datasets: extract texts, run SBERT, save .pt with unified schema.
- For PCBA: implement chosen strategy.
- Validate each .pt: shape [N, 768], float32, no NaN/Inf, num_nodes matches loader.
- Status: script skeleton written; dataset-specific sections are TODOs.

### Phase C — BGRL integration  (→ repos/bgrl/)
- Add `--feature-source {raw,sbert}` CLI flag.
- After loading dataset, if sbert: load .pt, assert shape[0] == num_nodes, replace data.x.
- Do NOT change models.py or augmentation logic.
- Status: TODO — requires Phase A + B.

### Phase D — GraphMAE integration  (→ repos/graphmae/)
- Add `--sbert-feat-path` CLI flag to main_transductive.py.
- In load_dataset(), after graph build: conditionally replace graph.ndata["feat"],
  skip scale_feats(), return corrected num_features.
- Do NOT change model architecture.
- Status: TODO — requires Phase A + B.

### Phase E — Smoke tests  (→ notes/experiment_log.md)
- Reload each .pt standalone; assert shape/dtype.
- Run 1 training epoch for BGRL + GraphMAE with sbert features on one dataset.
- Record: loss finite? feature dim correct? no assertion errors.

---

## Reproducibility rules
- Log encoder name, embedding dim, preprocessing date, script git commit hash.
- Never overwrite .pt files silently; bump a version key or use timestamped filenames.
- Keep all assumptions explicit in notes/ and in code comments.
- Do not hardcode num_features; always derive from the loaded tensor.
