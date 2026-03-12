# Experiment Log

Record each run here: date, command, outcome, notes.

---

## Template entry

```
### Run YYYY-MM-DD — <short description>
- Command:   <exact command>
- Dataset:   <dataset name>
- Model:     <BGRL | GraphMAE>
- Feature:   <raw | sbert>
- Encoder:   <sbert model name if applicable>
- Outcome:   <success | error | ...>
- Loss (ep1): <value or "n/a">
- Notes:     <any anomalies, shape checks, warnings>
```

---

### 2026-02-23 — TSGFM Amazon-Products SBERT features generated (Colab A100)

- File: `amazon_products_tsgfm_sbert.pt`
- Shape: `(316513, 768)` float32
- Encoder: `sentence-transformers/all-mpnet-base-v2`
- Source text: `texts.pkl` (TSGFM bundle), slot 0 = node texts
- Validation: finite=True
- Note: This is **NOT** `ogbn-products` (node count mismatch vs 2,449,029). Use only with TSGFM Amazon-Products graph.

## Phase A: Dataset audit runs

<!-- Fill in after running scripts/run_audit.sh -->

---

## Phase B: SBERT preprocessing runs

<!-- Fill in after running scripts/run_preprocess_sbert.sh -->

---

## Phase C/D: Smoke tests — BGRL

<!-- Fill in after Phase C integration -->

---

## Phase C/D: Smoke tests — GraphMAE

<!-- Fill in after Phase D integration -->
