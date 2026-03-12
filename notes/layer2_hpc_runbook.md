# Layer 2 HPC Runbook

## Why These Are The Next HPC Targets

The local Layer 2 bundle is already coherent enough to stop refactoring and move to compute-backed refreshes. The remaining gap is not output existence, but proving that the next ingestion pass came from the intended fresh HPC rerun instead of from already-present local execution-backed artifacts.

This runbook therefore keeps the current runnable queue and priorities unchanged:

| Priority | Target | Why now |
| --- | --- | --- |
| P0 | `graphmae_arxiv_sbert_node` | Refresh the current GraphMAE arXiv official-candidate evidence with a forced fresh export and keep the canonical arXiv suite manifest execution-backed. |
| P0 | `bgrl_arxiv_sbert_node` | Refresh the current BGRL arXiv official-candidate evidence inside the same canonical arXiv suite manifest. |
| P1 | `graphmae_pcba_native_graph` (`full_local_non_debug`) | Refresh the fuller non-debug local PCBA surface that already backs the current comparison/report flow. |

WN18RR is still excluded from the runnable official/local HPC queue. It remains experimental-only because `experimental_fence_still_enabled`, so it must not be promoted into any `official_candidate_*` surface.

## Fresh-Run Provenance Contract

`results/baseline/layer2_hpc_plan.json` is now the contract for what counts as a fresh HPC refresh. Each runnable target includes a `freshness` block with deterministic fields:

- `contract_version`
- `required_provenance_tag`
- `launch_provenance_tag`
- `expected_launch_id`
- `expected_run_mode`
- `expected_requested_target`
- `expected_profile`
- `expected_force_export=true`
- `requested_hpc_refresh=true`
- `required_fresh_export=true`
- `freshness_evidence_requirements`

The launch command itself is intentionally unchanged. The plan now supplies the provenance contract through `launch.env`, and that env must be present when the unchanged launch command is executed on HPC:

- `LAYER2_HPC_PROVENANCE_CONTRACT_VERSION`
- `LAYER2_HPC_EXPECTED_LAUNCH_ID`
- `LAYER2_HPC_EXPECTED_RUN_MODE`
- `LAYER2_HPC_EXPECTED_REQUESTED_TARGET`
- `LAYER2_HPC_LAUNCH_PROVENANCE_TAG`
- `LAYER2_HPC_REQUESTED_REFRESH=true`
- `LAYER2_HPC_REQUIRED_FRESH_EXPORT=true`
- `LAYER2_HPC_PLAN_PATH`
- `LAYER2_HPC_PLAN_SCHEMA_VERSION`

When that env is present, `scripts/run_layer2_suite.py` writes additive provenance markers into the stable suite manifest and rollup. The manifest is the ingestion source of truth; the rollup mirrors the same contract for quick inspection.

Top-level manifest and rollup fields written for planned HPC reruns:

- `provenance_contract_version`
- `requested_hpc_refresh`
- `expected_launch_id`
- `expected_run_mode`
- `expected_requested_target`
- `launch_provenance_tag`
- `required_fresh_export`
- `provenance_plan_path`
- `provenance_plan_schema_version`

Per-target fields written for planned HPC reruns:

- `requested_hpc_refresh`
- `expected_launch_id`
- `launch_provenance_tag`
- `provenance_tag`
- `required_fresh_export`

## Target Order

1. Run the arXiv official refresh first with `official_candidate_arxiv` so the stable manifest `results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json` is rebuilt in one pass.
2. Run the PCBA fuller rerun second with `graphmae_pcba_native_graph --mode official`, which maps to the `full_local_non_debug` profile and the dedicated checkpoint `checkpoints/graphmae_ogbg-molpcba.official_local.pt`.
3. After the reruns complete, ingest only the stable suite manifests through `scripts/refresh_layer2_artifacts.py`, then publish the updated `work/layer2/` snapshot through `scripts/sync_layer2_bundle.py`.

## Environments And Working Directories

All commands are generated from the current Layer 2 target registry and are recorded in `results/baseline/layer2_hpc_plan.json`.

Before launching on HPC, export the exact `launch.env` block for the planned target, then run the unchanged `launch.command`.

| Surface | Env | Working directory | Command |
| --- | --- | --- | --- |
| arXiv orchestrator | repo shell + `launch.env` | `.` | `python scripts/run_layer2_suite.py --target official_candidate_arxiv --mode official --force-export` |
| GraphMAE arXiv export | `graphmae_env` | `repos/graphmae` | `conda run -n graphmae_env python main_transductive.py --seeds 0 --dataset ogbn-arxiv --device -1 --max_epoch 30 --max_epoch_f 30 --num_hidden 512 --num_heads 4 --num_layers 2 --lr 0.005 --weight_decay 0.0005 --lr_f 0.001 --weight_decay_f 0.0 --in_drop 0.2 --attn_drop 0.1 --mask_rate 0.5 --replace_rate 0.0 --encoder gat --decoder gat --loss_fn sce --alpha_l 2.0 --optimizer adam --feat-pt ../../data/arxiv_sbert.pt --export-encoder-ckpt ../../checkpoints/graphmae_ogbn-arxiv.pt` |
| GraphMAE arXiv eval | `graphmae_env` | `.` | `conda run -n graphmae_env python eval/run_lp.py --model graphmae --dataset ogbn-arxiv --ckpt checkpoints/graphmae_ogbn-arxiv.pt --out_json results/baseline/graphmae_ogbn-arxiv.json --feat-pt data/arxiv_sbert.pt` |
| BGRL arXiv export | `llm` | `repos/bgrl` | `conda run -n llm python train.py --name arxiv --root ../../data --epochs 100 --cache-step 10 --feat-pt ../../data/arxiv_sbert.pt --export-encoder-ckpt ../../checkpoints/bgrl_ogbn-arxiv.pt` |
| BGRL arXiv eval | `llm` | `.` | `conda run -n llm python eval/run_lp.py --model bgrl --dataset ogbn-arxiv --ckpt checkpoints/bgrl_ogbn-arxiv.pt --out_json results/baseline/bgrl_ogbn-arxiv.json --feat-pt data/arxiv_sbert.pt` |
| PCBA orchestrator | repo shell + `launch.env` | `.` | `python scripts/run_layer2_suite.py --target graphmae_pcba_native_graph --mode official --force-export` |
| PCBA export | `graphmae_env` | `repos/graphmae` | `conda run -n graphmae_env python main_graph.py --dataset ogbg-molpcba --device -1 --max_epoch 1 --eval none --export-encoder-ckpt ../../checkpoints/graphmae_ogbg-molpcba.official_local.pt` |
| PCBA eval | `graphmae_env` | `.` | `conda run -n graphmae_env python eval/run_lp.py --model graphmae --dataset ogbg-molpcba --ckpt checkpoints/graphmae_ogbg-molpcba.official_local.pt --out_json results/baseline/graphmae_ogbg-molpcba.official_local.json --max_train_steps 32` |

## Expected Output Artifacts

The ingestion path uses the stable suite manifests, not the rolling `layer2_suite_official_manifest.json` alias.

| Target | Checkpoint | Result JSON | Stable manifest | Stable rollup |
| --- | --- | --- | --- | --- |
| GraphMAE arXiv official refresh | `checkpoints/graphmae_ogbn-arxiv.pt` | `results/baseline/graphmae_ogbn-arxiv.json` | `results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json` | `results/baseline/layer2_suite_official_candidate_arxiv_official_rollup.json` |
| BGRL arXiv official refresh | `checkpoints/bgrl_ogbn-arxiv.pt` | `results/baseline/bgrl_ogbn-arxiv.json` | `results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json` | `results/baseline/layer2_suite_official_candidate_arxiv_official_rollup.json` |
| GraphMAE PCBA fuller rerun | `checkpoints/graphmae_ogbg-molpcba.official_local.pt` | `results/baseline/graphmae_ogbg-molpcba.official_local.json` | `results/baseline/layer2_suite_graphmae_pcba_native_graph_official_manifest.json` | `results/baseline/layer2_suite_graphmae_pcba_native_graph_official_rollup.json` |

## Copy Results Back Into This Checkout

Copy the stable outputs back into the same repo-relative paths that the plan expects. The safest approach is to copy the exact checkpoint, result JSON, stable manifest, and stable rollup for each completed launch.

Example pattern from an HPC checkout back into this checkout:

```bash
rsync -av <hpc_repo>/checkpoints/graphmae_ogbn-arxiv.pt checkpoints/graphmae_ogbn-arxiv.pt
rsync -av <hpc_repo>/results/baseline/graphmae_ogbn-arxiv.json results/baseline/graphmae_ogbn-arxiv.json
rsync -av <hpc_repo>/results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json
rsync -av <hpc_repo>/results/baseline/layer2_suite_official_candidate_arxiv_official_rollup.json results/baseline/layer2_suite_official_candidate_arxiv_official_rollup.json
```

Do the same for the BGRL arXiv checkpoint/result pair and for the PCBA checkpoint/result/manifest/rollup pair. For the shared arXiv launch, both P0 targets rely on the same stable manifest and rollup.

## How Ingestion Decides Freshness

`scripts/ingest_layer2_hpc_results.py --dry-run` now distinguishes between artifact existence and fresh-run proof.

Target status meanings:

- `fresh_hpc_rerun_complete`: all required outputs exist, the stable manifest matches the plan, and the manifest proves the planned rerun through matching provenance tags plus `force_export_requested=true`, `fresh_export_used=true`, `stage_export_status=fresh_export_success`, and successful eval/parse status.
- `ready_preexisting_local_execution`: outputs are execution-backed and internally consistent, but they do not prove the planned HPC rerun. This is the current local state.
- `fresh_hpc_rerun_missing_provenance`: outputs exist and the manifest claims some planned refresh metadata, but the provenance contract is incomplete or mismatched, so the run does not count as the planned fresh HPC refresh.
- `missing_outputs`: one or more required checkpoint/result/manifest/rollup files are absent.
- `stale_or_mismatched_outputs`: the files exist, but manifest/result/rollup metadata disagree with the plan or no longer describe the intended target surface.

Refresh gating is intentionally strict:

- `--run-refresh` only proceeds when at least one launch is `fresh_hpc_rerun_complete`.
- full-refresh readiness requires every planned target to be `fresh_hpc_rerun_complete`, unless `--allow-partial-refresh` is used.

## Why Current Local Outputs Do Not Count Yet

The current repo already has execution-backed outputs for all runnable targets, but those manifests were not produced under the fresh-run HPC contract. Today they show:

- `fresh_export_used=false`
- `stage_export_status=skipped_existing`
- no matching planned HPC provenance fields such as `expected_launch_id`, `launch_provenance_tag`, and per-target `provenance_tag`

That is why ingestion now reports the current state as `ready_preexisting_local_execution` instead of treating it as a generic partial failure. The artifacts are real and usable as local evidence, but they are not yet a contract-satisfying fresh HPC refresh.

## Post-Run Ingestion Back Into Layer 2

Once the HPC outputs are copied back into this checkout:

1. Regenerate the plan so the frozen contract in `results/baseline/layer2_hpc_plan.json` is current:
   `python scripts/generate_layer2_hpc_plan.py`
2. Inspect what is fresh-complete vs preexisting vs blocked without mutating artifacts:
   `python scripts/ingest_layer2_hpc_results.py --dry-run`
3. If at least one launch is `fresh_hpc_rerun_complete`, refresh the existing Layer 2 artifacts:
   `python scripts/ingest_layer2_hpc_results.py --run-refresh`
4. Publish the updated bundle snapshot:
   `python scripts/ingest_layer2_hpc_results.py --run-sync-bundle`

The ingestion script still prepares or executes the existing artifact refresh pipeline:

- `scripts/refresh_layer2_artifacts.py`
- `scripts/sync_layer2_bundle.py`

That preserves the current manifest-backed bundle logic, the current summary/protocol reports, the current target priorities, and the WN18RR experimental-only publication rule.
