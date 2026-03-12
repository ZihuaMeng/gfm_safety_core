# PCBA Graph Protocol Report

## Scope
- Comparison profile: `pcba_graph_compare` for `ogbg-molpcba`.
- Compared target/profile surfaces: `graphmae_pcba_native_graph` `default` vs `full_local_non_debug`.
- Sources: `results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json` and `results/baseline/layer2_suite_graphmae_pcba_native_graph_official_manifest.json`.

## Checkpoint Provenance
- Debug checkpoint path: `checkpoints/graphmae_pcba_native_debug.pt`.
- Full-local non-debug checkpoint path: `checkpoints/graphmae_ogbg-molpcba.official_local.pt`.
- Checkpoint paths distinct=true; full_local_debug_checkpoint_surface_removed=true.

## What The Current Debug Profile Proves
- Status=debug_success; ap=0.129023; evidence_surface=local_debug; manifest_backed=true.
- Mode=debug; manifest_path=results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json; checkpoint_path=checkpoints/graphmae_pcba_native_debug.pt; debug_mode=true; debug_max_graphs=64; max_eval_batches=8; split_truncation=per_split_first_n.
- This proves the Layer 2 GraphMAE PCBA path can export a usable checkpoint, run the unified graph evaluator, and emit AP on a deterministic local slice.

## What The Full-Local Non-Debug Profile Adds
- Status=success; ap=0.034201; official_metric=false; evidence_surface=full_local_non_debug; manifest_backed=true.
- Mode=official; manifest_path=results/baseline/layer2_suite_graphmae_pcba_native_graph_official_manifest.json; checkpoint_path=checkpoints/graphmae_ogbg-molpcba.official_local.pt; debug_mode=false; debug_max_graphs=none; max_eval_batches=none; split_truncation=disabled.
- Additions over the debug profile: non_debug_execution_surface, checkpoint_provenance_separated, debug_checkpoint_surface_removed, debug_graph_cap_removed, split_truncation_removed, eval_batch_cap_removed, manifest_backed_profile_available, side_by_side_ap_delta_available.
- AP delta (full-local non-debug minus debug): -0.094822.

## Why This Is Still Not A Locked Official Result
- Shared remaining blockers: the artifact still carries official_metric=false, the evidence surface is local, not a locked official candidate.
- Full-local non-debug specific blockers: none.
- The fuller profile is intentionally reported as `full_local_non_debug`: it is a stronger local execution surface, not an official locked run or official-candidate row.

## Layer 2 Protocol Fit
- PCBA now has two first-class Layer 2 artifact surfaces under the same public target: a fast truncated debug profile and a full-local non-debug comparison profile.
- This mirrors the existing WN18RR comparison pattern, but PCBA remains outside the locked official-candidate surface until the remaining blockers are cleared.
