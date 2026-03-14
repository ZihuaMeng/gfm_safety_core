# Meeting Progress: arxiv_sbert

## Official Candidate Coverage
- GraphMAE arXiv SBERT node: status=success; accuracy=0.651565; source=results/baseline/graphmae_ogbn-arxiv.hpc.json (manual_hpc_result_json; provenance=fresh_manual_hpc_execution); manifest_backed=false.
- GraphMAE arXiv SBERT node supporting bootstrap context: `state/layer2_bootstrap/results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json`; refresh_reason=official_candidate_row_refreshed_from_manual_hpc_execution.
- BGRL arXiv SBERT node: status=success; accuracy=0.493488; source=results/baseline/bgrl_ogbn-arxiv.hpc.json (manual_hpc_result_json; provenance=fresh_manual_hpc_execution); manifest_backed=false.
- BGRL arXiv SBERT node supporting bootstrap context: `state/layer2_bootstrap/results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json`; refresh_reason=official_candidate_row_refreshed_from_manual_hpc_execution.
- These official-candidate rows were refreshed from fresh manual HPC execution JSONs without fabricating suite manifests.

## Local Debug / Full-Local Coverage
- GraphMAE PCBA native graph: status=debug_success; ap=0.129023; surface=local_debug; source=state/layer2_bootstrap/results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json (suite_manifest; provenance=bootstrap_inherited_unchanged); provenance_status=bootstrap_inherited_unchanged.
- GraphMAE PCBA native graph (full-local non-debug): status=success; ap=0.034122; surface=full_local_non_debug; source=results/baseline/graphmae_ogbg-molpcba.official_local.hpc.json (manual_hpc_result_json; provenance=fresh_manual_hpc_execution); provenance_status=fresh_manual_hpc_execution.
- WN18RR alignment audit: status=success; n/a; surface=none; source=state/layer2_bootstrap/results/baseline/wn18rr_alignment_audit.json (audit_json; provenance=bootstrap_inherited_unchanged); provenance_status=bootstrap_inherited_unchanged.
- WN18RR link-eval: status=success; mrr=0.000159; surface=full_local_non_debug; source=state/layer2_bootstrap/results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json (suite_manifest; provenance=bootstrap_inherited_unchanged); provenance_status=bootstrap_inherited_unchanged.
- WN18RR relation-aware link-eval: status=success; mrr=0.000515; surface=full_local_non_debug; source=state/layer2_bootstrap/results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json (suite_manifest; provenance=bootstrap_inherited_unchanged); provenance_status=bootstrap_inherited_unchanged.
- WN18RR full-scale link-eval: status=success; mrr=0.000159; surface=full_local_non_debug; source=state/layer2_bootstrap/results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json (suite_manifest; provenance=bootstrap_inherited_unchanged); provenance_status=bootstrap_inherited_unchanged.
- WN18RR relation-aware full-scale link-eval: status=success; mrr=0.000515; surface=full_local_non_debug; source=state/layer2_bootstrap/results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json (suite_manifest; provenance=bootstrap_inherited_unchanged); provenance_status=bootstrap_inherited_unchanged.

## PCBA Comparison
- Comparison profile: `pcba_graph_compare` pairs `default` and `full_local_non_debug` under `graphmae_pcba_native_graph`.
- Local debug remains inherited from bootstrap: source_kind=suite_manifest; bootstrap_inherited=true; source_path=state/layer2_bootstrap/results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json.
- Full-local non-debug was refreshed from manual HPC execution: status=success; ap=0.034122243717244045; source_kind=manual_hpc_result_json; manifest_backed=false; source_path=results/baseline/graphmae_ogbg-molpcba.official_local.hpc.json.
- Checkpoint provenance: debug=checkpoints/graphmae_pcba_native_debug.pt; full_local=checkpoints/graphmae_ogbg-molpcba.official_local.pt; distinct=true; debug_checkpoint_surface_removed=true.
- AP delta (full-local non-debug minus debug): -0.09490093655179083; still_not_locked=manifest_backed_execution_missing, official_metric_flag_false, not_locked_official_candidate_surface.

## Experimental Coverage
- WN18RR is included in `all_proven_local`. Baseline dot-product path retains `relation_types_ignored=true`.

## WN18RR Comparison
- Comparison profile: `wn18rr_experimental_compare` pairs `graphmae_wn18rr_sbert_link` and `graphmae_wn18rr_sbert_link_relaware`.
- Relation-aware delta: mrr_delta=0.000356; hits@1_delta=0.0; hits@3_delta=0.0; hits@10_delta=0.000319; metric_delta_source=fullscale.
- Shared remaining blockers: none.
- Remaining experimental reasons: none.
- Semantic alignment: verdict=verified_by_provenance; verified=true.
- Negative-sampling contract: defined=true; blocker_cleared=true.
- Official metric: full_scale_eval_completed=true; blocker_retained=false.

## Ingestion Inputs
- bootstrap root: `state/layer2_bootstrap`
- fresh GraphMAE arXiv result: `results/baseline/graphmae_ogbn-arxiv.hpc.json`
- fresh BGRL arXiv result: `results/baseline/bgrl_ogbn-arxiv.hpc.json`
- fresh GraphMAE PCBA official-local result: `results/baseline/graphmae_ogbg-molpcba.official_local.hpc.json`
- bootstrap arxiv_official_manifest: `state/layer2_bootstrap/results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json`
- bootstrap pcba_debug_manifest: `state/layer2_bootstrap/results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json`
- bootstrap pcba_official_manifest: `state/layer2_bootstrap/results/baseline/layer2_suite_graphmae_pcba_native_graph_official_manifest.json`
- bootstrap pcba_comparison_json: `state/layer2_bootstrap/results/baseline/pcba_graph_comparison.json`
- bootstrap wn18rr_alignment_audit: `state/layer2_bootstrap/results/baseline/wn18rr_alignment_audit.json`
- bootstrap wn18rr_semantic_alignment_audit: `state/layer2_bootstrap/results/baseline/wn18rr_semantic_alignment_audit.json`
- bootstrap wn18rr_debug_manifest: `state/layer2_bootstrap/results/baseline/layer2_suite_wn18rr_experimental_compare_debug_manifest.json`
- bootstrap wn18rr_official_manifest: `state/layer2_bootstrap/results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json`
- bootstrap wn18rr_comparison_json: `state/layer2_bootstrap/results/baseline/wn18rr_link_comparison.json`
- bootstrap meeting_progress_note: `state/layer2_bootstrap/notes/meeting_progress_arxiv_sbert.md`
- bootstrap pcba_report_note: `state/layer2_bootstrap/notes/pcba_graph_protocol_report.md`
- bootstrap wn18rr_report_note: `state/layer2_bootstrap/notes/wn18rr_link_protocol_report.md`
