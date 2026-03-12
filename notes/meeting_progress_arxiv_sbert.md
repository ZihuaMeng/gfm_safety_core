# Meeting Progress: arxiv_sbert

## Official Candidate Coverage
- GraphMAE arXiv SBERT node: status=success; accuracy=0.640413; fresh_export_used=false.
- BGRL arXiv SBERT node: status=success; accuracy=0.494969; fresh_export_used=false.
- These remain the current registry-backed official-candidate rows.

## Local Debug / Full-Local Coverage
- GraphMAE PCBA native graph: status=debug_success; ap=0.129023; surface=local-debug; official_metric=false; manifest_mode=debug; evidence=results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json (suite_manifest).
- GraphMAE PCBA native graph (full-local non-debug): status=success; ap=0.034201; surface=full-local-non-debug; official_metric=false; manifest_mode=official; evidence=results/baseline/layer2_suite_graphmae_pcba_native_graph_official_manifest.json (suite_manifest).

## PCBA Comparison
- Comparison profile: `pcba_graph_compare` pairs `default` and `full_local_non_debug` under `graphmae_pcba_native_graph`.
- Full-local non-debug status=success; ap=0.034201105224578345; surface=full_local_non_debug; manifest_backed=true; debug_mode=false; debug_max_graphs=none; max_eval_batches=none; split_truncation=disabled.
- Checkpoint provenance: debug=checkpoints/graphmae_pcba_native_debug.pt; full_local=checkpoints/graphmae_ogbg-molpcba.official_local.pt; distinct=true; debug_checkpoint_surface_removed=true.
- AP delta (full-local non-debug minus debug): -0.09482207504445653; still_not_locked=official_metric_flag_false, not_locked_official_candidate_surface.

## Experimental Coverage
- WN18RR alignment audit: status=success; checks=13/13; num_entities=40943; feat_rows=40943; relation_count=11.
- WN18RR alignment audit caveat: semantic_alignment_verified=false; ordering_evidence_passed=true.
- WN18RR experimental link-eval: status=debug_success; mrr=0.000104; hits@1=0.0; hits@3=0.0; hits@10=0.0; relation_types_ignored=true; official_metric=false.
- WN18RR relation-aware experimental link-eval: status=debug_success; mrr=0.000104; hits@1=0.0; hits@3=0.0; hits@10=0.0; relation_types_ignored=false; official_metric=false.
- WN18RR full-scale experimental link-eval: status=success; mrr=0.000159; hits@1=0.0; hits@3=0.0; hits@10=0.00016; relation_types_ignored=true; official_metric=true.
- WN18RR relation-aware full-scale experimental link-eval: status=success; mrr=0.000515; hits@1=0.0; hits@3=0.0; hits@10=0.000479; relation_types_ignored=false; official_metric=true.
- Experimental datasets remain excluded from `official_candidate_*` and `all_proven_local`: wn18rr.

## WN18RR Comparison
- Comparison profile: `wn18rr_experimental_compare` pairs `graphmae_wn18rr_sbert_link` and `graphmae_wn18rr_sbert_link_relaware`.
- Relation-aware delta: improved_only_by_relation_aware=relation_types_ignored; mrr_delta=0.000356; hits@1_delta=0.0; hits@3_delta=0.0; hits@10_delta=0.000319; scorer_trained=true; scorer_train_steps=10.
- Shared remaining blockers: none.
- Semantic alignment: verdict=verified_by_provenance; verified=true.
- Negative-sampling contract: defined=true; blocker_cleared=true.
- Official metric: full_scale_eval_completed=true; blocker_retained=false.

## Refresh Inputs
- official manifest: `results/baseline/layer2_suite_official_manifest.json`
- debug manifest: `results/baseline/layer2_suite_debug_manifest.json`
- additional manifests: none
- auto-discovered suite manifests: `results/baseline/layer2_suite_all_proven_local_debug_manifest.json`, `results/baseline/layer2_suite_all_proven_local_debug_preview_manifest.json`, `results/baseline/layer2_suite_debug_preview_manifest.json`, `results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json`, `results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_preview_manifest.json`, `results/baseline/layer2_suite_graphmae_pcba_native_graph_official_manifest.json`, `results/baseline/layer2_suite_graphmae_pcba_native_graph_official_preview_manifest.json`, `results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json`, `results/baseline/layer2_suite_official_candidate_arxiv_official_preview_manifest.json`, `results/baseline/layer2_suite_official_candidate_local_debug_manifest.json`, `results/baseline/layer2_suite_official_candidate_local_debug_preview_manifest.json`, `results/baseline/layer2_suite_official_preview_manifest.json`, `results/baseline/layer2_suite_pcba_graph_compare_debug_manifest.json`, `results/baseline/layer2_suite_pcba_graph_compare_debug_preview_manifest.json`, `results/baseline/layer2_suite_pcba_graph_compare_official_manifest.json`, `results/baseline/layer2_suite_wn18rr_experimental_compare_debug_manifest.json`, `results/baseline/layer2_suite_wn18rr_experimental_compare_debug_preview_manifest.json`, `results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json`, `results/baseline/layer2_suite_wn18rr_experimental_compare_official_preview_manifest.json`, `results/baseline/layer2_suite_wn18rr_experimental_debug_manifest.json`, `results/baseline/layer2_suite_wn18rr_experimental_debug_preview_manifest.json`
- manifest search order: `results/baseline/layer2_suite_official_manifest.json`, `results/baseline/layer2_suite_debug_manifest.json`, `results/baseline/layer2_suite_all_proven_local_debug_manifest.json`, `results/baseline/layer2_suite_all_proven_local_debug_preview_manifest.json`, `results/baseline/layer2_suite_debug_preview_manifest.json`, `results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json`, `results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_preview_manifest.json`, `results/baseline/layer2_suite_graphmae_pcba_native_graph_official_manifest.json`, `results/baseline/layer2_suite_graphmae_pcba_native_graph_official_preview_manifest.json`, `results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json`, `results/baseline/layer2_suite_official_candidate_arxiv_official_preview_manifest.json`, `results/baseline/layer2_suite_official_candidate_local_debug_manifest.json`, `results/baseline/layer2_suite_official_candidate_local_debug_preview_manifest.json`, `results/baseline/layer2_suite_official_preview_manifest.json`, `results/baseline/layer2_suite_pcba_graph_compare_debug_manifest.json`, `results/baseline/layer2_suite_pcba_graph_compare_debug_preview_manifest.json`, `results/baseline/layer2_suite_pcba_graph_compare_official_manifest.json`, `results/baseline/layer2_suite_wn18rr_experimental_compare_debug_manifest.json`, `results/baseline/layer2_suite_wn18rr_experimental_compare_debug_preview_manifest.json`, `results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json`, `results/baseline/layer2_suite_wn18rr_experimental_compare_official_preview_manifest.json`, `results/baseline/layer2_suite_wn18rr_experimental_debug_manifest.json`, `results/baseline/layer2_suite_wn18rr_experimental_debug_preview_manifest.json`
- WN18RR alignment audit: `results/baseline/wn18rr_alignment_audit.json`
- PCBA debug result fallback: `results/baseline/graphmae_ogbg-molpcba.native.debug.json`
- PCBA full-local result fallback: `results/baseline/graphmae_ogbg-molpcba.official_local.json`
- WN18RR debug result fallback: `results/baseline/graphmae_wn18rr.experimental.debug.json`
- WN18RR relaware debug result fallback: `results/baseline/graphmae_wn18rr.relaware.experimental.debug.json`
