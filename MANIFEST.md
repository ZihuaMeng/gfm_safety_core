# Layer 2 Bundle Manifest

Local Layer 2 files remain the editing source of truth. Published `work/layer2/` is a snapshot that should only be refreshed by `scripts/sync_layer2_bundle.py`.

Stable bundle evidence should use target-specific suite manifests and comparison artifacts.
`results/baseline/layer2_suite_official_manifest.json` and
`results/baseline/layer2_suite_debug_manifest.json` are rolling primary outputs that get
overwritten by the latest executed suite target, so they are not safe as canonical
cross-target evidence paths.

WN18RR entries in this manifest are copied as experimental evidence only. They must not be promoted into `official_candidate_*` or `all_proven_local`.

<!-- layer2-bundle-manifest:start -->
```json
{
  "bundle_root": "work/layer2",
  "mappings": [
    {
      "source": "MANIFEST.md",
      "destination": "MANIFEST.md",
      "kind": "bundle_manifest"
    },
    {
      "source": "README.md",
      "destination": "README.md",
      "kind": "artifact"
    },
    {
      "source": "scripts/layer2_artifact_utils.py",
      "destination": "scripts/layer2_artifact_utils.py",
      "kind": "code"
    },
    {
      "source": "scripts/refresh_layer2_artifacts.py",
      "destination": "scripts/refresh_layer2_artifacts.py",
      "kind": "code"
    },
    {
      "source": "scripts/sync_layer2_bundle.py",
      "destination": "scripts/sync_layer2_bundle.py",
      "kind": "code"
    },
    {
      "source": "scripts/layer2_hpc_plan_utils.py",
      "destination": "scripts/layer2_hpc_plan_utils.py",
      "kind": "code"
    },
    {
      "source": "scripts/generate_layer2_hpc_plan.py",
      "destination": "scripts/generate_layer2_hpc_plan.py",
      "kind": "code"
    },
    {
      "source": "scripts/ingest_layer2_hpc_results.py",
      "destination": "scripts/ingest_layer2_hpc_results.py",
      "kind": "code"
    },
    {
      "source": "scripts/layer2_suite_targets.py",
      "destination": "scripts/layer2_suite_targets.py",
      "kind": "code"
    },
    {
      "source": "scripts/run_layer2_suite.py",
      "destination": "scripts/run_layer2_suite.py",
      "kind": "code"
    },
    {
      "source": "scripts/verify_wn18rr_alignment.py",
      "destination": "scripts/verify_wn18rr_alignment.py",
      "kind": "code"
    },
    {
      "source": "outputs/summary.json",
      "destination": "artifacts/summary.json",
      "kind": "artifact"
    },
    {
      "source": "outputs/meeting_table.md",
      "destination": "artifacts/meeting_table.md",
      "kind": "artifact"
    },
    {
      "source": "notes/meeting_progress_arxiv_sbert.md",
      "destination": "artifacts/meeting_progress_arxiv_sbert.md",
      "kind": "artifact"
    },
    {
      "source": "notes/pcba_graph_protocol_report.md",
      "destination": "artifacts/pcba_graph_protocol_report.md",
      "kind": "artifact"
    },
    {
      "source": "notes/wn18rr_link_protocol_report.md",
      "destination": "artifacts/wn18rr_link_protocol_report.md",
      "kind": "artifact"
    },
    {
      "source": "notes/layer2_hpc_runbook.md",
      "destination": "artifacts/layer2_hpc_runbook.md",
      "kind": "artifact"
    },
    {
      "source": "results/baseline/layer2_hpc_plan.json",
      "destination": "artifacts/layer2_hpc_plan.json",
      "kind": "artifact"
    },
    {
      "source": "results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json",
      "destination": "evidence/layer2_suite_official_candidate_arxiv_official_manifest.json",
      "kind": "evidence"
    },
    {
      "source": "results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json",
      "destination": "evidence/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json",
      "kind": "evidence"
    },
    {
      "source": "results/baseline/layer2_suite_graphmae_pcba_native_graph_official_manifest.json",
      "destination": "evidence/layer2_suite_graphmae_pcba_native_graph_official_manifest.json",
      "kind": "evidence"
    },
    {
      "source": "results/baseline/pcba_graph_comparison.json",
      "destination": "evidence/pcba_graph_comparison.json",
      "kind": "evidence"
    },
    {
      "source": "results/baseline/wn18rr_alignment_audit.json",
      "destination": "evidence/wn18rr_alignment_audit.json",
      "kind": "evidence"
    },
    {
      "source": "results/baseline/wn18rr_semantic_alignment_audit.json",
      "destination": "evidence/wn18rr_semantic_alignment_audit.json",
      "kind": "evidence"
    },
    {
      "source": "results/baseline/layer2_suite_wn18rr_experimental_compare_debug_manifest.json",
      "destination": "evidence/layer2_suite_wn18rr_experimental_compare_debug_manifest.json",
      "kind": "evidence"
    },
    {
      "source": "results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json",
      "destination": "evidence/layer2_suite_wn18rr_experimental_compare_official_manifest.json",
      "kind": "evidence"
    },
    {
      "source": "results/baseline/wn18rr_link_comparison.json",
      "destination": "evidence/wn18rr_link_comparison.json",
      "kind": "evidence"
    }
  ]
}
```
<!-- layer2-bundle-manifest:end -->
