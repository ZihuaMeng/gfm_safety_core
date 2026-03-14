# GFM Safety - Layer 2 Local State

This repo's current Layer 2 narrative is anchored to execution-backed evidence under
`results/baseline/`. As of March 10, 2026, the stable source-of-truth files are the
target-specific suite manifests, comparison JSON artifacts, and alignment audits, not
the rolling `layer2_suite_official_manifest.json` / `layer2_suite_debug_manifest.json`
aliases.

## Current Layer 2 State

- arXiv official-candidate coverage is execution-backed and current:
  - GraphMAE arXiv SBERT node: `success`, `accuracy=0.640413`,
    `fresh_export_used=false`, evidence:
    `results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json`
  - BGRL arXiv SBERT node: `success`, `accuracy=0.494969`,
    `fresh_export_used=false`, evidence:
    `results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json`
- PCBA now has two manifest-backed local surfaces under the same public target:
  - `local_debug`: `debug_success`, `ap=0.129023`, checkpoint
    `checkpoints/graphmae_pcba_native_debug.pt`, evidence:
    `results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json`
  - `full_local_non_debug`: `success`, `ap=0.034201`, dedicated non-debug checkpoint
    `checkpoints/graphmae_ogbg-molpcba.official_local.pt`, evidence:
    `results/baseline/layer2_suite_graphmae_pcba_native_graph_official_manifest.json`
  - The full-local non-debug surface proves non-debug local execution and separated
    checkpoint provenance, but it is still not a locked official result because
    `official_metric=false` and the surface is not an `official_candidate_*` row.
- WN18RR is included in `all_proven_local` with execution-backed evidence (via legacy group `wn18rr_experimental_compare`):
  - Structural alignment audit: `success`, `13/13 checks`, `num_entities=40943`,
    evidence: `results/baseline/wn18rr_alignment_audit.json`
  - Semantic alignment audit: `success`, `verdict=verified_by_provenance`, evidence:
    `results/baseline/wn18rr_semantic_alignment_audit.json`
  - Full-scale baseline link eval: `success`, `mrr=0.000159`,
    `relation_types_ignored=true`, evidence:
    `results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json`
  - Full-scale relation-aware link eval: `success`, `mrr=0.000515`,
    `hits@10=0.000479`, evidence:
    `results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json`
  - The comparison and audit artifacts have cleared all technical blockers
    (semantic-alignment, negative-sampling-contract, official-metric).
    WN18RR is now included in `all_proven_local`. Baseline dot-product path
    retains `relation_types_ignored=true`; relation-aware path clears all blockers.

## Evidence Priority

- Highest trust: target-specific suite manifests, comparison JSON, and audit JSON in
  `results/baseline/`
- Derived summaries: `outputs/summary.json`, `outputs/meeting_table.md`,
  `notes/meeting_progress_arxiv_sbert.md`,
  `notes/pcba_graph_protocol_report.md`, `notes/wn18rr_link_protocol_report.md`
- Review-only older handoff snapshots: `notes/NEW_CHAT_CONTEXT.md`,
  `notes/claude/*.md`, `work/layer2_update/**`, `work/layer2/artifacts/**`

## Canonical Narrative Docs

- `README.md`
- `MANIFEST.md`
- `notes/meeting_progress_arxiv_sbert.md`
- `notes/pcba_graph_protocol_report.md`
- `notes/wn18rr_link_protocol_report.md`
