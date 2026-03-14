#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from layer2_artifact_utils import (
    ACCEPTABLE_EXECUTION_STATUSES,
    PROTECTED_OFFICIAL_ENTRY_KEYS,
    PROJECT_ROOT,
    StructuredError,
    apply_text_write,
    classify_bundle_destination_policy,
    extract_manifest_spec,
    infer_manifest_run_type,
    json_dumps_stable,
    load_json_required,
    parse_note_fields,
    print_json,
    relpath_str,
    render_text_write_plan,
    require_mapping,
    sha256_bytes,
    summary_entry_evidence_quality,
)


SUMMARY_SOURCE = "outputs/summary.json"
MEETING_TABLE_SOURCE = "outputs/meeting_table.md"
README_SOURCE = "README.md"
MANIFEST_SOURCE = "MANIFEST.md"
PROGRESS_NOTE_SOURCE = "notes/meeting_progress_arxiv_sbert.md"
PCBA_REPORT_SOURCE = "notes/pcba_graph_protocol_report.md"
WN18RR_REPORT_SOURCE = "notes/wn18rr_link_protocol_report.md"
LAYER2_HPC_RUNBOOK_SOURCE = "notes/layer2_hpc_runbook.md"

ARXIV_OFFICIAL_MANIFEST_SOURCE = (
    "results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json"
)
PCBA_DEBUG_MANIFEST_SOURCE = (
    "results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json"
)
PCBA_OFFICIAL_MANIFEST_SOURCE = (
    "results/baseline/layer2_suite_graphmae_pcba_native_graph_official_manifest.json"
)
PCBA_COMPARISON_SOURCE = "results/baseline/pcba_graph_comparison.json"
WN18RR_ALIGNMENT_AUDIT_SOURCE = "results/baseline/wn18rr_alignment_audit.json"
WN18RR_SEMANTIC_AUDIT_SOURCE = "results/baseline/wn18rr_semantic_alignment_audit.json"
WN18RR_DEBUG_MANIFEST_SOURCE = (
    "results/baseline/layer2_suite_wn18rr_experimental_compare_debug_manifest.json"
)
WN18RR_OFFICIAL_MANIFEST_SOURCE = (
    "results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json"
)
WN18RR_COMPARISON_SOURCE = "results/baseline/wn18rr_link_comparison.json"
LAYER2_HPC_PLAN_SOURCE = "results/baseline/layer2_hpc_plan.json"
LAYER2_HPC_PLAN_UTILS_SOURCE = "scripts/layer2_hpc_plan_utils.py"
GENERATE_LAYER2_HPC_PLAN_SOURCE = "scripts/generate_layer2_hpc_plan.py"
INGEST_LAYER2_HPC_RESULTS_SOURCE = "scripts/ingest_layer2_hpc_results.py"

NON_CANONICAL_EVIDENCE_SOURCES = frozenset(
    {
        "results/baseline/layer2_suite_official_manifest.json",
        "results/baseline/layer2_suite_debug_manifest.json",
    }
)
REQUIRED_BUNDLE_SOURCES = frozenset(
    {
        README_SOURCE,
        MANIFEST_SOURCE,
        SUMMARY_SOURCE,
        MEETING_TABLE_SOURCE,
        PROGRESS_NOTE_SOURCE,
        PCBA_REPORT_SOURCE,
        WN18RR_REPORT_SOURCE,
        LAYER2_HPC_RUNBOOK_SOURCE,
        LAYER2_HPC_PLAN_UTILS_SOURCE,
        GENERATE_LAYER2_HPC_PLAN_SOURCE,
        INGEST_LAYER2_HPC_RESULTS_SOURCE,
        LAYER2_HPC_PLAN_SOURCE,
        ARXIV_OFFICIAL_MANIFEST_SOURCE,
        PCBA_DEBUG_MANIFEST_SOURCE,
        PCBA_OFFICIAL_MANIFEST_SOURCE,
        PCBA_COMPARISON_SOURCE,
        WN18RR_ALIGNMENT_AUDIT_SOURCE,
        WN18RR_SEMANTIC_AUDIT_SOURCE,
        WN18RR_DEBUG_MANIFEST_SOURCE,
        WN18RR_OFFICIAL_MANIFEST_SOURCE,
        WN18RR_COMPARISON_SOURCE,
    }
)
TITLE_REQUIRED_SOURCES = frozenset(
    {
        README_SOURCE,
        MANIFEST_SOURCE,
        MEETING_TABLE_SOURCE,
        PROGRESS_NOTE_SOURCE,
        PCBA_REPORT_SOURCE,
        WN18RR_REPORT_SOURCE,
        LAYER2_HPC_RUNBOOK_SOURCE,
    }
)
SUMMARY_REQUIRED_ENTRY_KEYS = frozenset(
    {
        "graphmae_pcba_native_full_local_non_debug",
        "wn18rr_fullscale_experimental_link_eval",
        "wn18rr_relaware_fullscale_experimental_link_eval",
    }
)
WN18RR_EXPERIMENTAL_POLICY_SOURCES = frozenset(
    {
        WN18RR_ALIGNMENT_AUDIT_SOURCE,
        WN18RR_SEMANTIC_AUDIT_SOURCE,
        WN18RR_DEBUG_MANIFEST_SOURCE,
        WN18RR_OFFICIAL_MANIFEST_SOURCE,
        WN18RR_COMPARISON_SOURCE,
        WN18RR_REPORT_SOURCE,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync Layer 2 local artifacts and code snapshots into the published "
            "`work/layer2/` bundle using MANIFEST.md as the path authority."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / MANIFEST_SOURCE,
        help="Path to the MANIFEST.md file that defines local-to-bundle mappings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the MANIFEST-based copy/removal plan without writing bundle files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest_text = args.manifest.read_text(encoding="utf-8")
    except OSError as exc:
        print_json(
            StructuredError(
                "missing_manifest",
                "Failed to read MANIFEST.md.",
                {
                    "path": relpath_str(args.manifest),
                    "reason": f"{type(exc).__name__}: {exc}",
                },
            ).to_payload()
        )
        return 1

    try:
        spec = extract_manifest_spec(manifest_text, manifest_path=args.manifest)
        bundle_root = _resolve_bundle_root(args.manifest, spec)
        mappings = _normalize_manifest_mappings(args.manifest, bundle_root, spec)
        preflight = _run_preflight_validation(mappings)
        if preflight["blocking_issue_count"] > 0:
            if args.dry_run:
                print_json({
                    "dry_run": True,
                    "integrity_ok": False,
                    "manifest": relpath_str(args.manifest),
                    "bundle_root": relpath_str(bundle_root),
                    "warning_count": preflight["warning_count"],
                    "blocking_issue_count": preflight["blocking_issue_count"],
                    "blocking_issues": preflight["blocking_issues"],
                    "preflight": preflight,
                })
                return 1
            raise StructuredError(
                "bundle_preflight_failed",
                "Bundle preflight validation failed.",
                {
                    "manifest": relpath_str(args.manifest),
                    "bundle_root": relpath_str(bundle_root),
                    "preflight": preflight,
                },
            )

        plans = _build_copy_plans(mappings)
        bundle_integrity = _build_bundle_integrity_report(plans)
        evidence_freeze = _run_evidence_freeze_checks(plans)
        validation = _combine_validation_reports(preflight, bundle_integrity, evidence_freeze)
        stale_files = _build_stale_file_plans(bundle_root, plans)

        if validation["blocking_issue_count"] > 0 and not args.dry_run:
            raise StructuredError(
                "bundle_sync_blocked",
                "Bundle sync validation failed.",
                {
                    "manifest": relpath_str(args.manifest),
                    "bundle_root": relpath_str(bundle_root),
                    "validation": validation,
                    "preflight": preflight,
                    "bundle_integrity": bundle_integrity,
                    "evidence_freeze": evidence_freeze,
                    "copies": _serialize_plans(plans),
                    "stale_files": _serialize_stale_files(stale_files, dry_run=args.dry_run),
                },
            )

        if not args.dry_run:
            _apply_copy_plans(plans)
            _apply_stale_file_plans(stale_files, bundle_root=bundle_root)

        snapshot = _verify_snapshot(bundle_root, plans)
        if not args.dry_run and not snapshot["exact_match"]:
            raise StructuredError(
                "bundle_snapshot_mismatch",
                "Published bundle does not match the MANIFEST source-of-truth after sync.",
                {
                    "manifest": relpath_str(args.manifest),
                    "bundle_root": relpath_str(bundle_root),
                    "snapshot": snapshot,
                    "copies": _serialize_plans(plans),
                    "stale_files": _serialize_stale_files(stale_files, dry_run=False),
                },
            )

        report_payload = _build_report_payload(
            dry_run=args.dry_run,
            manifest_path=args.manifest,
            bundle_root=bundle_root,
            preflight=preflight,
            bundle_integrity=bundle_integrity,
            evidence_freeze=evidence_freeze,
            validation=validation,
            plans=plans,
            stale_files=stale_files,
            snapshot=snapshot,
        )
        report_payload["integrity_ok"] = validation["blocking_issue_count"] == 0
        report_payload["integrity_reports"] = _write_integrity_sidecars(
            bundle_root=bundle_root,
            report_payload=report_payload,
            dry_run=args.dry_run,
        )
        print_json(report_payload)
        if args.dry_run and not report_payload["integrity_ok"]:
            return 1
        return 0
    except StructuredError as exc:
        print_json(exc.to_payload())
        return 1


def _resolve_bundle_root(manifest_path: Path, spec: dict[str, Any]) -> Path:
    bundle_root = spec.get("bundle_root")
    if not isinstance(bundle_root, str) or not bundle_root:
        raise StructuredError(
            "invalid_manifest_shape",
            "The layer2-bundle-manifest JSON block must define a non-empty string `bundle_root`.",
            {
                "path": relpath_str(manifest_path),
            },
        )
    return (manifest_path.parent / bundle_root).resolve()


def _normalize_manifest_mappings(
    manifest_path: Path,
    bundle_root: Path,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    mappings = spec.get("mappings")
    if not isinstance(mappings, list):
        raise StructuredError(
            "invalid_manifest_shape",
            "The layer2-bundle-manifest JSON block must define `mappings` as a list.",
            {
                "path": relpath_str(manifest_path),
                "actual_type": type(mappings).__name__,
            },
        )

    normalized: list[dict[str, Any]] = []
    for index, mapping in enumerate(mappings):
        if not isinstance(mapping, dict):
            raise StructuredError(
                "invalid_manifest_shape",
                "Each MANIFEST mapping must be a JSON object.",
                {
                    "path": relpath_str(manifest_path),
                    "mapping_index": index,
                    "actual_type": type(mapping).__name__,
                },
            )
        source_rel = mapping.get("source")
        destination_rel = mapping.get("destination")
        kind = mapping.get("kind")
        if not isinstance(source_rel, str) or not source_rel:
            raise StructuredError(
                "invalid_manifest_shape",
                "Each MANIFEST mapping must define a non-empty string `source`.",
                {
                    "path": relpath_str(manifest_path),
                    "mapping_index": index,
                },
            )
        if not isinstance(destination_rel, str) or not destination_rel:
            raise StructuredError(
                "invalid_manifest_shape",
                "Each MANIFEST mapping must define a non-empty string `destination`.",
                {
                    "path": relpath_str(manifest_path),
                    "mapping_index": index,
                },
            )
        if not isinstance(kind, str) or not kind:
            raise StructuredError(
                "invalid_manifest_shape",
                "Each MANIFEST mapping must define a non-empty string `kind`.",
                {
                    "path": relpath_str(manifest_path),
                    "mapping_index": index,
                },
            )

        source_abs = (manifest_path.parent / source_rel).resolve()
        destination_abs = (bundle_root / destination_rel).resolve()
        try:
            destination_abs.relative_to(bundle_root)
        except ValueError as exc:
            raise StructuredError(
                "invalid_destination",
                "MANIFEST destination escapes the bundle root.",
                {
                    "path": relpath_str(manifest_path),
                    "mapping_index": index,
                    "destination": destination_rel,
                    "bundle_root": relpath_str(bundle_root),
                },
            ) from exc

        normalized.append(
            {
                "mapping_index": index,
                "kind": kind,
                "source": source_rel,
                "destination": destination_rel,
                "source_abs": source_abs,
                "destination_abs": destination_abs,
            }
        )

    return normalized


def _run_preflight_validation(mappings: list[dict[str, Any]]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []

    source_rels = [mapping["source"] for mapping in mappings]
    source_rel_set = set(source_rels)
    destination_rels = [mapping["destination"] for mapping in mappings]

    missing_required_sources = sorted(REQUIRED_BUNDLE_SOURCES - source_rel_set)
    if missing_required_sources:
        blocking_issues.append(
            {
                "kind": "required_sources_missing_from_manifest",
                "sources": missing_required_sources,
            }
        )

    duplicate_sources = _find_duplicates(source_rels)
    if duplicate_sources:
        blocking_issues.append(
            {
                "kind": "duplicate_manifest_sources",
                "sources": duplicate_sources,
            }
        )

    duplicate_destinations = _find_duplicates(destination_rels)
    if duplicate_destinations:
        blocking_issues.append(
            {
                "kind": "duplicate_manifest_destinations",
                "destinations": duplicate_destinations,
            }
        )

    non_canonical_sources = sorted(
        mapping["source"] for mapping in mappings if mapping["source"] in NON_CANONICAL_EVIDENCE_SOURCES
    )
    if non_canonical_sources:
        blocking_issues.append(
            {
                "kind": "non_canonical_evidence_sources_listed",
                "sources": non_canonical_sources,
            }
        )

    missing_sources: list[dict[str, Any]] = []
    for mapping in mappings:
        if not mapping["source_abs"].exists():
            missing_sources.append(
                {
                    "source": mapping["source"],
                    "kind": mapping["kind"],
                    "destination": mapping["destination"],
                }
            )
    if missing_sources:
        blocking_issues.append(
            {
                "kind": "missing_manifest_sources",
                "sources": missing_sources,
            }
        )
        return {
            "source_count": len(mappings),
            "required_source_count": len(REQUIRED_BUNDLE_SOURCES),
            "missing_source_count": len(missing_sources),
            "warnings": warnings,
            "warning_count": len(warnings),
            "blocking_issues": blocking_issues,
            "blocking_issue_count": len(blocking_issues),
        }

    for mapping in mappings:
        file_report = _validate_manifest_source_file(mapping, source_rel_set)
        warnings.extend(file_report["warnings"])
        blocking_issues.extend(file_report["blocking_issues"])

    return {
        "source_count": len(mappings),
        "required_source_count": len(REQUIRED_BUNDLE_SOURCES),
        "missing_source_count": 0,
        "warnings": warnings,
        "warning_count": len(warnings),
        "blocking_issues": blocking_issues,
        "blocking_issue_count": len(blocking_issues),
    }


def _validate_manifest_source_file(
    mapping: dict[str, Any],
    source_rel_set: set[str],
) -> dict[str, list[dict[str, Any]]]:
    source_rel = mapping["source"]
    source_abs = mapping["source_abs"]
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []

    if source_abs.suffix.lower() == ".json":
        try:
            payload = load_json_required(source_abs, label=f"manifest_source:{source_rel}")
            mapping_payload = require_mapping(
                payload,
                label=f"manifest_source:{source_rel}",
                path=source_abs,
            )
            json_report = _validate_json_source(
                source_rel,
                source_abs,
                mapping_payload,
                source_rel_set=source_rel_set,
            )
            warnings.extend(json_report["warnings"])
            blocking_issues.extend(json_report["blocking_issues"])
        except StructuredError as exc:
            blocking_issues.append(
                {
                    "kind": "invalid_json_source",
                    "source": source_rel,
                    "details": exc.to_payload()["error"],
                }
            )
        return {
            "warnings": warnings,
            "blocking_issues": blocking_issues,
        }

    try:
        text = source_abs.read_text(encoding="utf-8")
    except OSError as exc:
        blocking_issues.append(
            {
                "kind": "read_error",
                "source": source_rel,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return {
            "warnings": warnings,
            "blocking_issues": blocking_issues,
        }

    if not text.strip():
        blocking_issues.append(
            {
                "kind": "empty_text_source",
                "source": source_rel,
            }
        )
        return {
            "warnings": warnings,
            "blocking_issues": blocking_issues,
        }

    if source_rel in TITLE_REQUIRED_SOURCES and not text.lstrip().startswith("#"):
        blocking_issues.append(
            {
                "kind": "missing_markdown_title",
                "source": source_rel,
            }
        )

    if source_rel == MEETING_TABLE_SOURCE and "| Group | Entry | Status | Metric | Evidence | Caveats |" not in text:
        blocking_issues.append(
            {
                "kind": "meeting_table_missing_header",
                "source": source_rel,
            }
        )

    return {
        "warnings": warnings,
        "blocking_issues": blocking_issues,
    }


def _validate_json_source(
    source_rel: str,
    source_abs: Path,
    payload: dict[str, Any],
    *,
    source_rel_set: set[str],
) -> dict[str, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []

    if source_rel == SUMMARY_SOURCE:
        entries = payload.get("entries")
        generated_artifacts = payload.get("generated_artifacts")
        if not isinstance(entries, list):
            blocking_issues.append(
                {
                    "kind": "summary_entries_missing",
                    "source": source_rel,
                    "actual_type": type(entries).__name__,
                }
            )
            return {"warnings": warnings, "blocking_issues": blocking_issues}
        if not isinstance(generated_artifacts, dict):
            blocking_issues.append(
                {
                    "kind": "summary_generated_artifacts_missing",
                    "source": source_rel,
                    "actual_type": type(generated_artifacts).__name__,
                }
            )
        summary_entries = _summary_entries_by_key(payload)
        missing_summary_keys = sorted(SUMMARY_REQUIRED_ENTRY_KEYS - set(summary_entries))
        if missing_summary_keys:
            blocking_issues.append(
                {
                    "kind": "summary_required_entries_missing",
                    "source": source_rel,
                    "entry_keys": missing_summary_keys,
                }
            )
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            evidence = entry.get("evidence")
            if not isinstance(evidence, dict):
                continue
            evidence_path = evidence.get("source_path")
            if not isinstance(evidence_path, str) or not evidence_path:
                continue
            if not (PROJECT_ROOT / evidence_path).exists():
                blocking_issues.append(
                    {
                        "kind": "summary_evidence_missing",
                        "source": source_rel,
                        "entry_key": entry.get("key"),
                        "evidence_path": evidence_path,
                    }
                )
                continue
            if evidence_path in NON_CANONICAL_EVIDENCE_SOURCES:
                warnings.append(
                    {
                        "kind": "summary_uses_rolling_alias",
                        "source": source_rel,
                        "entry_key": entry.get("key"),
                        "evidence_path": evidence_path,
                    }
                )
            if evidence.get("source_kind") in {"suite_manifest", "audit_json"} and evidence_path not in source_rel_set:
                warnings.append(
                    {
                        "kind": "summary_evidence_outside_bundle_manifest",
                        "source": source_rel,
                        "entry_key": entry.get("key"),
                        "evidence_path": evidence_path,
                    }
                )
        for artifact_name, artifact_path in sorted(generated_artifacts.items()):
            if not isinstance(artifact_path, str):
                blocking_issues.append(
                    {
                        "kind": "summary_generated_artifact_invalid",
                        "source": source_rel,
                        "artifact_name": artifact_name,
                        "actual_type": type(artifact_path).__name__,
                    }
                )
                continue
            if not (PROJECT_ROOT / artifact_path).exists():
                blocking_issues.append(
                    {
                        "kind": "summary_generated_artifact_missing",
                        "source": source_rel,
                        "artifact_name": artifact_name,
                        "artifact_path": artifact_path,
                    }
                )
        return {"warnings": warnings, "blocking_issues": blocking_issues}

    if source_abs.name.startswith("layer2_suite_") and source_abs.name.endswith("_manifest.json"):
        targets = payload.get("targets")
        if not isinstance(payload.get("suite_name"), str):
            blocking_issues.append(
                {
                    "kind": "suite_manifest_missing_suite_name",
                    "source": source_rel,
                }
            )
        if not isinstance(targets, list):
            blocking_issues.append(
                {
                    "kind": "suite_manifest_missing_targets",
                    "source": source_rel,
                    "actual_type": type(targets).__name__,
                }
            )
        if infer_manifest_run_type(payload) not in {"execution", "preview"}:
            blocking_issues.append(
                {
                    "kind": "suite_manifest_invalid_run_type",
                    "source": source_rel,
                    "run_type": payload.get("run_type"),
                }
            )
        return {"warnings": warnings, "blocking_issues": blocking_issues}

    if source_rel == PCBA_COMPARISON_SOURCE:
        if _nested_get(payload, "comparison_profile", "name") != "pcba_graph_compare":
            blocking_issues.append(
                {
                    "kind": "pcba_comparison_profile_mismatch",
                    "source": source_rel,
                    "actual": _nested_get(payload, "comparison_profile", "name"),
                }
            )
        profiles = payload.get("profiles")
        if not isinstance(profiles, dict) or {"local_debug", "full_local_non_debug"} - set(profiles):
            blocking_issues.append(
                {
                    "kind": "pcba_comparison_profiles_missing",
                    "source": source_rel,
                }
            )
        return {"warnings": warnings, "blocking_issues": blocking_issues}

    if source_rel == WN18RR_ALIGNMENT_AUDIT_SOURCE:
        checks = payload.get("checks")
        if payload.get("dataset") != "wn18rr" or payload.get("status") != "success":
            blocking_issues.append(
                {
                    "kind": "wn18rr_alignment_audit_invalid",
                    "source": source_rel,
                    "dataset": payload.get("dataset"),
                    "status": payload.get("status"),
                }
            )
        if not isinstance(checks, dict):
            blocking_issues.append(
                {
                    "kind": "wn18rr_alignment_audit_missing_checks",
                    "source": source_rel,
                    "actual_type": type(checks).__name__,
                }
            )
        return {"warnings": warnings, "blocking_issues": blocking_issues}

    if source_rel == WN18RR_SEMANTIC_AUDIT_SOURCE:
        checks = payload.get("checks")
        if payload.get("dataset") != "wn18rr" or payload.get("status") != "success":
            blocking_issues.append(
                {
                    "kind": "wn18rr_semantic_audit_invalid",
                    "source": source_rel,
                    "dataset": payload.get("dataset"),
                    "status": payload.get("status"),
                }
            )
        if payload.get("semantic_alignment_verified") is not True:
            blocking_issues.append(
                {
                    "kind": "wn18rr_semantic_alignment_not_verified",
                    "source": source_rel,
                    "semantic_alignment_verified": payload.get("semantic_alignment_verified"),
                }
            )
        if not isinstance(checks, dict):
            blocking_issues.append(
                {
                    "kind": "wn18rr_semantic_audit_missing_checks",
                    "source": source_rel,
                    "actual_type": type(checks).__name__,
                }
            )
        return {"warnings": warnings, "blocking_issues": blocking_issues}

    if source_rel == WN18RR_COMPARISON_SOURCE:
        if _nested_get(payload, "comparison_profile", "name") != "wn18rr_experimental_compare":
            blocking_issues.append(
                {
                    "kind": "wn18rr_comparison_profile_mismatch",
                    "source": source_rel,
                    "actual": _nested_get(payload, "comparison_profile", "name"),
                }
            )
        for key in (
            "alignment_evidence",
            "negative_sampling_assessment",
            "official_metric_assessment",
            "comparison",
        ):
            if not isinstance(payload.get(key), dict):
                blocking_issues.append(
                    {
                        "kind": "wn18rr_comparison_missing_section",
                        "source": source_rel,
                        "section": key,
                        "actual_type": type(payload.get(key)).__name__,
                    }
                )
        return {"warnings": warnings, "blocking_issues": blocking_issues}

    return {"warnings": warnings, "blocking_issues": blocking_issues}


def _build_copy_plans(mappings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for mapping in mappings:
        source_abs = mapping["source_abs"]
        destination_abs = mapping["destination_abs"]
        source_bytes = source_abs.read_bytes()
        destination_bytes = destination_abs.read_bytes() if destination_abs.exists() else None
        changed = destination_bytes != source_bytes
        action = "unchanged"
        if not destination_abs.exists():
            action = "create"
        elif changed:
            action = "update"
        plans.append(
            {
                **mapping,
                "action": action,
                "changed": changed,
                "size_bytes": len(source_bytes),
                "bytes_copied": len(source_bytes) if changed else 0,
                "sha256": sha256_bytes(source_bytes),
                "source_bytes": source_bytes,
            }
        )
    return plans


def _build_bundle_integrity_report(plans: list[dict[str, Any]]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []

    summary_plan = _find_plan(plans, source_rel=SUMMARY_SOURCE)
    if summary_plan is not None:
        report = _check_summary_integrity(summary_plan)
        warnings.extend(report["warnings"])
        blocking_issues.extend(report["blocking_issues"])

    for plan in plans:
        if plan["kind"] != "evidence":
            continue
        if not plan["source_abs"].name.startswith("layer2_suite_"):
            continue
        report = _check_manifest_integrity(plan)
        warnings.extend(report["warnings"])
        blocking_issues.extend(report["blocking_issues"])

    return {
        "warnings": warnings,
        "warning_count": len(warnings),
        "blocking_issues": blocking_issues,
        "blocking_issue_count": len(blocking_issues),
    }


def _run_evidence_freeze_checks(plans: list[dict[str, Any]]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    summary_payload = _load_json_mapping(
        _require_plan(plans, source_rel=SUMMARY_SOURCE)["source_abs"],
        label="bundle_source_summary",
    )
    summary_entries = _summary_entries_by_key(summary_payload)

    arxiv_plan = _require_plan(plans, source_rel=ARXIV_OFFICIAL_MANIFEST_SOURCE)
    arxiv_manifest = _load_json_mapping(
        arxiv_plan["source_abs"],
        label=f"bundle_source_manifest:{ARXIV_OFFICIAL_MANIFEST_SOURCE}",
    )
    arxiv_targets = _targets_by_name(arxiv_manifest)
    arxiv_ok = (
        infer_manifest_run_type(arxiv_manifest) == "execution"
        and arxiv_manifest.get("requested_target") == "official_candidate_arxiv"
        and set(arxiv_targets) == {"graphmae_arxiv_sbert_node", "bgrl_arxiv_sbert_node"}
    )
    arxiv_entry_details: list[dict[str, Any]] = []
    for entry_key in sorted(PROTECTED_OFFICIAL_ENTRY_KEYS):
        entry = summary_entries.get(entry_key)
        quality = summary_entry_evidence_quality(entry) if entry is not None else None
        entry_ok = (
            entry is not None
            and quality is not None
            and quality["acceptable_execution_evidence"]
        )
        arxiv_ok = arxiv_ok and entry_ok
        arxiv_entry_details.append(
            {
                "entry_key": entry_key,
                "present": entry is not None,
                "status": entry.get("status") if isinstance(entry, dict) else None,
                "run_type": quality["run_type"] if quality is not None else None,
                "evidence_path": _nested_get(entry, "evidence", "source_path"),
                "acceptable_execution_evidence": (
                    quality["acceptable_execution_evidence"] if quality is not None else False
                ),
            }
        )
    checks.append(
        {
            "kind": "arxiv_official_execution_backed",
            "status": "pass" if arxiv_ok else "fail",
            "source_path": ARXIV_OFFICIAL_MANIFEST_SOURCE,
            "run_type": infer_manifest_run_type(arxiv_manifest),
            "overall_status": arxiv_manifest.get("overall_status"),
            "requested_target": arxiv_manifest.get("requested_target"),
            "entries": arxiv_entry_details,
        }
    )
    if not arxiv_ok:
        blocking_issues.append(
            {
                "kind": "arxiv_official_manifest_not_execution_backed",
                "source_path": ARXIV_OFFICIAL_MANIFEST_SOURCE,
                "run_type": infer_manifest_run_type(arxiv_manifest),
                "overall_status": arxiv_manifest.get("overall_status"),
                "requested_target": arxiv_manifest.get("requested_target"),
                "entries": arxiv_entry_details,
            }
        )

    pcba_debug_manifest = _load_json_mapping(
        _require_plan(plans, source_rel=PCBA_DEBUG_MANIFEST_SOURCE)["source_abs"],
        label=f"bundle_source_manifest:{PCBA_DEBUG_MANIFEST_SOURCE}",
    )
    pcba_official_manifest = _load_json_mapping(
        _require_plan(plans, source_rel=PCBA_OFFICIAL_MANIFEST_SOURCE)["source_abs"],
        label=f"bundle_source_manifest:{PCBA_OFFICIAL_MANIFEST_SOURCE}",
    )
    pcba_comparison = _load_json_mapping(
        _require_plan(plans, source_rel=PCBA_COMPARISON_SOURCE)["source_abs"],
        label=f"bundle_source_artifact:{PCBA_COMPARISON_SOURCE}",
    )
    pcba_debug_target = _targets_by_name(pcba_debug_manifest).get("graphmae_pcba_native_graph", {})
    pcba_official_target = _targets_by_name(pcba_official_manifest).get("graphmae_pcba_native_graph", {})
    pcba_official_notes = parse_note_fields(str(pcba_official_target.get("notes") or ""))
    pcba_ok = (
        infer_manifest_run_type(pcba_official_manifest) == "execution"
        and pcba_official_target.get("profile_name") == "full_local_non_debug"
        and pcba_official_target.get("checkpoint_path")
        == "checkpoints/graphmae_ogbg-molpcba.official_local.pt"
        and pcba_official_target.get("checkpoint_path") != pcba_debug_target.get("checkpoint_path")
        and _nested_get(
            pcba_comparison,
            "checkpoint_provenance",
            "full_local_non_debug_checkpoint_path",
        )
        == "checkpoints/graphmae_ogbg-molpcba.official_local.pt"
        and _nested_get(pcba_comparison, "checkpoint_provenance", "checkpoint_paths_distinct") is True
        and _nested_get(
            pcba_comparison,
            "checkpoint_provenance",
            "debug_checkpoint_surface_removed_for_full_local",
        )
        is True
        and bool(
            _nested_get(
                pcba_comparison,
                "profiles",
                "full_local_non_debug",
                "evidence",
                "source_path",
            )
        )
    )
    checks.append(
        {
            "kind": "pcba_full_local_non_debug_checkpoint_frozen",
            "status": "pass" if pcba_ok else "fail",
            "source_path": PCBA_OFFICIAL_MANIFEST_SOURCE,
            "checkpoint_path": pcba_official_target.get("checkpoint_path"),
            "debug_checkpoint_path": pcba_debug_target.get("checkpoint_path"),
            "profile_name": pcba_official_target.get("profile_name"),
            "note_fields": {
                "dedicated_non_debug_checkpoint": pcba_official_notes.get(
                    "dedicated_non_debug_checkpoint"
                ),
                "debug_mode": pcba_official_notes.get("debug_mode"),
            },
            "comparison_evidence_path": _nested_get(
                pcba_comparison,
                "profiles",
                "full_local_non_debug",
                "evidence",
                "source_path",
            ),
        }
    )
    if not pcba_ok:
        blocking_issues.append(
            {
                "kind": "pcba_official_manifest_not_frozen_to_dedicated_checkpoint",
                "source_path": PCBA_OFFICIAL_MANIFEST_SOURCE,
                "checkpoint_path": pcba_official_target.get("checkpoint_path"),
                "debug_checkpoint_path": pcba_debug_target.get("checkpoint_path"),
                "profile_name": pcba_official_target.get("profile_name"),
                "note_fields": pcba_official_notes,
            }
        )

    wn18rr_official_manifest = _load_json_mapping(
        _require_plan(plans, source_rel=WN18RR_OFFICIAL_MANIFEST_SOURCE)["source_abs"],
        label=f"bundle_source_manifest:{WN18RR_OFFICIAL_MANIFEST_SOURCE}",
    )
    wn18rr_comparison = _load_json_mapping(
        _require_plan(plans, source_rel=WN18RR_COMPARISON_SOURCE)["source_abs"],
        label=f"bundle_source_artifact:{WN18RR_COMPARISON_SOURCE}",
    )
    wn18rr_semantic_audit = _load_json_mapping(
        _require_plan(plans, source_rel=WN18RR_SEMANTIC_AUDIT_SOURCE)["source_abs"],
        label=f"bundle_source_artifact:{WN18RR_SEMANTIC_AUDIT_SOURCE}",
    )
    wn18rr_report_text = _read_text_required(
        _require_plan(plans, source_rel=WN18RR_REPORT_SOURCE)["source_abs"],
        label=WN18RR_REPORT_SOURCE,
    )
    progress_note_text = _read_text_required(
        _require_plan(plans, source_rel=PROGRESS_NOTE_SOURCE)["source_abs"],
        label=PROGRESS_NOTE_SOURCE,
    )
    wn18rr_targets = list(_targets_by_name(wn18rr_official_manifest).values())
    still_experimental_reasons = _nested_get(
        wn18rr_comparison,
        "comparison",
        "still_experimental_reasons",
    )
    wn18rr_destination_policy = _evaluate_wn18rr_destination_policy(plans)
    wn18rr_freeze_requirements = {
        "semantic_audit_success": wn18rr_semantic_audit.get("status") == "success",
        "semantic_audit_verified": wn18rr_semantic_audit.get("semantic_alignment_verified") is True,
        "comparison_semantic_alignment_verified": _nested_get(
            wn18rr_comparison,
            "alignment_evidence",
            "semantic_alignment_verified",
        )
        is True,
        "negative_sampling_contract_defined": _nested_get(
            wn18rr_comparison,
            "negative_sampling_assessment",
            "contract_defined",
        )
        is True,
        "negative_sampling_blocker_cleared": _nested_get(
            wn18rr_comparison,
            "negative_sampling_assessment",
            "blocker_cleared",
        )
        is True,
        "official_metric_protocol_matches_benchmark": _nested_get(
            wn18rr_comparison,
            "official_metric_assessment",
            "metric_protocol_matches_benchmark",
        )
        is True,
        "official_metric_full_scale_eval_completed": _nested_get(
            wn18rr_comparison,
            "official_metric_assessment",
            "full_scale_eval_completed",
        )
        is True,
        "official_metric_blocker_cleared": _nested_get(
            wn18rr_comparison,
            "official_metric_assessment",
            "blocker_retained",
        )
        is False,
        "remaining_reasons_cleared": (
            len(still_experimental_reasons) == 0
        ),
        "manifest_execution_backed": infer_manifest_run_type(wn18rr_official_manifest) == "execution",
        "manifest_success": wn18rr_official_manifest.get("overall_status") == "success",
        "manifest_targets_present": bool(wn18rr_targets),
        "targets_have_valid_artifact_group": all(
            target.get("artifact_group") in {"local_debug", "experimental"}
            for target in wn18rr_targets
        ),
        "report_records_semantic_alignment": "semantic_alignment_verified=true" in wn18rr_report_text,
        "report_records_negative_sampling_clearance": (
            "contract_defined=true; blocker_cleared=true." in wn18rr_report_text
        ),
        "report_records_metric_clearance": "blocker_retained=false." in wn18rr_report_text,
        "report_records_promotion_status": (
            "WN18RR Promotion Status" in wn18rr_report_text
            or "Remaining reasons: experimental_fence_still_enabled." in wn18rr_report_text
        ),
        "progress_note_records_semantic_alignment": (
            "Semantic alignment: verdict=verified_by_provenance; verified=true."
            in progress_note_text
        ),
        "progress_note_records_negative_sampling_clearance": (
            "Negative-sampling contract: defined=true; blocker_cleared=true."
            in progress_note_text
        ),
        "progress_note_records_metric_clearance": (
            "Official metric: full_scale_eval_completed=true; blocker_retained=false."
            in progress_note_text
        ),
        "progress_note_records_wn18rr_status": (
            "included in `all_proven_local`" in progress_note_text
            or "remaining fence=experimental_fence_still_enabled." in progress_note_text
        ),
    }
    failed_wn18rr_freeze_requirements = sorted(
        name for name, satisfied in wn18rr_freeze_requirements.items() if not satisfied
    )
    if failed_wn18rr_freeze_requirements:
        wn18rr_policy_decision = "true_blocking_issue"
        wn18rr_status = "fail"
    elif wn18rr_destination_policy["status"] != "pass":
        wn18rr_policy_decision = "promotion_violation"
        wn18rr_status = "fail"
    else:
        wn18rr_policy_decision = "experimental_evidence_accepted"
        wn18rr_status = "pass"
    checks.append(
        {
            "kind": "wn18rr_experimental_evidence_frozen",
            "status": wn18rr_status,
            "dataset": "wn18rr",
            "policy_decision": wn18rr_policy_decision,
            "comparison_path": WN18RR_COMPARISON_SOURCE,
            "semantic_audit_path": WN18RR_SEMANTIC_AUDIT_SOURCE,
            "report_path": WN18RR_REPORT_SOURCE,
            "semantic_alignment_verified": _nested_get(
                wn18rr_comparison,
                "alignment_evidence",
                "semantic_alignment_verified",
            ),
            "negative_sampling_blocker_cleared": _nested_get(
                wn18rr_comparison,
                "negative_sampling_assessment",
                "blocker_cleared",
            ),
            "official_metric_blocker_retained": _nested_get(
                wn18rr_comparison,
                "official_metric_assessment",
                "blocker_retained",
            ),
            "still_experimental_reasons": still_experimental_reasons,
            "target_groups": [target.get("target_groups") for target in wn18rr_targets],
            "failed_freeze_requirements": failed_wn18rr_freeze_requirements,
            "freeze_requirements": wn18rr_freeze_requirements,
            "destination_policy": wn18rr_destination_policy,
        }
    )
    if wn18rr_policy_decision == "promotion_violation":
        blocking_issues.append(
            {
                "kind": "wn18rr_experimental_publication_policy_violation",
                "comparison_path": WN18RR_COMPARISON_SOURCE,
                "semantic_audit_path": WN18RR_SEMANTIC_AUDIT_SOURCE,
                "policy_decision": wn18rr_policy_decision,
                "still_experimental_reasons": still_experimental_reasons,
                "destination_policy": wn18rr_destination_policy,
            }
        )
    elif wn18rr_policy_decision == "true_blocking_issue":
        blocking_issues.append(
            {
                "kind": "wn18rr_evidence_freeze_failed",
                "comparison_path": WN18RR_COMPARISON_SOURCE,
                "semantic_audit_path": WN18RR_SEMANTIC_AUDIT_SOURCE,
                "policy_decision": wn18rr_policy_decision,
                "semantic_alignment_verified": _nested_get(
                    wn18rr_comparison,
                    "alignment_evidence",
                    "semantic_alignment_verified",
                ),
                "negative_sampling_blocker_cleared": _nested_get(
                    wn18rr_comparison,
                    "negative_sampling_assessment",
                    "blocker_cleared",
                ),
                "official_metric_blocker_retained": _nested_get(
                    wn18rr_comparison,
                    "official_metric_assessment",
                    "blocker_retained",
                ),
                "still_experimental_reasons": still_experimental_reasons,
                "failed_freeze_requirements": failed_wn18rr_freeze_requirements,
            }
        )

    return {
        "checks": checks,
        "policy_summary": _build_policy_summary(checks),
        "warnings": warnings,
        "warning_count": len(warnings),
        "blocking_issues": blocking_issues,
        "blocking_issue_count": len(blocking_issues),
    }


def _evaluate_wn18rr_destination_policy(plans: list[dict[str, Any]]) -> dict[str, Any]:
    destinations: list[dict[str, Any]] = []
    for plan in plans:
        if plan["source"] not in WN18RR_EXPERIMENTAL_POLICY_SOURCES:
            continue
        policy = classify_bundle_destination_policy(plan["destination"])
        destinations.append(
            {
                "source": plan["source"],
                "destination": plan["destination"],
                "kind": plan["kind"],
                **policy,
            }
        )

    validated_sources = {item["source"] for item in destinations}
    missing_sources = sorted(WN18RR_EXPERIMENTAL_POLICY_SOURCES - validated_sources)
    experimental_only_destinations = [
        item for item in destinations if item["classification"] == "experimental_only"
    ]
    violations = [item for item in destinations if item["classification"] != "experimental_only"]
    promoted_surface_violations = [
        item for item in violations if item["classification"] == "promoted_surface"
    ]
    unclassified_surface_violations = [
        item for item in violations if item["classification"] == "unclassified_surface"
    ]

    return {
        "status": "pass" if not violations and not missing_sources else "fail",
        "validated_source_count": len(destinations),
        "missing_source_count": len(missing_sources),
        "experimental_only_destination_count": len(experimental_only_destinations),
        "violation_count": len(violations),
        "promoted_surface_violation_count": len(promoted_surface_violations),
        "unclassified_surface_violation_count": len(unclassified_surface_violations),
        "missing_sources": missing_sources,
        "experimental_only_destinations": experimental_only_destinations,
        "violations": violations,
    }


def _build_policy_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    promotion_violations: list[dict[str, Any]] = []
    true_blocking_issues: list[dict[str, Any]] = []
    for check in checks:
        policy_decision = check.get("policy_decision")
        if not isinstance(policy_decision, str):
            continue
        entry = {
            "dataset": check.get("dataset"),
            "kind": check.get("kind"),
            "status": check.get("status"),
        }
        if policy_decision == "experimental_evidence_accepted":
            accepted.append(entry)
        elif policy_decision == "promotion_violation":
            promotion_violations.append(entry)
        elif policy_decision == "true_blocking_issue":
            true_blocking_issues.append(entry)
    return {
        "experimental_evidence_accepted": accepted,
        "experimental_evidence_accepted_count": len(accepted),
        "promotion_violations": promotion_violations,
        "promotion_violation_count": len(promotion_violations),
        "true_blocking_issues": true_blocking_issues,
        "true_blocking_issue_count": len(true_blocking_issues),
    }


def _build_stale_file_plans(bundle_root: Path, plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_paths = {plan["destination_abs"].resolve() for plan in plans}
    stale_files: list[dict[str, Any]] = []
    if not bundle_root.exists():
        return stale_files
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in expected_paths:
            continue
        stale_files.append(
            {
                "path": relpath_str(resolved),
                "path_abs": resolved,
            }
        )
    return stale_files


def _apply_copy_plans(plans: list[dict[str, Any]]) -> None:
    for plan in plans:
        if not plan["changed"]:
            continue
        destination_abs = plan["destination_abs"]
        destination_abs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan["source_abs"], destination_abs)


def _apply_stale_file_plans(stale_files: list[dict[str, Any]], *, bundle_root: Path) -> None:
    for stale in stale_files:
        path_abs = stale["path_abs"]
        if path_abs.exists():
            path_abs.unlink()
        _prune_empty_parents(path_abs.parent, stop_at=bundle_root)


def _verify_snapshot(bundle_root: Path, plans: list[dict[str, Any]]) -> dict[str, Any]:
    expected_by_destination = {plan["destination_abs"].resolve(): plan for plan in plans}
    missing_destinations: list[dict[str, Any]] = []
    checksum_mismatches: list[dict[str, Any]] = []

    for destination_abs, plan in expected_by_destination.items():
        if not destination_abs.exists():
            missing_destinations.append(
                {
                    "destination": relpath_str(destination_abs),
                    "source": plan["source"],
                }
            )
            continue
        destination_bytes = destination_abs.read_bytes()
        if destination_bytes != plan["source_bytes"]:
            checksum_mismatches.append(
                {
                    "destination": relpath_str(destination_abs),
                    "source": plan["source"],
                    "expected_sha256": plan["sha256"],
                    "actual_sha256": sha256_bytes(destination_bytes),
                }
            )

    unexpected_files: list[str] = []
    if bundle_root.exists():
        for path in sorted(bundle_root.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved not in expected_by_destination:
                unexpected_files.append(relpath_str(resolved))

    actual_file_count = len(list(bundle_root.rglob("*"))) if bundle_root.exists() else 0
    actual_files_only = len([path for path in bundle_root.rglob("*") if path.is_file()]) if bundle_root.exists() else 0

    return {
        "exact_match": not missing_destinations and not checksum_mismatches and not unexpected_files,
        "expected_file_count": len(expected_by_destination),
        "actual_path_count": actual_file_count,
        "actual_file_count": actual_files_only,
        "missing_destinations": missing_destinations,
        "missing_destination_count": len(missing_destinations),
        "checksum_mismatches": checksum_mismatches,
        "checksum_mismatch_count": len(checksum_mismatches),
        "unexpected_files": unexpected_files,
        "unexpected_file_count": len(unexpected_files),
    }


def _build_report_payload(
    *,
    dry_run: bool,
    manifest_path: Path,
    bundle_root: Path,
    preflight: dict[str, Any],
    bundle_integrity: dict[str, Any],
    evidence_freeze: dict[str, Any],
    validation: dict[str, Any],
    plans: list[dict[str, Any]],
    stale_files: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    copies = _serialize_plans(plans)
    stale = _serialize_stale_files(stale_files, dry_run=dry_run)
    return {
        "dry_run": dry_run,
        "manifest": relpath_str(manifest_path),
        "bundle_root": relpath_str(bundle_root),
        "preflight": preflight,
        "bundle_integrity": bundle_integrity,
        "evidence_freeze": evidence_freeze,
        "validation": validation,
        "summary": {
            "mapped_file_count": len(plans),
            "copy_count": len([plan for plan in plans if plan["changed"]]),
            "create_count": len([plan for plan in plans if plan["action"] == "create"]),
            "update_count": len([plan for plan in plans if plan["action"] == "update"]),
            "unchanged_count": len([plan for plan in plans if not plan["changed"]]),
            "stale_file_count": len(stale_files),
            "bytes_copied": sum(plan["bytes_copied"] for plan in plans),
        },
        "snapshot": snapshot,
        "copies": copies,
        "stale_files": stale,
    }


def _write_integrity_sidecars(
    *,
    bundle_root: Path,
    report_payload: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    json_path = bundle_root.parent / f"{bundle_root.name}.integrity.json"
    markdown_path = bundle_root.parent / f"{bundle_root.name}.integrity.md"
    json_content = json_dumps_stable(
        {
            "manifest": report_payload["manifest"],
            "bundle_root": report_payload["bundle_root"],
            "summary": report_payload["summary"],
            "snapshot": report_payload["snapshot"],
            "validation": report_payload["validation"],
            "preflight": report_payload["preflight"],
            "bundle_integrity": report_payload["bundle_integrity"],
            "evidence_freeze": report_payload["evidence_freeze"],
            "copies": report_payload["copies"],
            "stale_files": report_payload["stale_files"],
        }
    )
    markdown_content = _render_markdown_report(report_payload)

    json_plan = render_text_write_plan(json_path, json_content)
    markdown_plan = render_text_write_plan(markdown_path, markdown_content)

    if not dry_run:
        try:
            apply_text_write(json_path, json_content)
            apply_text_write(markdown_path, markdown_content)
        except OSError as exc:
            raise StructuredError(
                "integrity_report_write_failed",
                "Failed to write bundle integrity sidecars.",
                {
                    "json_path": relpath_str(json_path),
                    "markdown_path": relpath_str(markdown_path),
                    "reason": f"{type(exc).__name__}: {exc}",
                },
            ) from exc

    return {
        "json": {
            **json_plan,
            "path": relpath_str(json_path),
            "action": json_plan["action"] if not dry_run else f"would_{json_plan['action']}",
        },
        "markdown": {
            **markdown_plan,
            "path": relpath_str(markdown_path),
            "action": markdown_plan["action"] if not dry_run else f"would_{markdown_plan['action']}",
        },
    }


def _render_markdown_report(report_payload: dict[str, Any]) -> str:
    policy_summary = report_payload["evidence_freeze"].get("policy_summary", {})
    lines = [
        "# Layer 2 Bundle Integrity",
        "",
        f"- Manifest: `{report_payload['manifest']}`",
        f"- Bundle root: `{report_payload['bundle_root']}`",
        f"- Dry run: `{str(report_payload['dry_run']).lower()}`",
        f"- Exact match: `{str(report_payload['snapshot']['exact_match']).lower()}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| mapped_file_count | {report_payload['summary']['mapped_file_count']} |",
        f"| copy_count | {report_payload['summary']['copy_count']} |",
        f"| unchanged_count | {report_payload['summary']['unchanged_count']} |",
        f"| stale_file_count | {report_payload['summary']['stale_file_count']} |",
        f"| bytes_copied | {report_payload['summary']['bytes_copied']} |",
        "",
        "## Policy Summary",
        "",
        "| Outcome | Count | Items |",
        "| --- | --- | --- |",
        (
            "| experimental_evidence_accepted | "
            f"{policy_summary.get('experimental_evidence_accepted_count', 0)} | "
            f"{_format_policy_summary_items(policy_summary.get('experimental_evidence_accepted'))} |"
        ),
        (
            "| promotion_violations | "
            f"{policy_summary.get('promotion_violation_count', 0)} | "
            f"{_format_policy_summary_items(policy_summary.get('promotion_violations'))} |"
        ),
        (
            "| true_blocking_issues | "
            f"{policy_summary.get('true_blocking_issue_count', 0)} | "
            f"{_format_policy_summary_items(policy_summary.get('true_blocking_issues'))} |"
        ),
        "",
        "## Evidence Freeze",
        "",
        "| Check | Status | Policy | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for check in report_payload["evidence_freeze"]["checks"]:
        detail = ""
        policy = check.get("policy_decision", "")
        if check["kind"] == "arxiv_official_execution_backed":
            detail = check["source_path"]
        elif check["kind"] == "pcba_full_local_non_debug_checkpoint_frozen":
            detail = str(check.get("checkpoint_path"))
        elif check["kind"] == "wn18rr_experimental_evidence_frozen":
            detail_parts = [",".join(check.get("still_experimental_reasons") or []) or "none"]
            destination_policy = check.get("destination_policy", {})
            if check.get("policy_decision") == "experimental_evidence_accepted":
                detail_parts.append(
                    "experimental_only_destinations="
                    f"{destination_policy.get('experimental_only_destination_count', 0)}"
                )
            elif check.get("policy_decision") == "promotion_violation":
                detail_parts.append(f"violations={destination_policy.get('violation_count', 0)}")
            elif check.get("failed_freeze_requirements"):
                detail_parts.append(
                    "failed_requirements="
                    f"{','.join(check.get('failed_freeze_requirements') or [])}"
                )
            detail = "; ".join(detail_parts)
        lines.append(f"| {check['kind']} | {check['status']} | {policy} | {detail} |")

    lines.extend(
        [
            "",
            "## Files",
            "",
            "| Kind | Source | Destination | Status | Bytes Copied | SHA256 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for copy in report_payload["copies"]:
        lines.append(
            "| "
            f"{copy['kind']} | `{copy['source']}` | `{copy['destination']}` | "
            f"{copy['action']} | {copy['bytes_copied']} | `{copy['sha256']}` |"
        )

    lines.extend(
        [
            "",
            "## Stale Files",
            "",
            "| Path | Status |",
            "| --- | --- |",
        ]
    )
    stale_files = report_payload["stale_files"]
    if stale_files:
        for stale in stale_files:
            lines.append(f"| `{stale['path']}` | {stale['action']} |")
    else:
        lines.append("| none | none |")
    lines.append("")
    return "\n".join(lines)


def _format_policy_summary_items(items: Any) -> str:
    if not isinstance(items, list) or not items:
        return "none"
    labels: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        dataset = item.get("dataset")
        kind = item.get("kind")
        if isinstance(dataset, str) and dataset:
            labels.append(dataset)
        elif isinstance(kind, str) and kind:
            labels.append(kind)
    return ", ".join(labels) if labels else "none"


def _combine_validation_reports(*reports: dict[str, Any]) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    for report in reports:
        warnings.extend(report.get("warnings", []))
        blocking_issues.extend(report.get("blocking_issues", []))
    return {
        "warnings": warnings,
        "warning_count": len(warnings),
        "blocking_issues": blocking_issues,
        "blocking_issue_count": len(blocking_issues),
    }


def _find_plan(plans: list[dict[str, Any]], *, source_rel: str) -> dict[str, Any] | None:
    for plan in plans:
        if plan["source"] == source_rel:
            return plan
    return None


def _require_plan(plans: list[dict[str, Any]], *, source_rel: str) -> dict[str, Any]:
    plan = _find_plan(plans, source_rel=source_rel)
    if plan is None:
        raise StructuredError(
            "manifest_mapping_missing",
            "Expected MANIFEST mapping is missing.",
            {
                "source": source_rel,
            },
        )
    return plan


def _check_summary_integrity(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    source_summary = _load_json_mapping(plan["source_abs"], label="bundle_source_summary")
    destination_summary = None
    if plan["destination_abs"].exists():
        destination_summary = _load_json_mapping(
            plan["destination_abs"],
            label="bundle_destination_summary",
        )

    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    source_entries = _summary_entries_by_key(source_summary)
    destination_entries = _summary_entries_by_key(destination_summary) if destination_summary else {}

    for entry in source_summary.get("entries", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "evaluation":
            continue
        quality = summary_entry_evidence_quality(entry)
        if quality["acceptable_execution_evidence"]:
            continue
        warnings.append(
            {
                "kind": "summary_entry_degraded",
                "entry_key": entry.get("key"),
                "status": quality["status"],
                "run_type": quality["run_type"],
                "metric_present": quality["metric_present"],
                "source_path": relpath_str(plan["source_abs"]),
            }
        )

    for entry_key in sorted(PROTECTED_OFFICIAL_ENTRY_KEYS):
        source_entry = source_entries.get(entry_key)
        if source_entry is None:
            blocking_issues.append(
                {
                    "kind": "protected_entry_missing",
                    "entry_key": entry_key,
                    "source_path": relpath_str(plan["source_abs"]),
                }
            )
            continue

        source_quality = summary_entry_evidence_quality(source_entry)
        if not source_quality["acceptable_execution_evidence"]:
            blocking_issues.append(
                {
                    "kind": "protected_entry_degraded",
                    "entry_key": entry_key,
                    "source_path": relpath_str(plan["source_abs"]),
                    "source_status": source_quality["status"],
                    "source_run_type": source_quality["run_type"],
                    "source_metric_present": source_quality["metric_present"],
                    "source_evidence_path": source_entry.get("evidence", {}).get("source_path"),
                }
            )

        destination_entry = destination_entries.get(entry_key)
        if destination_entry is None:
            continue
        destination_quality = summary_entry_evidence_quality(destination_entry)
        if source_quality["quality_key"] < destination_quality["quality_key"]:
            blocking_issues.append(
                {
                    "kind": "protected_entry_regressed_vs_bundle",
                    "entry_key": entry_key,
                    "source_path": relpath_str(plan["source_abs"]),
                    "destination_path": relpath_str(plan["destination_abs"]),
                    "source_quality_key": list(source_quality["quality_key"]),
                    "destination_quality_key": list(destination_quality["quality_key"]),
                    "source_status": source_quality["status"],
                    "destination_status": destination_quality["status"],
                    "source_run_type": source_quality["run_type"],
                    "destination_run_type": destination_quality["run_type"],
                }
            )

    return {
        "warnings": warnings,
        "blocking_issues": blocking_issues,
    }


def _check_manifest_integrity(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    source_manifest = _load_json_mapping(
        plan["source_abs"],
        label=f"bundle_source_manifest:{plan['source']}",
    )
    source_run_type = infer_manifest_run_type(source_manifest)
    source_status = str(source_manifest.get("overall_status") or "error")
    source_suite_name = source_manifest.get("suite_name")

    warnings: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []

    if source_run_type != "execution":
        blocking_issues.append(
            {
                "kind": "manifest_preview_regression",
                "source_path": relpath_str(plan["source_abs"]),
                "destination_path": relpath_str(plan["destination_abs"]),
                "suite_name": source_suite_name,
                "status": source_status,
                "run_type": source_run_type,
            }
        )

    _acceptable_manifest_statuses = ACCEPTABLE_EXECUTION_STATUSES | {"partial_failure"}
    if source_status not in _acceptable_manifest_statuses:
        blocking_issues.append(
            {
                "kind": "manifest_execution_status_unacceptable",
                "source_path": relpath_str(plan["source_abs"]),
                "destination_path": relpath_str(plan["destination_abs"]),
                "suite_name": source_suite_name,
                "status": source_status,
                "run_type": source_run_type,
            }
        )
    elif source_status == "partial_failure":
        warnings.append(
            {
                "kind": "manifest_execution_partial_failure",
                "source_path": relpath_str(plan["source_abs"]),
                "destination_path": relpath_str(plan["destination_abs"]),
                "suite_name": source_suite_name,
                "status": source_status,
                "run_type": source_run_type,
            }
        )

    if plan["destination_abs"].exists():
        destination_manifest = _load_json_mapping(
            plan["destination_abs"],
            label=f"bundle_destination_manifest:{plan['destination']}",
        )
        destination_run_type = infer_manifest_run_type(destination_manifest)
        destination_status = str(destination_manifest.get("overall_status") or "error")
        if source_run_type != "execution" and destination_run_type == "execution":
            blocking_issues.append(
                {
                    "kind": "manifest_regressed_vs_bundle",
                    "source_path": relpath_str(plan["source_abs"]),
                    "destination_path": relpath_str(plan["destination_abs"]),
                    "source_status": source_status,
                    "destination_status": destination_status,
                    "source_run_type": source_run_type,
                    "destination_run_type": destination_run_type,
                }
            )
        destination_suite_name = destination_manifest.get("suite_name")
        if source_suite_name != destination_suite_name:
            warnings.append(
                {
                    "kind": "manifest_suite_changed",
                    "source_path": relpath_str(plan["source_abs"]),
                    "destination_path": relpath_str(plan["destination_abs"]),
                    "source_suite_name": source_suite_name,
                    "destination_suite_name": destination_suite_name,
                }
            )

    return {
        "warnings": warnings,
        "blocking_issues": blocking_issues,
    }


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    payload = load_json_required(path, label=label)
    return require_mapping(payload, label=label, path=path)


def _read_text_required(path: Path, *, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StructuredError(
            "read_error",
            f"Failed to read required text input: {label}.",
            {
                "label": label,
                "path": relpath_str(path),
                "reason": f"{type(exc).__name__}: {exc}",
            },
        ) from exc


def _summary_entries_by_key(summary_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not summary_payload:
        return {}
    entries = summary_payload.get("entries")
    if not isinstance(entries, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if isinstance(key, str) and key:
            result[key] = entry
    return result


def _targets_by_name(manifest_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    targets = manifest_payload.get("targets")
    if not isinstance(targets, list):
        return result
    for target in targets:
        if not isinstance(target, dict):
            continue
        target_name = target.get("target_name")
        if isinstance(target_name, str) and target_name:
            result[target_name] = target
    return result


def _nested_get(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _serialize_plans(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "kind": plan["kind"],
            "source": plan["source"],
            "destination": plan["destination"],
            "action": plan["action"],
            "changed": plan["changed"],
            "size_bytes": plan["size_bytes"],
            "bytes_copied": plan["bytes_copied"],
            "sha256": plan["sha256"],
        }
        for plan in plans
    ]


def _serialize_stale_files(stale_files: list[dict[str, Any]], *, dry_run: bool) -> list[dict[str, Any]]:
    action = "would_remove" if dry_run else "removed"
    return [
        {
            "path": stale["path"],
            "action": action,
        }
        for stale in stale_files
    ]


def _find_duplicates(items: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return sorted(item for item, count in counts.items() if count > 1)


def _prune_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = path
    stop_at_resolved = stop_at.resolve()
    while True:
        current_resolved = current.resolve()
        if current_resolved == stop_at_resolved:
            break
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


if __name__ == "__main__":
    sys.exit(main())
