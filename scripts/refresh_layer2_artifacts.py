#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path
from typing import Any

from layer2_artifact_utils import (
    ACCEPTABLE_EXECUTION_STATUSES,
    EVIDENCE_STATUS_PRECEDENCE,
    PROTECTED_OFFICIAL_ENTRY_KEYS,
    PROJECT_ROOT,
    RUN_TYPE_PRECEDENCE,
    StructuredError,
    apply_text_write,
    evidence_quality_key,
    infer_manifest_preview,
    infer_manifest_run_type,
    is_acceptable_execution_evidence,
    json_dumps_stable,
    load_json_optional,
    load_json_required,
    metric_value_present,
    merge_note_fields,
    parse_note_fields,
    parse_note_items,
    print_json,
    relpath_str,
    render_text_write_plan,
    require_mapping,
)
from layer2_suite_targets import build_target_plan, get_artifact_items, get_target_metadata
from pcba_comparison_artifacts import (
    build_pcba_comparison_payload,
    build_pcba_summary_section,
    render_pcba_protocol_report,
)
from wn18rr_comparison_artifacts import (
    build_wn18rr_comparison_payload,
    build_wn18rr_summary_section,
    render_wn18rr_protocol_report,
)


GROUP_ORDER = ("official_candidate", "local_debug", "experimental")
_GENERIC_MANIFEST_NAMES = frozenset(
    {
        "layer2_suite_debug_manifest.json",
        "layer2_suite_official_manifest.json",
        "layer2_suite_debug_preview_manifest.json",
        "layer2_suite_official_preview_manifest.json",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh Layer 2 local artifacts from suite manifests, the WN18RR alignment audit, "
            "and the local PCBA/WN18RR debug result surfaces."
        )
    )
    parser.add_argument(
        "--official-manifest",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline" / "layer2_suite_official_manifest.json",
        help="Path to the official Layer 2 suite manifest.",
    )
    parser.add_argument(
        "--debug-manifest",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline" / "layer2_suite_debug_manifest.json",
        help="Path to the debug/experimental Layer 2 suite manifest.",
    )
    parser.add_argument(
        "--additional-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional Layer 2 suite manifests to search after the canonical official/debug "
            "manifests and before auto-discovered suite-specific manifests."
        ),
    )
    parser.add_argument(
        "--wn18rr-audit",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline" / "wn18rr_alignment_audit.json",
        help="Path to the WN18RR alignment audit JSON.",
    )
    parser.add_argument(
        "--wn18rr-semantic-audit",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline" / "wn18rr_semantic_alignment_audit.json",
        help="Path to the WN18RR semantic alignment audit JSON (optional).",
    )
    parser.add_argument(
        "--pcba-debug-result",
        type=Path,
        default=_default_target_path("graphmae_pcba_native_graph", "debug"),
        help="Fallback path for the GraphMAE PCBA native graph local/debug result JSON.",
    )
    parser.add_argument(
        "--pcba-full-local-result",
        "--pcba-official-local-result",
        dest="pcba_full_local_result",
        type=Path,
        default=_default_target_path(
            "graphmae_pcba_native_graph",
            "official",
            profile="full_local_non_debug",
        ),
        help=(
            "Fallback path for the GraphMAE PCBA native graph full-local non-debug "
            "comparison result JSON."
        ),
    )
    parser.add_argument(
        "--wn18rr-debug-result",
        type=Path,
        default=_default_target_path("graphmae_wn18rr_sbert_link", "debug"),
        help="Fallback path for the WN18RR experimental link-eval result JSON.",
    )
    parser.add_argument(
        "--wn18rr-relaware-debug-result",
        type=Path,
        default=_default_target_path("graphmae_wn18rr_sbert_link_relaware", "debug"),
        help="Fallback path for the WN18RR relation-aware experimental link-eval result JSON.",
    )
    parser.add_argument(
        "--wn18rr-fullscale-result",
        type=Path,
        default=_default_target_path("graphmae_wn18rr_sbert_link", "official"),
        help="Fallback path for the WN18RR full-scale experimental link-eval result JSON.",
    )
    parser.add_argument(
        "--wn18rr-relaware-fullscale-result",
        type=Path,
        default=_default_target_path("graphmae_wn18rr_sbert_link_relaware", "official"),
        help="Fallback path for the WN18RR relation-aware full-scale experimental link-eval result JSON.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "summary.json",
        help="Output path for the refreshed summary JSON.",
    )
    parser.add_argument(
        "--meeting-table-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "meeting_table.md",
        help="Output path for the refreshed meeting table markdown.",
    )
    parser.add_argument(
        "--progress-out",
        type=Path,
        default=PROJECT_ROOT / "notes" / "meeting_progress_arxiv_sbert.md",
        help="Output path for the refreshed meeting progress markdown.",
    )
    parser.add_argument(
        "--pcba-comparison-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline" / "pcba_graph_comparison.json",
        help="Output path for the refreshed PCBA graph comparison JSON.",
    )
    parser.add_argument(
        "--pcba-report-out",
        type=Path,
        default=PROJECT_ROOT / "notes" / "pcba_graph_protocol_report.md",
        help="Output path for the refreshed PCBA graph comparison report.",
    )
    parser.add_argument(
        "--wn18rr-comparison-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline" / "wn18rr_link_comparison.json",
        help="Output path for the refreshed WN18RR experimental comparison JSON.",
    )
    parser.add_argument(
        "--wn18rr-report-out",
        type=Path,
        default=PROJECT_ROOT / "notes" / "wn18rr_link_protocol_report.md",
        help="Output path for the refreshed WN18RR link protocol comparison report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned artifact updates without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        official_manifest = _load_suite_manifest_source(
            args.official_manifest, label="official_manifest",
        )
        debug_manifest = _load_suite_manifest_source(args.debug_manifest, label="debug_manifest")
        additional_manifests = [
            _load_suite_manifest_source(path, label=f"additional_manifest[{index}]")
            for index, path in enumerate(args.additional_manifest)
        ]
        discovered_manifests = [
            _load_suite_manifest_source(path, label=f"discovered_manifest[{index}]")
            for index, path in enumerate(
                _discover_suite_manifest_paths(
                    [
                        args.official_manifest,
                        args.debug_manifest,
                        *args.additional_manifest,
                    ]
                )
            )
        ]
        manifest_catalog = {
            "primary": [official_manifest, debug_manifest],
            "additional": additional_manifests,
            "discovered": discovered_manifests,
        }
        manifest_sources = _dedupe_manifest_sources(
            [
                *manifest_catalog["primary"],
                *manifest_catalog["additional"],
                *manifest_catalog["discovered"],
            ]
        )
        audit_payload = require_mapping(
            load_json_required(args.wn18rr_audit, label="wn18rr_alignment_audit"),
            label="wn18rr_alignment_audit",
            path=args.wn18rr_audit,
        )
        semantic_audit_payload = None
        semantic_audit_raw = load_json_optional(
            args.wn18rr_semantic_audit, label="wn18rr_semantic_alignment_audit",
        )
        if semantic_audit_raw is not None and isinstance(semantic_audit_raw, dict):
            semantic_audit_payload = semantic_audit_raw
            semantic_audit_payload["_source_path"] = relpath_str(args.wn18rr_semantic_audit)
        pcba_result = _load_result_json(args.pcba_debug_result, label="pcba_debug_result")
        pcba_full_local_result = _load_optional_result_json(
            args.pcba_full_local_result,
            label="pcba_full_local_result",
        )
        wn18rr_result = _load_result_json(args.wn18rr_debug_result, label="wn18rr_debug_result")
        wn18rr_relaware_result = _load_result_json(
            args.wn18rr_relaware_debug_result, label="wn18rr_relaware_debug_result",
        )
        wn18rr_fullscale_result = _load_optional_result_json(
            args.wn18rr_fullscale_result, label="wn18rr_fullscale_result",
        )
        wn18rr_relaware_fullscale_result = _load_optional_result_json(
            args.wn18rr_relaware_fullscale_result, label="wn18rr_relaware_fullscale_result",
        )

        entries = _build_entries(
            manifest_sources=manifest_sources,
            audit_payload=audit_payload,
            audit_path=args.wn18rr_audit,
            pcba_result=pcba_result,
            pcba_result_path=args.pcba_debug_result,
            pcba_full_local_result=pcba_full_local_result,
            pcba_full_local_result_path=args.pcba_full_local_result,
            wn18rr_result=wn18rr_result,
            wn18rr_result_path=args.wn18rr_debug_result,
            wn18rr_relaware_result=wn18rr_relaware_result,
            wn18rr_relaware_result_path=args.wn18rr_relaware_debug_result,
            wn18rr_fullscale_result=wn18rr_fullscale_result,
            wn18rr_fullscale_result_path=args.wn18rr_fullscale_result,
            wn18rr_relaware_fullscale_result=wn18rr_relaware_fullscale_result,
            wn18rr_relaware_fullscale_result_path=args.wn18rr_relaware_fullscale_result,
        )
        pcba_comparison_payload = build_pcba_comparison_payload(
            entries=entries,
            comparison_json_path=relpath_str(args.pcba_comparison_out),
            report_path=relpath_str(args.pcba_report_out),
        )
        pcba_summary_section = build_pcba_summary_section(pcba_comparison_payload)
        pcba_report = render_pcba_protocol_report(pcba_comparison_payload)
        wn18rr_comparison_payload = build_wn18rr_comparison_payload(
            entries=entries,
            audit_payload=audit_payload,
            semantic_audit_payload=semantic_audit_payload,
            comparison_json_path=relpath_str(args.wn18rr_comparison_out),
            report_path=relpath_str(args.wn18rr_report_out),
        )
        wn18rr_summary_section = build_wn18rr_summary_section(wn18rr_comparison_payload)
        wn18rr_report = render_wn18rr_protocol_report(wn18rr_comparison_payload)

        summary_payload = _build_summary_payload(
            entries=entries,
            official_manifest_path=args.official_manifest,
            debug_manifest_path=args.debug_manifest,
            additional_manifest_paths=[path for path in args.additional_manifest],
            discovered_manifest_paths=[source["path"] for source in discovered_manifests],
            manifest_search_order=[source["path"] for source in manifest_sources],
            audit_path=args.wn18rr_audit,
            pcba_result_path=args.pcba_debug_result,
            pcba_full_local_result_path=args.pcba_full_local_result,
            wn18rr_result_path=args.wn18rr_debug_result,
            wn18rr_relaware_result_path=args.wn18rr_relaware_debug_result,
            pcba_comparison_json_path=args.pcba_comparison_out,
            pcba_report_path=args.pcba_report_out,
            pcba_summary_section=pcba_summary_section,
            wn18rr_comparison_json_path=args.wn18rr_comparison_out,
            wn18rr_report_path=args.wn18rr_report_out,
            wn18rr_summary_section=wn18rr_summary_section,
        )
        write_specs = [
            {
                "output_type": "summary_json",
                "path": args.summary_out,
                "content": json_dumps_stable(summary_payload),
            },
            {
                "output_type": "meeting_table_markdown",
                "path": args.meeting_table_out,
                "content": _render_meeting_table(entries),
            },
            {
                "output_type": "meeting_progress_markdown",
                "path": args.progress_out,
                "content": _render_progress_markdown(
                    entries=entries,
                    pcba_comparison_payload=pcba_comparison_payload,
                    wn18rr_comparison_payload=wn18rr_comparison_payload,
                    official_manifest_path=args.official_manifest,
                    debug_manifest_path=args.debug_manifest,
                    additional_manifest_paths=[path for path in args.additional_manifest],
                    discovered_manifest_paths=[source["path"] for source in discovered_manifests],
                    manifest_search_order=[source["path"] for source in manifest_sources],
                    audit_path=args.wn18rr_audit,
                    pcba_result_path=args.pcba_debug_result,
                    pcba_full_local_result_path=args.pcba_full_local_result,
                    wn18rr_result_path=args.wn18rr_debug_result,
                    wn18rr_relaware_result_path=args.wn18rr_relaware_debug_result,
                ),
            },
            {
                "output_type": "pcba_comparison_json",
                "path": args.pcba_comparison_out,
                "content": json_dumps_stable(pcba_comparison_payload),
            },
            {
                "output_type": "pcba_protocol_report_markdown",
                "path": args.pcba_report_out,
                "content": pcba_report,
            },
            {
                "output_type": "wn18rr_comparison_json",
                "path": args.wn18rr_comparison_out,
                "content": json_dumps_stable(wn18rr_comparison_payload),
            },
            {
                "output_type": "wn18rr_protocol_report_markdown",
                "path": args.wn18rr_report_out,
                "content": wn18rr_report,
            },
        ]
        write_plans = [
            {
                **render_text_write_plan(spec["path"], spec["content"]),
                "output_type": spec["output_type"],
            }
            for spec in write_specs
        ]

        if not args.dry_run:
            for spec, plan in zip(write_specs, write_plans):
                if plan["changed"]:
                    apply_text_write(spec["path"], spec["content"])

        inputs_dict: dict[str, Any] = {
            "official_manifest": relpath_str(args.official_manifest),
            "debug_manifest": relpath_str(args.debug_manifest),
            "wn18rr_alignment_audit": relpath_str(args.wn18rr_audit),
            "pcba_debug_result": relpath_str(args.pcba_debug_result),
            "pcba_full_local_result": relpath_str(args.pcba_full_local_result),
            "wn18rr_debug_result": relpath_str(args.wn18rr_debug_result),
        }
        if semantic_audit_payload is not None:
            inputs_dict["wn18rr_semantic_alignment_audit"] = relpath_str(args.wn18rr_semantic_audit)
        if args.additional_manifest:
            inputs_dict["additional_manifests"] = [
                relpath_str(path) for path in args.additional_manifest
            ]
        if discovered_manifests:
            inputs_dict["discovered_manifests"] = [
                relpath_str(source["path"]) for source in discovered_manifests
            ]
        inputs_dict["manifest_search_order"] = [
            relpath_str(source["path"]) for source in manifest_sources
        ]
        if hasattr(args, "wn18rr_relaware_debug_result"):
            inputs_dict["wn18rr_relaware_debug_result"] = relpath_str(
                args.wn18rr_relaware_debug_result,
            )
        if wn18rr_fullscale_result is not None:
            inputs_dict["wn18rr_fullscale_result"] = relpath_str(args.wn18rr_fullscale_result)
        if wn18rr_relaware_fullscale_result is not None:
            inputs_dict["wn18rr_relaware_fullscale_result"] = relpath_str(
                args.wn18rr_relaware_fullscale_result,
            )
        print_json(
            {
                "dry_run": args.dry_run,
                "inputs": inputs_dict,
                "entry_keys": [entry["key"] for entry in entries],
                "writes": write_plans,
            }
        )
        return 0
    except StructuredError as exc:
        print_json(exc.to_payload())
        return 1


def _default_target_path(target_name: str, mode: str, *, profile: str = "default") -> Path:
    return build_target_plan(
        project_root=PROJECT_ROOT,
        target_name=target_name,
        mode=mode,
        profile=profile,
    ).out_json_path


def _load_suite_manifest(path: Path, *, label: str) -> dict[str, Any]:
    payload = require_mapping(load_json_required(path, label=label), label=label, path=path)
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise StructuredError(
            "invalid_input_shape",
            f"Expected `targets` to be a list in {label}.",
            {
                "label": label,
                "path": relpath_str(path),
                "actual_type": type(targets).__name__,
            },
        )
    return payload


def _load_suite_manifest_source(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_suite_manifest(path, label=label)
    requested_target = payload.get("requested_target")
    if not isinstance(requested_target, str) or not requested_target.strip():
        requested_target = _requested_target_from_suite_name(payload.get("suite_name"))
    finished_at = _parse_timestamp(payload.get("finished_at"))
    targets = payload.get("targets", [])
    run_type = infer_manifest_run_type(payload)
    preview = infer_manifest_preview(payload)
    return {
        "label": label,
        "path": path,
        "payload": payload,
        "suite_name": payload.get("suite_name"),
        "requested_target": requested_target,
        "mode": str(payload.get("mode")) if payload.get("mode") is not None else None,
        "run_type": run_type,
        "preview": preview,
        "target_count": len(targets) if isinstance(targets, list) else 0,
        "finished_at": finished_at.isoformat() if finished_at is not None else None,
        "finished_at_sort_key": finished_at.timestamp() if finished_at is not None else float("-inf"),
        "is_generic_path": path.name in _GENERIC_MANIFEST_NAMES,
    }


def _load_result_json(path: Path, *, label: str) -> dict[str, Any]:
    return require_mapping(load_json_required(path, label=label), label=label, path=path)


def _load_optional_result_json(path: Path, *, label: str) -> dict[str, Any] | None:
    payload = load_json_optional(path, label=label)
    if payload is None:
        return None
    return require_mapping(payload, label=label, path=path)


def _discover_suite_manifest_paths(excluded_paths: list[Path]) -> list[Path]:
    baseline_dir = PROJECT_ROOT / "results" / "baseline"
    excluded = {path.resolve() for path in excluded_paths}
    return [
        path
        for path in sorted(baseline_dir.glob("layer2_suite_*_manifest.json"))
        if path.resolve() not in excluded
    ]


def _dedupe_manifest_sources(manifest_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for source in manifest_sources:
        resolved_path = source["path"].resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        deduped.append(source)
    return deduped


def _build_entries(
    *,
    manifest_sources: list[dict[str, Any]],
    audit_payload: dict[str, Any],
    audit_path: Path,
    pcba_result: dict[str, Any],
    pcba_result_path: Path,
    pcba_full_local_result: dict[str, Any] | None,
    pcba_full_local_result_path: Path,
    wn18rr_result: dict[str, Any],
    wn18rr_result_path: Path,
    wn18rr_relaware_result: dict[str, Any] | None = None,
    wn18rr_relaware_result_path: Path | None = None,
    wn18rr_fullscale_result: dict[str, Any] | None = None,
    wn18rr_fullscale_result_path: Path | None = None,
    wn18rr_relaware_fullscale_result: dict[str, Any] | None = None,
    wn18rr_relaware_fullscale_result_path: Path | None = None,
) -> list[dict[str, Any]]:
    explicit_fallback_results: dict[str, tuple[dict[str, Any], Path]] = {
        "pcba_debug_result": (pcba_result, pcba_result_path),
        "wn18rr_debug_result": (wn18rr_result, wn18rr_result_path),
    }
    if pcba_full_local_result is not None:
        explicit_fallback_results["pcba_full_local_result"] = (
            pcba_full_local_result, pcba_full_local_result_path,
        )
    if wn18rr_fullscale_result is not None and wn18rr_fullscale_result_path is not None:
        explicit_fallback_results["wn18rr_fullscale_result"] = (
            wn18rr_fullscale_result, wn18rr_fullscale_result_path,
        )
    if wn18rr_relaware_fullscale_result is not None and wn18rr_relaware_fullscale_result_path is not None:
        explicit_fallback_results["wn18rr_relaware_fullscale_result"] = (
            wn18rr_relaware_fullscale_result, wn18rr_relaware_fullscale_result_path,
        )
    if wn18rr_relaware_result is not None and wn18rr_relaware_result_path is not None:
        explicit_fallback_results["wn18rr_relaware_debug_result"] = (
            wn18rr_relaware_result, wn18rr_relaware_result_path,
        )

    entries: list[dict[str, Any]] = []
    for item in get_artifact_items():
        target_metadata = get_target_metadata(item.target_name)
        if item.kind == "alignment_audit":
            entries.append(
                _entry_from_audit(
                    item=item,
                    target_metadata=target_metadata,
                    audit_payload=audit_payload,
                    audit_path=audit_path,
                )
            )
            continue

        if item.mode is None or item.profile is None or item.manifest_kind is None:
            raise StructuredError(
                "invalid_artifact_spec",
                "Registry-backed artifact item is missing execution metadata.",
                {
                    "item_key": item.key,
                    "target_name": item.target_name,
                },
            )
        target_plan = build_target_plan(
            project_root=PROJECT_ROOT,
            target_name=item.target_name,
            mode=item.mode,
            profile=item.profile,
        )
        resolution = _resolve_manifest_record(
            item=item,
            target_metadata=target_metadata,
            manifest_sources=manifest_sources,
        )
        fallback_candidate = _resolve_result_json_fallback(
            item=item,
            target_plan=target_plan,
            explicit_fallback_results=explicit_fallback_results,
        )

        if resolution is not None and resolution["acceptable"]:
            entries.append(
                _entry_from_manifest(
                    item=item,
                    target_metadata=target_metadata,
                    target_plan=target_plan,
                    resolution=resolution,
                )
            )
            continue

        if fallback_candidate is not None and fallback_candidate["acceptable"]:
            entries.append(
                _entry_from_result_json(
                    item=item,
                    target_metadata=target_metadata,
                    result_payload=fallback_candidate["payload"],
                    result_path=fallback_candidate["path"],
                    target_plan=target_plan,
                    supporting_manifest_sources=manifest_sources,
                    preferred_manifest_modes=_preferred_manifest_modes(item, target_metadata),
                    fallback_reason=_fallback_reason_for_result_json(
                        item=item,
                        resolution=resolution,
                        fallback_candidate=fallback_candidate,
                    ),
                )
            )
            continue

        if resolution is not None:
            entries.append(
                _entry_from_manifest(
                    item=item,
                    target_metadata=target_metadata,
                    target_plan=target_plan,
                    resolution=resolution,
                )
            )
            continue

        if fallback_candidate is not None:
            entries.append(
                _entry_from_result_json(
                    item=item,
                    target_metadata=target_metadata,
                    result_payload=fallback_candidate["payload"],
                    result_path=fallback_candidate["path"],
                    target_plan=target_plan,
                    supporting_manifest_sources=manifest_sources,
                    preferred_manifest_modes=_preferred_manifest_modes(item, target_metadata),
                    fallback_reason=_fallback_reason_for_result_json(
                        item=item,
                        resolution=None,
                        fallback_candidate=fallback_candidate,
                    ),
                )
            )
            continue

        if item.allow_missing:
            entries.append(
                _entry_from_missing_evidence(
                    item=item,
                    target_metadata=target_metadata,
                    target_plan=target_plan,
                    supporting_manifest_sources=manifest_sources,
                    preferred_manifest_modes=_preferred_manifest_modes(item, target_metadata),
                )
            )
            continue

        raise StructuredError(
            "missing_target",
            "Required Layer 2 target is absent from the searched suite manifests and result surfaces.",
            {
                "target_name": item.target_name,
                "preferred_manifest_modes": list(_preferred_manifest_modes(item, target_metadata)),
                "searched_manifest_paths": [
                    relpath_str(source["path"]) for source in manifest_sources
                ],
                "result_json_path": relpath_str(target_plan.out_json_path),
            },
        )
    return entries


def _resolve_result_json_fallback(
    *,
    item,
    target_plan,
    explicit_fallback_results: dict[str, tuple[dict[str, Any], Path]],
) -> dict[str, Any] | None:
    if item.fallback_result_label is not None:
        explicit_result = explicit_fallback_results.get(item.fallback_result_label)
        if explicit_result is None:
            return None
        payload, path = explicit_result
    else:
        optional_payload = load_json_optional(
            target_plan.out_json_path,
            label=f"{item.key}_result_json",
        )
        if optional_payload is None:
            return None
        payload = require_mapping(
            optional_payload,
            label=f"{item.key}_result_json",
            path=target_plan.out_json_path,
        )
        path = target_plan.out_json_path

    status = str(payload.get("status") or "error")
    metric_present = metric_value_present(payload.get("metric_value"))
    return {
        "payload": payload,
        "path": path,
        "status": status,
        "metric_present": metric_present,
        "quality_key": evidence_quality_key(
            status,
            run_type="execution",
            metric_present=metric_present,
        ),
        "acceptable": is_acceptable_execution_evidence(
            status=status,
            run_type="execution",
            metric_present=metric_present,
        ),
    }


def _fallback_reason_for_result_json(
    *,
    item,
    resolution: dict[str, Any] | None,
    fallback_candidate: dict[str, Any],
) -> str:
    result_path = relpath_str(fallback_candidate["path"])
    if resolution is None:
        return (
            f"No manifest-backed evidence matched {item.target_name}; using "
            f"{result_path} as the real execution evidence source."
        )
    manifest_path = relpath_str(resolution["manifest_source"]["path"])
    manifest_status = resolution["status"]
    manifest_run_type = resolution["manifest_source"]["run_type"]
    if item.key in PROTECTED_OFFICIAL_ENTRY_KEYS:
        return (
            f"Protected official-candidate entry refused degraded manifest evidence "
            f"({manifest_path}, run_type={manifest_run_type}, status={manifest_status}); "
            f"using {result_path} instead."
        )
    return (
        f"Manifest evidence for {item.target_name} was lower quality "
        f"({manifest_path}, run_type={manifest_run_type}, status={manifest_status}); "
        f"using {result_path} instead."
    )


def _preferred_manifest_modes(item, target_metadata) -> tuple[str, ...]:
    preferred = [item.mode]
    if item.group == "local_debug":
        preferred.extend(
            mode for mode in target_metadata.supported_modes if mode != item.mode
        )
    return tuple(preferred)


def _resolve_manifest_record(
    *,
    item,
    target_metadata,
    manifest_sources: list[dict[str, Any]],
) -> dict[str, Any] | None:
    preferred_modes = _preferred_manifest_modes(item, target_metadata)
    candidates: list[dict[str, Any]] = []
    for manifest_source in manifest_sources:
        records = _find_manifest_targets(manifest_source["payload"], item.target_name)
        for record in records:
            resolved_mode = str(record.get("mode") or manifest_source.get("mode") or "")
            resolved_profile = str(
                record.get("profile") or record.get("profile_name") or item.profile
            )
            if resolved_profile != item.profile:
                continue
            if resolved_mode not in preferred_modes:
                continue

            status = _manifest_record_status(record)
            metric_present = metric_value_present(record.get("parsed_metric_value"))
            candidates.append(
                {
                    "record": record,
                    "manifest_source": manifest_source,
                    "status": status,
                    "resolved_mode": resolved_mode,
                    "resolved_profile": resolved_profile,
                    "metric_present": metric_present,
                    "quality_key": evidence_quality_key(
                        status,
                        run_type=manifest_source["run_type"],
                        metric_present=metric_present,
                    ),
                    "acceptable": is_acceptable_execution_evidence(
                        status=status,
                        run_type=manifest_source["run_type"],
                        metric_present=metric_present,
                    ),
                }
            )

    if not candidates:
        return None

    acceptable_candidates = [
        candidate for candidate in candidates if candidate["acceptable"]
    ]
    selected_pool = acceptable_candidates or candidates
    selected = max(
        selected_pool,
        key=lambda candidate: _manifest_candidate_sort_key(
            candidate=candidate,
            target_name=item.target_name,
            preferred_modes=preferred_modes,
        ),
    )
    return {
        **selected,
        "candidate_count": len(candidates),
        "preferred_modes": preferred_modes,
        "used_secondary_mode": selected["resolved_mode"] != item.mode,
    }


def _manifest_candidate_sort_key(
    *,
    candidate: dict[str, Any],
    target_name: str,
    preferred_modes: tuple[str, ...],
) -> tuple[Any, ...]:
    manifest_source = candidate["manifest_source"]
    resolved_mode = candidate["resolved_mode"]
    requested_target = manifest_source.get("requested_target")
    return (
        candidate["quality_key"],
        -preferred_modes.index(resolved_mode),
        1 if requested_target == target_name else 0,
        -manifest_source.get("target_count", 0),
        1 if not manifest_source.get("is_generic_path") else 0,
        manifest_source.get("finished_at_sort_key", float("-inf")),
        relpath_str(manifest_source["path"]),
    )


def _manifest_record_status(record: dict[str, Any]) -> str:
    return str(
        record.get("parsed_status")
        or record.get("stage_eval_status")
        or record.get("overall_status")
        or "error"
    )


def _requested_target_from_suite_name(suite_name: Any) -> str | None:
    if not isinstance(suite_name, str):
        return None
    prefix = "layer2_suite:"
    if suite_name.startswith(prefix):
        return suite_name[len(prefix):]
    return None


def _parse_timestamp(raw_value: Any) -> datetime | None:
    if not isinstance(raw_value, str) or not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _entry_from_manifest(
    *,
    item,
    target_metadata,
    target_plan,
    resolution: dict[str, Any],
) -> dict[str, Any]:
    manifest_source = resolution["manifest_source"]
    manifest = manifest_source["payload"]
    manifest_path = manifest_source["path"]
    record = resolution["record"]
    resolved_mode = resolution["resolved_mode"]
    resolved_profile = resolution["resolved_profile"]
    notes = str(record.get("notes") or "")
    note_fields = parse_note_fields(notes)
    official_metric = _resolve_official_metric(
        note_fields=note_fields,
        target_metadata=target_metadata,
        resolved_mode=resolved_mode,
        resolved_profile=resolved_profile,
    )
    evidence_surface = _artifact_surface(
        item.group,
        resolved_mode,
        resolved_profile=resolved_profile,
    )
    caveat_extras = {"official_metric": official_metric}
    if item.group == "local_debug":
        caveat_extras["evidence_surface"] = evidence_surface
    caveat_fields = merge_note_fields(
        target_metadata.gate_caveats(),
        _registry_note_fields(target_metadata),
        note_fields,
        caveat_extras,
    )
    status = _manifest_record_status(record)

    return {
        "key": item.key,
        "label": item.label,
        "group": item.group,
        "kind": "evaluation",
        "status": status,
        "classification": target_metadata.classification(status=status),
        "dataset": str(record.get("dataset") or target_metadata.dataset_name),
        "model": str(record.get("model") or target_metadata.model_name),
        "task": str(record.get("task") or target_metadata.task_type),
        "metric": {
            "name": (
                record.get("parsed_metric_name")
                or record.get("registry_metric_name")
                or target_metadata.metric_name
            ),
            "value": record.get("parsed_metric_value"),
            "official_metric": official_metric,
        },
        "evidence": {
            "source_kind": "suite_manifest",
            "source_path": relpath_str(manifest_path),
            "suite_name": manifest.get("suite_name"),
            "target_name": item.target_name,
            "result_path": str(record.get("out_json") or relpath_str(target_plan.out_json_path)),
            "checkpoint_path": str(record.get("checkpoint_path") or relpath_str(target_plan.checkpoint_path)),
            "fresh_export_used": record.get("fresh_export_used"),
            "manifest_kind": resolved_mode,
            "preferred_manifest_kind": item.manifest_kind,
            "manifest_target_present": True,
            "resolved_mode": resolved_mode,
            "resolved_profile": resolved_profile,
            "requested_target": manifest_source.get("requested_target"),
            "run_type": manifest_source.get("run_type"),
            "preview": manifest_source.get("preview"),
            "quality_key": list(resolution["quality_key"]),
            "evidence_surface": evidence_surface,
        },
        "caveats": _build_caveats(caveat_fields),
        "details": {
            "mode": record.get("mode") or resolved_mode,
            "profile": record.get("profile") or record.get("profile_name") or resolved_profile,
            "artifact_group": record.get("artifact_group") or target_metadata.artifact_group,
            "stage_export_status": record.get("stage_export_status"),
            "stage_eval_status": record.get("stage_eval_status"),
            "return_code_export": record.get("return_code_export"),
            "return_code_eval": record.get("return_code_eval"),
            "notes": notes,
            "note_fields": note_fields,
            "registry_note_fields": _registry_note_fields(target_metadata),
            "registry_metric_name": record.get("registry_metric_name") or target_metadata.metric_name,
            "registry_metric_additional_names": (
                record.get("registry_metric_additional_names")
                or list(target_metadata.metric_additional_names)
            ),
            "registry_dataset_experimental": record.get(
                "registry_dataset_experimental",
                target_metadata.dataset_experimental,
            ),
            "registry_requires_alignment_audit": record.get(
                "registry_requires_alignment_audit",
                target_metadata.requires_alignment_audit,
            ),
            "registry_caveats": list(target_metadata.registry_caveats),
            "target_groups": list(target_metadata.suite_groups),
            "alignment_audit_status": record.get("alignment_audit_status"),
            "alignment_audit_json": record.get("alignment_audit_json"),
            "evidence_surface": evidence_surface,
            "manifest_resolution": {
                "selected_manifest": relpath_str(manifest_path),
                "selected_requested_target": manifest_source.get("requested_target"),
                "selected_target_count": manifest_source.get("target_count"),
                "selected_manifest_is_generic": manifest_source.get("is_generic_path"),
                "selected_run_type": manifest_source.get("run_type"),
                "selected_preview": manifest_source.get("preview"),
                "selected_status": resolution["status"],
                "selected_metric_present": resolution["metric_present"],
                "selected_acceptable": resolution["acceptable"],
                "selected_quality_key": list(resolution["quality_key"]),
                "candidate_count": resolution["candidate_count"],
                "preferred_modes": list(resolution["preferred_modes"]),
                "used_secondary_mode": resolution["used_secondary_mode"],
            },
            "readiness_gate": target_metadata.readiness_gate,
            "promotion_blockers": target_metadata.readiness_gate.get("promotion_blockers", []),
        },
    }


def _entry_from_result_json(
    *,
    item,
    target_metadata,
    result_payload: dict[str, Any],
    result_path: Path,
    target_plan,
    supporting_manifest_sources: list[dict[str, Any]],
    preferred_manifest_modes: tuple[str, ...],
    fallback_reason: str,
) -> dict[str, Any]:
    notes = str(result_payload.get("notes") or "")
    note_fields = parse_note_fields(notes)
    status = str(result_payload.get("status") or "error")
    official_metric = _resolve_official_metric(
        note_fields=note_fields,
        target_metadata=target_metadata,
        resolved_mode=item.mode,
        resolved_profile=item.profile,
    )
    evidence_surface = _artifact_surface(
        item.group,
        item.mode,
        resolved_profile=item.profile,
    )
    caveat_extras = {"official_metric": official_metric}
    if item.group == "local_debug":
        caveat_extras["evidence_surface"] = evidence_surface
    caveat_fields = merge_note_fields(
        target_metadata.gate_caveats(),
        _registry_note_fields(target_metadata),
        note_fields,
        caveat_extras,
    )
    supporting_manifest_paths = [
        relpath_str(source["path"]) for source in supporting_manifest_sources
    ]
    supporting_manifest_path = supporting_manifest_paths[0] if supporting_manifest_paths else None

    return {
        "key": item.key,
        "label": item.label,
        "group": item.group,
        "kind": "evaluation",
        "status": status,
        "classification": target_metadata.classification(status=status),
        "dataset": str(result_payload.get("dataset") or target_metadata.dataset_name),
        "model": str(result_payload.get("model") or target_metadata.model_name),
        "task": str(result_payload.get("task") or target_metadata.task_type),
        "metric": {
            "name": result_payload.get("metric_name") or target_metadata.metric_name,
            "value": result_payload.get("metric_value"),
            "official_metric": official_metric,
        },
        "evidence": {
            "source_kind": "result_json_fallback",
            "source_path": relpath_str(result_path),
            "suite_name": None,
            "target_name": item.target_name,
            "result_path": relpath_str(result_path),
            "checkpoint_path": relpath_str(target_plan.checkpoint_path),
            "fresh_export_used": None,
            "manifest_kind": item.manifest_kind,
            "manifest_target_present": False,
            "resolved_mode": item.mode,
            "resolved_profile": item.profile,
            "requested_target": None,
            "run_type": "execution",
            "preview": False,
            "quality_key": list(
                evidence_quality_key(
                    status,
                    run_type="execution",
                    metric_present=metric_value_present(result_payload.get("metric_value")),
                )
            ),
            "evidence_surface": evidence_surface,
        },
        "caveats": _build_caveats(caveat_fields),
        "details": {
            "notes": notes,
            "note_fields": note_fields,
            "registry_note_fields": _registry_note_fields(target_metadata),
            "registry_metric_name": target_metadata.metric_name,
            "registry_metric_additional_names": list(target_metadata.metric_additional_names),
            "registry_dataset_experimental": target_metadata.dataset_experimental,
            "registry_requires_alignment_audit": target_metadata.requires_alignment_audit,
            "registry_caveats": list(target_metadata.registry_caveats),
            "artifact_group": target_metadata.artifact_group,
            "target_groups": list(target_metadata.suite_groups),
            "supporting_manifest_path": supporting_manifest_path,
            "supporting_manifest_paths": supporting_manifest_paths,
            "preferred_manifest_modes": list(preferred_manifest_modes),
            "fallback_reason": fallback_reason,
            "protected_official_candidate": item.key in PROTECTED_OFFICIAL_ENTRY_KEYS,
            "evidence_surface": evidence_surface,
            "readiness_gate": target_metadata.readiness_gate,
            "promotion_blockers": target_metadata.readiness_gate.get("promotion_blockers", []),
        },
    }


def _entry_from_missing_evidence(
    *,
    item,
    target_metadata,
    target_plan,
    supporting_manifest_sources: list[dict[str, Any]],
    preferred_manifest_modes: tuple[str, ...],
) -> dict[str, Any]:
    evidence_surface = _artifact_surface(
        item.group,
        item.mode or target_metadata.artifact_mode,
        resolved_profile=item.profile,
    )
    supporting_manifest_paths = [
        relpath_str(source["path"]) for source in supporting_manifest_sources
    ]
    return {
        "key": item.key,
        "label": item.label,
        "group": item.group,
        "kind": "evaluation",
        "status": "blocked",
        "classification": target_metadata.classification(status="blocked"),
        "dataset": target_metadata.dataset_name,
        "model": target_metadata.model_name,
        "task": target_metadata.task_type,
        "metric": {
            "name": target_metadata.metric_name,
            "value": None,
            "official_metric": False,
        },
        "evidence": {
            "source_kind": "missing_evidence",
            "source_path": relpath_str(target_plan.out_json_path),
            "suite_name": None,
            "target_name": item.target_name,
            "result_path": relpath_str(target_plan.out_json_path),
            "checkpoint_path": relpath_str(target_plan.checkpoint_path),
            "fresh_export_used": None,
            "manifest_kind": item.manifest_kind,
            "manifest_target_present": False,
            "resolved_mode": item.mode,
            "resolved_profile": item.profile,
            "requested_target": None,
            "run_type": "execution",
            "preview": False,
            "quality_key": list(
                evidence_quality_key(
                    "blocked",
                    run_type="execution",
                    metric_present=False,
                )
            ),
            "evidence_surface": evidence_surface,
        },
        "caveats": {
            "official_metric": False,
            "evidence_surface": evidence_surface,
        },
        "details": {
            "mode": item.mode,
            "profile": item.profile,
            "notes": "comparison_profile_not_yet_materialized",
            "note_fields": {},
            "registry_note_fields": _registry_note_fields(target_metadata),
            "registry_metric_name": target_metadata.metric_name,
            "registry_metric_additional_names": list(target_metadata.metric_additional_names),
            "registry_dataset_experimental": target_metadata.dataset_experimental,
            "registry_requires_alignment_audit": target_metadata.requires_alignment_audit,
            "registry_caveats": list(target_metadata.registry_caveats),
            "artifact_group": target_metadata.artifact_group,
            "target_groups": list(target_metadata.suite_groups),
            "supporting_manifest_paths": supporting_manifest_paths,
            "preferred_manifest_modes": list(preferred_manifest_modes),
            "missing_reason": (
                "No manifest-backed or fallback-backed evidence exists yet for this optional "
                "comparison profile."
            ),
            "evidence_surface": evidence_surface,
            "readiness_gate": target_metadata.readiness_gate,
            "promotion_blockers": target_metadata.readiness_gate.get("promotion_blockers", []),
        },
    }


def _resolve_official_metric(
    *,
    note_fields: dict[str, Any],
    target_metadata,
    resolved_mode: str,
    resolved_profile: str | None,
) -> Any:
    official_metric = note_fields.get("official_metric")
    if official_metric is not None:
        return official_metric
    if resolved_profile == "full_local_non_debug":
        return False
    if not target_metadata.readiness_gate.get("official_metric_available", True):
        return False
    return resolved_mode == "official" or target_metadata.official_candidate


def _artifact_surface(
    group: str,
    resolved_mode: str,
    *,
    resolved_profile: str | None = None,
) -> str:
    if group == "official_candidate":
        return "official_candidate"
    if group == "experimental":
        return "experimental"
    if resolved_profile == "full_local_non_debug":
        return "full_local_non_debug"
    if resolved_profile == "official_local":
        return "official_local"
    if resolved_mode == "official":
        return "full_local_non_debug"
    return "local_debug"


def _entry_from_audit(
    *,
    item,
    target_metadata,
    audit_payload: dict[str, Any],
    audit_path: Path,
) -> dict[str, Any]:
    checks = audit_payload.get("checks")
    if not isinstance(checks, dict):
        raise StructuredError(
            "invalid_input_shape",
            "Expected `checks` to be a JSON object in the WN18RR alignment audit.",
            {
                "path": relpath_str(audit_path),
                "actual_type": type(checks).__name__,
            },
        )
    total_checks = len(checks)
    passed_checks = sum(
        1
        for value in checks.values()
        if isinstance(value, dict) and value.get("passed") is True
    )
    ordering_check = checks.get("sbert_ordering_evidence", {})
    loader_check = checks.get("graphmae_loader_consistent", {})
    status = str(audit_payload.get("status") or "error")

    return {
        "key": item.key,
        "label": item.label,
        "group": item.group,
        "kind": "alignment_audit",
        "status": status,
        "classification": target_metadata.classification(status=status),
        "dataset": target_metadata.dataset_name,
        "model": None,
        "task": "alignment_audit",
        "metric": {
            "name": None,
            "value": None,
            "official_metric": False,
        },
        "evidence": {
            "source_kind": "audit_json",
            "source_path": relpath_str(audit_path),
            "suite_name": None,
            "target_name": item.target_name,
            "result_path": relpath_str(audit_path),
            "checkpoint_path": None,
            "fresh_export_used": None,
            "manifest_kind": None,
            "manifest_target_present": None,
            "run_type": "execution",
            "preview": False,
            "quality_key": list(
                evidence_quality_key(
                    status,
                    run_type="execution",
                    metric_present=False,
                )
            ),
        },
        "caveats": {
            "experimental": target_metadata.experimental,
            "semantic_alignment_verified": False,
            "official_metric": False,
            "promotion_ready": not bool(target_metadata.readiness_gate.get("promotion_blockers")),
        },
        "details": {
            "notes": str(audit_payload.get("notes") or ""),
            "note_fields": {},
            "registry_metric_name": target_metadata.metric_name,
            "registry_metric_additional_names": list(target_metadata.metric_additional_names),
            "registry_dataset_experimental": target_metadata.dataset_experimental,
            "registry_requires_alignment_audit": target_metadata.requires_alignment_audit,
            "registry_caveats": list(target_metadata.registry_caveats),
            "artifact_group": target_metadata.artifact_group,
            "target_groups": list(target_metadata.suite_groups),
            "audit_checks_passed": passed_checks,
            "audit_checks_total": total_checks,
            "missing_pieces": audit_payload.get("missing_pieces", []),
            "num_entities": audit_payload.get("num_entities"),
            "feat_rows": audit_payload.get("feat_rows"),
            "feat_dim": audit_payload.get("feat_dim"),
            "train_edges": audit_payload.get("train_edges"),
            "valid_edges": audit_payload.get("valid_edges"),
            "test_edges": audit_payload.get("test_edges"),
            "relation_count": audit_payload.get("relation_count"),
            "ordering_evidence_passed": ordering_check.get("passed"),
            "graphmae_loader_consistent": loader_check.get("passed"),
            "readiness_gate": target_metadata.readiness_gate,
            "promotion_blockers": target_metadata.readiness_gate.get("promotion_blockers", []),
        },
    }


def _find_manifest_targets(manifest: dict[str, Any], target_name: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for target in manifest.get("targets", []):
        if isinstance(target, dict) and target.get("target_name") == target_name:
            matches.append(target)
    return matches


def _registry_note_fields(target_metadata) -> dict[str, Any]:
    return parse_note_items(target_metadata.registry_caveats)


def _build_summary_payload(
    *,
    entries: list[dict[str, Any]],
    official_manifest_path: Path,
    debug_manifest_path: Path,
    additional_manifest_paths: list[Path],
    discovered_manifest_paths: list[Path],
    manifest_search_order: list[Path],
    audit_path: Path,
    pcba_result_path: Path,
    pcba_full_local_result_path: Path,
    wn18rr_result_path: Path,
    pcba_comparison_json_path: Path,
    pcba_report_path: Path,
    pcba_summary_section: dict[str, Any],
    wn18rr_comparison_json_path: Path,
    wn18rr_report_path: Path,
    wn18rr_summary_section: dict[str, Any],
    wn18rr_relaware_result_path: Path | None = None,
) -> dict[str, Any]:
    source_files: dict[str, Any] = {
        "official_manifest": relpath_str(official_manifest_path),
        "debug_manifest": relpath_str(debug_manifest_path),
        "additional_manifests": [relpath_str(path) for path in additional_manifest_paths],
        "discovered_manifests": [relpath_str(path) for path in discovered_manifest_paths],
        "manifest_search_order": [relpath_str(path) for path in manifest_search_order],
        "wn18rr_alignment_audit": relpath_str(audit_path),
        "pcba_debug_result": relpath_str(pcba_result_path),
        "pcba_full_local_result": relpath_str(pcba_full_local_result_path),
        "pcba_official_local_result": relpath_str(pcba_full_local_result_path),
        "wn18rr_debug_result": relpath_str(wn18rr_result_path),
    }
    if wn18rr_relaware_result_path is not None:
        source_files["wn18rr_relaware_debug_result"] = relpath_str(
            wn18rr_relaware_result_path,
        )
    return {
        "schema_version": "layer2_artifact_summary/v1",
        "evidence_quality_policy": {
            "run_type_precedence": RUN_TYPE_PRECEDENCE,
            "status_precedence": EVIDENCE_STATUS_PRECEDENCE,
            "acceptable_execution_statuses": sorted(ACCEPTABLE_EXECUTION_STATUSES),
            "protected_official_entry_keys": sorted(PROTECTED_OFFICIAL_ENTRY_KEYS),
            "selection_rule": (
                "Manifest-backed evidence must be non-preview execution evidence with a "
                "present metric to outrank result JSON fallback."
            ),
        },
        "source_files": source_files,
        "generated_artifacts": {
            "pcba_comparison_json": relpath_str(pcba_comparison_json_path),
            "pcba_protocol_report": relpath_str(pcba_report_path),
            "wn18rr_comparison_json": relpath_str(wn18rr_comparison_json_path),
            "wn18rr_protocol_report": relpath_str(wn18rr_report_path),
        },
        "grouped_entry_keys": _grouped_entry_keys(entries),
        "comparison_sections": {
            "pcba_graph_compare": pcba_summary_section,
            "wn18rr_experimental_compare": wn18rr_summary_section,
        },
        "entries": entries,
    }


def _render_meeting_table(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# Layer 2 Meeting Table",
        "",
        "## Refreshed Summary",
        "| Group | Entry | Status | Metric | Evidence | Caveats |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            "| "
            + " | ".join(
                [
                    _group_label(entry["group"]),
                    entry["label"],
                    entry["status"],
                    _metric_display(entry),
                    _evidence_display(entry),
                    _caveat_display(entry),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Scope Notes",
            f"- Official candidate coverage: {_labels_for_group(entries, 'official_candidate')}.",
            f"- Local/debug coverage: {_labels_for_group(entries, 'local_debug')}.",
            f"- Experimental coverage: {_labels_for_group(entries, 'experimental')}. {_experimental_scope_note(entries)}",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_progress_markdown(
    *,
    entries: list[dict[str, Any]],
    pcba_comparison_payload: dict[str, Any],
    wn18rr_comparison_payload: dict[str, Any],
    official_manifest_path: Path,
    debug_manifest_path: Path,
    additional_manifest_paths: list[Path],
    discovered_manifest_paths: list[Path],
    manifest_search_order: list[Path],
    audit_path: Path,
    pcba_result_path: Path,
    pcba_full_local_result_path: Path,
    wn18rr_result_path: Path,
    wn18rr_relaware_result_path: Path | None = None,
) -> str:
    official_entries = [entry for entry in entries if entry["group"] == "official_candidate"]
    local_entries = [entry for entry in entries if entry["group"] == "local_debug"]
    experimental_entries = [entry for entry in entries if entry["group"] == "experimental"]

    lines = [
        "# Meeting Progress: arxiv_sbert",
        "",
        "## Official Candidate Coverage",
    ]
    for entry in official_entries:
        lines.append(
            f"- {entry['label']}: status={entry['status']}; {_metric_display(entry)}; "
            f"fresh_export_used={_format_scalar(entry['evidence']['fresh_export_used'])}."
        )
    if official_entries:
        lines.append("- These remain the current registry-backed official-candidate rows.")

    lines.extend(
        [
            "",
            "## Local Debug / Full-Local Coverage",
        ]
    )
    for entry in local_entries:
        evidence = entry["evidence"]
        lines.append(
            f"- {entry['label']}: status={entry['status']}; {_metric_display(entry)}; "
            f"surface={_evidence_surface_label(entry)}; "
            f"official_metric={_format_scalar(entry['metric']['official_metric'])}; "
            f"manifest_mode={_format_scalar(evidence.get('resolved_mode'))}; "
            f"evidence={evidence['source_path']} ({evidence['source_kind']})."
        )
        if evidence["source_kind"] == "result_json_fallback":
            lines.append(f"- {entry['label']} fallback: {entry['details']['fallback_reason']}")

    pcba_comparison = pcba_comparison_payload["comparison"]
    pcba_local = pcba_comparison_payload["profiles"]["local_debug"]
    pcba_full_local = pcba_comparison_payload["profiles"]["full_local_non_debug"]
    pcba_checkpoint_provenance = pcba_comparison_payload["checkpoint_provenance"]
    lines.extend(
        [
            "",
            "## PCBA Comparison",
            (
                f"- Comparison profile: `{pcba_comparison_payload['comparison_profile']['name']}` "
                f"pairs `{pcba_local['profile_name']}` and `{pcba_full_local['profile_name']}` "
                f"under `{pcba_local['target_name']}`."
            ),
            (
                f"- Full-local non-debug status={pcba_full_local['status']}; "
                f"ap={_format_scalar(pcba_full_local['metric']['value'])}; "
                f"surface={_format_scalar(pcba_full_local['evidence']['evidence_surface'])}; "
                f"manifest_backed={_format_scalar(pcba_full_local['evidence']['manifest_backed'])}; "
                f"debug_mode={_format_scalar(pcba_full_local['truncation']['debug_mode'])}; "
                f"debug_max_graphs={_format_scalar(pcba_full_local['truncation']['debug_max_graphs'])}; "
                f"max_eval_batches={_format_scalar(pcba_full_local['truncation']['max_eval_batches'])}; "
                f"split_truncation={_format_scalar(pcba_full_local['truncation']['split_truncation'])}."
            ),
            (
                f"- Checkpoint provenance: "
                f"debug={_format_scalar(pcba_checkpoint_provenance['local_debug_checkpoint_path'])}; "
                f"full_local={_format_scalar(pcba_checkpoint_provenance['full_local_non_debug_checkpoint_path'])}; "
                f"distinct={_format_scalar(pcba_checkpoint_provenance['checkpoint_paths_distinct'])}; "
                f"debug_checkpoint_surface_removed="
                f"{_format_scalar(pcba_checkpoint_provenance['debug_checkpoint_surface_removed_for_full_local'])}."
            ),
            (
                f"- AP delta (full-local non-debug minus debug): "
                f"{_format_scalar(pcba_comparison['metric_delta_full_local_non_debug_minus_local_debug']['ap'])}; "
                f"still_not_locked={', '.join(pcba_comparison['still_not_locked_reasons']) or 'none'}."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Experimental Coverage",
        ]
    )
    for entry in experimental_entries:
        if entry["kind"] == "alignment_audit":
            details = entry["details"]
            lines.append(
                f"- {entry['label']}: status={entry['status']}; "
                f"checks={details['audit_checks_passed']}/{details['audit_checks_total']}; "
                f"num_entities={details['num_entities']}; feat_rows={details['feat_rows']}; "
                f"relation_count={details['relation_count']}."
            )
            lines.append(
                f"- {entry['label']} caveat: semantic_alignment_verified=false; "
                f"ordering_evidence_passed={_format_scalar(details['ordering_evidence_passed'])}."
            )
            continue

        caveats = entry["caveats"]
        lines.append(
            f"- {entry['label']}: status={entry['status']}; {_metric_display(entry)}; "
            f"hits@1={_format_scalar(caveats.get('hits@1'))}; "
            f"hits@3={_format_scalar(caveats.get('hits@3'))}; "
            f"hits@10={_format_scalar(caveats.get('hits@10'))}; "
            f"relation_types_ignored={_format_scalar(caveats.get('relation_types_ignored'))}; "
            f"official_metric={_format_scalar(entry['metric']['official_metric'])}."
        )
    experimental_datasets = sorted({entry["dataset"] for entry in experimental_entries})
    if experimental_datasets:
        lines.append(
            "- Experimental datasets remain excluded from `official_candidate_*` and "
            f"`all_proven_local`: {', '.join(experimental_datasets)}."
        )
    comparison = wn18rr_comparison_payload["comparison"]
    relaware_path = wn18rr_comparison_payload["paths"]["relation_aware"]
    lines.extend(
        [
            "",
            "## WN18RR Comparison",
            (
                f"- Comparison profile: "
                f"`{wn18rr_comparison_payload['comparison_profile']['name']}` pairs "
                f"`{wn18rr_comparison_payload['paths']['baseline']['target_name']}` and "
                f"`{relaware_path['target_name']}`."
            ),
            (
                f"- Relation-aware delta: "
                f"improved_only_by_relation_aware="
                f"{', '.join(comparison['improved_only_by_relation_aware']) or 'none'}; "
                f"mrr_delta={_format_scalar(comparison['metric_delta_relation_aware_minus_baseline']['mrr'])}; "
                f"hits@1_delta={_format_scalar(comparison['metric_delta_relation_aware_minus_baseline']['hits@1'])}; "
                f"hits@3_delta={_format_scalar(comparison['metric_delta_relation_aware_minus_baseline']['hits@3'])}; "
                f"hits@10_delta={_format_scalar(comparison['metric_delta_relation_aware_minus_baseline']['hits@10'])}; "
                f"scorer_trained={_format_scalar(relaware_path['scorer_trained'])}; "
                f"scorer_train_steps={_format_scalar(relaware_path['scorer_train_steps'])}."
            ),
            (
                f"- Shared remaining blockers: "
                f"{', '.join(comparison['common_remaining_promotion_blockers']) or 'none'}."
            ),
        ]
    )
    alignment = wn18rr_comparison_payload.get("alignment_evidence", {})
    semantic_verdict = alignment.get("semantic_verdict")
    if semantic_verdict:
        lines.append(
            f"- Semantic alignment: verdict={semantic_verdict}; "
            f"verified={_format_scalar(alignment.get('semantic_alignment_verified'))}."
        )
    neg_sampling = wn18rr_comparison_payload.get("negative_sampling_assessment", {})
    if neg_sampling.get("contract_defined"):
        lines.append(
            f"- Negative-sampling contract: defined={_format_scalar(neg_sampling['contract_defined'])}; "
            f"blocker_cleared={_format_scalar(neg_sampling['blocker_cleared'])}."
        )
    official_metric = wn18rr_comparison_payload.get("official_metric_assessment", {})
    if official_metric:
        lines.append(
            f"- Official metric: full_scale_eval_completed="
            f"{_format_scalar(official_metric.get('full_scale_eval_completed'))}; "
            f"blocker_retained={_format_scalar(official_metric.get('blocker_retained'))}."
        )

    refresh_inputs = [
        "",
        "## Refresh Inputs",
        f"- official manifest: `{relpath_str(official_manifest_path)}`",
        f"- debug manifest: `{relpath_str(debug_manifest_path)}`",
        (
            "- additional manifests: "
            f"{_render_path_list(additional_manifest_paths)}"
        ),
        (
            "- auto-discovered suite manifests: "
            f"{_render_path_list(discovered_manifest_paths)}"
        ),
        (
            "- manifest search order: "
            f"{_render_path_list(manifest_search_order)}"
        ),
        f"- WN18RR alignment audit: `{relpath_str(audit_path)}`",
        f"- PCBA debug result fallback: `{relpath_str(pcba_result_path)}`",
        f"- PCBA full-local result fallback: `{relpath_str(pcba_full_local_result_path)}`",
        f"- WN18RR debug result fallback: `{relpath_str(wn18rr_result_path)}`",
    ]
    if wn18rr_relaware_result_path is not None:
        refresh_inputs.append(
            f"- WN18RR relaware debug result fallback: "
            f"`{relpath_str(wn18rr_relaware_result_path)}`"
        )
    lines.extend(refresh_inputs)
    return "\n".join(lines) + "\n"


def _group_label(group: str) -> str:
    if group == "official_candidate":
        return "official"
    if group == "local_debug":
        return "local/debug"
    return "experimental"


def _metric_display(entry: dict[str, Any]) -> str:
    metric = entry["metric"]
    if metric["name"] is not None and metric["value"] is not None:
        return f"{metric['name']}={_format_metric_value(metric['value'])}"
    if entry["kind"] == "alignment_audit":
        details = entry["details"]
        return f"checks={details['audit_checks_passed']}/{details['audit_checks_total']}"
    return "n/a"


def _evidence_display(entry: dict[str, Any]) -> str:
    evidence = entry["evidence"]
    source_kind = evidence["source_kind"]
    source_path = evidence["source_path"]
    resolved_mode = evidence.get("resolved_mode")
    mode_suffix = f"@{resolved_mode}" if resolved_mode is not None else ""
    if evidence["target_name"]:
        return f"{source_kind}:{source_path}:{evidence['target_name']}{mode_suffix}"
    return f"{source_kind}:{source_path}"


def _caveat_display(entry: dict[str, Any]) -> str:
    caveats = entry["caveats"]
    if not caveats:
        if entry["classification"]["official_candidate"]:
            return "official_candidate=true"
        return "none"
    items = [f"{key}={_format_scalar(caveats[key])}" for key in sorted(caveats)]
    return "; ".join(items)


def _build_caveats(note_fields: dict[str, Any]) -> dict[str, Any]:
    ignored = {"result_notes", "checkpoint_exists", "_fragments"}
    return {
        key: value
        for key, value in note_fields.items()
        if key not in ignored
    }


def _grouped_entry_keys(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        group: [entry["key"] for entry in entries if entry["group"] == group]
        for group in GROUP_ORDER
        if any(entry["group"] == group for entry in entries)
    }


def _labels_for_group(entries: list[dict[str, Any]], group: str) -> str:
    labels = [entry["label"] for entry in entries if entry["group"] == group]
    if not labels:
        return "none"
    return ", ".join(labels)


def _evidence_surface_label(entry: dict[str, Any]) -> str:
    surface = entry["evidence"].get("evidence_surface")
    if surface == "full_local_non_debug":
        return "full-local-non-debug"
    if surface == "official_local":
        return "official-local"
    if surface == "local_debug":
        return "local-debug"
    if surface == "official_candidate":
        return "official-candidate"
    if surface == "experimental":
        return "experimental"
    return "unknown"


def _render_path_list(paths: list[Path]) -> str:
    if not paths:
        return "none"
    return ", ".join(f"`{relpath_str(path)}`" for path in paths)


def _experimental_scope_note(entries: list[dict[str, Any]]) -> str:
    experimental_evals = [
        entry
        for entry in entries
        if entry["group"] == "experimental" and entry["kind"] == "evaluation"
    ]
    caveat_fragments: list[str] = []
    if any(entry["caveats"].get("relation_types_ignored") is True for entry in experimental_evals):
        caveat_fragments.append("relation_types_ignored=true")
    if any(entry["metric"]["official_metric"] is False for entry in experimental_evals):
        caveat_fragments.append("official_metric=false")
    if not caveat_fragments:
        return "Experimental rows remain fenced from official candidate groups."
    return (
        "Experimental rows remain fenced from official candidate groups; "
        f"current eval caveats: {', '.join(caveat_fragments)}."
    )


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
