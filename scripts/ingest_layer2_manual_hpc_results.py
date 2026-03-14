#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from layer2_artifact_utils import (
    ACCEPTABLE_EXECUTION_STATUSES,
    PROJECT_ROOT,
    StructuredError,
    apply_text_write,
    evidence_quality_key,
    json_dumps_stable,
    load_json_required,
    merge_note_fields,
    metric_value_present,
    parse_note_fields,
    print_json,
    relpath_str,
    render_text_write_plan,
    require_mapping,
    sha256_bytes,
)
from layer2_suite_targets import build_target_plan, get_artifact_items, get_target_metadata
from pcba_comparison_artifacts import (
    build_pcba_comparison_payload,
    build_pcba_summary_section,
    render_pcba_protocol_report,
)
from refresh_layer2_artifacts import (
    _artifact_surface,
    _build_caveats,
    _entry_from_audit,
    _entry_from_manifest,
    _grouped_entry_keys,
    _load_suite_manifest_source,
    _registry_note_fields,
    _render_meeting_table,
    _resolve_manifest_record,
    _resolve_official_metric,
)
from wn18rr_comparison_artifacts import (
    build_wn18rr_comparison_payload,
    build_wn18rr_summary_section,
    render_wn18rr_protocol_report,
)


BOOTSTRAP_ARTIFACTS: dict[str, dict[str, Any]] = {
    "arxiv_official_manifest": {
        "path": Path(
            "results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json"
        ),
        "kind": "json",
        "role": "supporting_context_for_manual_arxiv_refresh",
    },
    "pcba_debug_manifest": {
        "path": Path(
            "results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json"
        ),
        "kind": "json",
        "role": "bootstrap_local_debug_evidence",
    },
    "pcba_official_manifest": {
        "path": Path(
            "results/baseline/layer2_suite_graphmae_pcba_native_graph_official_manifest.json"
        ),
        "kind": "json",
        "role": "supporting_context_for_manual_pcba_refresh",
    },
    "pcba_comparison_json": {
        "path": Path("results/baseline/pcba_graph_comparison.json"),
        "kind": "json",
        "role": "bootstrap_comparison_reference",
    },
    "wn18rr_alignment_audit": {
        "path": Path("results/baseline/wn18rr_alignment_audit.json"),
        "kind": "json",
        "role": "bootstrap_alignment_audit",
    },
    "wn18rr_semantic_alignment_audit": {
        "path": Path("results/baseline/wn18rr_semantic_alignment_audit.json"),
        "kind": "json",
        "role": "bootstrap_semantic_alignment_audit",
    },
    "wn18rr_debug_manifest": {
        "path": Path(
            "results/baseline/layer2_suite_wn18rr_experimental_compare_debug_manifest.json"
        ),
        "kind": "json",
        "role": "bootstrap_experimental_debug_evidence",
    },
    "wn18rr_official_manifest": {
        "path": Path(
            "results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json"
        ),
        "kind": "json",
        "role": "bootstrap_experimental_fullscale_evidence",
    },
    "wn18rr_comparison_json": {
        "path": Path("results/baseline/wn18rr_link_comparison.json"),
        "kind": "json",
        "role": "bootstrap_comparison_reference",
    },
    "meeting_progress_note": {
        "path": Path("notes/meeting_progress_arxiv_sbert.md"),
        "kind": "text",
        "role": "bootstrap_note_reference",
    },
    "pcba_report_note": {
        "path": Path("notes/pcba_graph_protocol_report.md"),
        "kind": "text",
        "role": "bootstrap_note_reference",
    },
    "wn18rr_report_note": {
        "path": Path("notes/wn18rr_link_protocol_report.md"),
        "kind": "text",
        "role": "bootstrap_note_reference",
    },
}

MANUAL_RESULT_SPECS: dict[str, dict[str, Any]] = {
    "graphmae_arxiv_official_candidate": {
        "arg_name": "graphmae_arxiv_result",
        "bootstrap_context_label": "arxiv_official_manifest",
        "refresh_reason": "official_candidate_row_refreshed_from_manual_hpc_execution",
        "expected_model": "graphmae",
        "expected_dataset": "ogbn-arxiv",
        "expected_task": "node",
    },
    "bgrl_arxiv_official_candidate": {
        "arg_name": "bgrl_arxiv_result",
        "bootstrap_context_label": "arxiv_official_manifest",
        "refresh_reason": "official_candidate_row_refreshed_from_manual_hpc_execution",
        "expected_model": "bgrl",
        "expected_dataset": "ogbn-arxiv",
        "expected_task": "node",
    },
    "graphmae_pcba_native_full_local_non_debug": {
        "arg_name": "graphmae_pcba_official_local_result",
        "bootstrap_context_label": "pcba_official_manifest",
        "refresh_reason": "full_local_non_debug_row_refreshed_from_manual_hpc_execution",
        "expected_model": "graphmae",
        "expected_dataset": "ogbg-molpcba",
        "expected_task": "graph",
    },
}

BOOTSTRAP_MANIFEST_LABELS = (
    "arxiv_official_manifest",
    "pcba_debug_manifest",
    "pcba_official_manifest",
    "wn18rr_debug_manifest",
    "wn18rr_official_manifest",
)
OUTPUT_BOOTSTRAP_BASELINES = {
    "meeting_progress_markdown": "meeting_progress_note",
    "pcba_comparison_json": "pcba_comparison_json",
    "pcba_protocol_report_markdown": "pcba_report_note",
    "wn18rr_comparison_json": "wn18rr_comparison_json",
    "wn18rr_protocol_report_markdown": "wn18rr_report_note",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh Layer 2 artifacts from fresh manual HPC execution JSONs plus a curated "
            "bootstrap snapshot for unchanged debug/experimental evidence."
        )
    )
    parser.add_argument(
        "--graphmae-arxiv-result",
        type=Path,
        required=True,
        help="Fresh manual HPC GraphMAE arXiv result JSON.",
    )
    parser.add_argument(
        "--bgrl-arxiv-result",
        type=Path,
        required=True,
        help="Fresh manual HPC BGRL arXiv result JSON.",
    )
    parser.add_argument(
        "--graphmae-pcba-official-local-result",
        type=Path,
        required=True,
        help="Fresh manual HPC GraphMAE PCBA official-local/full-local result JSON.",
    )
    parser.add_argument(
        "--bootstrap-root",
        type=Path,
        default=PROJECT_ROOT / "state" / "layer2_bootstrap",
        help="Bootstrap snapshot root containing inherited Layer 2 evidence.",
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
        help="Output path for the refreshed PCBA protocol report.",
    )
    parser.add_argument(
        "--wn18rr-comparison-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline" / "wn18rr_link_comparison.json",
        help="Output path for the refreshed WN18RR comparison JSON.",
    )
    parser.add_argument(
        "--wn18rr-report-out",
        type=Path,
        default=PROJECT_ROOT / "notes" / "wn18rr_link_protocol_report.md",
        help="Output path for the refreshed WN18RR protocol report.",
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
        bootstrap_state = _load_bootstrap_state(args.bootstrap_root)
        manual_results = _load_manual_results(args)
        entries = _build_entries(
            bootstrap_state=bootstrap_state,
            manual_results=manual_results,
            bootstrap_root=args.bootstrap_root,
        )

        pcba_comparison_payload = build_pcba_comparison_payload(
            entries=entries,
            comparison_json_path=relpath_str(args.pcba_comparison_out),
            report_path=relpath_str(args.pcba_report_out),
        )
        pcba_summary_section = build_pcba_summary_section(pcba_comparison_payload)
        pcba_report = render_pcba_protocol_report(pcba_comparison_payload)

        semantic_audit_payload = dict(
            bootstrap_state["json_payloads"]["wn18rr_semantic_alignment_audit"]
        )
        semantic_audit_payload["_source_path"] = bootstrap_state["inventory"][
            "wn18rr_semantic_alignment_audit"
        ]["path"]
        wn18rr_comparison_payload = build_wn18rr_comparison_payload(
            entries=entries,
            audit_payload=bootstrap_state["json_payloads"]["wn18rr_alignment_audit"],
            semantic_audit_payload=semantic_audit_payload,
            comparison_json_path=relpath_str(args.wn18rr_comparison_out),
            report_path=relpath_str(args.wn18rr_report_out),
        )
        wn18rr_summary_section = build_wn18rr_summary_section(wn18rr_comparison_payload)
        wn18rr_report = render_wn18rr_protocol_report(wn18rr_comparison_payload)

        summary_payload = _build_summary_payload(
            entries=entries,
            manual_results=manual_results,
            bootstrap_state=bootstrap_state,
            bootstrap_root=args.bootstrap_root,
            pcba_summary_section=pcba_summary_section,
            wn18rr_summary_section=wn18rr_summary_section,
            summary_out=args.summary_out,
            meeting_table_out=args.meeting_table_out,
            progress_out=args.progress_out,
            pcba_comparison_out=args.pcba_comparison_out,
            pcba_report_out=args.pcba_report_out,
            wn18rr_comparison_out=args.wn18rr_comparison_out,
            wn18rr_report_out=args.wn18rr_report_out,
        )
        generated_payloads = {
            "summary_json": summary_payload,
            "meeting_table_markdown": _render_meeting_table(entries),
            "meeting_progress_markdown": _render_progress_markdown(
                entries=entries,
                manual_results=manual_results,
                bootstrap_state=bootstrap_state,
                bootstrap_root=args.bootstrap_root,
                pcba_comparison_payload=pcba_comparison_payload,
                wn18rr_comparison_payload=wn18rr_comparison_payload,
            ),
            "pcba_comparison_json": pcba_comparison_payload,
            "pcba_protocol_report_markdown": pcba_report,
            "wn18rr_comparison_json": wn18rr_comparison_payload,
            "wn18rr_protocol_report_markdown": wn18rr_report,
        }
        summary_payload["bootstrap_output_deltas"] = _bootstrap_output_deltas(
            bootstrap_state=bootstrap_state,
            generated_payloads=generated_payloads,
        )

        write_specs = [
            {
                "output_type": "summary_json",
                "path": args.summary_out,
                "content": json_dumps_stable(generated_payloads["summary_json"]),
            },
            {
                "output_type": "meeting_table_markdown",
                "path": args.meeting_table_out,
                "content": generated_payloads["meeting_table_markdown"],
            },
            {
                "output_type": "meeting_progress_markdown",
                "path": args.progress_out,
                "content": generated_payloads["meeting_progress_markdown"],
            },
            {
                "output_type": "pcba_comparison_json",
                "path": args.pcba_comparison_out,
                "content": json_dumps_stable(generated_payloads["pcba_comparison_json"]),
            },
            {
                "output_type": "pcba_protocol_report_markdown",
                "path": args.pcba_report_out,
                "content": generated_payloads["pcba_protocol_report_markdown"],
            },
            {
                "output_type": "wn18rr_comparison_json",
                "path": args.wn18rr_comparison_out,
                "content": json_dumps_stable(generated_payloads["wn18rr_comparison_json"]),
            },
            {
                "output_type": "wn18rr_protocol_report_markdown",
                "path": args.wn18rr_report_out,
                "content": generated_payloads["wn18rr_protocol_report_markdown"],
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

        print_json(
            {
                "dry_run": args.dry_run,
                "mode": "manual_hpc_result_ingestion",
                "manual_result_inputs": {
                    key: value["report"]
                    for key, value in manual_results.items()
                },
                "bootstrap_root": relpath_str(args.bootstrap_root),
                "bootstrap_artifacts": bootstrap_state["inventory"],
                "refreshed_entry_keys": [
                    key for key in MANUAL_RESULT_SPECS
                ],
                "inherited_entry_keys": [
                    entry["key"]
                    for entry in entries
                    if entry["evidence"].get("bootstrap_inherited") is True
                ],
                "writes": write_plans,
                "bootstrap_output_deltas": summary_payload["bootstrap_output_deltas"],
            }
        )
        return 0
    except StructuredError as exc:
        print_json(exc.to_payload())
        return 1


def _load_bootstrap_state(bootstrap_root: Path) -> dict[str, Any]:
    inventory: dict[str, dict[str, Any]] = {}
    json_payloads: dict[str, dict[str, Any]] = {}
    text_payloads: dict[str, str] = {}

    for label, spec in BOOTSTRAP_ARTIFACTS.items():
        path = (bootstrap_root / spec["path"]).resolve()
        if spec["kind"] == "json":
            payload = require_mapping(
                load_json_required(path, label=f"bootstrap_{label}"),
                label=f"bootstrap_{label}",
                path=path,
            )
            json_payloads[label] = payload
        else:
            text_payloads[label] = _read_text_required(path, label=f"bootstrap_{label}")

        inventory[label] = {
            "path": relpath_str(path),
            "kind": spec["kind"],
            "role": spec["role"],
            "sha256": sha256_bytes(path.read_bytes()),
        }

    manifest_sources = [
        _load_suite_manifest_source(
            (bootstrap_root / BOOTSTRAP_ARTIFACTS[label]["path"]).resolve(),
            label=f"bootstrap_{label}",
        )
        for label in BOOTSTRAP_MANIFEST_LABELS
    ]
    return {
        "inventory": inventory,
        "json_payloads": json_payloads,
        "text_payloads": text_payloads,
        "manifest_sources": manifest_sources,
    }


def _load_manual_results(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    artifact_items = {item.key: item for item in get_artifact_items()}
    manual_results: dict[str, dict[str, Any]] = {}
    for entry_key, spec in MANUAL_RESULT_SPECS.items():
        item = artifact_items[entry_key]
        result_path = getattr(args, spec["arg_name"])
        payload = _load_manual_result(
            path=result_path,
            entry_key=entry_key,
            expected_model=spec["expected_model"],
            expected_dataset=spec["expected_dataset"],
            expected_task=spec["expected_task"],
        )
        manual_results[entry_key] = {
            "path": result_path,
            "payload": payload,
            "item": item,
            "report": {
                "path": relpath_str(result_path),
                "source_kind": "manual_hpc_result_json",
                "manifest_backed": False,
                "run_type": "execution",
                "status": payload.get("status"),
                "metric_name": payload.get("metric_name"),
                "metric_value": payload.get("metric_value"),
            },
        }
    return manual_results


def _load_manual_result(
    *,
    path: Path,
    entry_key: str,
    expected_model: str,
    expected_dataset: str,
    expected_task: str,
) -> dict[str, Any]:
    payload = require_mapping(
        load_json_required(path, label=f"{entry_key}_manual_result"),
        label=f"{entry_key}_manual_result",
        path=path,
    )
    status = str(payload.get("status") or "")
    metric_value = payload.get("metric_value")
    if status not in ACCEPTABLE_EXECUTION_STATUSES or not metric_value_present(metric_value):
        raise StructuredError(
            "invalid_manual_result",
            "Fresh manual HPC result JSON must be successful execution evidence with a metric.",
            {
                "entry_key": entry_key,
                "path": relpath_str(path),
                "status": status,
                "metric_value": metric_value,
            },
        )

    actual_model = str(payload.get("model") or "")
    actual_dataset = str(payload.get("dataset") or "")
    actual_task = str(payload.get("task") or "")
    mismatches: dict[str, Any] = {}
    if actual_model != expected_model:
        mismatches["model"] = {"expected": expected_model, "actual": actual_model}
    if actual_dataset != expected_dataset:
        mismatches["dataset"] = {"expected": expected_dataset, "actual": actual_dataset}
    if actual_task != expected_task:
        mismatches["task"] = {"expected": expected_task, "actual": actual_task}
    if mismatches:
        raise StructuredError(
            "manual_result_mismatch",
            "Fresh manual HPC result JSON does not match the expected Layer 2 target surface.",
            {
                "entry_key": entry_key,
                "path": relpath_str(path),
                "mismatches": mismatches,
            },
        )
    return payload


def _build_entries(
    *,
    bootstrap_state: dict[str, Any],
    manual_results: dict[str, dict[str, Any]],
    bootstrap_root: Path,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for item in get_artifact_items():
        target_metadata = get_target_metadata(item.target_name)
        if item.kind == "alignment_audit":
            entry = _entry_from_audit(
                item=item,
                target_metadata=target_metadata,
                audit_payload=bootstrap_state["json_payloads"]["wn18rr_alignment_audit"],
                audit_path=(bootstrap_root / BOOTSTRAP_ARTIFACTS["wn18rr_alignment_audit"]["path"]).resolve(),
            )
            entries.append(
                _annotate_bootstrap_entry(
                    entry,
                    bootstrap_label="wn18rr_alignment_audit",
                    provenance_status="bootstrap_inherited_unchanged",
                    inheritance_reason="unchanged_wn18rr_alignment_audit_inherited_from_bootstrap",
                )
            )
            continue

        if item.mode is None or item.profile is None or item.manifest_kind is None:
            raise StructuredError(
                "invalid_artifact_spec",
                "Artifact item is missing execution metadata.",
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

        if item.key in manual_results:
            bootstrap_context_label = MANUAL_RESULT_SPECS[item.key]["bootstrap_context_label"]
            resolution = _resolve_manifest_record(
                item=item,
                target_metadata=target_metadata,
                manifest_sources=bootstrap_state["manifest_sources"],
            )
            if resolution is None:
                raise StructuredError(
                    "missing_bootstrap_context",
                    "Bootstrap manifest context for a fresh manual HPC result is missing.",
                    {
                        "entry_key": item.key,
                        "bootstrap_context_label": bootstrap_context_label,
                    },
                )
            entries.append(
                _entry_from_manual_result_json(
                    item=item,
                    target_metadata=target_metadata,
                    result_payload=manual_results[item.key]["payload"],
                    result_path=manual_results[item.key]["path"],
                    target_plan=target_plan,
                    bootstrap_reference=_bootstrap_reference_from_resolution(
                        resolution=resolution,
                        bootstrap_label=bootstrap_context_label,
                        refresh_reason=MANUAL_RESULT_SPECS[item.key]["refresh_reason"],
                    ),
                )
            )
            continue

        resolution = _resolve_manifest_record(
            item=item,
            target_metadata=target_metadata,
            manifest_sources=bootstrap_state["manifest_sources"],
        )
        if resolution is None:
            raise StructuredError(
                "missing_bootstrap_evidence",
                "Bootstrap snapshot is missing required inherited Layer 2 evidence.",
                {
                    "entry_key": item.key,
                    "target_name": item.target_name,
                    "mode": item.mode,
                    "profile": item.profile,
                },
            )
        entry = _entry_from_manifest(
            item=item,
            target_metadata=target_metadata,
            target_plan=target_plan,
            resolution=resolution,
        )
        entries.append(
            _annotate_bootstrap_entry(
                entry,
                bootstrap_label=_bootstrap_label_for_manifest_path(
                    relpath_str(resolution["manifest_source"]["path"]),
                ),
                provenance_status="bootstrap_inherited_unchanged",
                inheritance_reason=f"{item.key}_inherited_unchanged_from_bootstrap",
            )
        )
    return entries


def _entry_from_manual_result_json(
    *,
    item,
    target_metadata,
    result_payload: dict[str, Any],
    result_path: Path,
    target_plan,
    bootstrap_reference: dict[str, Any],
) -> dict[str, Any]:
    notes = str(result_payload.get("notes") or "")
    note_fields = merge_note_fields(
        parse_note_fields(notes),
        _manual_default_note_fields(item),
    )
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
            "source_kind": "manual_hpc_result_json",
            "source_path": relpath_str(result_path),
            "suite_name": None,
            "target_name": item.target_name,
            "result_path": relpath_str(result_path),
            "checkpoint_path": relpath_str(target_plan.checkpoint_path),
            "fresh_export_used": None,
            "manifest_kind": item.manifest_kind,
            "preferred_manifest_kind": item.manifest_kind,
            "manifest_target_present": False,
            "manifest_backed": False,
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
            "bootstrap_inherited": False,
            "provenance_status": "fresh_manual_hpc_execution",
        },
        "caveats": _build_caveats(caveat_fields),
        "details": {
            "mode": item.mode,
            "profile": item.profile,
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
            "bootstrap_reference": bootstrap_reference,
            "manual_hpc_provenance": {
                "source_kind": "manual_hpc_result_json",
                "manifest_backed": False,
                "run_type": "execution",
                "refresh_reason": bootstrap_reference["refresh_reason"],
                "supporting_bootstrap_evidence": [bootstrap_reference],
            },
            "evidence_surface": evidence_surface,
            "readiness_gate": target_metadata.readiness_gate,
            "promotion_blockers": target_metadata.readiness_gate.get("promotion_blockers", []),
        },
    }


def _bootstrap_reference_from_resolution(
    *,
    resolution: dict[str, Any],
    bootstrap_label: str,
    refresh_reason: str,
) -> dict[str, Any]:
    record = resolution["record"]
    manifest_source = resolution["manifest_source"]
    return {
        "bootstrap_label": bootstrap_label,
        "bootstrap_path": relpath_str(manifest_source["path"]),
        "refresh_reason": refresh_reason,
        "source_kind": "bootstrap_suite_manifest_context",
        "run_type": manifest_source.get("run_type"),
        "preview": manifest_source.get("preview"),
        "status": resolution.get("status"),
        "metric_name": record.get("parsed_metric_name") or record.get("registry_metric_name"),
        "metric_value": record.get("parsed_metric_value"),
        "resolved_mode": resolution.get("resolved_mode"),
        "resolved_profile": resolution.get("resolved_profile"),
    }


def _annotate_bootstrap_entry(
    entry: dict[str, Any],
    *,
    bootstrap_label: str | None,
    provenance_status: str,
    inheritance_reason: str,
) -> dict[str, Any]:
    entry["evidence"]["manifest_backed"] = entry["evidence"]["source_kind"] == "suite_manifest"
    entry["evidence"]["bootstrap_inherited"] = True
    entry["evidence"]["provenance_status"] = provenance_status
    entry["details"]["bootstrap_inheritance"] = {
        "bootstrap_label": bootstrap_label,
        "inheritance_reason": inheritance_reason,
        "source_kind": entry["evidence"]["source_kind"],
        "source_path": entry["evidence"]["source_path"],
    }
    return entry


def _bootstrap_label_for_manifest_path(path: str) -> str | None:
    for label, spec in BOOTSTRAP_ARTIFACTS.items():
        if Path(path).as_posix().endswith(spec["path"].as_posix()):
            return label
    return None


def _manual_default_note_fields(item) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    if item.mode == "official":
        defaults["debug_mode"] = False
    if item.profile == "full_local_non_debug":
        defaults["debug_max_graphs"] = None
        defaults["max_eval_batches"] = None
        defaults["split_truncation"] = "disabled"
    return defaults


def _build_summary_payload(
    *,
    entries: list[dict[str, Any]],
    manual_results: dict[str, dict[str, Any]],
    bootstrap_state: dict[str, Any],
    bootstrap_root: Path,
    pcba_summary_section: dict[str, Any],
    wn18rr_summary_section: dict[str, Any],
    summary_out: Path,
    meeting_table_out: Path,
    progress_out: Path,
    pcba_comparison_out: Path,
    pcba_report_out: Path,
    wn18rr_comparison_out: Path,
    wn18rr_report_out: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "layer2_manual_hpc_ingestion_summary/v1",
        "refresh_method": "manual_hpc_result_ingestion",
        "provenance_policy": {
            "fresh_source_kind": "manual_hpc_result_json",
            "fresh_manifest_backed": False,
            "fresh_run_type": "execution",
            "inherited_provenance_status": "bootstrap_inherited_unchanged",
            "no_fake_manifests_created": True,
            "no_dummy_jsons_required": True,
            "wn18rr_publication_rule": (
                "WN18RR remains experimental-only and excluded from official_candidate_* "
                "and all_proven_local."
            ),
        },
        "manual_result_inputs": {
            key: value["report"]
            for key, value in manual_results.items()
        },
        "bootstrap_snapshot": {
            "root": relpath_str(bootstrap_root),
            "artifacts": bootstrap_state["inventory"],
        },
        "generated_artifacts": {
            "summary_json": relpath_str(summary_out),
            "meeting_table_markdown": relpath_str(meeting_table_out),
            "meeting_progress_markdown": relpath_str(progress_out),
            "pcba_comparison_json": relpath_str(pcba_comparison_out),
            "pcba_protocol_report": relpath_str(pcba_report_out),
            "wn18rr_comparison_json": relpath_str(wn18rr_comparison_out),
            "wn18rr_protocol_report": relpath_str(wn18rr_report_out),
        },
        "grouped_entry_keys": _grouped_entry_keys(entries),
        "comparison_sections": {
            "pcba_graph_compare": pcba_summary_section,
            "wn18rr_experimental_compare": wn18rr_summary_section,
        },
        "refresh_plan": {
            "refreshed_from_manual_hpc_execution": list(MANUAL_RESULT_SPECS),
            "inherited_from_bootstrap": [
                entry["key"]
                for entry in entries
                if entry["evidence"].get("bootstrap_inherited") is True
            ],
        },
        "entries": entries,
    }


def _bootstrap_output_deltas(
    *,
    bootstrap_state: dict[str, Any],
    generated_payloads: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    deltas: dict[str, dict[str, Any]] = {}
    for output_type, bootstrap_label in OUTPUT_BOOTSTRAP_BASELINES.items():
        bootstrap_spec = BOOTSTRAP_ARTIFACTS[bootstrap_label]
        if bootstrap_spec["kind"] == "json":
            generated_payload = generated_payloads[output_type]
            bootstrap_payload = bootstrap_state["json_payloads"][bootstrap_label]
            matches = generated_payload == bootstrap_payload
            generated_bytes = json_dumps_stable(generated_payload).encode("utf-8")
        else:
            generated_text = str(generated_payloads[output_type])
            bootstrap_text = bootstrap_state["text_payloads"][bootstrap_label]
            matches = generated_text == bootstrap_text
            generated_bytes = generated_text.encode("utf-8")
        deltas[output_type] = {
            "bootstrap_label": bootstrap_label,
            "bootstrap_path": bootstrap_state["inventory"][bootstrap_label]["path"],
            "matches_bootstrap": matches,
            "generated_sha256": sha256_bytes(generated_bytes),
            "bootstrap_sha256": bootstrap_state["inventory"][bootstrap_label]["sha256"],
        }
    return deltas


def _render_progress_markdown(
    *,
    entries: list[dict[str, Any]],
    manual_results: dict[str, dict[str, Any]],
    bootstrap_state: dict[str, Any],
    bootstrap_root: Path,
    pcba_comparison_payload: dict[str, Any],
    wn18rr_comparison_payload: dict[str, Any],
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
            f"source={_entry_source_label(entry)}; manifest_backed={_format_bool(entry['evidence'].get('manifest_backed'))}."
        )
        bootstrap_reference = entry["details"].get("bootstrap_reference", {})
        if bootstrap_reference:
            lines.append(
                f"- {entry['label']} supporting bootstrap context: "
                f"`{bootstrap_reference['bootstrap_path']}`; "
                f"refresh_reason={bootstrap_reference['refresh_reason']}."
            )
    lines.append(
        "- These official-candidate rows were refreshed from fresh manual HPC execution JSONs "
        "without fabricating suite manifests."
    )

    lines.extend(
        [
            "",
            "## Local Debug / Full-Local Coverage",
        ]
    )
    for entry in local_entries:
        lines.append(
            f"- {entry['label']}: status={entry['status']}; {_metric_display(entry)}; "
            f"surface={_format_scalar(entry['evidence'].get('evidence_surface'))}; "
            f"source={_entry_source_label(entry)}; "
            f"provenance_status={entry['evidence'].get('provenance_status')}."
        )

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
                f"- Local debug remains inherited from bootstrap: "
                f"source_kind={pcba_local['evidence']['source_kind']}; "
                f"bootstrap_inherited={_format_bool(pcba_local['evidence'].get('bootstrap_inherited'))}; "
                f"source_path={pcba_local['evidence']['source_path']}."
            ),
            (
                f"- Full-local non-debug was refreshed from manual HPC execution: "
                f"status={pcba_full_local['status']}; "
                f"ap={_format_scalar(pcba_full_local['metric']['value'])}; "
                f"source_kind={pcba_full_local['evidence']['source_kind']}; "
                f"manifest_backed={_format_bool(pcba_full_local['evidence']['manifest_backed'])}; "
                f"source_path={pcba_full_local['evidence']['source_path']}."
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
                f"source={_entry_source_label(entry)}."
            )
            continue
        lines.append(
            f"- {entry['label']}: status={entry['status']}; {_metric_display(entry)}; "
            f"source={_entry_source_label(entry)}; "
            f"official_metric={_format_scalar(entry['metric']['official_metric'])}."
        )
    wn18rr_still_experimental = wn18rr_comparison_payload["comparison"]["still_experimental_reasons"]
    if wn18rr_still_experimental:
        lines.append(
            "- WN18RR remains inherited from bootstrap and stays excluded from "
            "`official_candidate_*` and `all_proven_local`."
        )
    else:
        lines.append(
            "- WN18RR is included in `all_proven_local`. "
            "Baseline dot-product path retains `relation_types_ignored=true`."
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
                f"mrr_delta={_format_scalar(comparison['metric_delta_relation_aware_minus_baseline']['mrr'])}; "
                f"hits@1_delta={_format_scalar(comparison['metric_delta_relation_aware_minus_baseline']['hits@1'])}; "
                f"hits@3_delta={_format_scalar(comparison['metric_delta_relation_aware_minus_baseline']['hits@3'])}; "
                f"hits@10_delta={_format_scalar(comparison['metric_delta_relation_aware_minus_baseline']['hits@10'])}; "
                f"metric_delta_source={comparison.get('metric_delta_source', 'debug')}."
            ),
            (
                f"- Shared remaining blockers: "
                f"{', '.join(comparison['common_remaining_promotion_blockers']) or 'none'}."
            ),
            (
                f"- Remaining experimental reasons: "
                f"{', '.join(comparison['still_experimental_reasons']) or 'none'}."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Ingestion Inputs",
            f"- bootstrap root: `{relpath_str(bootstrap_root)}`",
            (
                f"- fresh GraphMAE arXiv result: "
                f"`{manual_results['graphmae_arxiv_official_candidate']['report']['path']}`"
            ),
            (
                f"- fresh BGRL arXiv result: "
                f"`{manual_results['bgrl_arxiv_official_candidate']['report']['path']}`"
            ),
            (
                f"- fresh GraphMAE PCBA official-local result: "
                f"`{manual_results['graphmae_pcba_native_full_local_non_debug']['report']['path']}`"
            ),
        ]
    )
    for label in BOOTSTRAP_ARTIFACTS:
        lines.append(
            f"- bootstrap {label}: `{bootstrap_state['inventory'][label]['path']}`"
        )
    return "\n".join(lines) + "\n"


def _entry_source_label(entry: dict[str, Any]) -> str:
    return (
        f"{entry['evidence']['source_path']} "
        f"({entry['evidence']['source_kind']}; "
        f"provenance={entry['evidence'].get('provenance_status')})"
    )


def _metric_display(entry: dict[str, Any]) -> str:
    metric = entry["metric"]
    if metric["name"] is None or metric["value"] is None:
        return "n/a"
    return f"{metric['name']}={_format_metric_value(metric['value'])}"


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_bool(value: Any) -> str:
    return str(bool(value)).lower()


def _format_scalar(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _read_text_required(path: Path, *, label: str) -> str:
    if not path.exists():
        raise StructuredError(
            "missing_input",
            f"Missing required input: {label}.",
            {
                "label": label,
                "path": relpath_str(path),
            },
        )
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StructuredError(
            "read_error",
            f"Failed to read required input: {label}.",
            {
                "label": label,
                "path": relpath_str(path),
                "reason": f"{type(exc).__name__}: {exc}",
            },
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
