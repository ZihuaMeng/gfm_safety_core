#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from layer2_artifact_utils import (
    PROJECT_ROOT,
    StructuredError,
    apply_text_write,
    json_dumps_stable,
    load_json_required,
    print_json,
    relpath_str,
    require_mapping,
)
from layer2_hpc_plan_utils import (
    DEFAULT_PLAN_PATH,
    PLAN_SCHEMA_VERSION,
    SUCCESS_STATUSES,
    refresh_command,
    sync_bundle_command,
)


TARGET_STATUS_FRESH_COMPLETE = "fresh_hpc_rerun_complete"
TARGET_STATUS_PREEXISTING = "ready_preexisting_local_execution"
TARGET_STATUS_MISSING_PROVENANCE = "fresh_hpc_rerun_missing_provenance"
TARGET_STATUS_MISSING_OUTPUTS = "missing_outputs"
TARGET_STATUS_STALE_OR_MISMATCHED = "stale_or_mismatched_outputs"
LAUNCH_STATUS_MIXED = "mixed_target_statuses"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a generated Layer 2 HPC plan, verify which expected fresh rerun outputs "
            "exist, and optionally call the existing refresh/sync pipeline."
        )
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="Path to the generated Layer 2 HPC plan JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write an ingestion report or execute refresh/sync commands.",
    )
    parser.add_argument(
        "--run-refresh",
        action="store_true",
        help="Execute scripts/refresh_layer2_artifacts.py when readiness checks pass.",
    )
    parser.add_argument(
        "--run-sync-bundle",
        action="store_true",
        help="Execute scripts/sync_layer2_bundle.py after readiness checks pass.",
    )
    parser.add_argument(
        "--allow-partial-refresh",
        action="store_true",
        help="Allow refresh/sync to proceed when only a subset of planned targets are complete.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "baseline" / "layer2_hpc_ingestion_report.json",
        help="Output path for a persisted ingestion report when not running in dry-run mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        plan = _load_plan(args.plan)
        report = _build_report(plan=plan, plan_path=args.plan, dry_run=args.dry_run)
        actions = _execute_actions(args=args, report=report)
        report["actions"] = actions

        if not args.dry_run:
            apply_text_write(args.report_out, json_dumps_stable(report))
            report["report_path"] = relpath_str(args.report_out)

        print_json(report)
        return 0
    except StructuredError as exc:
        print_json(exc.to_payload())
        return 1


def _load_plan(plan_path: Path) -> dict[str, Any]:
    payload = require_mapping(
        load_json_required(plan_path, label="layer2_hpc_plan"),
        label="layer2_hpc_plan",
        path=plan_path,
    )
    if payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise StructuredError(
            "invalid_plan_schema",
            "Unsupported Layer 2 HPC plan schema version.",
            {
                "path": relpath_str(plan_path),
                "expected": PLAN_SCHEMA_VERSION,
                "actual": payload.get("schema_version"),
            },
        )
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise StructuredError(
            "invalid_plan_shape",
            "Layer 2 HPC plan must define a non-empty `targets` list.",
            {
                "path": relpath_str(plan_path),
                "actual_type": type(targets).__name__,
            },
        )
    return payload


def _count_statuses(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _build_report(*, plan: dict[str, Any], plan_path: Path, dry_run: bool) -> dict[str, Any]:
    inspected_targets = [_inspect_target(target) for target in plan["targets"]]
    inspected_launches = _build_launch_report(inspected_targets)

    complete_targets = [
        target for target in inspected_targets if target["readiness_phase"] == "complete"
    ]
    partial_targets = [target for target in inspected_targets if target["readiness_phase"] == "partial"]
    missing_targets = [target for target in inspected_targets if target["readiness_phase"] == "missing"]
    complete_launches = [
        launch for launch in inspected_launches if launch["readiness_phase"] == "complete"
    ]

    complete_manifest_paths = list(
        dict.fromkeys(
            launch["manifest_path"]
            for launch in complete_launches
            if isinstance(launch.get("manifest_path"), str)
        )
    )
    all_manifest_paths = list(
        dict.fromkeys(
            str(target["expected_outputs"]["manifest"])
            for target in plan["targets"]
            if isinstance(target.get("expected_outputs"), dict)
        )
    )
    target_status_counts = _count_statuses(inspected_targets)
    launch_status_counts = _count_statuses(inspected_launches)
    fully_ready = len(complete_targets) == len(inspected_targets)
    refresh_ready = bool(complete_manifest_paths)

    return {
        "plan_path": relpath_str(plan_path),
        "schema_version": plan["schema_version"],
        "dry_run": dry_run,
        "overall_status": _overall_status(inspected_targets),
        "summary": {
            "target_count": len(inspected_targets),
            "target_status_counts": target_status_counts,
            "complete_target_count": len(complete_targets),
            "partial_target_count": len(partial_targets),
            "missing_target_count": len(missing_targets),
            "fresh_hpc_rerun_complete_count": target_status_counts.get(
                TARGET_STATUS_FRESH_COMPLETE,
                0,
            ),
            "ready_preexisting_local_execution_count": target_status_counts.get(
                TARGET_STATUS_PREEXISTING,
                0,
            ),
            "fresh_hpc_rerun_missing_provenance_count": target_status_counts.get(
                TARGET_STATUS_MISSING_PROVENANCE,
                0,
            ),
            "stale_or_mismatched_outputs_count": target_status_counts.get(
                TARGET_STATUS_STALE_OR_MISMATCHED,
                0,
            ),
            "missing_outputs_count": target_status_counts.get(TARGET_STATUS_MISSING_OUTPUTS, 0),
            "launch_count": len(inspected_launches),
            "launch_status_counts": launch_status_counts,
            "complete_launch_count": len(complete_launches),
            "fully_ready_for_full_refresh": fully_ready,
            "refresh_ready_for_completed_launches": refresh_ready,
        },
        "targets": inspected_targets,
        "launches": inspected_launches,
        "prepared_commands": {
            "refresh_all_planned_manifests": refresh_command(all_manifest_paths),
            "refresh_completed_manifests": (
                refresh_command(complete_manifest_paths) if complete_manifest_paths else None
            ),
            "refresh_completed_manifests_dry_run": (
                refresh_command(complete_manifest_paths, dry_run=True)
                if complete_manifest_paths
                else None
            ),
            "sync_bundle": sync_bundle_command(),
            "sync_bundle_dry_run": sync_bundle_command(dry_run=True),
        },
        "excluded_targets": plan.get("excluded_targets", []),
        "policy_notes": plan.get("policy_notes", []),
    }


def _inspect_target(target: dict[str, Any]) -> dict[str, Any]:
    required_fields = (
        "plan_id",
        "launch_id",
        "target_name",
        "requested_target",
        "mode",
        "profile",
        "priority",
        "purpose",
    )
    for field in required_fields:
        if not isinstance(target.get(field), str) or not str(target[field]).strip():
            raise StructuredError(
                "invalid_plan_target",
                "Layer 2 HPC plan target is missing a required string field.",
                {
                    "plan_id": target.get("plan_id"),
                    "field": field,
                    "actual": target.get(field),
            },
        )

    expected_outputs = target.get("expected_outputs")
    if not isinstance(expected_outputs, dict):
        raise StructuredError(
            "invalid_plan_target",
            "Layer 2 HPC plan target is missing `expected_outputs`.",
            {
                "plan_id": target.get("plan_id"),
                "actual_type": type(expected_outputs).__name__,
            },
        )
    freshness = target.get("freshness")
    if not isinstance(freshness, dict):
        raise StructuredError(
            "invalid_plan_target",
            "Layer 2 HPC plan target is missing `freshness`.",
            {
                "plan_id": target.get("plan_id"),
                "actual_type": type(freshness).__name__,
            },
        )

    checkpoint_report = _inspect_file_path(str(expected_outputs["checkpoint"]))
    result_report = _inspect_result_json(target)
    manifest_report = _inspect_manifest(target, freshness=freshness)
    rollup_report = _inspect_rollup(target)

    integrity_issues = [
        *checkpoint_report["issues"],
        *result_report["issues"],
        *manifest_report["integrity_issues"],
        *rollup_report["issues"],
    ]
    freshness_issues = list(manifest_report["freshness_issues"])
    issues = [*integrity_issues, *freshness_issues]
    existing_required = sum(
        1
        for report in (checkpoint_report, result_report, manifest_report, rollup_report)
        if report["exists"]
    )
    if existing_required < 4:
        status = TARGET_STATUS_MISSING_OUTPUTS
        readiness_phase = "missing" if existing_required == 0 else "partial"
    elif integrity_issues:
        status = TARGET_STATUS_STALE_OR_MISMATCHED
        readiness_phase = "partial"
    elif manifest_report["freshness_contract_satisfied"]:
        status = TARGET_STATUS_FRESH_COMPLETE
        readiness_phase = "complete"
    elif manifest_report["provenance_marker_observed"]:
        status = TARGET_STATUS_MISSING_PROVENANCE
        readiness_phase = "partial"
    else:
        status = TARGET_STATUS_PREEXISTING
        readiness_phase = "partial"

    return {
        "plan_id": target["plan_id"],
        "launch_id": target["launch_id"],
        "priority": target["priority"],
        "purpose": target["purpose"],
        "target_name": target["target_name"],
        "requested_target": target["requested_target"],
        "mode": target["mode"],
        "profile": target["profile"],
        "readiness_phase": readiness_phase,
        "status": status,
        "integrity_issues": integrity_issues,
        "freshness_issues": freshness_issues,
        "issues": issues,
        "freshness": {
            "expected": freshness,
            "observed": manifest_report["freshness_observed"],
            "contract_satisfied": manifest_report["freshness_contract_satisfied"],
            "provenance_marker_observed": manifest_report["provenance_marker_observed"],
        },
        "expected_outputs": expected_outputs,
        "checks": {
            "checkpoint": checkpoint_report,
            "result_json": result_report,
            "manifest": manifest_report,
            "rollup": rollup_report,
        },
    }


def _inspect_file_path(path_rel: str) -> dict[str, Any]:
    path = (PROJECT_ROOT / path_rel).resolve()
    if not path.exists():
        return {
            "path": path_rel,
            "exists": False,
            "size_bytes": None,
            "issues": [f"missing:{path_rel}"],
        }
    return {
        "path": path_rel,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "issues": [],
    }


def _inspect_result_json(target: dict[str, Any]) -> dict[str, Any]:
    expected_outputs = target["expected_outputs"]
    path_rel = str(expected_outputs["result_json"])
    path = (PROJECT_ROOT / path_rel).resolve()
    if not path.exists():
        return {
            "path": path_rel,
            "exists": False,
            "result_status": None,
            "metric_name": None,
            "issues": [f"missing_result_json:{path_rel}"],
        }

    payload = require_mapping(
        load_json_required(path, label=f"result_json:{target['plan_id']}"),
        label=f"result_json:{target['plan_id']}",
        path=path,
    )
    issues: list[str] = []
    if payload.get("status") not in SUCCESS_STATUSES:
        issues.append(f"result_status_not_success:{payload.get('status')}")
    if payload.get("dataset") != target.get("dataset"):
        issues.append(f"result_dataset_mismatch:{payload.get('dataset')}")
    if payload.get("model") != target.get("model"):
        issues.append(f"result_model_mismatch:{payload.get('model')}")
    if payload.get("task") != target.get("task"):
        issues.append(f"result_task_mismatch:{payload.get('task')}")
    if payload.get("metric_name") != target.get("metric_name"):
        issues.append(f"result_metric_mismatch:{payload.get('metric_name')}")

    return {
        "path": path_rel,
        "exists": True,
        "result_status": payload.get("status"),
        "metric_name": payload.get("metric_name"),
        "metric_value": payload.get("metric_value"),
        "issues": issues,
    }


def _inspect_manifest(target: dict[str, Any], *, freshness: dict[str, Any]) -> dict[str, Any]:
    expected_outputs = target["expected_outputs"]
    path_rel = str(expected_outputs["manifest"])
    path = (PROJECT_ROOT / path_rel).resolve()
    if not path.exists():
        return {
            "path": path_rel,
            "exists": False,
            "integrity_issues": [f"missing_manifest:{path_rel}"],
            "freshness_issues": [],
            "issues": [f"missing_manifest:{path_rel}"],
            "freshness_contract_satisfied": False,
            "provenance_marker_observed": False,
            "freshness_observed": {},
            "fresh_export_used": None,
            "stage_export_status": None,
            "stage_eval_status": None,
            "parsed_status": None,
        }

    payload = require_mapping(
        load_json_required(path, label=f"suite_manifest:{target['plan_id']}"),
        label=f"suite_manifest:{target['plan_id']}",
        path=path,
    )
    integrity_issues: list[str] = []
    if payload.get("requested_target") != target["requested_target"]:
        integrity_issues.append(
            f"manifest_requested_target_mismatch:{payload.get('requested_target')}"
        )
    if payload.get("mode") != target["mode"]:
        integrity_issues.append(f"manifest_mode_mismatch:{payload.get('mode')}")
    if payload.get("run_type") != "execution":
        integrity_issues.append(f"manifest_run_type_not_execution:{payload.get('run_type')}")
    if payload.get("overall_status") not in SUCCESS_STATUSES:
        integrity_issues.append(
            f"manifest_overall_status_not_success:{payload.get('overall_status')}"
        )

    manifest_targets = payload.get("targets")
    target_entry: dict[str, Any] | None = None
    if not isinstance(manifest_targets, list):
        integrity_issues.append("manifest_targets_missing")
    else:
        for entry in manifest_targets:
            if isinstance(entry, dict) and entry.get("target_name") == target["target_name"]:
                target_entry = entry
                break
        if target_entry is None:
            integrity_issues.append(f"manifest_target_missing:{target['target_name']}")

    fresh_export_used = None
    stage_export_status = None
    stage_eval_status = None
    parsed_status = None
    force_export_requested = None
    target_requested_hpc_refresh = None
    target_expected_launch_id = None
    target_launch_provenance_tag = None
    target_provenance_tag = None
    target_required_fresh_export = None
    if target_entry is not None:
        if target_entry.get("profile_name") != target["profile"]:
            integrity_issues.append(f"manifest_profile_mismatch:{target_entry.get('profile_name')}")
        if target_entry.get("checkpoint_path") != expected_outputs["checkpoint"]:
            integrity_issues.append(
                f"manifest_checkpoint_mismatch:{target_entry.get('checkpoint_path')}"
            )
        if target_entry.get("out_json") != expected_outputs["result_json"]:
            integrity_issues.append(f"manifest_out_json_mismatch:{target_entry.get('out_json')}")
        export_command = _nested_get(target, "export", "command")
        if target_entry.get("command_export") != export_command:
            integrity_issues.append("manifest_export_command_mismatch")
        eval_command = _nested_get(target, "eval", "command")
        if target_entry.get("command_eval") != eval_command:
            integrity_issues.append("manifest_eval_command_mismatch")
        force_export_requested = target_entry.get("force_export_requested")
        stage_export_status = target_entry.get("stage_export_status")
        stage_eval_status = target_entry.get("stage_eval_status")
        parsed_status = target_entry.get("parsed_status")
        fresh_export_used = target_entry.get("fresh_export_used")
        target_requested_hpc_refresh = target_entry.get("requested_hpc_refresh")
        target_expected_launch_id = target_entry.get("expected_launch_id")
        target_launch_provenance_tag = target_entry.get("launch_provenance_tag")
        target_provenance_tag = target_entry.get("provenance_tag")
        target_required_fresh_export = target_entry.get("required_fresh_export")

    observed = {
        "provenance_contract_version": payload.get("provenance_contract_version"),
        "requested_hpc_refresh": payload.get("requested_hpc_refresh"),
        "expected_launch_id": payload.get("expected_launch_id"),
        "expected_run_mode": payload.get("expected_run_mode"),
        "expected_requested_target": payload.get("expected_requested_target"),
        "launch_provenance_tag": payload.get("launch_provenance_tag"),
        "required_fresh_export": payload.get("required_fresh_export"),
        "provenance_plan_path": payload.get("provenance_plan_path"),
        "provenance_plan_schema_version": payload.get("provenance_plan_schema_version"),
        "target_requested_hpc_refresh": target_requested_hpc_refresh,
        "target_expected_launch_id": target_expected_launch_id,
        "target_launch_provenance_tag": target_launch_provenance_tag,
        "target_provenance_tag": target_provenance_tag,
        "target_required_fresh_export": target_required_fresh_export,
        "force_export_requested": force_export_requested,
        "fresh_export_used": fresh_export_used,
        "stage_export_status": stage_export_status,
        "stage_eval_status": stage_eval_status,
        "parsed_status": parsed_status,
    }
    provenance_marker_observed = any(
        (
            observed["requested_hpc_refresh"] is True,
            isinstance(observed["expected_launch_id"], str)
            and bool(str(observed["expected_launch_id"]).strip()),
            isinstance(observed["launch_provenance_tag"], str)
            and bool(str(observed["launch_provenance_tag"]).strip()),
            isinstance(observed["target_provenance_tag"], str)
            and bool(str(observed["target_provenance_tag"]).strip()),
            observed["required_fresh_export"] is True,
            isinstance(observed["provenance_contract_version"], str)
            and bool(str(observed["provenance_contract_version"]).strip()),
        )
    )

    freshness_issues: list[str] = []
    if payload.get("provenance_contract_version") != freshness.get("contract_version"):
        freshness_issues.append(
            f"provenance_contract_version_mismatch:{payload.get('provenance_contract_version')}"
        )
    if payload.get("requested_hpc_refresh") is not True:
        freshness_issues.append(
            f"manifest_requested_hpc_refresh_not_true:{payload.get('requested_hpc_refresh')}"
        )
    if target_requested_hpc_refresh is not True:
        freshness_issues.append(
            f"target_requested_hpc_refresh_not_true:{target_requested_hpc_refresh}"
        )
    if payload.get("expected_launch_id") != freshness.get("expected_launch_id"):
        freshness_issues.append(
            f"manifest_expected_launch_id_mismatch:{payload.get('expected_launch_id')}"
        )
    if target_expected_launch_id != freshness.get("expected_launch_id"):
        freshness_issues.append(
            f"target_expected_launch_id_mismatch:{target_expected_launch_id}"
        )
    if payload.get("expected_run_mode") != freshness.get("expected_run_mode"):
        freshness_issues.append(
            f"manifest_expected_run_mode_mismatch:{payload.get('expected_run_mode')}"
        )
    if payload.get("expected_requested_target") != freshness.get("expected_requested_target"):
        freshness_issues.append(
            "manifest_expected_requested_target_mismatch:"
            f"{payload.get('expected_requested_target')}"
        )
    if payload.get("launch_provenance_tag") != freshness.get("launch_provenance_tag"):
        freshness_issues.append(
            f"manifest_launch_provenance_tag_mismatch:{payload.get('launch_provenance_tag')}"
        )
    if target_launch_provenance_tag != freshness.get("launch_provenance_tag"):
        freshness_issues.append(
            f"target_launch_provenance_tag_mismatch:{target_launch_provenance_tag}"
        )
    if payload.get("required_fresh_export") is not True:
        freshness_issues.append(
            f"manifest_required_fresh_export_not_true:{payload.get('required_fresh_export')}"
        )
    if target_required_fresh_export is not True:
        freshness_issues.append(
            f"target_required_fresh_export_not_true:{target_required_fresh_export}"
        )
    if target_provenance_tag != freshness.get("required_provenance_tag"):
        freshness_issues.append(f"provenance_tag_mismatch:{target_provenance_tag}")
    if force_export_requested is not True:
        freshness_issues.append(f"force_export_requested_not_true:{force_export_requested}")
    if stage_export_status != "fresh_export_success":
        freshness_issues.append(f"fresh_export_not_observed:{stage_export_status}")
    if fresh_export_used is not True:
        freshness_issues.append(f"fresh_export_used_not_true:{fresh_export_used}")
    if stage_eval_status not in SUCCESS_STATUSES:
        freshness_issues.append(f"manifest_stage_eval_not_success:{stage_eval_status}")
    if parsed_status not in SUCCESS_STATUSES:
        freshness_issues.append(f"manifest_parsed_status_not_success:{parsed_status}")

    return {
        "path": path_rel,
        "exists": True,
        "run_type": payload.get("run_type"),
        "overall_status": payload.get("overall_status"),
        "provenance_contract_version": payload.get("provenance_contract_version"),
        "requested_hpc_refresh": payload.get("requested_hpc_refresh"),
        "expected_launch_id": payload.get("expected_launch_id"),
        "launch_provenance_tag": payload.get("launch_provenance_tag"),
        "required_fresh_export": payload.get("required_fresh_export"),
        "fresh_export_used": fresh_export_used,
        "stage_export_status": stage_export_status,
        "stage_eval_status": stage_eval_status,
        "parsed_status": parsed_status,
        "provenance_tag": target_provenance_tag,
        "integrity_issues": integrity_issues,
        "freshness_issues": freshness_issues,
        "issues": [*integrity_issues, *freshness_issues],
        "freshness_contract_satisfied": not integrity_issues and not freshness_issues,
        "provenance_marker_observed": provenance_marker_observed,
        "freshness_observed": observed,
    }


def _inspect_rollup(target: dict[str, Any]) -> dict[str, Any]:
    expected_outputs = target["expected_outputs"]
    path_rel = str(expected_outputs["rollup"])
    path = (PROJECT_ROOT / path_rel).resolve()
    if not path.exists():
        return {
            "path": path_rel,
            "exists": False,
            "issues": [f"missing_rollup:{path_rel}"],
        }

    payload = require_mapping(
        load_json_required(path, label=f"suite_rollup:{target['plan_id']}"),
        label=f"suite_rollup:{target['plan_id']}",
        path=path,
    )
    issues: list[str] = []
    if payload.get("requested_target") != target["requested_target"]:
        issues.append(f"rollup_requested_target_mismatch:{payload.get('requested_target')}")
    if payload.get("mode") != target["mode"]:
        issues.append(f"rollup_mode_mismatch:{payload.get('mode')}")
    if payload.get("run_type") != "execution":
        issues.append(f"rollup_run_type_not_execution:{payload.get('run_type')}")
    if payload.get("overall_status") not in SUCCESS_STATUSES:
        issues.append(f"rollup_overall_status_not_success:{payload.get('overall_status')}")

    return {
        "path": path_rel,
        "exists": True,
        "run_type": payload.get("run_type"),
        "overall_status": payload.get("overall_status"),
        "requested_hpc_refresh": payload.get("requested_hpc_refresh"),
        "expected_launch_id": payload.get("expected_launch_id"),
        "launch_provenance_tag": payload.get("launch_provenance_tag"),
        "issues": issues,
    }


def _build_launch_report(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        grouped.setdefault(target["launch_id"], []).append(target)

    launches: list[dict[str, Any]] = []
    for launch_id, launch_targets in grouped.items():
        statuses = [target["status"] for target in launch_targets]
        readiness_phases = [target["readiness_phase"] for target in launch_targets]
        if len(set(statuses)) == 1:
            launch_status = statuses[0]
        else:
            launch_status = LAUNCH_STATUS_MIXED
        if all(phase == "complete" for phase in readiness_phases):
            readiness_phase = "complete"
        elif all(phase == "missing" for phase in readiness_phases):
            readiness_phase = "missing"
        else:
            readiness_phase = "partial"

        manifest_path = str(launch_targets[0]["expected_outputs"]["manifest"])
        launches.append(
            {
                "launch_id": launch_id,
                "status": launch_status,
                "readiness_phase": readiness_phase,
                "priority": launch_targets[0]["priority"],
                "manifest_path": manifest_path,
                "target_ids": [target["plan_id"] for target in launch_targets],
                "issues": [
                    {
                        "plan_id": target["plan_id"],
                        "status": target["status"],
                        "issues": target["issues"],
                    }
                    for target in launch_targets
                    if target["issues"]
                ],
            }
        )

    launches.sort(key=lambda item: item["launch_id"])
    return launches


def _overall_status(targets: list[dict[str, Any]]) -> str:
    statuses = {target["status"] for target in targets}
    if statuses == {TARGET_STATUS_FRESH_COMPLETE}:
        return TARGET_STATUS_FRESH_COMPLETE
    if statuses == {TARGET_STATUS_PREEXISTING}:
        return TARGET_STATUS_PREEXISTING
    if statuses == {TARGET_STATUS_MISSING_OUTPUTS}:
        return TARGET_STATUS_MISSING_OUTPUTS
    if TARGET_STATUS_STALE_OR_MISMATCHED in statuses:
        return TARGET_STATUS_STALE_OR_MISMATCHED
    if TARGET_STATUS_FRESH_COMPLETE in statuses:
        return f"partial_{TARGET_STATUS_FRESH_COMPLETE}"
    if TARGET_STATUS_MISSING_PROVENANCE in statuses:
        return TARGET_STATUS_MISSING_PROVENANCE
    return LAUNCH_STATUS_MIXED


def _execute_actions(*, args: argparse.Namespace, report: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if not args.run_refresh and not args.run_sync_bundle:
        return actions

    fully_ready = bool(_nested_get(report, "summary", "fully_ready_for_full_refresh"))
    refresh_ready = bool(_nested_get(report, "summary", "refresh_ready_for_completed_launches"))
    if not refresh_ready:
        raise StructuredError(
            "no_completed_hpc_outputs",
            "No contract-satisfying fresh HPC rerun outputs are ready for ingestion.",
            {
                "overall_status": report.get("overall_status"),
                "complete_target_count": _nested_get(report, "summary", "complete_target_count"),
            },
        )
    if not fully_ready and not args.allow_partial_refresh:
        raise StructuredError(
            "partial_hpc_outputs",
            "Planned HPC outputs are only partially complete. Re-run with --allow-partial-refresh to continue.",
            {
                "overall_status": report.get("overall_status"),
                "summary": report.get("summary"),
            },
        )

    if args.run_refresh:
        refresh_cmd = report["prepared_commands"]["refresh_completed_manifests"]
        actions.append(
            _run_prepared_command(
                label="refresh_layer2_artifacts",
                command=refresh_cmd,
                dry_run=args.dry_run,
            )
        )

    if args.run_sync_bundle:
        sync_cmd = report["prepared_commands"]["sync_bundle"]
        actions.append(
            _run_prepared_command(
                label="sync_layer2_bundle",
                command=sync_cmd,
                dry_run=args.dry_run,
            )
        )

    return actions


def _run_prepared_command(*, label: str, command: str | None, dry_run: bool) -> dict[str, Any]:
    if command is None:
        return {
            "label": label,
            "status": "skipped",
            "command": None,
            "reason": "command_not_prepared",
        }
    if dry_run:
        return {
            "label": label,
            "status": "dry_run",
            "command": command,
        }

    completed = subprocess.run(
        shlex.split(command),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_json = _parse_json_optional(completed.stdout)
    return {
        "label": label,
        "status": "success" if completed.returncode == 0 else "error",
        "command": command,
        "return_code": completed.returncode,
        "stdout_json": stdout_json,
        "stdout_excerpt": _tail_excerpt(completed.stdout),
        "stderr_excerpt": _tail_excerpt(completed.stderr),
    }


def _parse_json_optional(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _tail_excerpt(text: str, *, limit: int = 20) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= limit:
        return lines
    return lines[-limit:]


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


if __name__ == "__main__":
    raise SystemExit(main())
