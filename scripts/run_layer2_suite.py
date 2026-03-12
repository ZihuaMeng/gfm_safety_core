from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from layer2_hpc_plan_utils import LAUNCH_ENV_VAR_NAMES, target_provenance_tag
from layer2_suite_targets import build_target_plan, expand_target, get_alignment_audit, target_help_text


SUCCESS_STATUSES = frozenset({"success", "debug_success"})
EXPORT_SUCCESS_STATUSES = frozenset({"skipped_existing", "fresh_export_success"})


@dataclass(frozen=True)
class StageRun:
    return_code: int | None
    log_path: Path | None
    error_text: str | None = None


@dataclass(frozen=True)
class CheckpointSnapshot:
    exists: bool
    mtime: str | None
    size_bytes: int | None
    sha256: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _relpath(project_root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _manifest_path(project_root: Path, mode: str) -> Path:
    return project_root / "results" / "baseline" / f"layer2_suite_{mode}_manifest.json"


def _rollup_path(project_root: Path, mode: str) -> Path:
    return project_root / "results" / "baseline" / f"layer2_suite_{mode}_rollup.json"


def _preview_manifest_path(project_root: Path, mode: str) -> Path:
    return project_root / "results" / "baseline" / f"layer2_suite_{mode}_preview_manifest.json"


def _preview_rollup_path(project_root: Path, mode: str) -> Path:
    return project_root / "results" / "baseline" / f"layer2_suite_{mode}_preview_rollup.json"


def _suite_slug(requested_target: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", requested_target.strip().lower()).strip("_")
    return slug or "unknown_target"


def _target_run_slug(target_name: str, profile: str | None) -> str:
    base = target_name.strip().lower()
    if profile is not None and profile != "default":
        base = f"{base}_{profile}"
    slug = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return slug or "unknown_target"


def _suite_manifest_path(project_root: Path, mode: str, requested_target: str) -> Path:
    slug = _suite_slug(requested_target)
    return project_root / "results" / "baseline" / f"layer2_suite_{slug}_{mode}_manifest.json"


def _suite_rollup_path(project_root: Path, mode: str, requested_target: str) -> Path:
    slug = _suite_slug(requested_target)
    return project_root / "results" / "baseline" / f"layer2_suite_{slug}_{mode}_rollup.json"


def _suite_preview_manifest_path(project_root: Path, mode: str, requested_target: str) -> Path:
    slug = _suite_slug(requested_target)
    return project_root / "results" / "baseline" / f"layer2_suite_{slug}_{mode}_preview_manifest.json"


def _suite_preview_rollup_path(project_root: Path, mode: str, requested_target: str) -> Path:
    slug = _suite_slug(requested_target)
    return project_root / "results" / "baseline" / f"layer2_suite_{slug}_{mode}_preview_rollup.json"


def _command_hash(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def _parse_bool_env(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _load_hpc_refresh_context() -> dict[str, object] | None:
    raw = {key: os.environ.get(env_name) for key, env_name in LAUNCH_ENV_VAR_NAMES.items()}
    requested_hpc_refresh = _parse_bool_env(raw["requested_hpc_refresh"])
    required_fresh_export = _parse_bool_env(raw["required_fresh_export"])
    has_marker = any(
        [
            requested_hpc_refresh is True,
            required_fresh_export is True,
            bool(raw["contract_version"]),
            bool(raw["expected_launch_id"]),
            bool(raw["expected_run_mode"]),
            bool(raw["expected_requested_target"]),
            bool(raw["launch_provenance_tag"]),
            bool(raw["plan_path"]),
            bool(raw["plan_schema_version"]),
        ]
    )
    if not has_marker:
        return None
    return {
        "provenance_contract_version": raw["contract_version"],
        "requested_hpc_refresh": bool(requested_hpc_refresh),
        "expected_launch_id": raw["expected_launch_id"],
        "expected_run_mode": raw["expected_run_mode"],
        "expected_requested_target": raw["expected_requested_target"],
        "launch_provenance_tag": raw["launch_provenance_tag"],
        "required_fresh_export": bool(required_fresh_export),
        "provenance_plan_path": raw["plan_path"],
        "provenance_plan_schema_version": raw["plan_schema_version"],
    }


def _apply_hpc_refresh_record(
    record: dict[str, object],
    *,
    hpc_refresh_context: dict[str, object] | None,
) -> None:
    if hpc_refresh_context is None:
        return

    fields = {
        "provenance_contract_version": hpc_refresh_context.get("provenance_contract_version"),
        "requested_hpc_refresh": hpc_refresh_context.get("requested_hpc_refresh"),
        "expected_launch_id": hpc_refresh_context.get("expected_launch_id"),
        "required_fresh_export": hpc_refresh_context.get("required_fresh_export"),
    }
    launch_provenance_tag = hpc_refresh_context.get("launch_provenance_tag")
    if isinstance(launch_provenance_tag, str) and launch_provenance_tag.strip():
        fields["launch_provenance_tag"] = launch_provenance_tag

    launch_id = hpc_refresh_context.get("expected_launch_id")
    requested_target = hpc_refresh_context.get("expected_requested_target")
    expected_run_mode = hpc_refresh_context.get("expected_run_mode")
    target_name = record.get("target_name")
    profile_name = record.get("profile_name")
    if all(
        isinstance(value, str) and value.strip()
        for value in (launch_id, requested_target, expected_run_mode, target_name, profile_name)
    ):
        fields["provenance_tag"] = target_provenance_tag(
            str(launch_id),
            str(requested_target),
            str(expected_run_mode),
            str(target_name),
            str(profile_name),
        )

    for key, value in fields.items():
        if value is not None:
            record[key] = value


def _suite_hpc_refresh_fields(
    hpc_refresh_context: dict[str, object] | None,
) -> dict[str, object]:
    if hpc_refresh_context is None:
        return {}

    fields = {
        "provenance_contract_version": hpc_refresh_context.get("provenance_contract_version"),
        "requested_hpc_refresh": hpc_refresh_context.get("requested_hpc_refresh"),
        "expected_launch_id": hpc_refresh_context.get("expected_launch_id"),
        "expected_run_mode": hpc_refresh_context.get("expected_run_mode"),
        "expected_requested_target": hpc_refresh_context.get("expected_requested_target"),
        "launch_provenance_tag": hpc_refresh_context.get("launch_provenance_tag"),
        "required_fresh_export": hpc_refresh_context.get("required_fresh_export"),
        "provenance_plan_path": hpc_refresh_context.get("provenance_plan_path"),
        "provenance_plan_schema_version": hpc_refresh_context.get("provenance_plan_schema_version"),
    }
    return {key: value for key, value in fields.items() if value is not None}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_mtime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _snapshot_checkpoint(path: Path, *, include_sha256: bool) -> CheckpointSnapshot:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return CheckpointSnapshot(False, None, None, None)

    sha256 = _file_sha256(path) if include_sha256 else None
    return CheckpointSnapshot(
        exists=True,
        mtime=_format_mtime(stat.st_mtime),
        size_bytes=stat.st_size,
        sha256=sha256,
    )


def _apply_snapshot(record: dict[str, object], prefix: str, snapshot: CheckpointSnapshot) -> None:
    record[f"checkpoint_exists_{prefix}"] = snapshot.exists
    record[f"checkpoint_mtime_{prefix}"] = snapshot.mtime
    record[f"checkpoint_size_bytes_{prefix}"] = snapshot.size_bytes
    if prefix == "after":
        record["checkpoint_size_bytes"] = snapshot.size_bytes
        record["checkpoint_sha256"] = snapshot.sha256


def _default_record(
    project_root: Path,
    plan,
    *,
    hpc_refresh_context: dict[str, object] | None,
) -> dict[str, object]:
    metadata = plan.metadata
    command_export = plan.export.shell_command()
    command_eval = plan.eval.shell_command()
    record = {
        "target_name": plan.target_name,
        "mode": plan.mode,
        "profile": plan.profile,
        "profile_name": plan.profile,
        "target_label": metadata.label,
        **metadata.to_manifest_metadata(),
        "force_export_requested": False,
        "fresh_export_attempted": False,
        "fresh_export_used": False,
        "stage_export_status": "pending",
        "stage_eval_status": "pending",
        "return_code": None,
        "return_code_export": None,
        "return_code_eval": None,
        "checkpoint_path": _relpath(project_root, plan.checkpoint_path),
        "checkpoint_exists_before": None,
        "checkpoint_exists_after": None,
        "checkpoint_mtime_before": None,
        "checkpoint_mtime_after": None,
        "checkpoint_size_bytes_before": None,
        "checkpoint_size_bytes_after": None,
        "checkpoint_size_bytes": None,
        "checkpoint_sha256": None,
        "out_json": _relpath(project_root, plan.out_json_path),
        "command_export": command_export,
        "command_eval": command_eval,
        "command_export_hash": _command_hash(command_export),
        "command_eval_hash": _command_hash(command_eval),
        "cwd_export": str(plan.export.cwd),
        "cwd_eval": str(plan.eval.cwd),
        "env_export": plan.export.conda_env,
        "env_eval": plan.eval.conda_env,
        "log_path_export": None,
        "log_path_eval": None,
        "parsed_metric_name": None,
        "parsed_metric_value": None,
        "parsed_status": None,
        "notes": "",
    }
    _apply_hpc_refresh_record(record, hpc_refresh_context=hpc_refresh_context)
    return record


def _append_note(record: dict[str, object], note: str) -> None:
    note = note.strip()
    if not note:
        return
    existing = str(record.get("notes") or "")
    record["notes"] = f"{existing}; {note}" if existing else note


def _apply_profile_result_overrides(record: dict[str, object], *, plan) -> None:
    if plan.target_name == "graphmae_pcba_native_graph" and plan.profile == "full_local_non_debug":
        for note in (
            "full_local_non_debug=true",
            "dedicated_non_debug_checkpoint=true",
            "debug_mode=false",
            "debug_max_graphs=none",
            "max_train_steps=32",
            "max_eval_batches=none",
            "split_truncation=disabled",
            "official_metric=false",
            "locked_official=false",
        ):
            _append_note(record, note)


def _run_stage(*, spec, log_dir: Path, target_name: str, stage_name: str) -> StageRun:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{target_name}_{stage_name}_{_timestamp_slug()}.log"
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"$ {spec.shell_command()}\n")
            handle.write(f"[cwd] {spec.cwd}\n")
            handle.write(f"[conda_env] {spec.conda_env}\n\n")
            handle.flush()
            completed = subprocess.run(
                spec.argv,
                cwd=spec.cwd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    except OSError as exc:
        return StageRun(return_code=None, log_path=log_path, error_text=f"{type(exc).__name__}: {exc}")
    return StageRun(return_code=completed.returncode, log_path=log_path)


def _parse_eval_json(out_json_path: Path) -> dict[str, object]:
    if not out_json_path.exists():
        return {}
    try:
        return json.loads(out_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_parse_error": f"{type(exc).__name__}: {exc}"}


def _run_alignment_audit(
    *,
    project_root: Path,
    audit_plan,
    target_name: str,
    log_dir: Path,
    record: dict[str, object],
    dry_run: bool,
) -> bool:
    """Run alignment audit if required. Returns True if target may proceed."""
    if dry_run:
        record["alignment_audit_status"] = "dry_run"
        record["alignment_audit_json"] = _relpath(project_root, audit_plan.out_json_path)
        return True

    print(f"[audit] running alignment audit for {target_name}")
    audit_run = _run_stage(
        spec=audit_plan.command,
        log_dir=log_dir,
        target_name=target_name,
        stage_name="alignment_audit",
    )
    record["alignment_audit_log_path"] = _relpath(project_root, audit_run.log_path)

    if audit_run.error_text is not None:
        record["alignment_audit_status"] = "error"
        _append_note(record, f"alignment_audit_error={audit_run.error_text}")
        return False

    # Try to parse audit JSON regardless of return code
    audit_data: dict = {}
    if audit_plan.out_json_path.exists():
        try:
            audit_data = json.loads(audit_plan.out_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    if audit_run.return_code != 0:
        audit_status = audit_data.get("status", "error")
        record["alignment_audit_status"] = audit_status
        if audit_data:
            record["alignment_audit_json"] = _relpath(project_root, audit_plan.out_json_path)
        missing = audit_data.get("missing_pieces", [])
        if missing:
            _append_note(record, f"alignment_missing={'; '.join(missing)}")
        else:
            _append_note(record, f"alignment_audit_return_code={audit_run.return_code}")
        return False

    if not audit_data:
        record["alignment_audit_status"] = "error"
        _append_note(record, "alignment audit rc=0 but JSON not created or unparseable")
        return False

    audit_status = audit_data.get("status", "error")
    record["alignment_audit_status"] = audit_status
    record["alignment_audit_json"] = _relpath(project_root, audit_plan.out_json_path)

    if audit_status != "success":
        missing = audit_data.get("missing_pieces", [])
        if missing:
            _append_note(record, f"alignment_missing={'; '.join(missing)}")
        return False

    print(f"[audit] alignment audit passed for {target_name}")
    return True


def _print_target_preview(record: dict[str, object]) -> None:
    target_name = str(record["target_name"])
    profile_name = record.get("profile_name")
    if profile_name is not None and profile_name != "default":
        print(f"[target] {target_name} [{profile_name}]")
    else:
        print(f"[target] {target_name}")
    if record.get("model") and record.get("dataset") and record.get("task"):
        print(f"  spec={record['model']}:{record['dataset']}:{record['task']}")
    print(f"  profile={record['profile_name']}")
    print(f"  export_status={record['stage_export_status']}")
    print(f"  export_env={record['env_export']}")
    print(f"  export_cwd={record['cwd_export']}")
    print(f"  export_cmd={record['command_export']}")
    print(f"  eval_status={record['stage_eval_status']}")
    print(f"  eval_env={record['env_eval']}")
    print(f"  eval_cwd={record['cwd_eval']}")
    print(f"  eval_cmd={record['command_eval']}")
    if record.get("checkpoint_path"):
        print(f"  checkpoint={record['checkpoint_path']}")
    if record.get("out_json"):
        print(f"  out_json={record['out_json']}")
    print(f"  fresh_export_used={record['fresh_export_used']}")
    if record.get("alignment_audit_status"):
        print(f"  alignment_audit_status={record['alignment_audit_status']}")
    if record.get("notes"):
        print(f"  notes={record['notes']}")


def _overall_status(target_records: list[dict[str, object]], *, dry_run: bool) -> str:
    if dry_run:
        return "dry_run"

    blocked = False
    failed = False
    for record in target_records:
        export_status = str(record.get("stage_export_status") or "")
        eval_status = str(record.get("stage_eval_status") or "")
        parsed_status = str(record.get("parsed_status") or "")
        if export_status not in EXPORT_SUCCESS_STATUSES:
            failed = True
        if eval_status == "blocked" or parsed_status == "blocked":
            blocked = True
        elif eval_status not in SUCCESS_STATUSES:
            failed = True

    if failed:
        return "partial_failure"
    if blocked:
        return "blocked"
    return "success"


def _build_rollup(
    *,
    suite_name: str,
    mode: str,
    started_at: str,
    finished_at: str,
    overall_status: str,
    targets: list[dict[str, object]],
    hpc_refresh_context: dict[str, object] | None,
) -> dict[str, object]:
    rollup = {
        "suite_name": suite_name,
        "mode": mode,
        "started_at": started_at,
        "finished_at": finished_at,
        "overall_status": overall_status,
        "targets": [],
    }
    rollup.update(_suite_hpc_refresh_fields(hpc_refresh_context))
    for target in targets:
        target_summary = {
            "target_name": target["target_name"],
            "target_label": target.get("target_label"),
            "mode": target["mode"],
            "profile_name": target["profile_name"],
            "dataset": target.get("dataset"),
            "model": target.get("model"),
            "task": target.get("task"),
            "artifact_group": target.get("artifact_group"),
            "registry_metric_name": target.get("registry_metric_name"),
            "parsed_status": target["parsed_status"],
            "parsed_metric_name": target["parsed_metric_name"],
            "parsed_metric_value": target["parsed_metric_value"],
            "checkpoint_path": target["checkpoint_path"],
            "fresh_export_used": target["fresh_export_used"],
            "notes": target["notes"],
        }
        for key in (
            "requested_hpc_refresh",
            "expected_launch_id",
            "launch_provenance_tag",
            "provenance_tag",
            "required_fresh_export",
        ):
            if key in target:
                target_summary[key] = target[key]
        rollup["targets"].append(target_summary)
    return rollup


def _write_suite_outputs(
    *,
    project_root: Path,
    mode: str,
    requested_target: str,
    suite_name: str,
    started_at: str,
    targets: list[dict[str, object]],
    dry_run: bool,
    hpc_refresh_context: dict[str, object] | None,
) -> tuple[Path, Path]:
    finished_at = _utc_now()
    overall_status = _overall_status(targets, dry_run=dry_run)
    run_type = "preview" if dry_run else "execution"
    if dry_run:
        manifest_path = _preview_manifest_path(project_root, mode)
        rollup_path = _preview_rollup_path(project_root, mode)
        suite_manifest_path = _suite_preview_manifest_path(project_root, mode, requested_target)
        suite_rollup_path = _suite_preview_rollup_path(project_root, mode, requested_target)
    else:
        manifest_path = _manifest_path(project_root, mode)
        rollup_path = _rollup_path(project_root, mode)
        suite_manifest_path = _suite_manifest_path(project_root, mode, requested_target)
        suite_rollup_path = _suite_rollup_path(project_root, mode, requested_target)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "suite_name": suite_name,
        "requested_target": requested_target,
        "mode": mode,
        "run_type": run_type,
        "preview": dry_run,
        "started_at": started_at,
        "finished_at": finished_at,
        "overall_status": overall_status,
        "target_count": len(targets),
        "target_names": [str(target.get("target_name")) for target in targets],
        "output_paths": {
            "primary_manifest": _relpath(project_root, manifest_path),
            "primary_rollup": _relpath(project_root, rollup_path),
            "suite_manifest": _relpath(project_root, suite_manifest_path),
            "suite_rollup": _relpath(project_root, suite_rollup_path),
        },
        "targets": targets,
    }
    manifest_payload.update(_suite_hpc_refresh_fields(hpc_refresh_context))
    rollup_payload = _build_rollup(
        suite_name=suite_name,
        mode=mode,
        started_at=started_at,
        finished_at=finished_at,
        overall_status=overall_status,
        targets=targets,
        hpc_refresh_context=hpc_refresh_context,
    )
    rollup_payload["requested_target"] = requested_target
    rollup_payload["run_type"] = run_type
    rollup_payload["preview"] = dry_run
    rollup_payload["target_count"] = len(targets)
    rollup_payload["output_paths"] = {
        "primary_manifest": _relpath(project_root, manifest_path),
        "primary_rollup": _relpath(project_root, rollup_path),
        "suite_manifest": _relpath(project_root, suite_manifest_path),
        "suite_rollup": _relpath(project_root, suite_rollup_path),
    }

    manifest_text = json.dumps(manifest_payload, indent=2) + "\n"
    rollup_text = json.dumps(rollup_payload, indent=2) + "\n"
    for path in (manifest_path, suite_manifest_path):
        path.write_text(manifest_text, encoding="utf-8")
    for path in (rollup_path, suite_rollup_path):
        path.write_text(rollup_text, encoding="utf-8")
    return manifest_path, rollup_path


def _should_continue(record: dict[str, object]) -> bool:
    export_status = str(record.get("stage_export_status") or "")
    eval_status = str(record.get("stage_eval_status") or "")
    if export_status not in EXPORT_SUCCESS_STATUSES | {"dry_run"}:
        return False
    if eval_status in SUCCESS_STATUSES | {"dry_run"}:
        return True
    return False


def _blocked_record(target_name: str, mode: str, message: str) -> dict[str, object]:
    return {
        "target_name": target_name,
        "mode": mode,
        "profile": None,
        "profile_name": None,
        "force_export_requested": False,
        "fresh_export_attempted": False,
        "fresh_export_used": False,
        "stage_export_status": "blocked",
        "stage_eval_status": "blocked",
        "return_code": None,
        "return_code_export": None,
        "return_code_eval": None,
        "checkpoint_path": None,
        "checkpoint_exists_before": None,
        "checkpoint_exists_after": None,
        "checkpoint_mtime_before": None,
        "checkpoint_mtime_after": None,
        "checkpoint_size_bytes_before": None,
        "checkpoint_size_bytes_after": None,
        "checkpoint_size_bytes": None,
        "checkpoint_sha256": None,
        "out_json": None,
        "command_export": None,
        "command_eval": None,
        "command_export_hash": None,
        "command_eval_hash": None,
        "cwd_export": None,
        "cwd_eval": None,
        "env_export": None,
        "env_eval": None,
        "log_path_export": None,
        "log_path_eval": None,
        "parsed_metric_name": None,
        "parsed_metric_value": None,
        "parsed_status": "blocked",
        "notes": message,
    }


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the Layer 2 suite for debug and official-candidate targets with checkpoint provenance.",
        epilog=target_help_text(),
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--mode", required=True, choices=["debug", "official"])
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--continue-on-error", action="store_true", dest="continue_on_error")
    parser.add_argument("--force-export", action="store_true", dest="force_export")
    parser.add_argument("--project-root", default=str(default_root))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    started_at = _utc_now()
    suite_name = f"layer2_suite:{args.target}"
    log_dir = project_root / "logs" / "layer2_suite"
    manifest_targets: list[dict[str, object]] = []
    hpc_refresh_context = _load_hpc_refresh_context()

    try:
        expanded_targets = expand_target(args.target, args.mode)
    except ValueError as exc:
        manifest_targets.append(_blocked_record(args.target, args.mode, str(exc)))
        manifest_path, rollup_path = _write_suite_outputs(
            project_root=project_root,
            mode=args.mode,
            requested_target=args.target,
            suite_name=suite_name,
            started_at=started_at,
            targets=manifest_targets,
            dry_run=False,
            hpc_refresh_context=hpc_refresh_context,
        )
        print(str(exc), file=sys.stderr)
        print(f"manifest={manifest_path}")
        print(f"rollup={rollup_path}")
        return 2

    for expanded_target in expanded_targets:
        target_name = expanded_target.target_name
        target_mode = expanded_target.mode
        profile = expanded_target.profile
        run_slug = _target_run_slug(target_name, profile)
        plan = build_target_plan(
            project_root=project_root,
            target_name=target_name,
            mode=target_mode,
            profile=profile,
        )
        record = _default_record(
            project_root,
            plan,
            hpc_refresh_context=hpc_refresh_context,
        )
        record["force_export_requested"] = args.force_export

        snapshot_before = _snapshot_checkpoint(plan.checkpoint_path, include_sha256=False)
        _apply_snapshot(record, "before", snapshot_before)

        # -- Alignment audit (if required for this target) --
        audit_plan = get_alignment_audit(project_root, target_name)
        if audit_plan is not None:
            audit_ok = _run_alignment_audit(
                project_root=project_root,
                audit_plan=audit_plan,
                target_name=target_name,
                log_dir=log_dir,
                record=record,
                dry_run=args.dry_run,
            )
            if not audit_ok:
                record["stage_export_status"] = "skipped_audit_failed"
                record["stage_eval_status"] = "skipped_audit_failed"
                manifest_targets.append(record)
                _write_suite_outputs(
                    project_root=project_root,
                    mode=args.mode,
                    requested_target=args.target,
                    suite_name=suite_name,
                    started_at=started_at,
                    targets=manifest_targets,
                    dry_run=False,
                    hpc_refresh_context=hpc_refresh_context,
                )
                _print_target_preview(record)
                if not args.continue_on_error:
                    manifest_path, rollup_path = _write_suite_outputs(
                        project_root=project_root,
                        mode=args.mode,
                        requested_target=args.target,
                        suite_name=suite_name,
                        started_at=started_at,
                        targets=manifest_targets,
                        dry_run=False,
                        hpc_refresh_context=hpc_refresh_context,
                    )
                    print(f"manifest={manifest_path}")
                    print(f"rollup={rollup_path}")
                    return 1
                continue

        if args.dry_run:
            if snapshot_before.exists and not args.force_export:
                record["stage_export_status"] = "skipped_existing"
                _append_note(record, f"checkpoint_exists={_relpath(project_root, plan.checkpoint_path)}")
            else:
                record["stage_export_status"] = "dry_run"
                if args.force_export:
                    _append_note(record, "force_export=true")
            record["stage_eval_status"] = "dry_run"
            snapshot_after = _snapshot_checkpoint(plan.checkpoint_path, include_sha256=True)
            _apply_snapshot(record, "after", snapshot_after)
            manifest_targets.append(record)
            _write_suite_outputs(
                project_root=project_root,
                mode=args.mode,
                requested_target=args.target,
                suite_name=suite_name,
                started_at=started_at,
                targets=manifest_targets,
                dry_run=True,
                hpc_refresh_context=hpc_refresh_context,
            )
            _print_target_preview(record)
            continue

        if snapshot_before.exists and not args.force_export:
            record["stage_export_status"] = "skipped_existing"
            _append_note(record, f"checkpoint_exists={_relpath(project_root, plan.checkpoint_path)}")
        else:
            record["fresh_export_attempted"] = True
            export_result = _run_stage(
                spec=plan.export,
                log_dir=log_dir,
                target_name=run_slug,
                stage_name="export",
            )
            record["return_code_export"] = export_result.return_code
            record["log_path_export"] = _relpath(project_root, export_result.log_path)
            if export_result.error_text is not None:
                record["stage_export_status"] = "fresh_export_error"
                _append_note(record, export_result.error_text)
            elif export_result.return_code != 0:
                record["stage_export_status"] = "fresh_export_error"
                _append_note(record, f"export_return_code={export_result.return_code}")
            else:
                record["stage_export_status"] = "fresh_export_success"
                record["fresh_export_used"] = True

        snapshot_after = _snapshot_checkpoint(plan.checkpoint_path, include_sha256=True)
        _apply_snapshot(record, "after", snapshot_after)

        if record["stage_export_status"] == "fresh_export_success" and not snapshot_after.exists:
            record["stage_export_status"] = "fresh_export_error"
            record["fresh_export_used"] = False
            _append_note(record, "export finished but checkpoint was not created")

        if record["stage_export_status"] not in EXPORT_SUCCESS_STATUSES:
            record["stage_eval_status"] = "skipped_due_to_export_error"
            record["return_code"] = record["return_code_export"]
            manifest_targets.append(record)
            _write_suite_outputs(
                project_root=project_root,
                mode=args.mode,
                requested_target=args.target,
                suite_name=suite_name,
                started_at=started_at,
                targets=manifest_targets,
                dry_run=False,
                hpc_refresh_context=hpc_refresh_context,
            )
            _print_target_preview(record)
            if not args.continue_on_error:
                manifest_path, rollup_path = _write_suite_outputs(
                    project_root=project_root,
                    mode=args.mode,
                    requested_target=args.target,
                    suite_name=suite_name,
                    started_at=started_at,
                    targets=manifest_targets,
                    dry_run=False,
                    hpc_refresh_context=hpc_refresh_context,
                )
                print(f"manifest={manifest_path}")
                print(f"rollup={rollup_path}")
                return 1
            continue

        eval_result = _run_stage(
            spec=plan.eval,
            log_dir=log_dir,
            target_name=run_slug,
            stage_name="eval",
        )
        record["return_code_eval"] = eval_result.return_code
        record["return_code"] = eval_result.return_code
        record["log_path_eval"] = _relpath(project_root, eval_result.log_path)

        if eval_result.error_text is not None:
            record["stage_eval_status"] = "error"
            _append_note(record, eval_result.error_text)
        elif eval_result.return_code != 0:
            record["stage_eval_status"] = "error"
            _append_note(record, f"eval_return_code={eval_result.return_code}")
        else:
            parsed = _parse_eval_json(plan.out_json_path)
            if not parsed:
                record["stage_eval_status"] = "error"
                _append_note(record, "eval completed but out_json was not created")
            elif "_parse_error" in parsed:
                record["stage_eval_status"] = "error"
                _append_note(record, str(parsed["_parse_error"]))
            else:
                record["parsed_status"] = parsed.get("status")
                record["parsed_metric_name"] = parsed.get("metric_name")
                record["parsed_metric_value"] = parsed.get("metric_value")
                record["stage_eval_status"] = (
                    parsed.get("status") if parsed.get("status") is not None else "success"
                )
                parsed_notes = parsed.get("notes")
                if parsed_notes:
                    _append_note(record, f"result_notes={parsed_notes}")
                _apply_profile_result_overrides(record, plan=plan)

        manifest_targets.append(record)
        _write_suite_outputs(
            project_root=project_root,
            mode=args.mode,
            requested_target=args.target,
            suite_name=suite_name,
            started_at=started_at,
            targets=manifest_targets,
            dry_run=False,
            hpc_refresh_context=hpc_refresh_context,
        )
        _print_target_preview(record)

        if not _should_continue(record) and not args.continue_on_error:
            manifest_path, rollup_path = _write_suite_outputs(
                project_root=project_root,
                mode=args.mode,
                requested_target=args.target,
                suite_name=suite_name,
                started_at=started_at,
                targets=manifest_targets,
                dry_run=False,
                hpc_refresh_context=hpc_refresh_context,
            )
            print(f"manifest={manifest_path}")
            print(f"rollup={rollup_path}")
            return 1

    manifest_path, rollup_path = _write_suite_outputs(
        project_root=project_root,
        mode=args.mode,
        requested_target=args.target,
        suite_name=suite_name,
        started_at=started_at,
        targets=manifest_targets,
        dry_run=args.dry_run,
        hpc_refresh_context=hpc_refresh_context,
    )
    print(f"manifest={manifest_path}")
    print(f"rollup={rollup_path}")

    overall_status = _overall_status(manifest_targets, dry_run=args.dry_run)
    return 0 if overall_status in {"success", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
