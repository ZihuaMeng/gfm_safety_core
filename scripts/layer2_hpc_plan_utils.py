from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any

from layer2_artifact_utils import PROJECT_ROOT
from layer2_suite_targets import build_target_plan


PLAN_SCHEMA_VERSION = "layer2_hpc_plan/v2"
DEFAULT_PLAN_PATH = PROJECT_ROOT / "results" / "baseline" / "layer2_hpc_plan.json"
PRIORITY_ORDER = ("P0", "P1", "P2")
SUCCESS_STATUSES = frozenset({"success", "debug_success"})
FRESH_RUN_PROVENANCE_CONTRACT_VERSION = "layer2_hpc_fresh_run/v1"
FRESHNESS_EVIDENCE_REQUIREMENTS = (
    "manifest.run_type=execution",
    "manifest.requested_hpc_refresh=true",
    "manifest.expected_launch_id matches plan",
    "manifest.launch_provenance_tag matches plan",
    "manifest.required_fresh_export=true",
    "target.provenance_tag matches plan",
    "target.force_export_requested=true",
    "target.fresh_export_used=true",
    "target.stage_export_status=fresh_export_success",
    "target.stage_eval_status in success statuses",
    "target.parsed_status in success statuses",
)
LAUNCH_ENV_VAR_NAMES = {
    "contract_version": "LAYER2_HPC_PROVENANCE_CONTRACT_VERSION",
    "expected_launch_id": "LAYER2_HPC_EXPECTED_LAUNCH_ID",
    "expected_run_mode": "LAYER2_HPC_EXPECTED_RUN_MODE",
    "expected_requested_target": "LAYER2_HPC_EXPECTED_REQUESTED_TARGET",
    "launch_provenance_tag": "LAYER2_HPC_LAUNCH_PROVENANCE_TAG",
    "requested_hpc_refresh": "LAYER2_HPC_REQUESTED_REFRESH",
    "required_fresh_export": "LAYER2_HPC_REQUIRED_FRESH_EXPORT",
    "plan_path": "LAYER2_HPC_PLAN_PATH",
    "plan_schema_version": "LAYER2_HPC_PLAN_SCHEMA_VERSION",
}

HPC_TARGET_SPECS: tuple[dict[str, str], ...] = (
    {
        "plan_id": "graphmae_arxiv_official_refresh",
        "launch_id": "official_candidate_arxiv_official_refresh",
        "target_name": "graphmae_arxiv_sbert_node",
        "requested_target": "official_candidate_arxiv",
        "mode": "official",
        "profile": "default",
        "priority": "P0",
        "purpose": (
            "Refresh the execution-backed GraphMAE arXiv official-candidate result on HPC "
            "and republish the shared arXiv official suite manifest."
        ),
    },
    {
        "plan_id": "bgrl_arxiv_official_refresh",
        "launch_id": "official_candidate_arxiv_official_refresh",
        "target_name": "bgrl_arxiv_sbert_node",
        "requested_target": "official_candidate_arxiv",
        "mode": "official",
        "profile": "default",
        "priority": "P0",
        "purpose": (
            "Refresh the execution-backed BGRL arXiv official-candidate result on HPC "
            "and republish the shared arXiv official suite manifest."
        ),
    },
    {
        "plan_id": "graphmae_pcba_full_local_official_rerun",
        "launch_id": "graphmae_pcba_full_local_official_rerun",
        "target_name": "graphmae_pcba_native_graph",
        "requested_target": "graphmae_pcba_native_graph",
        "mode": "official",
        "profile": "full_local_non_debug",
        "priority": "P1",
        "purpose": (
            "Rerun the fuller GraphMAE PCBA full-local non-debug surface on HPC so the "
            "existing comparison and manifest-backed local evidence can be refreshed."
        ),
    },
)

EXCLUDED_TARGET_SPECS: tuple[dict[str, str], ...] = (
    {
        "requested_target": "wn18rr_experimental_compare",
        "priority": "P2",
        "classification": "experimental_only",
        "reason": "experimental_fence_still_enabled",
        "included_in_runnable_queue": "false",
    },
)


def plan_target_specs() -> tuple[dict[str, str], ...]:
    return HPC_TARGET_SPECS


def excluded_target_specs() -> tuple[dict[str, str], ...]:
    return EXCLUDED_TARGET_SPECS


def relpath_from_root(path: Path, *, project_root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return os.path.relpath(path.resolve(), start=project_root.resolve())


def suite_slug(requested_target: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", requested_target.strip().lower()).strip("_")
    return slug or "unknown_target"


def provenance_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def launch_provenance_tag(launch_id: str, requested_target: str, mode: str) -> str:
    return ":".join(
        (
            "layer2_hpc_fresh_run",
            provenance_slug(launch_id),
            suite_slug(requested_target),
            provenance_slug(mode),
        )
    )


def target_provenance_tag(
    launch_id: str,
    requested_target: str,
    mode: str,
    target_name: str,
    profile: str,
) -> str:
    return ":".join(
        (
            launch_provenance_tag(launch_id, requested_target, mode),
            suite_slug(target_name),
            provenance_slug(profile),
        )
    )


def suite_manifest_relpath(requested_target: str, mode: str, *, preview: bool = False) -> str:
    suffix = "_preview_manifest.json" if preview else "_manifest.json"
    return f"results/baseline/layer2_suite_{suite_slug(requested_target)}_{mode}{suffix}"


def suite_rollup_relpath(requested_target: str, mode: str, *, preview: bool = False) -> str:
    suffix = "_preview_rollup.json" if preview else "_rollup.json"
    return f"results/baseline/layer2_suite_{suite_slug(requested_target)}_{mode}{suffix}"


def launch_command(requested_target: str, mode: str, *, force_export: bool = True) -> str:
    argv = ["python", "scripts/run_layer2_suite.py", "--target", requested_target, "--mode", mode]
    if force_export:
        argv.append("--force-export")
    return shlex.join(argv)


def refresh_command(
    additional_manifests: list[str],
    *,
    dry_run: bool = False,
) -> str:
    argv = ["python", "scripts/refresh_layer2_artifacts.py"]
    for manifest_path in additional_manifests:
        argv.extend(["--additional-manifest", manifest_path])
    if dry_run:
        argv.append("--dry-run")
    return shlex.join(argv)


def sync_bundle_command(*, dry_run: bool = False) -> str:
    argv = ["python", "scripts/sync_layer2_bundle.py"]
    if dry_run:
        argv.append("--dry-run")
    return shlex.join(argv)


def launch_env(
    *,
    launch_id: str,
    requested_target: str,
    mode: str,
    plan_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, str]:
    env = {
        LAUNCH_ENV_VAR_NAMES["contract_version"]: FRESH_RUN_PROVENANCE_CONTRACT_VERSION,
        LAUNCH_ENV_VAR_NAMES["expected_launch_id"]: launch_id,
        LAUNCH_ENV_VAR_NAMES["expected_run_mode"]: mode,
        LAUNCH_ENV_VAR_NAMES["expected_requested_target"]: requested_target,
        LAUNCH_ENV_VAR_NAMES["launch_provenance_tag"]: launch_provenance_tag(
            launch_id,
            requested_target,
            mode,
        ),
        LAUNCH_ENV_VAR_NAMES["requested_hpc_refresh"]: "true",
        LAUNCH_ENV_VAR_NAMES["required_fresh_export"]: "true",
        LAUNCH_ENV_VAR_NAMES["plan_path"]: relpath_from_root(plan_path, project_root=project_root),
        LAUNCH_ENV_VAR_NAMES["plan_schema_version"]: PLAN_SCHEMA_VERSION,
    }
    return dict(sorted(env.items()))


def freshness_contract(
    *,
    launch_id: str,
    requested_target: str,
    mode: str,
    target_name: str,
    profile: str,
) -> dict[str, Any]:
    return {
        "contract_version": FRESH_RUN_PROVENANCE_CONTRACT_VERSION,
        "required_provenance_tag": target_provenance_tag(
            launch_id,
            requested_target,
            mode,
            target_name,
            profile,
        ),
        "launch_provenance_tag": launch_provenance_tag(launch_id, requested_target, mode),
        "expected_launch_id": launch_id,
        "expected_run_mode": mode,
        "expected_requested_target": requested_target,
        "expected_profile": profile,
        "expected_force_export": True,
        "requested_hpc_refresh": True,
        "required_fresh_export": True,
        "freshness_evidence_requirements": list(FRESHNESS_EVIDENCE_REQUIREMENTS),
    }


def build_plan_payload(
    *,
    project_root: Path = PROJECT_ROOT,
    plan_path: Path = DEFAULT_PLAN_PATH,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    plan_path = plan_path.resolve()
    targets: list[dict[str, Any]] = []
    combined_manifests: list[str] = []

    for spec in plan_target_specs():
        target_plan = build_target_plan(
            project_root=project_root,
            target_name=spec["target_name"],
            mode=spec["mode"],
            profile=spec["profile"],
        )
        suite_manifest = suite_manifest_relpath(spec["requested_target"], spec["mode"])
        suite_rollup = suite_rollup_relpath(spec["requested_target"], spec["mode"])
        combined_manifests.append(suite_manifest)
        launch_env_vars = launch_env(
            launch_id=spec["launch_id"],
            requested_target=spec["requested_target"],
            mode=spec["mode"],
            plan_path=plan_path,
            project_root=project_root,
        )

        targets.append(
            {
                "plan_id": spec["plan_id"],
                "launch_id": spec["launch_id"],
                "priority": spec["priority"],
                "purpose": spec["purpose"],
                "target_name": target_plan.target_name,
                "target_label": target_plan.metadata.label,
                "requested_target": spec["requested_target"],
                "mode": target_plan.mode,
                "profile": target_plan.profile,
                "dataset": target_plan.metadata.dataset_name,
                "model": target_plan.metadata.model_name,
                "task": target_plan.metadata.task_type,
                "metric_name": target_plan.metadata.metric_name,
                "artifact_group": target_plan.metadata.artifact_group,
                "classification": {
                    "official_candidate": target_plan.metadata.official_candidate,
                    "debug_local": target_plan.metadata.debug_local,
                    "experimental_only": target_plan.metadata.experimental,
                },
                "freshness": freshness_contract(
                    launch_id=spec["launch_id"],
                    requested_target=spec["requested_target"],
                    mode=target_plan.mode,
                    target_name=target_plan.target_name,
                    profile=target_plan.profile,
                ),
                "launch": {
                    "command": launch_command(spec["requested_target"], spec["mode"]),
                    "cwd": ".",
                    "env": launch_env_vars,
                },
                "export": {
                    "command": target_plan.export.shell_command(),
                    "cwd": relpath_from_root(target_plan.export.cwd, project_root=project_root),
                    "env": target_plan.export.conda_env,
                },
                "eval": {
                    "command": target_plan.eval.shell_command(),
                    "cwd": relpath_from_root(target_plan.eval.cwd, project_root=project_root),
                    "env": target_plan.eval.conda_env,
                },
                "expected_outputs": {
                    "checkpoint": relpath_from_root(
                        target_plan.checkpoint_path,
                        project_root=project_root,
                    ),
                    "result_json": relpath_from_root(
                        target_plan.out_json_path,
                        project_root=project_root,
                    ),
                    "manifest": suite_manifest,
                    "rollup": suite_rollup,
                },
                "post_run_integration": {
                    "refresh_command": refresh_command([suite_manifest]),
                    "sync_bundle_command": sync_bundle_command(),
                },
            }
        )

    unique_manifests = list(dict.fromkeys(combined_manifests))
    excluded_targets = [
        {
            "requested_target": item["requested_target"],
            "priority": item["priority"],
            "classification": item["classification"],
            "reason": item["reason"],
            "included_in_runnable_queue": False,
        }
        for item in excluded_target_specs()
    ]

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "repo_root": ".",
        "priority_order": list(PRIORITY_ORDER),
        "targets": targets,
        "target_order": [target["plan_id"] for target in targets],
        "excluded_targets": excluded_targets,
        "post_run_integration": {
            "refresh_command": refresh_command(unique_manifests),
            "sync_bundle_command": sync_bundle_command(),
            "stable_manifest_paths": unique_manifests,
        },
        "policy_notes": [
            "Stable suite manifests are the ingestion source of truth; do not rely on rolling layer2_suite_official_manifest.json aliases for cross-target evidence.",
            "WN18RR remains experimental-only and is intentionally excluded from the runnable official/local HPC queue.",
        ],
    }
