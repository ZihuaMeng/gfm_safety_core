from __future__ import annotations

from typing import Any


LOCAL_DEBUG_ENTRY_KEY = "graphmae_pcba_native_local_debug"
FULL_LOCAL_NON_DEBUG_ENTRY_KEY = "graphmae_pcba_native_full_local_non_debug"
COMPARE_PROFILE_NAME = "pcba_graph_compare"

_SUCCESS_STATUSES = frozenset({"success", "debug_success"})
_BLOCKER_ORDER = (
    "profile_execution_not_successful",
    "manifest_backed_execution_missing",
    "non_execution_or_preview_evidence",
    "debug_checkpoint_surface",
    "debug_mode_enabled",
    "split_truncation_enabled",
    "eval_batch_cap_enabled",
    "official_metric_flag_false",
    "not_locked_official_candidate_surface",
)
_BLOCKER_LABELS = {
    "profile_execution_not_successful": "profile execution is not yet successful",
    "manifest_backed_execution_missing": "manifest-backed execution evidence is missing",
    "non_execution_or_preview_evidence": "the selected evidence is preview-only or non-execution",
    "debug_checkpoint_surface": "the checkpoint surface is still a debug/local export",
    "debug_mode_enabled": "the evaluation still runs in debug mode",
    "split_truncation_enabled": "per-split truncation is still enabled",
    "eval_batch_cap_enabled": "evaluation still uses a batch cap",
    "official_metric_flag_false": "the artifact still carries official_metric=false",
    "not_locked_official_candidate_surface": "the evidence surface is local, not a locked official candidate",
}


def build_pcba_comparison_payload(
    *,
    entries: list[dict[str, Any]],
    comparison_json_path: str,
    report_path: str,
) -> dict[str, Any]:
    local_entry = _find_entry(entries, LOCAL_DEBUG_ENTRY_KEY)
    full_local_entry = _find_entry(entries, FULL_LOCAL_NON_DEBUG_ENTRY_KEY)

    local_summary = _build_profile_summary(local_entry, profile_family="local_debug")
    full_local_summary = _build_profile_summary(
        full_local_entry,
        profile_family="full_local_non_debug",
    )

    local_blockers = local_summary["remaining_blockers_toward_locked_run"]
    full_local_blockers = full_local_summary["remaining_blockers_toward_locked_run"]
    common_remaining = _sorted_blockers(set(local_blockers) & set(full_local_blockers))
    local_only = _sorted_blockers(set(local_blockers) - set(full_local_blockers))
    full_local_only = _sorted_blockers(set(full_local_blockers) - set(local_blockers))
    checkpoint_paths_distinct = _checkpoint_paths_distinct(
        local_summary["evidence"]["checkpoint_path"],
        full_local_summary["evidence"]["checkpoint_path"],
    )
    full_local_debug_checkpoint_blocker = "debug_checkpoint_surface" in full_local_blockers

    additions: list[str] = []
    if full_local_summary["mode"] != "debug":
        additions.append("non_debug_execution_surface")
    if checkpoint_paths_distinct:
        additions.append("checkpoint_provenance_separated")
    if (
        "debug_checkpoint_surface" in local_blockers
        and "debug_checkpoint_surface" not in full_local_blockers
    ):
        additions.append("debug_checkpoint_surface_removed")
    if _is_disabled(full_local_summary["truncation"]["debug_max_graphs"]) and not _is_disabled(
        local_summary["truncation"]["debug_max_graphs"]
    ):
        additions.append("debug_graph_cap_removed")
    if _is_disabled(full_local_summary["truncation"]["split_truncation"]) and not _is_disabled(
        local_summary["truncation"]["split_truncation"]
    ):
        additions.append("split_truncation_removed")
    if _is_disabled(full_local_summary["truncation"]["max_eval_batches"]) and not _is_disabled(
        local_summary["truncation"]["max_eval_batches"]
    ):
        additions.append("eval_batch_cap_removed")
    if full_local_summary["evidence"]["manifest_backed"]:
        additions.append("manifest_backed_profile_available")
    if _metric_delta(full_local_summary["metric"]["value"], local_summary["metric"]["value"]) is not None:
        additions.append("side_by_side_ap_delta_available")

    still_not_locked = _sorted_blockers(
        set(common_remaining) | set(full_local_only) | {"not_locked_official_candidate_surface"}
    )

    return {
        "schema_version": "pcba_graph_comparison/v1",
        "comparison_profile": {
            "name": COMPARE_PROFILE_NAME,
            "dataset": "ogbg-molpcba",
            "target_name": local_summary["target_name"],
            "profile_names": [
                local_summary["profile_name"],
                full_local_summary["profile_name"],
            ],
        },
        "artifacts": {
            "comparison_json": comparison_json_path,
            "protocol_report": report_path,
        },
        "checkpoint_provenance": {
            "local_debug_checkpoint_path": local_summary["evidence"]["checkpoint_path"],
            "full_local_non_debug_checkpoint_path": full_local_summary["evidence"]["checkpoint_path"],
            "checkpoint_paths_distinct": checkpoint_paths_distinct,
            "local_debug_uses_debug_checkpoint_surface": (
                "debug_checkpoint_surface" in local_blockers
            ),
            "full_local_non_debug_uses_debug_checkpoint_surface": full_local_debug_checkpoint_blocker,
            "debug_checkpoint_surface_removed_for_full_local": not full_local_debug_checkpoint_blocker,
        },
        "profiles": {
            "local_debug": local_summary,
            "full_local_non_debug": full_local_summary,
        },
        "comparison": {
            "metric_delta_full_local_non_debug_minus_local_debug": {
                "ap": _metric_delta(
                    full_local_summary["metric"]["value"],
                    local_summary["metric"]["value"],
                )
            },
            "common_remaining_locked_run_blockers": common_remaining,
            "local_debug_only_remaining_blockers": local_only,
            "full_local_non_debug_only_remaining_blockers": full_local_only,
            "full_local_non_debug_additions": additions,
            "still_not_locked_reasons": still_not_locked,
        },
    }


def build_pcba_summary_section(comparison_payload: dict[str, Any]) -> dict[str, Any]:
    comparison = comparison_payload["comparison"]
    return {
        "label": "PCBA graph dual-profile comparison",
        "profile": comparison_payload["comparison_profile"]["name"],
        "dataset": comparison_payload["comparison_profile"]["dataset"],
        "artifact_path": comparison_payload["artifacts"]["comparison_json"],
        "report_path": comparison_payload["artifacts"]["protocol_report"],
        "local_debug_entry_key": LOCAL_DEBUG_ENTRY_KEY,
        "full_local_non_debug_entry_key": FULL_LOCAL_NON_DEBUG_ENTRY_KEY,
        "checkpoint_paths_distinct": comparison_payload["checkpoint_provenance"]["checkpoint_paths_distinct"],
        "debug_checkpoint_surface_removed_for_full_local": comparison_payload["checkpoint_provenance"][
            "debug_checkpoint_surface_removed_for_full_local"
        ],
        "metric_delta_full_local_non_debug_minus_local_debug": comparison[
            "metric_delta_full_local_non_debug_minus_local_debug"
        ],
        "full_local_non_debug_additions": comparison["full_local_non_debug_additions"],
        "still_not_locked_reasons": comparison["still_not_locked_reasons"],
    }


def render_pcba_protocol_report(comparison_payload: dict[str, Any]) -> str:
    local_profile = comparison_payload["profiles"]["local_debug"]
    full_local_profile = comparison_payload["profiles"]["full_local_non_debug"]
    comparison = comparison_payload["comparison"]
    checkpoint_provenance = comparison_payload["checkpoint_provenance"]

    lines = [
        "# PCBA Graph Protocol Report",
        "",
        "## Scope",
        (
            f"- Comparison profile: `{comparison_payload['comparison_profile']['name']}` "
            f"for `{comparison_payload['comparison_profile']['dataset']}`."
        ),
        (
            f"- Compared target/profile surfaces: `{local_profile['target_name']}` "
            f"`{local_profile['profile_name']}` vs `{full_local_profile['profile_name']}`."
        ),
        (
            f"- Sources: `{local_profile['evidence']['source_path']}` and "
            f"`{full_local_profile['evidence']['source_path']}`."
        ),
        "",
        "## Checkpoint Provenance",
        (
            f"- Debug checkpoint path: "
            f"`{_format_scalar(checkpoint_provenance['local_debug_checkpoint_path'])}`."
        ),
        (
            f"- Full-local non-debug checkpoint path: "
            f"`{_format_scalar(checkpoint_provenance['full_local_non_debug_checkpoint_path'])}`."
        ),
        (
            f"- Checkpoint paths distinct="
            f"{_format_scalar(checkpoint_provenance['checkpoint_paths_distinct'])}; "
            f"full_local_debug_checkpoint_surface_removed="
            f"{_format_scalar(checkpoint_provenance['debug_checkpoint_surface_removed_for_full_local'])}."
        ),
        "",
        "## What The Current Debug Profile Proves",
        (
            f"- Status={local_profile['status']}; "
            f"ap={_format_optional_float(local_profile['metric']['value'])}; "
            f"evidence_surface={local_profile['evidence']['evidence_surface']}; "
            f"manifest_backed={_format_bool(local_profile['evidence']['manifest_backed'])}."
        ),
        (
            f"- Mode={_format_scalar(local_profile['mode'])}; "
            f"manifest_path={_format_scalar(local_profile['evidence']['source_path'])}; "
            f"checkpoint_path={_format_scalar(local_profile['evidence']['checkpoint_path'])}; "
            f"debug_mode={_format_scalar(local_profile['truncation']['debug_mode'])}; "
            f"debug_max_graphs={_format_scalar(local_profile['truncation']['debug_max_graphs'])}; "
            f"max_eval_batches={_format_scalar(local_profile['truncation']['max_eval_batches'])}; "
            f"split_truncation={_format_scalar(local_profile['truncation']['split_truncation'])}."
        ),
        (
            "- This proves the Layer 2 GraphMAE PCBA path can export a usable checkpoint, "
            "run the unified graph evaluator, and emit AP on a deterministic local slice."
        ),
        "",
        "## What The Full-Local Non-Debug Profile Adds",
        (
            f"- Status={full_local_profile['status']}; "
            f"ap={_format_optional_float(full_local_profile['metric']['value'])}; "
            f"official_metric={_format_scalar(full_local_profile['metric']['official_metric'])}; "
            f"evidence_surface={full_local_profile['evidence']['evidence_surface']}; "
            f"manifest_backed={_format_bool(full_local_profile['evidence']['manifest_backed'])}."
        ),
        (
            f"- Mode={_format_scalar(full_local_profile['mode'])}; "
            f"manifest_path={_format_scalar(full_local_profile['evidence']['source_path'])}; "
            f"checkpoint_path={_format_scalar(full_local_profile['evidence']['checkpoint_path'])}; "
            f"debug_mode={_format_scalar(full_local_profile['truncation']['debug_mode'])}; "
            f"debug_max_graphs={_format_scalar(full_local_profile['truncation']['debug_max_graphs'])}; "
            f"max_eval_batches={_format_scalar(full_local_profile['truncation']['max_eval_batches'])}; "
            f"split_truncation={_format_scalar(full_local_profile['truncation']['split_truncation'])}."
        ),
        (
            f"- Additions over the debug profile: "
            f"{_render_list(comparison['full_local_non_debug_additions'])}."
        ),
        (
            f"- AP delta (full-local non-debug minus debug): "
            f"{_format_signed_float(comparison['metric_delta_full_local_non_debug_minus_local_debug']['ap'])}."
        ),
        "",
        "## Why This Is Still Not A Locked Official Result",
        (
            f"- Shared remaining blockers: "
            f"{_render_blocker_list(comparison['common_remaining_locked_run_blockers'])}."
        ),
        (
            f"- Full-local non-debug specific blockers: "
            f"{_render_blocker_list(comparison['full_local_non_debug_only_remaining_blockers'])}."
        ),
        (
            "- The fuller profile is intentionally reported as `full_local_non_debug`: it is "
            "a stronger local execution surface, not an official locked run or official-candidate row."
        ),
        "",
        "## Layer 2 Protocol Fit",
        (
            "- PCBA now has two first-class Layer 2 artifact surfaces under the same public "
            "target: a fast truncated debug profile and a full-local non-debug comparison profile."
        ),
        (
            "- This mirrors the existing WN18RR comparison pattern, but PCBA remains outside "
            "the locked official-candidate surface until the remaining blockers are cleared."
        ),
    ]
    return "\n".join(lines) + "\n"


def _build_profile_summary(entry: dict[str, Any], *, profile_family: str) -> dict[str, Any]:
    note_fields = entry["details"].get("note_fields", {})
    evidence = entry["evidence"]
    blockers = _profile_blockers(entry)
    return {
        "entry_key": entry["key"],
        "label": entry["label"],
        "target_name": evidence.get("target_name") or "graphmae_pcba_native_graph",
        "profile_name": evidence.get("resolved_profile") or entry["details"].get("profile"),
        "profile_family": profile_family,
        "mode": evidence.get("resolved_mode") or entry["details"].get("mode"),
        "status": entry["status"],
        "evidence": {
            "source_kind": evidence.get("source_kind"),
            "source_path": evidence.get("source_path"),
            "manifest_backed": evidence.get("source_kind") == "suite_manifest",
            "fallback_backed": evidence.get("source_kind") == "result_json_fallback",
            "run_type": evidence.get("run_type"),
            "preview": evidence.get("preview"),
            "evidence_surface": evidence.get("evidence_surface"),
            "checkpoint_path": evidence.get("checkpoint_path"),
            "result_path": evidence.get("result_path"),
        },
        "truncation": {
            "debug_mode": note_fields.get("debug_mode"),
            "debug_max_graphs": note_fields.get("debug_max_graphs"),
            "batch_size": note_fields.get("batch_size"),
            "num_workers": note_fields.get("num_workers"),
            "max_train_steps": note_fields.get("max_train_steps"),
            "max_eval_batches": note_fields.get("max_eval_batches"),
            "split_truncation": note_fields.get("split_truncation"),
        },
        "metric": {
            "name": entry["metric"]["name"],
            "value": entry["metric"]["value"],
            "official_metric": entry["metric"]["official_metric"],
        },
        "evaluator_surface": _evaluator_surface(entry),
        "remaining_blockers_toward_locked_run": blockers,
    }


def _profile_blockers(entry: dict[str, Any]) -> list[str]:
    blockers: set[str] = set()
    evidence = entry["evidence"]
    note_fields = entry["details"].get("note_fields", {})

    if entry["status"] not in _SUCCESS_STATUSES:
        blockers.add("profile_execution_not_successful")
    if evidence.get("source_kind") != "suite_manifest":
        blockers.add("manifest_backed_execution_missing")
    if evidence.get("run_type") != "execution" or bool(evidence.get("preview")):
        blockers.add("non_execution_or_preview_evidence")
    if _uses_debug_checkpoint_surface(evidence.get("checkpoint_path")):
        blockers.add("debug_checkpoint_surface")
    if note_fields.get("debug_mode") is True or evidence.get("resolved_mode") == "debug":
        blockers.add("debug_mode_enabled")
    if note_fields.get("split_truncation") not in (None, "disabled"):
        blockers.add("split_truncation_enabled")
    if note_fields.get("max_eval_batches") not in (None, "none"):
        blockers.add("eval_batch_cap_enabled")
    if entry["metric"]["official_metric"] is not True:
        blockers.add("official_metric_flag_false")
    if evidence.get("evidence_surface") != "official_candidate":
        blockers.add("not_locked_official_candidate_surface")
    return _sorted_blockers(blockers)


def _evaluator_surface(entry: dict[str, Any]) -> str:
    note_fields = entry["details"].get("note_fields", {})
    profile_name = entry["evidence"].get("resolved_profile") or entry["details"].get("profile")
    if note_fields.get("debug_mode") is True or entry["evidence"].get("resolved_mode") == "debug":
        return "run_lp_ap_local_debug_subset"
    if profile_name == "full_local_non_debug":
        return "run_lp_ap_full_local_non_debug_surface"
    if entry["metric"]["official_metric"] is True:
        return "run_lp_ap_official_surface"
    return "run_lp_ap_local_surface"


def _find_entry(entries: list[dict[str, Any]], entry_key: str) -> dict[str, Any]:
    for entry in entries:
        if entry["key"] == entry_key:
            return entry
    raise ValueError(f"Missing required PCBA comparison entry: {entry_key}")


def _sorted_blockers(blockers: set[str]) -> list[str]:
    return sorted(
        blockers,
        key=lambda blocker: (
            _BLOCKER_ORDER.index(blocker) if blocker in _BLOCKER_ORDER else len(_BLOCKER_ORDER),
            blocker,
        ),
    )


def _metric_delta(lhs: Any, rhs: Any) -> float | None:
    if not isinstance(lhs, (int, float)) or not isinstance(rhs, (int, float)):
        return None
    return float(lhs) - float(rhs)


def _checkpoint_paths_distinct(lhs: Any, rhs: Any) -> bool:
    if not isinstance(lhs, str) or not isinstance(rhs, str):
        return False
    return lhs != rhs


def _uses_debug_checkpoint_surface(checkpoint_path: Any) -> bool:
    return str(checkpoint_path or "").endswith("_debug.pt")


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _is_disabled(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, int):
        return value == 0
    if isinstance(value, str):
        return value.lower() in {"none", "disabled", "0", "false"}
    return False


def _format_bool(value: Any) -> str:
    return str(bool(value)).lower()


def _format_scalar(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _format_optional_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "n/a"


def _format_signed_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):+.6f}"
    return "n/a"


def _render_list(items: list[str]) -> str:
    if not items:
        return "none"
    return ", ".join(items)


def _render_blocker_list(items: list[str]) -> str:
    if not items:
        return "none"
    return ", ".join(_BLOCKER_LABELS.get(item, item) for item in items)
