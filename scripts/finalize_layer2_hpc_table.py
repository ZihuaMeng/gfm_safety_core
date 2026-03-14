#!/usr/bin/env python3
"""finalize_layer2_hpc_table.py — Layer 2 HPC closure finalization.

Post-processes Layer 2 artifact pipeline outputs to:
1. Resolve WN18RR evidence with fresh-over-bootstrap precedence
2. Harmonize WN18RR semantic alignment caveats across surfaces
3. Export a flat all-dataset CSV (outputs/layer2_results_table.csv)

Diff summary: New script — no existing files modified.
Smoke command:
    cd ~/projects/gfm_safety_core/scripts && python finalize_layer2_hpc_table.py --dry-run
Expected key output lines:
    "mode": "finalize_layer2_hpc_table"
    "harmonization": {"semantic_alignment_verified": true, ...}
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from layer2_artifact_utils import (
    PROJECT_ROOT,
    json_dumps_stable,
    load_json_optional,
    load_json_required,
    print_json,
    relpath_str,
    render_text_write_plan,
    apply_text_write,
    require_mapping,
)
from refresh_layer2_artifacts import _render_meeting_table


# ---------------------------------------------------------------------------
# WN18RR evidence resolution: prefer fresh current-repo over bootstrap
# ---------------------------------------------------------------------------

WN18RR_EVIDENCE_ARTIFACTS: dict[str, str] = {
    "wn18rr_alignment_audit": "results/baseline/wn18rr_alignment_audit.json",
    "wn18rr_semantic_alignment_audit": (
        "results/baseline/wn18rr_semantic_alignment_audit.json"
    ),
    "wn18rr_official_manifest": (
        "results/baseline/"
        "layer2_suite_wn18rr_experimental_compare_official_manifest.json"
    ),
    "wn18rr_debug_manifest": (
        "results/baseline/"
        "layer2_suite_wn18rr_experimental_compare_debug_manifest.json"
    ),
    "wn18rr_link_comparison": "results/baseline/wn18rr_link_comparison.json",
}


def _resolve_wn18rr_evidence(
    bootstrap_root: Path,
) -> dict[str, dict[str, Any]]:
    """For each WN18RR artifact, pick current-repo copy if it exists, else bootstrap."""
    resolved: dict[str, dict[str, Any]] = {}
    for label, rel_path in WN18RR_EVIDENCE_ARTIFACTS.items():
        current_path = PROJECT_ROOT / rel_path
        bootstrap_path = bootstrap_root / rel_path
        if current_path.exists():
            resolved[label] = {
                "path": current_path,
                "source": "current_repo",
                "rel_path": relpath_str(current_path),
            }
        elif bootstrap_path.exists():
            resolved[label] = {
                "path": bootstrap_path,
                "source": "bootstrap",
                "rel_path": relpath_str(bootstrap_path),
            }
        else:
            resolved[label] = {
                "path": None,
                "source": "missing",
                "rel_path": None,
            }
    return resolved


# ---------------------------------------------------------------------------
# WN18RR harmonization
# ---------------------------------------------------------------------------


def _determine_semantic_alignment(
    wn18rr_evidence: dict[str, dict[str, Any]],
) -> tuple[bool, str | None]:
    """Determine semantic_alignment_verified from best available evidence."""
    # Try semantic alignment audit first
    sa = wn18rr_evidence.get("wn18rr_semantic_alignment_audit", {})
    sa_path = sa.get("path")
    if sa_path is not None:
        sa_payload = load_json_optional(sa_path, label="wn18rr_semantic_alignment_audit")
        if isinstance(sa_payload, dict):
            if sa_payload.get("semantic_alignment_verified"):
                verdict = sa_payload.get("verdict")
                return True, str(verdict) if verdict else "verified_by_provenance"

    # Fall back to comparison JSON
    comp = wn18rr_evidence.get("wn18rr_link_comparison", {})
    comp_path = comp.get("path")
    if comp_path is not None:
        comp_payload = load_json_optional(comp_path, label="wn18rr_link_comparison")
        if isinstance(comp_payload, dict):
            alignment = comp_payload.get("alignment_evidence", {})
            if alignment.get("semantic_alignment_verified"):
                return True, alignment.get("semantic_verdict")

    return False, None


def _harmonize_entries(
    entries: list[dict[str, Any]],
    wn18rr_evidence: dict[str, dict[str, Any]],
    semantic_verified: bool,
    semantic_verdict: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Patch entries for WN18RR harmonization.

    Returns (patched_entries, changes_applied).
    """
    changes: list[dict[str, Any]] = []
    patched: list[dict[str, Any]] = []

    for entry in entries:
        entry = _shallow_copy_entry(entry)

        # --- Alignment audit: fix semantic_alignment_verified caveat ---
        if (
            entry.get("key") == "wn18rr_alignment_audit"
            and entry.get("kind") == "alignment_audit"
        ):
            old_val = entry["caveats"].get("semantic_alignment_verified")
            if semantic_verified and old_val is False:
                entry["caveats"]["semantic_alignment_verified"] = True
                changes.append({
                    "entry_key": entry["key"],
                    "field": "caveats.semantic_alignment_verified",
                    "old": False,
                    "new": True,
                    "reason": (
                        f"harmonized_from_semantic_audit "
                        f"(verdict={semantic_verdict})"
                    ),
                })

            # Update evidence path if fresh audit exists in current repo
            audit_ev = wn18rr_evidence.get("wn18rr_alignment_audit", {})
            if audit_ev.get("source") == "current_repo":
                _update_evidence_path(entry, audit_ev, changes)

        # --- WN18RR eval entries: update evidence if fresh manifest exists ---
        elif (
            entry.get("group") == "experimental"
            and entry.get("kind") == "evaluation"
            and "wn18rr" in str(entry.get("dataset", "")).lower()
        ):
            source_path = entry["evidence"].get("source_path", "")
            if "layer2_bootstrap" in source_path:
                manifest_label = _manifest_label_for_source(source_path)
                if manifest_label:
                    fresh = wn18rr_evidence.get(manifest_label, {})
                    if fresh.get("source") == "current_repo":
                        _update_evidence_path(entry, fresh, changes)

        patched.append(entry)

    return patched, changes


def _shallow_copy_entry(entry: dict[str, Any]) -> dict[str, Any]:
    result = dict(entry)
    result["caveats"] = dict(result.get("caveats", {}))
    result["evidence"] = dict(result.get("evidence", {}))
    return result


def _update_evidence_path(
    entry: dict[str, Any],
    evidence_info: dict[str, Any],
    changes: list[dict[str, Any]],
) -> None:
    new_path = evidence_info.get("rel_path")
    if not new_path:
        return
    old_path = entry["evidence"].get("source_path")
    if old_path == new_path:
        return
    entry["evidence"]["source_path"] = new_path
    if entry["evidence"].get("result_path") == old_path:
        entry["evidence"]["result_path"] = new_path
    entry["evidence"]["bootstrap_inherited"] = False
    entry["evidence"]["provenance_status"] = "fresh_current_repo"
    changes.append({
        "entry_key": entry["key"],
        "field": "evidence.source_path",
        "old": old_path,
        "new": new_path,
        "reason": "fresh_current_repo_preferred_over_bootstrap",
    })


def _manifest_label_for_source(source_path: str) -> str | None:
    if "debug_manifest" in source_path:
        return "wn18rr_debug_manifest"
    if "official_manifest" in source_path:
        return "wn18rr_official_manifest"
    return None


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "group",
    "label",
    "dataset",
    "model",
    "task",
    "status",
    "metric_name",
    "metric_value",
    "official_metric",
    "evidence_source_kind",
    "evidence_source_path",
    "provenance_status",
    "manifest_backed",
    "experimental",
    "promotion_ready",
    "remaining_blockers",
    "notes_caveats",
]


def _entry_to_csv_row(entry: dict[str, Any]) -> dict[str, str]:
    evidence = entry.get("evidence", {})
    classification = entry.get("classification", {})
    caveats = entry.get("caveats", {})
    metric = entry.get("metric", {})

    experimental = bool(
        classification.get("experimental")
        or caveats.get("experimental")
        or entry.get("group") == "experimental"
    )

    # Metric display
    metric_name = metric.get("name") or ""
    metric_value = metric.get("value")
    if entry.get("kind") == "alignment_audit" and not metric_name:
        details = entry.get("details", {})
        metric_name = "checks"
        passed = details.get("audit_checks_passed", 0)
        total = details.get("audit_checks_total", 0)
        mv_str = f"{passed}/{total}"
    elif isinstance(metric_value, float):
        mv_str = f"{metric_value:.6f}"
    elif metric_value is not None:
        mv_str = str(metric_value)
    else:
        mv_str = ""

    remaining_blockers = _compute_remaining_blockers(entry)
    promotion_ready = (
        entry.get("group") == "official_candidate" and not remaining_blockers
    )

    caveat_items = [
        f"{k}={_fmt(caveats[k])}" for k in sorted(caveats)
    ]

    return {
        "group": entry.get("group", ""),
        "label": entry.get("label", ""),
        "dataset": entry.get("dataset", ""),
        "model": str(entry.get("model") or ""),
        "task": entry.get("task", ""),
        "status": entry.get("status", ""),
        "metric_name": metric_name,
        "metric_value": mv_str,
        "official_metric": _fmt(metric.get("official_metric", False)),
        "evidence_source_kind": evidence.get("source_kind", ""),
        "evidence_source_path": evidence.get("source_path", ""),
        "provenance_status": evidence.get("provenance_status", ""),
        "manifest_backed": _fmt(evidence.get("manifest_backed", False)),
        "experimental": _fmt(experimental),
        "promotion_ready": _fmt(promotion_ready),
        "remaining_blockers": remaining_blockers,
        "notes_caveats": "; ".join(caveat_items),
    }


def _compute_remaining_blockers(entry: dict[str, Any]) -> str:
    if entry.get("group") == "official_candidate":
        return ""

    blockers: list[str] = []
    caveats = entry.get("caveats", {})

    if entry.get("group") == "experimental":
        blockers.append("experimental_fence_still_enabled")
        if caveats.get("relation_types_ignored") is True:
            blockers.append("relation_types_ignored")

    elif entry.get("group") == "local_debug":
        if caveats.get("debug_mode") is True:
            blockers.append("debug_mode_enabled")
        if entry.get("metric", {}).get("official_metric") is not True:
            blockers.append("official_metric_flag_false")
        if not entry.get("evidence", {}).get("manifest_backed"):
            blockers.append("manifest_backed_execution_missing")
        blockers.append("not_locked_official_candidate_surface")

    return "; ".join(blockers)


def _export_csv(entries: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for entry in entries:
        writer.writerow(_entry_to_csv_row(entry))
    return buf.getvalue()


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Finalize Layer 2 HPC closure: harmonize WN18RR semantics, "
            "export all-dataset CSV."
        ),
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "summary.json",
        help="Path to the existing summary.json (from manual HPC ingestion).",
    )
    p.add_argument(
        "--bootstrap-root",
        type=Path,
        default=PROJECT_ROOT / "state" / "layer2_bootstrap",
        help="Bootstrap snapshot root.",
    )
    p.add_argument(
        "--meeting-table-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "meeting_table.md",
        help="Output path for the harmonized meeting table.",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "layer2_results_table.csv",
        help="Output path for the flat CSV results table.",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help=(
            "Output path for the updated summary JSON.  "
            "Defaults to overwriting --summary-json (in-place update)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without writing files.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary_out = args.summary_out or args.summary_json

    # 1. Load existing summary produced by ingest_layer2_manual_hpc_results.py
    summary = require_mapping(
        load_json_required(args.summary_json, label="summary_json"),
        label="summary_json",
        path=args.summary_json,
    )
    entries: list[dict[str, Any]] = summary.get("entries", [])
    if not entries:
        print_json({"error": "summary.json contains no entries"})
        return 1

    # 2. Resolve WN18RR evidence (fresh current-repo > bootstrap)
    wn18rr_evidence = _resolve_wn18rr_evidence(args.bootstrap_root)

    # 3. Determine semantic alignment state from best available evidence
    semantic_verified, semantic_verdict = _determine_semantic_alignment(
        wn18rr_evidence,
    )

    # 4. Harmonize WN18RR entries
    patched_entries, changes = _harmonize_entries(
        entries,
        wn18rr_evidence,
        semantic_verified,
        semantic_verdict,
    )

    # 5. Stamp finalization metadata into summary
    summary["entries"] = patched_entries
    summary["finalization"] = {
        "schema_version": "layer2_finalization/v1",
        "wn18rr_evidence_resolution": {
            label: {"source": info["source"], "path": info["rel_path"]}
            for label, info in wn18rr_evidence.items()
        },
        "harmonization": {
            "semantic_alignment_verified": semantic_verified,
            "semantic_verdict": semantic_verdict,
            "changes": changes,
        },
        "experimental_fence_preserved": True,
        "wn18rr_excluded_from_official_candidate": True,
    }

    # 6. Render outputs
    meeting_table = _render_meeting_table(patched_entries)
    csv_content = _export_csv(patched_entries)
    summary_content = json_dumps_stable(summary)

    write_specs = [
        {
            "output_type": "summary_json",
            "path": summary_out,
            "content": summary_content,
        },
        {
            "output_type": "meeting_table",
            "path": args.meeting_table_out,
            "content": meeting_table,
        },
        {
            "output_type": "csv_table",
            "path": args.csv_out,
            "content": csv_content,
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

    print_json({
        "mode": "finalize_layer2_hpc_table",
        "dry_run": args.dry_run,
        "summary_source": relpath_str(args.summary_json),
        "bootstrap_root": relpath_str(args.bootstrap_root),
        "wn18rr_evidence_resolution": {
            label: info["source"]
            for label, info in wn18rr_evidence.items()
        },
        "harmonization": {
            "semantic_alignment_verified": semantic_verified,
            "semantic_verdict": semantic_verdict,
            "changes_count": len(changes),
            "changes": changes,
        },
        "writes": write_plans,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
