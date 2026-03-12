from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent

_MANIFEST_BLOCK_RE = re.compile(
    r"<!--\s*layer2-bundle-manifest:start\s*-->\s*```json\s*(.*?)\s*```\s*<!--\s*layer2-bundle-manifest:end\s*-->",
    flags=re.DOTALL,
)
_MULTI_SLASH_RE = re.compile(r"/+")
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")

EVIDENCE_STATUS_PRECEDENCE: dict[str, int] = {
    "success": 500,
    "debug_success": 400,
    "error": 300,
    "blocked": 200,
    "dry_run": 100,
}
RUN_TYPE_PRECEDENCE: dict[str, int] = {
    "execution": 100,
    "preview": 0,
}
ACCEPTABLE_EXECUTION_STATUSES = frozenset({"success", "debug_success"})
PROTECTED_OFFICIAL_ENTRY_KEYS = frozenset(
    {
        "graphmae_arxiv_official_candidate",
        "bgrl_arxiv_official_candidate",
    }
)
EXPERIMENTAL_ONLY_BUNDLE_ROOTS = frozenset({"artifacts", "evidence"})
PROMOTED_BUNDLE_SURFACE_MARKERS = frozenset(
    {
        "official_candidate",
        "all_proven_local",
        "proven_local",
        "full_local_non_debug",
        "full_local",
        "local_debug",
    }
)


class StructuredError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


def relpath_str(path: Path, *, root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def json_dumps_stable(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


def print_json(payload: Any) -> None:
    print(json_dumps_stable(payload), end="")


def load_json_required(path: Path, *, label: str) -> Any:
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
        return json.loads(path.read_text(encoding="utf-8"))
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
    except json.JSONDecodeError as exc:
        raise StructuredError(
            "invalid_json",
            f"Failed to parse JSON input: {label}.",
            {
                "label": label,
                "path": relpath_str(path),
                "reason": f"{type(exc).__name__}: {exc}",
            },
        ) from exc


def load_json_optional(path: Path, *, label: str) -> Any | None:
    if not path.exists():
        return None
    return load_json_required(path, label=label)


def require_mapping(payload: Any, *, label: str, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise StructuredError(
            "invalid_input_shape",
            f"Expected a JSON object for {label}.",
            {
                "label": label,
                "path": relpath_str(path),
                "actual_type": type(payload).__name__,
            },
        )
    return payload


def parse_note_fields(note_text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    fragments: list[str] = []
    for raw_part in note_text.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            fragments.append(part)
            continue
        key, raw_value = part.split("=", 1)
        fields[key.strip()] = _coerce_scalar(raw_value.strip())
    if fragments:
        fields["_fragments"] = fragments
    return fields


def parse_note_items(note_items: Iterable[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    fragments: list[str] = []
    for item in note_items:
        item_fields = parse_note_fields(item)
        item_fragments = item_fields.pop("_fragments", [])
        merged.update(item_fields)
        fragments.extend(item_fragments)
    if fragments:
        merged["_fragments"] = fragments
    return merged


def merge_note_fields(*field_sets: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    fragments: list[str] = []
    for field_set in field_sets:
        item_fragments = field_set.get("_fragments", [])
        for key, value in field_set.items():
            if key == "_fragments":
                continue
            merged[key] = value
        fragments.extend(item_fragments)
    if fragments:
        merged["_fragments"] = fragments
    return merged


def render_text_write_plan(path: Path, content: str) -> dict[str, Any]:
    new_bytes = content.encode("utf-8")
    exists = path.exists()
    old_bytes = path.read_bytes() if exists else None
    changed = old_bytes != new_bytes
    action = "unchanged"
    if not exists:
        action = "create"
    elif changed:
        action = "update"
    return {
        "path": relpath_str(path),
        "action": action,
        "changed": changed,
        "size_bytes": len(new_bytes),
        "sha256": sha256_bytes(new_bytes),
    }


def apply_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def extract_manifest_spec(markdown_text: str, *, manifest_path: Path) -> dict[str, Any]:
    match = _MANIFEST_BLOCK_RE.search(markdown_text)
    if not match:
        raise StructuredError(
            "invalid_manifest",
            "MANIFEST.md is missing the layer2-bundle-manifest JSON block.",
            {
                "path": relpath_str(manifest_path),
            },
        )
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise StructuredError(
            "invalid_manifest_json",
            "Failed to parse the layer2-bundle-manifest JSON block.",
            {
                "path": relpath_str(manifest_path),
                "reason": f"{type(exc).__name__}: {exc}",
            },
        ) from exc
    if not isinstance(payload, dict):
        raise StructuredError(
            "invalid_manifest_shape",
            "The layer2-bundle-manifest JSON block must be a JSON object.",
            {
                "path": relpath_str(manifest_path),
                "actual_type": type(payload).__name__,
            },
        )
    return payload


def classify_bundle_destination_policy(
    destination: str,
    *,
    experimental_roots: Iterable[str] | None = None,
    promoted_markers: Iterable[str] | None = None,
) -> dict[str, Any]:
    normalized = _normalize_bundle_destination(destination)
    lowered = normalized.lower()
    roots = frozenset(
        root.strip("/").lower()
        for root in (experimental_roots or EXPERIMENTAL_ONLY_BUNDLE_ROOTS)
        if isinstance(root, str) and root.strip("/")
    )
    markers = frozenset(
        marker.lower()
        for marker in (promoted_markers or PROMOTED_BUNDLE_SURFACE_MARKERS)
        if isinstance(marker, str) and marker
    )
    root = lowered.split("/", 1)[0] if lowered else ""
    matched_markers = sorted(marker for marker in markers if marker in lowered)
    allowed_root = root in roots
    experimental_only = allowed_root and not matched_markers
    if experimental_only:
        classification = "experimental_only"
    elif matched_markers:
        classification = "promoted_surface"
    else:
        classification = "unclassified_surface"
    return {
        "destination": normalized,
        "root": root,
        "allowed_root": allowed_root,
        "matched_promoted_markers": matched_markers,
        "experimental_only": experimental_only,
        "classification": classification,
    }


def evidence_status_precedence(status: Any) -> int:
    normalized = _normalize_status(status)
    if normalized is None:
        return 0
    return EVIDENCE_STATUS_PRECEDENCE.get(normalized, 0)


def run_type_precedence(run_type: Any) -> int:
    normalized = normalize_run_type(run_type)
    return RUN_TYPE_PRECEDENCE.get(normalized, 0)


def evidence_quality_key(
    status: Any,
    *,
    run_type: Any,
    metric_present: bool | None = None,
) -> tuple[int, int, int]:
    metric_rank = 1 if metric_present else 0
    return (
        run_type_precedence(run_type),
        evidence_status_precedence(status),
        metric_rank,
    )


def is_acceptable_execution_evidence(
    *,
    status: Any,
    run_type: Any,
    metric_present: bool | None = None,
) -> bool:
    normalized_status = _normalize_status(status)
    if normalize_run_type(run_type) != "execution":
        return False
    if normalized_status not in ACCEPTABLE_EXECUTION_STATUSES:
        return False
    if metric_present is False:
        return False
    return True


def normalize_run_type(
    run_type: Any,
    *,
    preview: Any | None = None,
    overall_status: Any | None = None,
    target_statuses: Iterable[Any] | None = None,
) -> str:
    if isinstance(run_type, str):
        normalized = run_type.strip().lower()
        if normalized in RUN_TYPE_PRECEDENCE:
            return normalized
    if preview is True:
        return "preview"
    if preview is False:
        return "execution"
    if _normalize_status(overall_status) == "dry_run":
        return "preview"
    normalized_target_statuses = [
        status
        for status in (_normalize_status(status) for status in (target_statuses or []))
        if status is not None
    ]
    if normalized_target_statuses and all(status == "dry_run" for status in normalized_target_statuses):
        return "preview"
    return "execution"


def infer_manifest_run_type(payload: dict[str, Any]) -> str:
    targets = payload.get("targets")
    target_statuses: list[Any] = []
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, dict):
                target_statuses.append(
                    target.get("parsed_status")
                    or target.get("stage_eval_status")
                    or target.get("overall_status")
                )
    return normalize_run_type(
        payload.get("run_type"),
        preview=payload.get("preview"),
        overall_status=payload.get("overall_status"),
        target_statuses=target_statuses,
    )


def infer_manifest_preview(payload: dict[str, Any]) -> bool:
    return infer_manifest_run_type(payload) == "preview"


def metric_value_present(value: Any) -> bool:
    return value is not None


def summary_entry_evidence_quality(entry: dict[str, Any]) -> dict[str, Any]:
    evidence = entry.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    status = entry.get("status")
    run_type = normalize_run_type(
        evidence.get("run_type"),
        preview=evidence.get("preview"),
        overall_status=status,
    )
    metric = entry.get("metric", {})
    metric_value = metric.get("value") if isinstance(metric, dict) else None
    metric_present = metric_value_present(metric_value)
    return {
        "status": _normalize_status(status),
        "run_type": run_type,
        "preview": run_type == "preview",
        "metric_present": metric_present,
        "quality_key": evidence_quality_key(
            status,
            run_type=run_type,
            metric_present=metric_present,
        ),
        "acceptable_execution_evidence": is_acceptable_execution_evidence(
            status=status,
            run_type=run_type,
            metric_present=metric_present,
        ),
    }


def _normalize_status(status: Any) -> str | None:
    if not isinstance(status, str):
        return None
    normalized = status.strip().lower()
    if not normalized:
        return None
    return normalized


def _normalize_bundle_destination(destination: str) -> str:
    normalized = _MULTI_SLASH_RE.sub("/", destination.replace("\\", "/")).strip("/")
    return normalized


def _coerce_scalar(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if _INT_RE.match(raw_value):
        try:
            return int(raw_value)
        except ValueError:
            return raw_value
    if _FLOAT_RE.match(raw_value):
        try:
            return float(raw_value)
        except ValueError:
            return raw_value
    return raw_value
