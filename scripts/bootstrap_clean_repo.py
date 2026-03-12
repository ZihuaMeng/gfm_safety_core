#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_FILE_PATHS = (
    Path(".gitignore"),
    Path("hpc_env.yml"),
    Path("README.md"),
    Path("ARCHITECTURE.md"),
    Path("MANIFEST.md"),
    Path("CLAUDE.md"),
)
TREE_PATHS = (
    Path("eval"),
    Path("src"),
    Path("scripts"),
    Path("tools"),
    Path("repos/bgrl"),
    Path("repos/graphmae"),
)
NOTE_GLOB = "notes/*.md"
STATE_BOOTSTRAP_ROOT = Path("state/layer2_bootstrap")
STATE_BOOTSTRAP_README = STATE_BOOTSTRAP_ROOT / "README.md"
STATE_BOOTSTRAP_README_TEXT = """# Layer 2 Bootstrap Snapshot

This directory is a lightweight bootstrap snapshot of the current Layer 2 evidence.

Heavy generated artifacts are intentionally excluded from the clean GitHub migration,
including full `results/`, `outputs/`, `work/`, datasets, checkpoints, and logs.

Per `MANIFEST.md`, local source paths remain the editable source of truth and
`work/layer2/` remains a derived snapshot.

Future official Layer 2 state should continue to be refreshed from real manifests and
result artifacts produced on HPC rather than edited here.
"""
CANONICAL_EVIDENCE_PATHS = (
    Path("results/baseline/layer2_suite_official_candidate_arxiv_official_manifest.json"),
    Path("results/baseline/layer2_suite_graphmae_pcba_native_graph_debug_manifest.json"),
    Path("results/baseline/layer2_suite_graphmae_pcba_native_graph_official_manifest.json"),
    Path("results/baseline/pcba_graph_comparison.json"),
    Path("results/baseline/wn18rr_alignment_audit.json"),
    Path("results/baseline/wn18rr_semantic_alignment_audit.json"),
    Path("results/baseline/layer2_suite_wn18rr_experimental_compare_debug_manifest.json"),
    Path("results/baseline/layer2_suite_wn18rr_experimental_compare_official_manifest.json"),
    Path("results/baseline/wn18rr_link_comparison.json"),
    Path("notes/meeting_progress_arxiv_sbert.md"),
    Path("notes/pcba_graph_protocol_report.md"),
    Path("notes/wn18rr_link_protocol_report.md"),
)
EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".claude",
        ".codex",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".venv",
        ".vscode",
        "__pycache__",
        "conda-meta",
    }
)
EXCLUDED_RELATIVE_DIRS = frozenset(
    {
        "repos/bgrl_backup_before_graphmae_patch",
        "repos/bgrl/.git",
        "repos/bgrl/runs",
        "repos/bgrl/img",
        "repos/graphmae/.git",
        "repos/graphmae/data",
        "repos/graphmae/dataset",
        "repos/graphmae/imgs",
        "repos/graphmae/chem/init_weights",
    }
)
EXCLUDED_FILE_PATHS = frozenset({"scripts/pack_colab_payload.py"})
EXCLUDED_FILE_NAMES = frozenset({".DS_Store", "Thumbs.db"})
EXCLUDED_FILE_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".pyc",
    ".pyd",
    ".pyo",
    ".safetensors",
    ".so",
    ".tgz",
    ".zip",
)


class BootstrapError(Exception):
    pass


@dataclass(frozen=True)
class Operation:
    dest_rel: Path
    source_rel: Path | None = None
    content: str | None = None

    @property
    def kind(self) -> str:
        return "WRITE" if self.source_rel is None else "COPY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a clean GitHub-ready copy of the repo with curated Layer 2 "
            "bootstrap evidence and heavyweight artifacts excluded."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the source repository.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Path where the clean repo copy should be created.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the deterministic copy/write plan without creating the clean repo.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source_root = _resolve_source_root(args.source)
        dest_root = _resolve_dest_root(args.dest, source_root=source_root)
        operations, warnings = _build_operations(source_root)
    except BootstrapError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1

    mkdir_targets = _collect_missing_directories(dest_root=dest_root, operations=operations)
    _print_plan(
        source_root=source_root,
        dest_root=dest_root,
        mkdir_targets=mkdir_targets,
        operations=operations,
        warnings=warnings,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        return 0

    try:
        _apply_plan(
            source_root=source_root,
            dest_root=dest_root,
            mkdir_targets=mkdir_targets,
            operations=operations,
        )
    except OSError as exc:
        print(f"ERROR failed to materialize clean repo: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def _resolve_source_root(source_root: Path) -> Path:
    resolved = source_root.expanduser().resolve()
    if not resolved.exists():
        raise BootstrapError(f"source repo does not exist: {resolved}")
    if not resolved.is_dir():
        raise BootstrapError(f"source repo is not a directory: {resolved}")
    return resolved


def _resolve_dest_root(dest_root: Path, *, source_root: Path) -> Path:
    resolved = dest_root.expanduser().resolve()
    if resolved == source_root:
        raise BootstrapError("--dest must be different from --source")
    if source_root in resolved.parents:
        raise BootstrapError("--dest cannot be created inside --source")
    if resolved in source_root.parents:
        raise BootstrapError("--dest cannot be an ancestor of --source")
    if resolved.exists():
        if not resolved.is_dir():
            raise BootstrapError(f"destination exists and is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise BootstrapError(
                f"destination already exists and is not empty: {resolved}"
            )
    return resolved


def _build_operations(source_root: Path) -> tuple[list[Operation], list[str]]:
    operations: list[Operation] = []
    warnings: list[str] = []
    seen_destinations: set[Path] = set()

    def add_copy(source_rel: Path, dest_rel: Path | None = None) -> None:
        destination = dest_rel or source_rel
        if destination in seen_destinations:
            raise BootstrapError(f"duplicate destination planned: {destination.as_posix()}")
        operations.append(Operation(dest_rel=destination, source_rel=source_rel))
        seen_destinations.add(destination)

    def add_write(dest_rel: Path, content: str) -> None:
        if dest_rel in seen_destinations:
            raise BootstrapError(f"duplicate destination planned: {dest_rel.as_posix()}")
        operations.append(Operation(dest_rel=dest_rel, content=content))
        seen_destinations.add(dest_rel)

    for source_rel in ROOT_FILE_PATHS:
        source_path = source_root / source_rel
        if not source_path.exists():
            warnings.append(f"WARNING missing required source file: {source_rel.as_posix()}")
            continue
        if not source_path.is_file():
            warnings.append(f"WARNING expected file but found non-file path: {source_rel.as_posix()}")
            continue
        add_copy(source_rel)

    for tree_rel in TREE_PATHS:
        tree_root = source_root / tree_rel
        if not tree_root.exists():
            warnings.append(f"WARNING missing required source tree: {tree_rel.as_posix()}")
            continue
        if not tree_root.is_dir():
            warnings.append(f"WARNING expected directory but found non-directory path: {tree_rel.as_posix()}")
            continue
        for source_rel in _iter_tree_files(source_root, tree_rel):
            add_copy(source_rel)

    notes_root = source_root / "notes"
    if notes_root.exists() and notes_root.is_dir():
        for note_path in sorted(notes_root.glob("*.md"), key=lambda path: path.as_posix()):
            source_rel = note_path.relative_to(source_root)
            if _should_exclude_file(source_rel):
                continue
            add_copy(source_rel)
    else:
        warnings.append(f"WARNING missing notes directory for glob {NOTE_GLOB}")

    add_write(STATE_BOOTSTRAP_README, STATE_BOOTSTRAP_README_TEXT)
    for source_rel in CANONICAL_EVIDENCE_PATHS:
        source_path = source_root / source_rel
        if not source_path.exists():
            warnings.append(
                "WARNING missing canonical Layer 2 evidence file: "
                f"{source_rel.as_posix()}"
            )
            continue
        if not source_path.is_file():
            warnings.append(
                "WARNING expected canonical evidence file but found non-file path: "
                f"{source_rel.as_posix()}"
            )
            continue
        add_copy(source_rel, dest_rel=STATE_BOOTSTRAP_ROOT / source_rel)

    operations.sort(key=lambda item: item.dest_rel.as_posix())
    warnings.sort()
    return operations, warnings


def _iter_tree_files(source_root: Path, tree_rel: Path) -> list[Path]:
    collected: list[Path] = []
    walk_root = source_root / tree_rel
    for current_root, dirnames, filenames in os.walk(walk_root, topdown=True):
        current_path = Path(current_root)
        current_rel = current_path.relative_to(source_root)
        dirnames[:] = sorted(
            directory_name
            for directory_name in dirnames
            if _should_descend_dir(current_rel / directory_name)
        )
        for file_name in sorted(filenames):
            source_rel = current_rel / file_name
            if _should_exclude_file(source_rel):
                continue
            collected.append(source_rel)
    return collected


def _should_descend_dir(path: Path) -> bool:
    normalized = path.as_posix()
    if normalized in EXCLUDED_RELATIVE_DIRS:
        return False
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return False
    return True


def _should_exclude_file(path: Path) -> bool:
    normalized = path.as_posix()
    if normalized in EXCLUDED_FILE_PATHS:
        return True
    if any(parent.as_posix() in EXCLUDED_RELATIVE_DIRS for parent in _path_and_parents(path.parent)):
        return True
    name = path.name
    if name in EXCLUDED_FILE_NAMES:
        return True
    if name.endswith(":Zone.Identifier"):
        return True
    if name.endswith(".tar.gz"):
        return True
    if any(name.endswith(suffix) for suffix in EXCLUDED_FILE_SUFFIXES):
        return True
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts[:-1]):
        return True
    if any(part.endswith(".egg-info") for part in path.parts):
        return True
    if name.endswith(".ipynb"):
        return True
    return False


def _path_and_parents(path: Path) -> list[Path]:
    parents: list[Path] = []
    current = path
    while current != Path("."):
        parents.append(current)
        current = current.parent
    return parents


def _collect_missing_directories(dest_root: Path, operations: list[Operation]) -> list[Path]:
    directories: set[Path] = set()
    for operation in operations:
        target_path = dest_root / operation.dest_rel
        current = target_path.parent
        while True:
            directories.add(current)
            if current == dest_root:
                break
            current = current.parent
    return sorted(
        (directory for directory in directories if not directory.exists()),
        key=lambda path: (len(path.parts), path.as_posix()),
    )


def _print_plan(
    *,
    source_root: Path,
    dest_root: Path,
    mkdir_targets: list[Path],
    operations: list[Operation],
    warnings: list[str],
    dry_run: bool,
) -> None:
    print(f"SOURCE {source_root}")
    print(f"DEST {dest_root}")
    print(f"MODE {'dry-run' if dry_run else 'apply'}")
    for directory in mkdir_targets:
        print(f"MKDIR {directory}")
    for operation in operations:
        destination = dest_root / operation.dest_rel
        if operation.source_rel is None:
            print(f"WRITE {destination}")
            continue
        print(f"COPY {operation.source_rel.as_posix()} -> {destination}")
    for warning in warnings:
        print(warning)
    copy_count = sum(1 for operation in operations if operation.source_rel is not None)
    write_count = sum(1 for operation in operations if operation.source_rel is None)
    print(
        "SUMMARY "
        f"mkdirs={len(mkdir_targets)} "
        f"copies={copy_count} "
        f"writes={write_count} "
        f"warnings={len(warnings)}"
    )
    if dry_run:
        print("DRY-RUN no files were written")
    else:
        print("DONE clean repo materialized")


def _apply_plan(
    source_root: Path,
    dest_root: Path,
    *,
    mkdir_targets: list[Path],
    operations: list[Operation],
) -> None:
    for directory in mkdir_targets:
        directory.mkdir(parents=True, exist_ok=True)
    for operation in operations:
        destination = dest_root / operation.dest_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if operation.source_rel is None:
            destination.write_text(operation.content or "", encoding="utf-8")
            continue
        shutil.copy2(source_root / operation.source_rel, destination)


if __name__ == "__main__":
    sys.exit(main())
