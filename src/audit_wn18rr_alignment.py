#!/usr/bin/env python3
"""
Read-only alignment audit for WN18RR external SBERT features.

What this script does:
1) Load native WN18RR (prefer DGL `WN18Dataset`, fallback to PyG `WordNet18RR`).
2) Load external features from `data/wn18rr_sbert.pt`.
3) Assert exact node-count equality.
4) Print top-5 standard dataset entities (if available) and top-5 row-level
   metadata entries from the external `.pt` payload (if present) for manual
   alignment inspection.

This script does not modify any dataset/model source files.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def _read_entity2id_file(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None

    pairs: List[Tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = line.strip()
        if not raw:
            continue

        # Robust parse: entity can include punctuation; assume final token is id.
        parts = re.split(r"\s+", raw)
        if len(parts) < 2:
            continue

        maybe_id = parts[-1]
        try:
            idx = int(maybe_id)
        except ValueError:
            continue

        name = " ".join(parts[:-1]).strip()
        if not name:
            continue
        pairs.append((idx, name))

    if not pairs:
        return None

    pairs.sort(key=lambda item: item[0])
    return [name for _, name in pairs]


def _extract_entities_from_entity2id(entity2id: Dict[object, object]) -> Optional[List[str]]:
    items: List[Tuple[int, str]] = []
    for key, value in entity2id.items():
        if isinstance(key, str) and isinstance(value, int):
            items.append((value, key))
        elif isinstance(key, int) and isinstance(value, str):
            items.append((key, value))

    if not items:
        return None

    items.sort(key=lambda item: item[0])
    return [name for _, name in items]


def _try_load_dgl_wn18rr(data_root: Path) -> Tuple[int, Optional[List[str]], str]:
    errors: List[str] = []

    try:
        from dgl.data import WN18Dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"DGL import failed: {exc}") from exc

    # Different DGL versions expose slightly different constructor signatures.
    kw_variants = [
        {"name": "wn18rr", "raw_dir": str(data_root), "verbose": False},
        {"name": "wn18rr", "raw_dir": str(data_root)},
        {"raw_dir": str(data_root), "verbose": False},
        {"raw_dir": str(data_root)},
        {"name": "wn18rr", "verbose": False},
        {"name": "wn18rr"},
        {},
    ]

    dataset = None
    for kwargs in kw_variants:
        try:
            dataset = WN18Dataset(**kwargs)
            break
        except Exception as exc:
            errors.append(f"kwargs={kwargs}: {exc}")

    if dataset is None:
        joined = "\n  - " + "\n  - ".join(errors) if errors else ""
        raise RuntimeError(f"DGL WN18Dataset load failed with all variants:{joined}")

    num_nodes = getattr(dataset, "num_nodes", None)
    if num_nodes is None:
        raise RuntimeError("DGL WN18Dataset loaded but has no `num_nodes` attribute.")

    entities: Optional[List[str]] = None
    if hasattr(dataset, "entity2id") and isinstance(dataset.entity2id, dict):
        entities = _extract_entities_from_entity2id(dataset.entity2id)

    dataset_name = str(getattr(dataset, "name", "")).lower()
    if dataset_name and dataset_name != "wn18rr":
        raise RuntimeError(
            "DGL WN18Dataset resolved to a non-WN18RR variant "
            f"(name={dataset_name!r})."
        )

    source = "DGL WN18Dataset(name='wn18rr')"
    return int(num_nodes), entities, source


def _try_load_pyg_wn18rr(data_root: Path) -> Tuple[int, Optional[List[str]], str]:
    try:
        from torch_geometric.datasets import WordNet18RR  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"PyG import failed: {exc}") from exc

    dataset = WordNet18RR(root=str(data_root))
    graph = dataset[0]
    num_nodes = int(graph.num_nodes)

    candidate_files = [
        data_root / "WN18RR" / "entity2id.txt",
        data_root / "pyg" / "WN18RR" / "entity2id.txt",
        Path(dataset.raw_dir) / "entity2id.txt",
        Path(dataset.raw_dir).parent / "entity2id.txt",
    ]

    entities: Optional[List[str]] = None
    for file_path in candidate_files:
        entities = _read_entity2id_file(file_path)
        if entities:
            break

    source = "PyG WordNet18RR"
    return num_nodes, entities, source


def _load_native_wn18rr(data_root: Path) -> Tuple[int, Optional[List[str]], str]:
    # Prefer DGL (GraphMAE stack), fallback to PyG.
    dgl_err: Optional[str] = None
    try:
        return _try_load_dgl_wn18rr(data_root)
    except Exception as exc:
        dgl_err = str(exc)

    try:
        num_nodes, entities, source = _try_load_pyg_wn18rr(data_root)
        source = f"{source} (DGL failed: {dgl_err})"
        return num_nodes, entities, source
    except Exception as exc:
        raise RuntimeError(
            "Failed to load native WN18RR via both DGL and PyG.\n"
            f"- DGL error: {dgl_err}\n"
            f"- PyG error: {exc}"
        ) from exc


def _load_external_payload(path: Path) -> Tuple[torch.Tensor, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(payload, dict):
        if "x" not in payload:
            raise KeyError(
                f"External payload is dict but missing key 'x'. Keys: {sorted(payload.keys())}"
            )
        x = payload["x"]
    elif torch.is_tensor(payload):
        x = payload
    else:
        raise TypeError(f"Unsupported payload type: {type(payload).__name__}")

    if not torch.is_tensor(x):
        raise TypeError(f"External feature field is not a tensor: {type(x).__name__}")
    if x.dim() != 2:
        raise ValueError(f"External feature tensor must be 2-D, got shape {tuple(x.shape)}")

    return x, payload


def _format_top5(items: Sequence[object]) -> List[str]:
    return [f"{index}: {repr(value)}" for index, value in enumerate(items[:5])]


def _extract_external_row_metadata_preview(payload: object) -> List[str]:
    if not isinstance(payload, dict):
        return ["(payload is not a dict; no row-level metadata keys)"]

    # Candidate keys that may hold per-node row metadata.
    candidate_keys = [
        "texts",
        "raw_texts",
        "entities",
        "entity_names",
        "entity_texts",
        "id2entity",
        "entity2id",
        "rows",
        "records",
    ]

    previews: List[str] = []

    for key in candidate_keys:
        if key not in payload:
            continue

        value = payload[key]
        if isinstance(value, (list, tuple)):
            previews.append(f"key='{key}' (len={len(value)}):")
            previews.extend([f"  {line}" for line in _format_top5(list(value))])
            continue

        if isinstance(value, dict):
            if key == "entity2id":
                entities = _extract_entities_from_entity2id(value)
                if entities:
                    previews.append(f"key='entity2id' (sorted by id):")
                    previews.extend([f"  {line}" for line in _format_top5(entities)])
                else:
                    previews.append("key='entity2id' present but not parseable as id mapping")
            elif key == "id2entity":
                pairs: List[Tuple[int, object]] = []
                for k, v in value.items():
                    try:
                        idx = int(k)
                    except Exception:
                        continue
                    pairs.append((idx, v))
                if pairs:
                    pairs.sort(key=lambda item: item[0])
                    previews.append("key='id2entity' (sorted by id):")
                    previews.extend([f"  {i}: {repr(v)}" for i, v in pairs[:5]])
                else:
                    previews.append("key='id2entity' present but not parseable")
            else:
                previews.append(f"key='{key}' is dict (size={len(value)}), preview skipped")
            continue

        previews.append(f"key='{key}' exists but unsupported type={type(value).__name__}")

    if not previews:
        return ["(no row-level metadata keys found in external payload dict)"]

    return previews


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit WN18RR node alignment for external SBERT features.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing local datasets (default: data)",
    )
    parser.add_argument(
        "--feat-pt",
        type=Path,
        default=Path("data/wn18rr_sbert.pt"),
        help="External feature .pt file to audit (default: data/wn18rr_sbert.pt)",
    )
    args = parser.parse_args()

    print("=== WN18RR Alignment Audit (read-only) ===")
    print(f"data_root: {args.data_root}")
    print(f"feat_pt:   {args.feat_pt}")

    if not args.feat_pt.exists():
        raise FileNotFoundError(f"External feature file not found: {args.feat_pt}")

    x_ext, payload = _load_external_payload(args.feat_pt)
    ext_num_nodes = int(x_ext.shape[0])
    print(f"external_x.shape: {tuple(x_ext.shape)}  dtype={x_ext.dtype}")

    native_num_nodes, native_entities, native_source = _load_native_wn18rr(args.data_root)
    print(f"native_loader: {native_source}")
    print(f"native_num_nodes: {native_num_nodes}")

    assert (
        native_num_nodes == ext_num_nodes
    ), (
        "Node-count mismatch: "
        f"native WN18RR has {native_num_nodes:,} nodes, "
        f"external features have {ext_num_nodes:,} rows."
    )
    print("[OK] num_nodes matches exactly.")

    print("\n--- Top 5 entities from native WN18RR (if available) ---")
    if native_entities:
        for line in _format_top5(native_entities):
            print(line)
    else:
        print("(native entity dictionary not exposed by loader/version)")

    print("\n--- Top 5 row-metadata entries from external .pt (if available) ---")
    for line in _extract_external_row_metadata_preview(payload):
        print(line)

    if isinstance(payload, dict):
        keys = sorted(payload.keys())
        print("\nexternal_payload_keys:", keys)

    print("\nAudit completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"[ASSERTION FAILED] {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
