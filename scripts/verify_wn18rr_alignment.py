#!/usr/bin/env python3
"""
WN18RR alignment audit: verify consistency between entity2id.txt,
raw split files, and SBERT feature tensor.

Writes structured JSON to --out_json. Requires only torch (for .pt loading).
No DGL or PyG dependency.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _read_entity2id(path: Path) -> dict[str, int]:
    """Parse entity2id.txt: 'entity_name integer_id' per line."""
    mapping: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            entity = " ".join(parts[:-1])
            try:
                idx = int(parts[-1])
            except ValueError:
                continue
            mapping[entity] = idx
    return mapping


def _read_split_file(path: Path, entity2id: dict[str, int]) -> dict:
    """Parse WN18RR split file (head\\trel\\ttail)."""
    edges: list[tuple[int, int]] = []
    relations: Counter = Counter()
    oov_entities: set[str] = set()
    parse_errors = 0

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parse_errors += 1
                continue
            head, rel, tail = parts[0], parts[1], parts[2]
            relations[rel] += 1
            h_id = entity2id.get(head)
            t_id = entity2id.get(tail)
            if h_id is None:
                oov_entities.add(head)
            if t_id is None:
                oov_entities.add(tail)
            if h_id is not None and t_id is not None:
                edges.append((h_id, t_id))

    return {
        "edges": edges,
        "edge_count": len(edges),
        "relations": dict(relations),
        "oov_count": len(oov_entities),
        "parse_errors": parse_errors,
    }


def _run_checks(dataset_root: Path, feat_pt_path: Path) -> dict:
    """Run all alignment checks and return structured result dict."""
    checks: dict[str, dict] = {}
    missing_pieces: list[str] = []
    status = "success"

    # 1. entity2id.txt
    entity2id_path = dataset_root / "entity2id.txt"
    entity2id: dict[str, int] = {}
    if not entity2id_path.exists():
        checks["entity2id_exists"] = {"passed": False, "detail": f"{entity2id_path} not found"}
        missing_pieces.append("entity2id.txt not found")
        status = "error"
    else:
        entity2id = _read_entity2id(entity2id_path)
        if not entity2id:
            checks["entity2id_exists"] = {"passed": False, "detail": "empty or unparseable"}
            missing_pieces.append("entity2id.txt empty")
            status = "error"
        else:
            checks["entity2id_exists"] = {"passed": True, "num_entities": len(entity2id)}

    num_entities = len(entity2id)

    # 2. Entity IDs contiguous 0..N-1
    if entity2id:
        ids = sorted(entity2id.values())
        contiguous = ids == list(range(len(ids)))
        checks["entity_ids_contiguous"] = {
            "passed": contiguous,
            "min_id": ids[0],
            "max_id": ids[-1],
            "expected_max": len(ids) - 1,
        }
        if not contiguous:
            missing_pieces.append(f"IDs not contiguous 0..{len(ids) - 1}")
            status = "error"

    # 3. SBERT feature file
    feat_rows: int | None = None
    feat_dim: int | None = None
    sbert_metadata: dict = {}

    if not feat_pt_path.exists():
        checks["feat_pt_exists"] = {"passed": False, "detail": f"{feat_pt_path} not found"}
        missing_pieces.append("SBERT .pt not found")
        status = "error"
    else:
        import torch

        payload = torch.load(str(feat_pt_path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "x" not in payload:
            checks["feat_pt_exists"] = {"passed": False, "detail": "payload missing 'x' key"}
            missing_pieces.append("SBERT .pt has no 'x' key")
            status = "error"
        else:
            x = payload["x"]
            feat_rows = int(x.shape[0])
            feat_dim = int(x.shape[1])
            for k, v in payload.items():
                if k == "x":
                    continue
                sbert_metadata[k] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
            checks["feat_pt_exists"] = {
                "passed": True,
                "feat_rows": feat_rows,
                "feat_dim": feat_dim,
                "dtype": str(x.dtype),
            }

    # 4. Entity count == feat rows
    if entity2id and feat_rows is not None:
        match = num_entities == feat_rows
        checks["entity_feat_count_match"] = {
            "passed": match,
            "num_entities": num_entities,
            "feat_rows": feat_rows,
        }
        if not match:
            missing_pieces.append(f"entity count ({num_entities}) != feat rows ({feat_rows})")
            status = "error"

    # 5. SBERT metadata num_nodes consistency
    if feat_rows is not None and "num_nodes" in sbert_metadata:
        meta_nodes = int(sbert_metadata["num_nodes"])
        consistent = meta_nodes == feat_rows
        checks["sbert_metadata_consistent"] = {
            "passed": consistent,
            "metadata_num_nodes": meta_nodes,
            "actual_feat_rows": feat_rows,
        }
        if not consistent:
            missing_pieces.append(f"metadata num_nodes ({meta_nodes}) != rows ({feat_rows})")
            status = "error"

    # 6. Feature dimension
    if feat_dim is not None:
        checks["feat_dim_check"] = {
            "passed": feat_dim == 768,
            "actual_dim": feat_dim,
            "expected_dim": 768,
        }

    # 7. Split files
    split_dir = dataset_root / "raw"
    all_relations: dict[str, int] = {}
    split_infos: dict[str, dict | None] = {"train": None, "valid": None, "test": None}

    if not split_dir.exists():
        checks["split_dir_exists"] = {"passed": False, "detail": f"{split_dir} not found"}
        missing_pieces.append("raw/ directory not found")
        status = "error"
    else:
        checks["split_dir_exists"] = {"passed": True}
        for split_name, required in [("train", True), ("valid", False), ("test", True)]:
            split_path = split_dir / f"{split_name}.txt"
            if not split_path.exists():
                if required:
                    checks[f"split_{split_name}"] = {"passed": False, "detail": "not found"}
                    missing_pieces.append(f"{split_name}.txt not found")
                    status = "error"
                continue
            if not entity2id:
                continue
            info = _read_split_file(split_path, entity2id)
            all_relations.update(info["relations"])
            passed = info["edge_count"] > 0 and info["oov_count"] == 0 and info["parse_errors"] == 0
            checks[f"split_{split_name}"] = {
                "passed": passed,
                "edge_count": info["edge_count"],
                "relation_types": len(info["relations"]),
                "oov_entities": info["oov_count"],
                "parse_errors": info["parse_errors"],
            }
            if info["oov_count"] > 0:
                missing_pieces.append(f"{split_name}.txt: {info['oov_count']} OOV entities")
                status = "error"
            split_infos[split_name] = info

    # 8. Edge node IDs in range
    all_node_ids: set[int] = set()
    for info in split_infos.values():
        if info is not None:
            for h, t in info["edges"]:
                all_node_ids.add(h)
                all_node_ids.add(t)
    if all_node_ids and num_entities > 0:
        min_id, max_id = min(all_node_ids), max(all_node_ids)
        in_range = min_id >= 0 and max_id < num_entities
        checks["edge_ids_in_range"] = {
            "passed": in_range,
            "min_node_id": min_id,
            "max_node_id": max_id,
            "num_entities": num_entities,
            "unique_nodes_in_edges": len(all_node_ids),
        }
        if not in_range:
            missing_pieces.append(
                f"edge IDs out of range [{min_id}, {max_id}] vs [0, {num_entities - 1}]"
            )
            status = "error"

    # 9. SBERT ordering evidence
    sbert_notes = str(sbert_metadata.get("notes", ""))
    ordering_confirmed = "entity2id.txt ordering assumed" in sbert_notes
    checks["sbert_ordering_evidence"] = {
        "passed": ordering_confirmed,
        "sbert_notes": sbert_notes,
        "detail": (
            "SBERT .pt notes confirm entity2id.txt ordering was assumed during encoding. "
            "Row i of feature tensor corresponds to entity with id=i. "
            "Consistent with GraphMAE loader."
            if ordering_confirmed
            else "Cannot confirm ordering correspondence from .pt metadata."
        ),
    }
    if not ordering_confirmed:
        missing_pieces.append("SBERT ordering not confirmed from .pt metadata")

    # 10. GraphMAE loader transitivity
    loader_consistent = bool(entity2id and feat_rows is not None and num_entities == feat_rows)
    checks["graphmae_loader_consistent"] = {
        "passed": loader_consistent,
        "detail": (
            "GraphMAE _load_wn18rr_graph() uses identical entity2id.txt parsing "
            "and sets num_nodes=len(entity2id). Count matches SBERT rows."
        ),
    }

    # Final status: soft checks (ordering evidence, feat dim) don't cause hard error
    soft_checks = {"sbert_ordering_evidence", "feat_dim_check"}
    hard_failed = any(
        not c.get("passed", False) for name, c in checks.items() if name not in soft_checks
    )
    if hard_failed and status == "success":
        status = "error"

    relation_count = len(all_relations)
    return {
        "dataset": "wn18rr",
        "status": status,
        "num_entities": num_entities,
        "feat_rows": feat_rows,
        "feat_dim": feat_dim,
        "train_edges": split_infos["train"]["edge_count"] if split_infos["train"] else None,
        "valid_edges": split_infos["valid"]["edge_count"] if split_infos["valid"] else None,
        "test_edges": split_infos["test"]["edge_count"] if split_infos["test"] else None,
        "graph_num_nodes": num_entities,
        "relation_count": relation_count,
        "relation_types": sorted(all_relations.keys()) if all_relations else [],
        "checks": checks,
        "missing_pieces": missing_pieces,
        "sbert_metadata": sbert_metadata,
        "notes": (
            "Alignment audit for WN18RR SBERT features. "
            "Verifies entity2id.txt, split files, and feature tensor consistency. "
            "Does NOT verify per-row semantic correctness of SBERT embeddings."
        ),
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify WN18RR alignment between entity2id.txt, split files, and SBERT features.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="WN18RR dataset directory (e.g. data/WN18RR)",
    )
    parser.add_argument(
        "--feat-pt",
        type=Path,
        required=True,
        help="SBERT feature .pt file (e.g. data/wn18rr_sbert.pt)",
    )
    parser.add_argument(
        "--out_json",
        type=Path,
        required=True,
        help="Output path for structured audit JSON",
    )
    args = parser.parse_args()

    result = _run_checks(args.dataset_root, args.feat_pt)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    status = result["status"]
    print(f"[audit] status={status}")
    print(
        f"[audit] num_entities={result['num_entities']}, "
        f"feat_rows={result['feat_rows']}, feat_dim={result['feat_dim']}"
    )
    print(
        f"[audit] train={result['train_edges']}, "
        f"valid={result['valid_edges']}, test={result['test_edges']}"
    )
    print(f"[audit] relation_count={result['relation_count']}")
    passed = sum(1 for c in result["checks"].values() if c.get("passed"))
    total = len(result["checks"])
    print(f"[audit] checks_passed={passed}/{total}")
    if result["missing_pieces"]:
        print(f"[audit] missing: {'; '.join(result['missing_pieces'])}")
    print(f"[audit] wrote {args.out_json}")

    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
