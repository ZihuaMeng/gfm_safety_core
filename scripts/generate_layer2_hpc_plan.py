#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from layer2_artifact_utils import apply_text_write, json_dumps_stable, print_json, relpath_str
from layer2_hpc_plan_utils import DEFAULT_PLAN_PATH, build_plan_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic Layer 2 HPC execution plan that reuses the existing "
            "run_layer2_suite.py command surfaces and stable suite-manifest outputs."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="Output path for the generated HPC plan JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated plan metadata without writing the plan file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_plan_payload(plan_path=args.out)
    plan_text = json_dumps_stable(payload)

    if not args.dry_run:
        apply_text_write(args.out, plan_text)

    priority_counts: dict[str, int] = {}
    for target in payload["targets"]:
        priority = str(target["priority"])
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    print_json(
        {
            "dry_run": args.dry_run,
            "plan_path": relpath_str(args.out),
            "schema_version": payload["schema_version"],
            "target_count": len(payload["targets"]),
            "priority_counts": priority_counts,
            "excluded_target_count": len(payload["excluded_targets"]),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
