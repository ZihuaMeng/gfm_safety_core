"""eval/run_lp.py — Thin CLI dispatch for Layer 2 linear-probe evaluation.

Execution logic lives in eval/runners.py. This module handles:
- CLI argument parsing
- Config resolution
- Runner dispatch
- Error handling and result I/O
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.registry import (  # noqa: E402
    REGISTRY,
    VALID_MODELS,
)
from eval.runners import (  # noqa: E402
    DEBUG_GRAPH_BATCH_SIZE,
    GRAPH_BATCH_SIZE,
    EvalResult,
    GraphEvalConfig,
    format_debug_notes,
    run_graph_eval,
    run_link_eval,
    run_node_eval,
)


def _default_out_path(model: str, dataset: str) -> Path:
    return REPO_ROOT / "results" / "baseline" / f"{model}_{dataset}.json"


def _write_result(result: EvalResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


def _normalize_optional_cap(value: int | None) -> int | None:
    if value is None or value == 0:
        return None
    return value


def _resolve_graph_eval_config(args: argparse.Namespace) -> GraphEvalConfig:
    if args.debug_max_graphs < 0:
        raise ValueError("--debug_max_graphs must be >= 0.")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0.")
    if args.max_train_steps is not None and args.max_train_steps < 0:
        raise ValueError("--max_train_steps must be >= 0.")
    if args.max_eval_batches is not None and args.max_eval_batches < 0:
        raise ValueError("--max_eval_batches must be >= 0.")

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else (DEBUG_GRAPH_BATCH_SIZE if args.debug else GRAPH_BATCH_SIZE)
    )
    num_workers = 0 if args.debug else (args.num_workers if args.num_workers is not None else 0)
    return GraphEvalConfig(
        debug=args.debug,
        debug_max_graphs=args.debug_max_graphs,
        batch_size=batch_size,
        num_workers=num_workers,
        max_train_steps=_normalize_optional_cap(args.max_train_steps),
        max_eval_batches=_normalize_optional_cap(args.max_eval_batches),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal unified linear-probe evaluation scaffold.")
    parser.add_argument("--model", required=True, choices=sorted(VALID_MODELS))
    parser.add_argument("--dataset", required=True, choices=REGISTRY.list_datasets())
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_max_graphs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--feat-pt", type=str, default=None, dest="feat_pt",
                        help=(
                            "Path to external node features (.pt file, same schema as "
                            "preprocess_sbert_features.py). Required when the encoder was "
                            "trained with SBERT features. Replaces dataset features before "
                            "encoder inference."
                        ))
    parser.add_argument("--link-scorer", type=str, default=None, dest="link_scorer",
                        help=(
                            "Link scorer for link-prediction evaluation. "
                            "Options: dot_product (default), relation_diagonal. "
                            "relation_diagonal trains a per-relation diagonal scorer "
                            "on frozen node embeddings before evaluation."
                        ))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.debug:
        try:
            import torch as _torch_debug
            _torch_debug.set_num_threads(1)
            try:
                _torch_debug.set_num_interop_threads(1)
            except RuntimeError:
                pass  # interop pool already started; cannot change
            print("[debug] thread caps applied: set_num_threads(1), set_num_interop_threads(1)")
        except ImportError:
            print("[debug] torch not importable at startup; thread caps skipped")

    adapter = REGISTRY.get_adapter(args.dataset)
    task = adapter.task_type
    out_path = Path(args.out_json) if args.out_json else _default_out_path(args.model, args.dataset)
    graph_eval_config: GraphEvalConfig | None = None

    result = EvalResult(
        model=args.model,
        dataset=args.dataset,
        task=task,
        status="error",
        metric_name=None,
        metric_value=None,
        notes="Evaluation did not complete.",
    )

    try:
        adapter.validate_model(args.model)
        if args.feat_pt is not None and not adapter.supports_external_feat_pt:
            raise ValueError(
                f"--feat-pt is not supported for {adapter.dataset_name}. "
                f"This dataset uses native features only."
            )
        graph_eval_config = _resolve_graph_eval_config(args)

        if task == "node":
            result = run_node_eval(
                args.model, args.ckpt, args.device,
                debug=graph_eval_config.debug,
                max_train_steps=graph_eval_config.max_train_steps,
                feat_pt=args.feat_pt,
            )
        elif task == "graph":
            result = run_graph_eval(
                args.model, args.ckpt, args.device,
                graph_eval_config,
                feat_pt=args.feat_pt,
            )
        elif task == "link":
            link_scorer = args.link_scorer or "dot_product"
            result = run_link_eval(
                args.model, args.ckpt, args.device,
                debug=graph_eval_config.debug,
                feat_pt=args.feat_pt,
                scorer_name=link_scorer,
            )
        else:
            raise ValueError(f"Unhandled task: {task}")

    except NotImplementedError as exc:
        result = EvalResult(
            model=args.model,
            dataset=args.dataset,
            task=task,
            status="blocked",
            metric_name=None,
            metric_value=None,
            notes=str(exc),
        )
        _write_result(result, out_path)
        raise
    except Exception as exc:
        notes = f"{type(exc).__name__}: {exc}"
        if args.debug and graph_eval_config is not None:
            notes = f"{format_debug_notes(graph_eval_config)}; failure={type(exc).__name__}: {exc}"
        result = EvalResult(
            model=args.model,
            dataset=args.dataset,
            task=task,
            status="error",
            metric_name=None,
            metric_value=None,
            notes=notes,
        )
        _write_result(result, out_path)
        if args.debug:
            return
        raise

    _write_result(result, out_path)


if __name__ == "__main__":
    main()
