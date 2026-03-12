"""
SBERT-only baseline on ogbn-arxiv.

Loads precomputed SBERT node features from data/arxiv_sbert.pt,
trains a linear classifier on official OGB splits, and reports per-seed results.

Example:
    python src/sbert_only_baseline_arxiv.py \
        --seeds 0 1 2 \
        --epochs 300 \
        --lr 0.01 \
        --wd 0.0 \
        --out_json outputs/sbert_only_arxiv.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, NodePropPredDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SBERT-only linear baseline for ogbn-arxiv")
    parser.add_argument("--feature_pt", type=str, default="data/arxiv_sbert.pt")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--out_json", type=str, default="outputs/sbert_only_arxiv.json")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sbert_features(path: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        if "x" not in payload:
            raise KeyError(f"Feature file '{path}' is dict-like but missing key 'x'.")
        x = payload["x"]
    elif torch.is_tensor(payload):
        x = payload
    else:
        raise TypeError(f"Unsupported feature payload type: {type(payload)}")

    if not torch.is_tensor(x):
        raise TypeError(f"Loaded 'x' is not a tensor, got: {type(x)}")
    if x.dtype != torch.float32:
        raise TypeError(f"Expected float32 features, got {x.dtype}")
    if x.dim() != 2:
        raise ValueError(f"Expected 2-D features [N, D], got shape {tuple(x.shape)}")
    if x.shape != (169343, 768):
        raise ValueError(
            f"Expected feature shape (169343, 768), got {tuple(x.shape)}"
        )
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise ValueError("Features contain NaN or Inf values.")
    return x.contiguous()


def accuracy(evaluator: Evaluator, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    out = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return float(out["acc"])


def run_one_seed(
    seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    idx_train: torch.Tensor,
    idx_valid: torch.Tensor,
    idx_test: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    wd: float,
) -> dict:
    set_seed(seed)

    model = torch.nn.Linear(x.size(1), num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    evaluator = Evaluator(name="ogbn-arxiv")

    best_val = -1.0
    best_test_at_best_val = -1.0
    best_epoch = -1

    y_train = y[idx_train].view(-1)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits_train = model(x[idx_train])
        loss = F.cross_entropy(logits_train, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1, keepdim=True)

            val_acc = accuracy(evaluator, y[idx_valid], pred[idx_valid])
            test_acc = accuracy(evaluator, y[idx_test], pred[idx_test])

        if val_acc > best_val:
            best_val = val_acc
            best_test_at_best_val = test_acc
            best_epoch = epoch

    result = {
        "seed": int(seed),
        "best_val_acc": float(best_val),
        "best_test_acc_at_best_val": float(best_test_at_best_val),
        "best_epoch": int(best_epoch),
    }
    return result


def main() -> None:
    args = parse_args()

    x = load_sbert_features(args.feature_pt)
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=args.data_root)
    split_idx = dataset.get_idx_split()
    if split_idx is None:
        raise RuntimeError("Failed to get official OGB splits (split_idx is None).")
    _, labels = dataset[0]

    y = torch.from_numpy(labels).long()
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    if y.size(0) != x.size(0):
        raise ValueError(
            f"Feature/label node count mismatch: x has {x.size(0)}, y has {y.size(0)}"
        )

    idx_train = torch.from_numpy(split_idx["train"])
    idx_valid = torch.from_numpy(split_idx["valid"])
    idx_test = torch.from_numpy(split_idx["test"])

    num_classes = int(dataset.num_classes)

    print("=== SBERT-only baseline: ogbn-arxiv ===")
    print(f"feature_pt={args.feature_pt}")
    print(f"data_root={args.data_root}")
    print(f"x_shape={tuple(x.shape)}, x_dtype={x.dtype}")
    print(f"num_classes={num_classes}")
    print(
        f"split_sizes train={idx_train.numel()} valid={idx_valid.numel()} test={idx_test.numel()}"
    )
    print(
        f"hyperparams epochs={args.epochs} lr={args.lr} wd={args.wd} seeds={args.seeds}"
    )

    seed_results: list[dict] = []
    for seed in args.seeds:
        result = run_one_seed(
            seed=seed,
            x=x,
            y=y,
            idx_train=idx_train,
            idx_valid=idx_valid,
            idx_test=idx_test,
            num_classes=num_classes,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
        )
        seed_results.append(result)
        print(
            "seed={seed} best_val_acc={best_val:.6f} "
            "best_test_acc_at_best_val={best_test:.6f} best_epoch={best_epoch}".format(
                seed=result["seed"],
                best_val=result["best_val_acc"],
                best_test=result["best_test_acc_at_best_val"],
                best_epoch=result["best_epoch"],
            )
        )

    best_vals = np.array([r["best_val_acc"] for r in seed_results], dtype=np.float64)
    best_tests = np.array(
        [r["best_test_acc_at_best_val"] for r in seed_results], dtype=np.float64
    )

    summary = {
        "val_acc_mean": float(best_vals.mean()),
        "val_acc_std": float(best_vals.std(ddof=0)),
        "test_acc_at_best_val_mean": float(best_tests.mean()),
        "test_acc_at_best_val_std": float(best_tests.std(ddof=0)),
    }

    payload = {
        "dataset": "ogbn-arxiv",
        "feature_pt": args.feature_pt,
        "data_root": args.data_root,
        "num_nodes": int(x.size(0)),
        "in_dim": int(x.size(1)),
        "num_classes": num_classes,
        "split_sizes": {
            "train": int(idx_train.numel()),
            "valid": int(idx_valid.numel()),
            "test": int(idx_test.numel()),
        },
        "hyperparams": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "wd": float(args.wd),
            "seeds": [int(s) for s in args.seeds],
            "model": "torch.nn.Linear",
            "optimizer": "AdamW",
            "loss": "cross_entropy",
        },
        "seed_results": seed_results,
        "summary": summary,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(
        "summary val_mean={val_mean:.6f}±{val_std:.6f} "
        "test_mean={test_mean:.6f}±{test_std:.6f}".format(
            val_mean=summary["val_acc_mean"],
            val_std=summary["val_acc_std"],
            test_mean=summary["test_acc_at_best_val_mean"],
            test_std=summary["test_acc_at_best_val_std"],
        )
    )
    print(f"wrote_json={args.out_json}")


if __name__ == "__main__":
    main()
