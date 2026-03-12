"""Encoder checkpoint export contract for Layer 2 Step 1 -> Step 2 handoff.

Saved dict contract (consumed by eval/load_encoder.py):
    encoder           - OrderedDict[str, Tensor], bare encoder state_dict (no prefix)
    model_name        - "graphmae" or "bgrl"
    dataset           - e.g. "ogbn-arxiv"
    task_type         - "node", "graph", "link", or None
    hidden_dim        - int, encoder output dimension
    encoder_input_dim - int, encoder expected input feature dimension
    backend           - "dgl" or "pyg"
    exported_at       - ISO-8601 UTC timestamp
"""
from __future__ import annotations

import datetime
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


def export_encoder_checkpoint(
    encoder_state_dict: dict[str, torch.Tensor],
    out_path: str | Path,
    *,
    model_name: str,
    dataset: str,
    task_type: str | None = None,
    hidden_dim: int | None = None,
    encoder_input_dim: int | None = None,
    backend: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save an encoder-only checkpoint for downstream eval consumption.

    The saved file is loadable by ``eval/load_encoder.py`` via its existing
    ``_extract_state_dict`` -> ``"encoder"`` key path.

    ``encoder_input_dim`` records the feature dimension the encoder was trained
    with, enabling Step 2 to validate features before inference.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "encoder": OrderedDict(encoder_state_dict),
        "model_name": model_name,
        "dataset": dataset,
        "task_type": task_type,
        "hidden_dim": hidden_dim,
        "encoder_input_dim": encoder_input_dim,
        "backend": backend,
        "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if extra_metadata:
        payload.update(extra_metadata)

    torch.save(payload, out_path)
    print(f"[export] Encoder checkpoint saved: {out_path}")
    print(f"[export]   model_name={model_name}, dataset={dataset}, task_type={task_type}")
    print(f"[export]   hidden_dim={hidden_dim}, encoder_input_dim={encoder_input_dim}")
    print(f"[export]   keys={len(encoder_state_dict)}")
    return out_path.resolve()
