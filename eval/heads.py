from __future__ import annotations

import torch
from torch import nn


def mean_pool(node_embeddings: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
    if node_embeddings.ndim != 2:
        raise ValueError(
            f"mean_pool expects a 2-D tensor, got shape={tuple(node_embeddings.shape)}"
        )

    if batch is None:
        return node_embeddings.mean(dim=0, keepdim=True)

    if batch.ndim != 1:
        raise ValueError(f"batch must be 1-D, got shape={tuple(batch.shape)}")
    if batch.numel() != node_embeddings.shape[0]:
        raise ValueError(
            "batch length must match the number of node embeddings: "
            f"{batch.numel()} != {node_embeddings.shape[0]}"
        )

    batch = batch.to(device=node_embeddings.device, dtype=torch.long)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    pooled = node_embeddings.new_zeros((num_graphs, node_embeddings.shape[1]))
    counts = node_embeddings.new_zeros((num_graphs, 1))

    pooled.index_add_(0, batch, node_embeddings)
    counts.index_add_(
        0,
        batch,
        torch.ones((batch.shape[0], 1), device=node_embeddings.device, dtype=node_embeddings.dtype),
    )
    return pooled / counts.clamp_min_(1.0)


class NodeHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(node_embeddings)


class GraphHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, node_embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        graph_embeddings = mean_pool(node_embeddings, batch)
        return self.linear(graph_embeddings)


class LinkHead(nn.Module):
    def forward(self, src_embeddings: torch.Tensor, dst_embeddings: torch.Tensor) -> torch.Tensor:
        if src_embeddings.shape != dst_embeddings.shape:
            raise ValueError(
                "src_embeddings and dst_embeddings must have the same shape, got "
                f"{tuple(src_embeddings.shape)} vs {tuple(dst_embeddings.shape)}"
            )
        return (src_embeddings * dst_embeddings).sum(dim=-1)
