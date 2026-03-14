"""eval/runners.py — Task execution engines for Layer 2 linear-probe evaluation.

Public entry points:
    run_node_eval()   — Node classification (GraphMAE / BGRL on ogbn-arxiv)
    run_graph_eval()  — Graph classification (GraphMAE on ogbg-molpcba)
    run_link_eval()   — Link prediction (experimental WN18RR, GraphMAE / BGRL)

Each runner owns: feature preparation, dataset/task validation, head construction,
train/eval loop execution, evaluator dispatch, and result JSON assembly.

Torch and model-specific dependencies are lazily imported inside each runner
to keep CLI startup fast.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.registry import (  # noqa: E402
    GRAPH_RUNNER,
    LINK_RUNNER,
    NODE_RUNNER,
    REGISTRY,
)
from eval.link_protocol import (  # noqa: E402
    DEFAULT_NEGATIVE_SAMPLING_CONTRACT,
    DEFAULT_RANKING_PROTOCOL,
    DotProductScorer,
    LinkDatasetProtocol,
    LinkScorer,
    NegativeSamplingContract,
    RankingProtocol,
    build_filter_sets,
    compute_link_metrics,
    get_scorer,
    train_link_scorer,
)

GRAPHMAE_ROOT = REPO_ROOT / "repos" / "graphmae"
BGRL_ROOT = REPO_ROOT / "repos" / "bgrl"

# Derived constants from task runner configs
NODE_EPOCHS = NODE_RUNNER.epochs
NODE_LR = NODE_RUNNER.lr
GRAPH_EPOCHS = GRAPH_RUNNER.epochs
GRAPH_LR = GRAPH_RUNNER.lr
GRAPH_BATCH_SIZE = GRAPH_RUNNER.batch_size
DEBUG_GRAPH_BATCH_SIZE = GRAPH_RUNNER.debug_batch_size
LINK_DEBUG_MAX_TEST_EDGES = LINK_RUNNER.debug_max_test_edges


# ---------------------------------------------------------------------------
# Data classes (no torch dependency)
# ---------------------------------------------------------------------------


def _eval_status(debug: bool) -> str:
    """Return the appropriate status string for a successful eval run."""
    return "debug_success" if debug else "success"


@dataclass
class EvalResult:
    model: str
    dataset: str
    task: str
    status: str
    metric_name: str | None
    metric_value: float | None
    notes: str


@dataclass(frozen=True)
class GraphEvalConfig:
    debug: bool
    debug_max_graphs: int
    batch_size: int
    num_workers: int
    max_train_steps: int | None
    max_eval_batches: int | None
    ft_epochs: int | None = None
    graph_eval_every: int | None = None


# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------

torch = None
F = None
GraphHead = None
LinkHead = None
NodeHead = None
mean_pool = None
load_encoder = None


def _ensure_eval_deps() -> None:
    global F, GraphHead, LinkHead, NodeHead, mean_pool, load_encoder, torch

    if torch is None or F is None:
        import torch as _torch
        import torch.nn.functional as _F

        torch = _torch
        F = _F

    if GraphHead is None or LinkHead is None or NodeHead is None or mean_pool is None:
        from eval.heads import (
            GraphHead as _GraphHead,
            LinkHead as _LinkHead,
            NodeHead as _NodeHead,
            mean_pool as _mean_pool,
        )

        GraphHead = _GraphHead
        LinkHead = _LinkHead
        NodeHead = _NodeHead
        mean_pool = _mean_pool

    if load_encoder is None:
        from eval.load_encoder import load_encoder as _load_encoder

        load_encoder = _load_encoder


# ---------------------------------------------------------------------------
# Context managers & module loading
# ---------------------------------------------------------------------------


def _load_module_from_path(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module {module_name} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _graphmae_context():
    previous = Path.cwd()
    path = GRAPHMAE_ROOT.resolve()
    added = False
    try:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
            added = True
        if previous != path:
            os.chdir(path)
        yield path
    finally:
        os.chdir(previous)
        if added and str(path) in sys.path:
            sys.path.remove(str(path))


@contextlib.contextmanager
def _bgrl_context():
    path = BGRL_ROOT.resolve()
    added = False
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
        added = True
    try:
        yield path
    finally:
        if added and str(path) in sys.path:
            sys.path.remove(str(path))


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def format_debug_notes(config: GraphEvalConfig) -> str:
    """Format debug-mode notes for result metadata."""
    truncation = "per_split_first_n" if config.debug and config.debug_max_graphs > 0 else "disabled"
    max_train_steps = config.max_train_steps if config.max_train_steps is not None else "none"
    max_eval_batches = config.max_eval_batches if config.max_eval_batches is not None else "none"
    graph_eval_every = config.graph_eval_every if config.graph_eval_every is not None else "every_epoch"
    return (
        "local_debug_run=true; "
        f"debug_mode={str(config.debug).lower()}; "
        f"debug_max_graphs={config.debug_max_graphs}; "
        f"batch_size={config.batch_size}; "
        f"num_workers={config.num_workers}; "
        f"max_train_steps={max_train_steps}; "
        f"max_eval_batches={max_eval_batches}; "
        f"graph_eval_every={graph_eval_every}; "
        f"split_truncation={truncation}; "
        "official_metric=false"
    )


def _format_note_value(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _structured_notes(code: str, **fields: object) -> str:
    parts = [f"code={code}"]
    parts.extend(f"{key}={_format_note_value(value)}" for key, value in fields.items())
    return "; ".join(parts)


def _mask_to_index(mask):
    if mask.dtype == torch.bool:
        return mask
    raise TypeError(f"Expected boolean mask, got dtype={mask.dtype}")


def _accuracy(logits, labels, mask) -> float:
    mask = _mask_to_index(mask)
    if mask.sum().item() == 0:
        raise ValueError("Evaluation mask is empty.")
    pred = logits[mask].argmax(dim=-1)
    truth = labels[mask]
    return float((pred == truth).float().mean().item())


def _select_best_state(best_state, module):
    if best_state is not None:
        return best_state
    return copy.deepcopy(module.state_dict())


def _batch_index_from_counts(counts, device: str):
    counts = counts.to(dtype=torch.long, device=device)
    return torch.repeat_interleave(
        torch.arange(counts.shape[0], device=device, dtype=torch.long),
        counts,
    )


def _masked_bce_loss(logits, labels):
    mask = torch.isfinite(labels)
    if mask.sum().item() == 0:
        raise ValueError("All graph labels are NaN for this batch.")
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def _validate_feature_dim(
    encoder,
    actual_dim: int,
    feat_pt_provided: bool,
    dataset: str,
) -> None:
    """Validate that input features match what the encoder expects."""
    expected_dim = getattr(encoder, "input_dim", None)
    if expected_dim is None:
        return
    if actual_dim == expected_dim:
        return
    if not feat_pt_provided:
        raise ValueError(
            f"Feature dimension mismatch: encoder expects {expected_dim}-dim input but "
            f"{dataset} provides {actual_dim}-dim features. This encoder was likely trained "
            f"with external features (e.g. SBERT {expected_dim}d). Provide --feat-pt with "
            f"the same feature file used during training."
        )
    raise ValueError(
        f"Feature dimension mismatch after --feat-pt replacement: encoder expects "
        f"{expected_dim}-dim input but features have {actual_dim}-dim."
    )


def _load_feat_pt(path: str, expected_nodes: int):
    """Load external node features from a .pt file."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if "x" not in payload:
            raise KeyError(f"--feat-pt dict has no 'x' key. Keys: {list(payload.keys())}")
        x = payload["x"]
        print(f"[feat_pt] encoder={payload.get('encoder', '?')}, dim={payload.get('dim', '?')}")
    elif torch.is_tensor(payload):
        x = payload
    else:
        raise TypeError(f"--feat-pt: expected dict or Tensor, got {type(payload).__name__}")
    if x.shape[0] != expected_nodes:
        raise ValueError(
            f"--feat-pt node count mismatch: file has {x.shape[0]:,}, dataset has {expected_nodes:,}"
        )
    return x.float()


def _checkpoint_metadata(encoder) -> dict[str, object]:
    metadata = getattr(encoder, "checkpoint_metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return {}


# ---------------------------------------------------------------------------
# Node evaluation
# ---------------------------------------------------------------------------


@dataclass
class _NodePreparedData:
    """Intermediate data for node classification after encoding."""
    node_embeddings: object  # torch.Tensor
    labels: object           # torch.Tensor
    train_mask: object       # torch.Tensor
    val_mask: object         # torch.Tensor
    test_mask: object        # torch.Tensor
    num_classes: int
    hidden_dim: int


def _prepare_node_graphmae(encoder, device: str, feat_pt: str | None) -> _NodePreparedData:
    """Load ogbn-arxiv via GraphMAE, encode nodes, return prepared data."""
    with _graphmae_context():
        from graphmae.datasets.data_util import load_dataset

        graph, (_, num_classes) = load_dataset("ogbn-arxiv")

    if feat_pt is not None:
        graph.ndata["feat"] = _load_feat_pt(feat_pt, graph.num_nodes())

    graph = graph.to(device)
    features = graph.ndata["feat"].to(device)
    labels = graph.ndata["label"].view(-1).long().to(device)
    train_mask = graph.ndata["train_mask"].bool()
    val_mask = graph.ndata["val_mask"].bool()
    test_mask = graph.ndata["test_mask"].bool()

    _validate_feature_dim(encoder, features.shape[1], feat_pt is not None, "ogbn-arxiv")

    with torch.no_grad():
        node_embeddings = encoder(graph, features).detach()

    return _NodePreparedData(
        node_embeddings=node_embeddings,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=int(num_classes),
        hidden_dim=encoder.hidden_dim,
    )


def _prepare_node_bgrl(encoder, device: str, feat_pt: str | None) -> _NodePreparedData:
    """Load ogbn-arxiv via BGRL, encode nodes, return prepared data."""
    with _bgrl_context():
        data_module = _load_module_from_path("_gfm_safety_bgrl_data", BGRL_ROOT / "data.py")
        dataset = data_module.Dataset(root=str(REPO_ROOT / "data"), name="ogbn-arxiv")[0]

    if feat_pt is not None:
        dataset.x = _load_feat_pt(feat_pt, dataset.num_nodes)

    dataset = dataset.to(device)
    labels = dataset.y.view(-1).long()
    train_mask = dataset.train_mask[0].bool()
    val_mask = dataset.val_mask[0].bool()
    test_mask = dataset.test_mask[0].bool()
    num_classes = int(labels.max().item()) + 1

    _validate_feature_dim(encoder, dataset.x.shape[1], feat_pt is not None, "ogbn-arxiv")

    with torch.no_grad():
        node_embeddings = encoder(dataset.x, dataset.edge_index, dataset.edge_attr).detach()

    return _NodePreparedData(
        node_embeddings=node_embeddings,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=num_classes,
        hidden_dim=encoder.hidden_dim,
    )


def _node_train_eval_loop(
    prepared: _NodePreparedData,
    device: str,
    *,
    debug: bool,
    max_train_steps: int | None,
) -> tuple[float, int]:
    """Shared node classification training loop. Returns (best_test_acc, train_steps)."""
    head = NodeHead(hidden_dim=prepared.hidden_dim, num_classes=prepared.num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=NODE_LR)

    best_val = float("-inf")
    best_state = None
    best_test = float("nan")
    train_steps = 0

    for _ in range(NODE_EPOCHS):
        head.train()
        logits = head(prepared.node_embeddings)
        loss = F.cross_entropy(logits[prepared.train_mask], prepared.labels[prepared.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_steps += 1

        head.eval()
        with torch.no_grad():
            logits = head(prepared.node_embeddings)
            val_acc = _accuracy(logits, prepared.labels, prepared.val_mask)
            test_acc = _accuracy(logits, prepared.labels, prepared.test_mask)
        if val_acc >= best_val:
            best_val = val_acc
            best_test = test_acc
            best_state = copy.deepcopy(head.state_dict())

        if max_train_steps is not None and train_steps >= max_train_steps:
            if debug:
                print(f"[debug] node max_train_steps reached: stopping after {max_train_steps} step(s).")
            break

    head.load_state_dict(_select_best_state(best_state, head))
    return best_test, train_steps


_NODE_PREPARE = {
    "graphmae": _prepare_node_graphmae,
    "bgrl": _prepare_node_bgrl,
}

_NODE_NOTES = {
    "graphmae": (
        "Frozen GraphMAE encoder with a linear node head on official ogbn-arxiv "
        "train/valid/test node masks."
    ),
    "bgrl": (
        "Frozen BGRL student encoder with a linear node head on the official "
        "ogbn-arxiv split cached through repos/bgrl/data.py."
    ),
}


def run_node_eval(
    model: str,
    ckpt: str,
    device: str,
    *,
    debug: bool = False,
    max_train_steps: int | None = None,
    feat_pt: str | None = None,
) -> EvalResult:
    """Run node classification linear-probe evaluation.

    Loads encoder from checkpoint, prepares data using model-specific loader,
    trains a shared linear head, and returns evaluation metrics.
    """
    adapter = REGISTRY.get_adapter("ogbn-arxiv")
    adapter.validate_model(model)

    prepare_fn = _NODE_PREPARE.get(model)
    if prepare_fn is None:
        raise ValueError(
            f"Unsupported node model: {model!r}. "
            f"Supported: {', '.join(adapter.supported_models)}"
        )

    _ensure_eval_deps()
    encoder = load_encoder(model, ckpt, device)
    prepared = prepare_fn(encoder, device, feat_pt)
    best_test, train_steps = _node_train_eval_loop(
        prepared, device, debug=debug, max_train_steps=max_train_steps,
    )

    if debug:
        notes = (
            f"local_debug_run=true; max_train_steps={max_train_steps}; "
            f"actual_steps={train_steps}; official_metric=false"
        )
    else:
        notes = _NODE_NOTES.get(
            model,
            f"Frozen {model} encoder with linear node head on ogbn-arxiv.",
        )

    return EvalResult(
        model=model,
        dataset="ogbn-arxiv",
        task="node",
        status=_eval_status(debug),
        metric_name="accuracy",
        metric_value=float(best_test),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Graph evaluation (PCBA)
# ---------------------------------------------------------------------------


class _PCBADatasetView:
    def __init__(self, dataset, indices):
        self._dataset = dataset
        self._indices = [int(idx) for idx in indices]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        dgl_graph, label = self._dataset[self._indices[idx]]
        dgl_graph = dgl_graph.remove_self_loop().add_self_loop()
        if "feat" in dgl_graph.ndata:
            node_attr = dgl_graph.ndata["feat"].float()
        else:
            node_attr = torch.ones((dgl_graph.num_nodes(), 1), dtype=torch.float32)
        dgl_graph.ndata["attr"] = node_attr
        return dgl_graph, torch.as_tensor(label).float().view(1, -1)


def _pcba_labels_tensor(dataset):
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise ValueError(
            _structured_notes(
                "pcba_labels_unavailable",
                detail=(
                    "DglGraphPropPredDataset.labels is unavailable; cannot build an AP-aware "
                    "debug subset without materializing graphs"
                ),
            )
        )
    labels_tensor = torch.as_tensor(labels).float()
    if labels_tensor.ndim == 1:
        labels_tensor = labels_tensor.view(-1, 1)
    return labels_tensor


def _pcba_ap_stats(labels) -> tuple[int, int, int, int]:
    if labels.ndim == 1:
        labels = labels.view(-1, 1)

    usable_tasks = 0
    positive_labels = 0
    negative_labels = 0
    labeled_positions = int(torch.isfinite(labels).sum().item())
    for task_id in range(labels.shape[1]):
        task_labels = labels[:, task_id]
        finite_mask = torch.isfinite(task_labels)
        if not finite_mask.any():
            continue
        task_labels = task_labels[finite_mask]
        task_pos = int((task_labels == 1).sum().item())
        task_neg = int((task_labels == 0).sum().item())
        positive_labels += task_pos
        negative_labels += task_neg
        if task_pos > 0 and task_neg > 0:
            usable_tasks += 1

    return usable_tasks, positive_labels, negative_labels, labeled_positions


def _pcba_eval_graph_cap(
    graph_eval_config: GraphEvalConfig,
    split_len: int,
) -> int:
    if not graph_eval_config.debug or graph_eval_config.debug_max_graphs <= 0:
        return split_len
    target = min(split_len, graph_eval_config.debug_max_graphs)
    if graph_eval_config.max_eval_batches is not None:
        target = min(target, graph_eval_config.batch_size * graph_eval_config.max_eval_batches)
    return max(target, 1)


def _find_pcba_ap_seed_positions(split_labels) -> tuple[int, int, int] | None:
    best_seed: tuple[int, int, int] | None = None
    best_sort_key: tuple[int, int, int] | None = None

    for task_id in range(split_labels.shape[1]):
        task_labels = split_labels[:, task_id]
        pos_positions = torch.nonzero(task_labels == 1, as_tuple=False).view(-1)
        neg_positions = torch.nonzero(task_labels == 0, as_tuple=False).view(-1)
        if pos_positions.numel() == 0 or neg_positions.numel() == 0:
            continue

        pos_idx = int(pos_positions[0].item())
        neg_idx = int(neg_positions[0].item())
        first_idx, second_idx = sorted((pos_idx, neg_idx))
        sort_key = (second_idx, first_idx, task_id)
        if best_sort_key is None or sort_key < best_sort_key:
            best_sort_key = sort_key
            best_seed = (first_idx, second_idx, task_id)

    return best_seed


def _select_pcba_debug_eval_indices(
    dataset,
    indices,
    graph_eval_config: GraphEvalConfig,
    split_name: str,
):
    if not graph_eval_config.debug or graph_eval_config.debug_max_graphs <= 0:
        return indices

    indices = torch.as_tensor(indices, dtype=torch.long)
    target_size = min(int(indices.numel()), graph_eval_config.debug_max_graphs)
    if target_size == 0:
        return indices[:0]

    consumed_graph_cap = _pcba_eval_graph_cap(graph_eval_config, target_size)
    truncated = indices[:target_size]
    labels = _pcba_labels_tensor(dataset)

    consumed_labels = labels[truncated[:consumed_graph_cap]]
    usable_tasks, positive_labels, negative_labels, labeled_positions = _pcba_ap_stats(consumed_labels)
    if usable_tasks > 0:
        print(
            f"[debug][pcba] split={split_name} using prefix subset: "
            f"graphs={target_size}, consumed_graphs={consumed_graph_cap}, usable_tasks={usable_tasks}"
        )
        return truncated

    if consumed_graph_cap < 2:
        print(
            f"[debug][pcba] split={split_name} prefix subset cannot be AP-valid with "
            f"consumed_graphs={consumed_graph_cap}; AP requires at least 2 consumed graphs."
        )
        return truncated

    split_labels = labels[indices]
    seed = _find_pcba_ap_seed_positions(split_labels)
    if seed is None:
        print(
            f"[debug][pcba] split={split_name} no AP-valid seed exists in the full split; "
            f"falling back to the deterministic prefix subset."
        )
        return truncated

    first_pos, second_pos, task_id = seed
    selected_positions = [first_pos, second_pos]
    used_positions = {first_pos, second_pos}
    for position in range(indices.numel()):
        if len(selected_positions) >= target_size:
            break
        if position in used_positions:
            continue
        selected_positions.append(position)

    selected = indices[torch.as_tensor(selected_positions, dtype=torch.long)]
    selected_labels = labels[selected[:consumed_graph_cap]]
    usable_tasks, positive_labels, negative_labels, labeled_positions = _pcba_ap_stats(selected_labels)
    print(
        f"[debug][pcba] split={split_name} reordered subset for AP stability: "
        f"graphs={target_size}, consumed_graphs={consumed_graph_cap}, "
        f"seed_task={task_id}, usable_tasks={usable_tasks}, "
        f"positive_labels={positive_labels}, negative_labels={negative_labels}, "
        f"labeled_positions={labeled_positions}"
    )
    return selected


def _collate_graph_batch(batch):
    import dgl

    graphs = [item[0] for item in batch]
    labels = torch.cat([item[1] for item in batch], dim=0)
    return dgl.batch(graphs), labels


# ---------------------------------------------------------------------------
# Frozen-encoder precompute for graph eval
# ---------------------------------------------------------------------------


def _precompute_graph_embeddings(encoder, dataloader, device, *, max_batches=None, debug=False):
    """Run frozen encoder + mean-pool once over a DataLoader.

    Returns ``(graph_embeddings, labels)`` as **CPU** tensors.
    ``graph_embeddings`` has shape ``[num_graphs, hidden_dim]``.
    """
    all_embeddings: list = []
    all_labels: list = []

    for batch_idx, (batch_graph, labels) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            if debug:
                print(f"[debug] precompute: stopped after {max_batches} batch(es)")
            break
        batch_graph = batch_graph.to(device)
        batch_index = _batch_index_from_counts(batch_graph.batch_num_nodes(), device)
        with torch.no_grad():
            node_emb = encoder(batch_graph, batch_graph.ndata["attr"].to(device))
            graph_emb = mean_pool(node_emb, batch_index)
        all_embeddings.append(graph_emb.cpu())
        all_labels.append(labels.float())

    if not all_embeddings:
        raise ValueError("No graph batches were available for precomputation.")
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def _evaluate_pcba_from_logits(logits, labels, evaluator, split_name):
    """Compute PCBA AP from pre-computed logits and label tensors."""
    y_pred = torch.sigmoid(logits).cpu()
    y_true = labels.cpu()

    usable_tasks, positive_labels, negative_labels, labeled_positions = _pcba_ap_stats(y_true)
    if usable_tasks == 0:
        raise ValueError(
            _structured_notes(
                "pcba_ap_unavailable",
                split=split_name,
                graphs_evaluated=int(y_true.shape[0]),
                labeled_positions=labeled_positions,
                positive_labels=positive_labels,
                negative_labels=negative_labels,
                detail=(
                    f"No ogbg-molpcba task in the {split_name} split has both positive and "
                    "negative labels after embedding precompute"
                ),
            )
        )
    try:
        return float(evaluator.eval({"y_true": y_true, "y_pred": y_pred})["ap"])
    except RuntimeError as exc:
        raise ValueError(
            _structured_notes(
                "pcba_ogb_evaluator_error",
                split=split_name,
                graphs_evaluated=int(y_true.shape[0]),
                detail=str(exc),
            )
        ) from exc


def _evaluate_pcba_split(
    encoder,
    head,
    dataloader,
    evaluator,
    device: str,
    split_name: str,
    max_batches: int | None = None,
    debug: bool = False,
) -> float:
    y_true = []
    y_pred = []

    head.eval()
    for batch_idx, (batch_graph, labels) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            prefix = "[debug]" if debug else "[eval]"
            print(
                f"{prefix} max_eval_batches reached on {split_name}: "
                f"stopping after {max_batches} batch(es)."
            )
            break
        batch_graph = batch_graph.to(device)
        labels = labels.to(device).float()
        batch_index = _batch_index_from_counts(batch_graph.batch_num_nodes(), device)
        with torch.no_grad():
            node_embeddings = encoder(batch_graph, batch_graph.ndata["attr"].to(device))
            logits = head(node_embeddings, batch_index)
        y_true.append(labels.cpu())
        y_pred.append(torch.sigmoid(logits).cpu())

    if not y_true or not y_pred:
        raise ValueError(f"No graph batches were available for the {split_name} split.")

    y_true_tensor = torch.cat(y_true, dim=0)
    y_pred_tensor = torch.cat(y_pred, dim=0)

    usable_tasks, positive_labels, negative_labels, labeled_positions = _pcba_ap_stats(
        y_true_tensor
    )

    if usable_tasks == 0:
        if y_true_tensor.shape[0] < 2:
            detail = (
                "No ogbg-molpcba AP metric can be computed from fewer than 2 consumed graphs; "
                "increase --batch_size or --max_eval_batches"
            )
        else:
            detail = (
                "No ogbg-molpcba task in the requested eval slice has both positive and "
                "negative labels; increase --debug_max_graphs or --max_eval_batches"
            )
        raise ValueError(
            _structured_notes(
                "pcba_ap_unavailable",
                split=split_name,
                graphs_evaluated=int(y_true_tensor.shape[0]),
                labeled_positions=labeled_positions,
                positive_labels=positive_labels,
                negative_labels=negative_labels,
                max_eval_batches=max_batches,
                detail=detail,
            )
        )

    try:
        return float(
            evaluator.eval(
                {
                    "y_true": y_true_tensor,
                    "y_pred": y_pred_tensor,
                }
            )["ap"]
        )
    except RuntimeError as exc:
        raise ValueError(
            _structured_notes(
                "pcba_ogb_evaluator_error",
                split=split_name,
                graphs_evaluated=int(y_true_tensor.shape[0]),
                max_eval_batches=max_batches,
                detail=str(exc),
            )
        ) from exc


def _truncate_split_indices(indices, graph_eval_config: GraphEvalConfig):
    if not graph_eval_config.debug or graph_eval_config.debug_max_graphs <= 0:
        return indices
    return indices[: graph_eval_config.debug_max_graphs]


def _validate_graph_checkpoint_task(
    encoder,
    *,
    dataset: str,
) -> None:
    metadata = _checkpoint_metadata(encoder)
    checkpoint_task = metadata.get("task_type")
    if checkpoint_task is None:
        return
    if str(checkpoint_task).lower() == "graph":
        return
    raise ValueError(
        _structured_notes(
            "checkpoint_task_mismatch",
            dataset=dataset,
            expected_task="graph",
            checkpoint_task_type=checkpoint_task,
            checkpoint_dataset=metadata.get("dataset"),
        )
    )


def _validate_pcba_native_graph_inputs(
    encoder,
    *,
    dataset: str,
    native_feature_dim: int,
    feat_pt: str | None,
) -> None:
    metadata = _checkpoint_metadata(encoder)
    if feat_pt is not None:
        raise ValueError(
            _structured_notes(
                "pcba_external_features_unsupported",
                dataset=dataset,
                expected_feature_source="native",
                provided_feat_pt=feat_pt,
                checkpoint_task_type=metadata.get("task_type"),
                checkpoint_dataset=metadata.get("dataset"),
                detail="Layer 2 PCBA graph eval does not support --feat-pt graph feature broadcasting",
            )
        )

    checkpoint_dataset = metadata.get("dataset")
    if checkpoint_dataset is not None and str(checkpoint_dataset).lower() != dataset.lower():
        raise ValueError(
            _structured_notes(
                "checkpoint_dataset_mismatch",
                dataset=dataset,
                checkpoint_dataset=checkpoint_dataset,
                checkpoint_task_type=metadata.get("task_type"),
            )
        )

    expected_dim = getattr(encoder, "input_dim", None)
    if expected_dim is None:
        return
    if int(expected_dim) == int(native_feature_dim):
        return
    raise ValueError(
        _structured_notes(
            "pcba_native_input_dim_mismatch",
            dataset=dataset,
            feature_source="native",
            native_input_dim=native_feature_dim,
            encoder_input_dim=expected_dim,
            checkpoint_dataset=checkpoint_dataset,
            checkpoint_task_type=metadata.get("task_type"),
            checkpoint_feat_pt_used=metadata.get("feat_pt_used"),
            detail=(
                "Native PCBA graph eval requires a checkpoint trained on native atom features; "
                "graph-level external feature broadcasting is intentionally unsupported in Layer 2"
            ),
        )
    )


def _run_graph_eval_graphmae(
    encoder,
    device: str,
    graph_eval_config: GraphEvalConfig,
    *,
    feat_pt: str | None = None,
) -> EvalResult:
    _validate_graph_checkpoint_task(encoder, dataset="ogbg-molpcba")
    if feat_pt is not None:
        _validate_pcba_native_graph_inputs(
            encoder,
            dataset="ogbg-molpcba",
            native_feature_dim=-1,
            feat_pt=feat_pt,
        )

    with _graphmae_context():
        from dgl.dataloading import GraphDataLoader
        from ogb.graphproppred import DglGraphPropPredDataset, Evaluator

        dataset = DglGraphPropPredDataset(name="ogbg-molpcba", root="dataset")

        _sample_g, _ = dataset[0]
        _native_feat_dim = int(_sample_g.ndata["feat"].shape[1]) if "feat" in _sample_g.ndata else 1
        _validate_pcba_native_graph_inputs(
            encoder,
            dataset="ogbg-molpcba",
            native_feature_dim=_native_feat_dim,
            feat_pt=None,
        )

        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name="ogbg-molpcba")
        num_classes = int(dataset.num_tasks)

        train_dataset = _PCBADatasetView(
            dataset,
            _truncate_split_indices(split_idx["train"], graph_eval_config),
        )
        valid_dataset = _PCBADatasetView(
            dataset,
            _select_pcba_debug_eval_indices(
                dataset,
                split_idx["valid"],
                graph_eval_config,
                split_name="valid",
            ),
        )
        test_dataset = _PCBADatasetView(
            dataset,
            _select_pcba_debug_eval_indices(
                dataset,
                split_idx["test"],
                graph_eval_config,
                split_name="test",
            ),
        )

        _debug_loader_kw = (
            {"pin_memory": False, "persistent_workers": False}
            if graph_eval_config.debug
            else {}
        )
        train_loader = GraphDataLoader(
            train_dataset,
            batch_size=graph_eval_config.batch_size,
            shuffle=True,
            collate_fn=_collate_graph_batch,
            num_workers=graph_eval_config.num_workers,
            **_debug_loader_kw,
        )
        valid_loader = GraphDataLoader(
            valid_dataset,
            batch_size=graph_eval_config.batch_size,
            shuffle=False,
            collate_fn=_collate_graph_batch,
            num_workers=graph_eval_config.num_workers,
            **_debug_loader_kw,
        )
        test_loader = GraphDataLoader(
            test_dataset,
            batch_size=graph_eval_config.batch_size,
            shuffle=False,
            collate_fn=_collate_graph_batch,
            num_workers=graph_eval_config.num_workers,
            **_debug_loader_kw,
        )

    # ------------------------------------------------------------------
    # Precompute frozen graph embeddings (encoder runs ONCE per split)
    # ------------------------------------------------------------------
    import time as _time

    _prefix = "[debug]" if graph_eval_config.debug else "[eval]"

    _t0 = _time.time()
    print(f"{_prefix} precomputing frozen graph embeddings (train) ...")
    train_embeds, train_labels = _precompute_graph_embeddings(
        encoder, train_loader, device,
    )
    print(f"{_prefix}   train: {train_embeds.shape[0]} graphs, dim={train_embeds.shape[1]}")

    print(f"{_prefix} precomputing frozen graph embeddings (valid) ...")
    valid_embeds, valid_labels = _precompute_graph_embeddings(
        encoder, valid_loader, device,
        max_batches=graph_eval_config.max_eval_batches,
        debug=graph_eval_config.debug,
    )
    print(f"{_prefix}   valid: {valid_embeds.shape[0]} graphs")

    print(f"{_prefix} precomputing frozen graph embeddings (test) ...")
    test_embeds, test_labels = _precompute_graph_embeddings(
        encoder, test_loader, device,
        max_batches=graph_eval_config.max_eval_batches,
        debug=graph_eval_config.debug,
    )
    _embed_elapsed = _time.time() - _t0
    print(
        f"{_prefix}   test: {test_embeds.shape[0]} graphs  "
        f"(precompute total: {_embed_elapsed:.1f}s)"
    )

    # Move cached embeddings to training device; free encoder from GPU.
    train_embeds = train_embeds.to(device)
    train_labels = train_labels.to(device)
    valid_embeds = valid_embeds.to(device)
    valid_labels = valid_labels.to(device)
    test_embeds = test_embeds.to(device)
    test_labels = test_labels.to(device)

    encoder.cpu()
    if device != "cpu":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Train linear head on cached graph embeddings
    # ------------------------------------------------------------------
    hidden_dim = int(train_embeds.shape[1])
    head = GraphHead(hidden_dim=hidden_dim, num_classes=int(num_classes)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=GRAPH_LR)

    effective_ft_epochs = (
        graph_eval_config.ft_epochs
        if graph_eval_config.ft_epochs is not None
        else GRAPH_EPOCHS
    )
    eval_every = graph_eval_config.graph_eval_every  # None → every epoch
    batch_size = graph_eval_config.batch_size

    best_valid = float("-inf")
    best_state = None
    train_steps = 0
    stop_training = False
    n_train = train_embeds.shape[0]

    _t_train = _time.time()
    for epoch in range(effective_ft_epochs):
        head.train()
        perm = torch.randperm(n_train, device=train_embeds.device)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            logits = head.linear(train_embeds[idx])
            loss = _masked_bce_loss(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_steps += 1

            if (
                graph_eval_config.max_train_steps is not None
                and train_steps >= graph_eval_config.max_train_steps
            ):
                print(
                    f"{_prefix} max_train_steps reached: "
                    f"stopping after {graph_eval_config.max_train_steps} optimizer step(s)."
                )
                stop_training = True
                break

        # Validate at cadence: every eval_every epochs, plus always on the
        # last epoch (or when early-stopped by max_train_steps).
        is_last = (epoch == effective_ft_epochs - 1) or stop_training
        should_eval = is_last or eval_every is None or ((epoch + 1) % eval_every == 0)

        if should_eval:
            head.eval()
            with torch.no_grad():
                valid_logits = head.linear(valid_embeds)
            valid_ap = _evaluate_pcba_from_logits(
                valid_logits, valid_labels, evaluator, "valid",
            )
            if valid_ap >= best_valid:
                best_valid = valid_ap
                best_state = copy.deepcopy(head.state_dict())

        if stop_training:
            break

    # Final test AP — computed once with the best head state.
    head.load_state_dict(_select_best_state(best_state, head))
    head.eval()
    with torch.no_grad():
        test_logits = head.linear(test_embeds)
    best_test = _evaluate_pcba_from_logits(
        test_logits, test_labels, evaluator, "test",
    )

    _train_elapsed = _time.time() - _t_train
    _total_elapsed = _time.time() - _t0
    print(
        f"{_prefix} graph eval done: {train_steps} steps in {_train_elapsed:.1f}s, "
        f"best_valid_ap={best_valid:.6f}, test_ap={best_test:.6f}  "
        f"(total wall: {_total_elapsed:.1f}s)"
    )

    eval_cadence_note = (
        f" graph_eval_every={eval_every}," if eval_every is not None else ""
    )
    return EvalResult(
        model="graphmae",
        dataset="ogbg-molpcba",
        task="graph",
        status=_eval_status(graph_eval_config.debug),
        metric_name="ap",
        metric_value=float(best_test),
        notes=format_debug_notes(graph_eval_config)
        if graph_eval_config.debug
        else (
            "Frozen GraphMAE encoder with mean-pool linear graph head; "
            "graph embeddings precomputed once under torch.no_grad(); "
            f"ft_epochs={effective_ft_epochs},{eval_cadence_note} "
            "AP via official ogbg-molpcba OGB evaluator on official splits."
        ),
    )


def _precompute_graph_embeddings_bgrl(
    encoder, dataloader, device, *, max_batches=None, debug=False,
):
    """Run frozen BGRL encoder + mean-pool once over a PyG DataLoader.

    BGRL's encoder interface: ``encoder(x, edge_index, edge_weight=None)``
    returns node embeddings.  These are aggregated per-graph via mean_pool
    using PyG's ``batch.batch`` tensor.

    Returns ``(graph_embeddings, labels)`` as CPU tensors.
    """
    all_embeddings: list = []
    all_labels: list = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            if debug:
                print(f"[debug] precompute_bgrl: stopped after {max_batches} batch(es)")
            break
        batch = batch.to(device)
        # PCBA atom features may be integer (Long); cast to float for GCN
        x = batch.x.float() if batch.x is not None else torch.ones((batch.num_nodes, 1), device=device)
        with torch.no_grad():
            node_emb = encoder(x, batch.edge_index, edge_weight=None)
            graph_emb = mean_pool(node_emb, batch.batch)
        all_embeddings.append(graph_emb.cpu())
        all_labels.append(batch.y.float().cpu())

    if not all_embeddings:
        raise ValueError("No graph batches were available for BGRL precomputation.")
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def _run_graph_eval_bgrl(
    encoder,
    device: str,
    graph_eval_config: GraphEvalConfig,
    *,
    feat_pt: str | None = None,
) -> EvalResult:
    """Graph classification eval for BGRL on ogbg-molpcba via PyG DataLoader."""
    _validate_graph_checkpoint_task(encoder, dataset="ogbg-molpcba")
    if feat_pt is not None:
        _validate_pcba_native_graph_inputs(
            encoder, dataset="ogbg-molpcba",
            native_feature_dim=-1, feat_pt=feat_pt,
        )

    # Load PCBA via PyG (not DGL — BGRL uses PyG backend)
    from ogb.graphproppred import PygGraphPropPredDataset
    from torch_geometric.loader import DataLoader as PyGDataLoader

    dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root="dataset")

    _sample = dataset[0]
    _native_feat_dim = int(_sample.x.shape[1]) if _sample.x is not None else 1
    _validate_pcba_native_graph_inputs(
        encoder, dataset="ogbg-molpcba",
        native_feature_dim=_native_feat_dim, feat_pt=None,
    )

    split_idx = dataset.get_idx_split()
    from ogb.graphproppred import Evaluator
    evaluator = Evaluator(name="ogbg-molpcba")
    num_classes = int(dataset.num_tasks)

    # Build subsets
    train_indices = _truncate_split_indices(split_idx["train"], graph_eval_config)
    valid_indices = split_idx["valid"]
    test_indices = split_idx["test"]
    if graph_eval_config.debug and graph_eval_config.debug_max_graphs > 0:
        valid_indices = valid_indices[: graph_eval_config.debug_max_graphs]
        test_indices = test_indices[: graph_eval_config.debug_max_graphs]

    train_loader = PyGDataLoader(
        dataset[train_indices], batch_size=graph_eval_config.batch_size,
        shuffle=True, num_workers=graph_eval_config.num_workers,
    )
    valid_loader = PyGDataLoader(
        dataset[valid_indices], batch_size=graph_eval_config.batch_size,
        shuffle=False, num_workers=graph_eval_config.num_workers,
    )
    test_loader = PyGDataLoader(
        dataset[test_indices], batch_size=graph_eval_config.batch_size,
        shuffle=False, num_workers=graph_eval_config.num_workers,
    )

    # ------------------------------------------------------------------
    # Precompute frozen graph embeddings (encoder runs ONCE per split)
    # ------------------------------------------------------------------
    import time as _time

    _prefix = "[debug]" if graph_eval_config.debug else "[eval]"

    _t0 = _time.time()
    print(f"{_prefix} [bgrl] precomputing frozen graph embeddings (train) ...")
    train_embeds, train_labels = _precompute_graph_embeddings_bgrl(
        encoder, train_loader, device,
    )
    print(f"{_prefix}   train: {train_embeds.shape[0]} graphs, dim={train_embeds.shape[1]}")

    print(f"{_prefix} [bgrl] precomputing frozen graph embeddings (valid) ...")
    valid_embeds, valid_labels = _precompute_graph_embeddings_bgrl(
        encoder, valid_loader, device,
        max_batches=graph_eval_config.max_eval_batches,
        debug=graph_eval_config.debug,
    )
    print(f"{_prefix}   valid: {valid_embeds.shape[0]} graphs")

    print(f"{_prefix} [bgrl] precomputing frozen graph embeddings (test) ...")
    test_embeds, test_labels = _precompute_graph_embeddings_bgrl(
        encoder, test_loader, device,
        max_batches=graph_eval_config.max_eval_batches,
        debug=graph_eval_config.debug,
    )
    _embed_elapsed = _time.time() - _t0
    print(
        f"{_prefix}   test: {test_embeds.shape[0]} graphs  "
        f"(precompute total: {_embed_elapsed:.1f}s)"
    )

    # Move cached embeddings to training device; free encoder from GPU.
    train_embeds = train_embeds.to(device)
    train_labels = train_labels.to(device)
    valid_embeds = valid_embeds.to(device)
    valid_labels = valid_labels.to(device)
    test_embeds = test_embeds.to(device)
    test_labels = test_labels.to(device)
    encoder.cpu()
    if device != "cpu":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Train linear head on cached graph embeddings
    # ------------------------------------------------------------------
    hidden_dim = int(train_embeds.shape[1])
    head = GraphHead(hidden_dim=hidden_dim, num_classes=int(num_classes)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=GRAPH_LR)

    effective_ft_epochs = (
        graph_eval_config.ft_epochs
        if graph_eval_config.ft_epochs is not None
        else GRAPH_EPOCHS
    )
    eval_every = graph_eval_config.graph_eval_every
    batch_size = graph_eval_config.batch_size

    best_valid = float("-inf")
    best_state = None
    train_steps = 0
    stop_training = False
    n_train = train_embeds.shape[0]

    _t_train = _time.time()
    for epoch in range(effective_ft_epochs):
        head.train()
        perm = torch.randperm(n_train, device=train_embeds.device)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            logits = head.linear(train_embeds[idx])
            loss = _masked_bce_loss(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_steps += 1

            if (
                graph_eval_config.max_train_steps is not None
                and train_steps >= graph_eval_config.max_train_steps
            ):
                print(
                    f"{_prefix} max_train_steps reached: "
                    f"stopping after {graph_eval_config.max_train_steps} optimizer step(s)."
                )
                stop_training = True
                break

        is_last = (epoch == effective_ft_epochs - 1) or stop_training
        should_eval = is_last or eval_every is None or ((epoch + 1) % eval_every == 0)

        if should_eval:
            head.eval()
            with torch.no_grad():
                valid_logits = head.linear(valid_embeds)
            valid_ap = _evaluate_pcba_from_logits(
                valid_logits, valid_labels, evaluator, "valid",
            )
            if valid_ap >= best_valid:
                best_valid = valid_ap
                best_state = copy.deepcopy(head.state_dict())

        if stop_training:
            break

    # Final test AP
    head.load_state_dict(_select_best_state(best_state, head))
    head.eval()
    with torch.no_grad():
        test_logits = head.linear(test_embeds)
    best_test = _evaluate_pcba_from_logits(
        test_logits, test_labels, evaluator, "test",
    )

    _train_elapsed = _time.time() - _t_train
    _total_elapsed = _time.time() - _t0
    print(
        f"{_prefix} bgrl graph eval done: {train_steps} steps in {_train_elapsed:.1f}s, "
        f"best_valid_ap={best_valid:.6f}, test_ap={best_test:.6f}  "
        f"(total wall: {_total_elapsed:.1f}s)"
    )

    eval_cadence_note = (
        f" graph_eval_every={eval_every}," if eval_every is not None else ""
    )
    return EvalResult(
        model="bgrl",
        dataset="ogbg-molpcba",
        task="graph",
        status=_eval_status(graph_eval_config.debug),
        metric_name="ap",
        metric_value=float(best_test),
        notes=format_debug_notes(graph_eval_config)
        if graph_eval_config.debug
        else (
            "Frozen BGRL encoder (node-level SSL) with mean-pool linear graph head; "
            "graph embeddings precomputed once under torch.no_grad(); "
            f"ft_epochs={effective_ft_epochs},{eval_cadence_note} "
            "AP via official ogbg-molpcba OGB evaluator on official splits; "
            "node_ssl_graph_transfer=true; edge_features_ignored=true"
        ),
    )


def run_graph_eval(
    model: str,
    ckpt: str,
    device: str,
    graph_eval_config: GraphEvalConfig,
    *,
    feat_pt: str | None = None,
) -> EvalResult:
    """Run graph classification linear-probe evaluation.

    Supports GraphMAE and BGRL on ogbg-molpcba.
    """
    adapter = REGISTRY.get_adapter("ogbg-molpcba")
    adapter.validate_model(model)

    _ensure_eval_deps()
    encoder = load_encoder(model, ckpt, device)

    if model == "graphmae":
        return _run_graph_eval_graphmae(
            encoder, device, graph_eval_config, feat_pt=feat_pt,
        )
    if model == "bgrl":
        return _run_graph_eval_bgrl(
            encoder, device, graph_eval_config, feat_pt=feat_pt,
        )
    raise NotImplementedError(
        f"Graph-level eval for model={model!r} is not supported."
    )


# ---------------------------------------------------------------------------
# Link evaluation — generalized protocol-based runner
# ---------------------------------------------------------------------------


def _read_entity2id_for_link(path: Path) -> dict[str, int]:
    """Read ``entity2id.txt`` (space-separated: entity_name integer_id)."""
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
            idx = int(parts[-1])
            mapping[entity] = idx
    return mapping


def _read_split_edges_for_link(
    path: Path,
    entity2id: dict[str, int],
    relation_map: dict[str, int] | None = None,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Read WN18RR split file (head\\trel\\ttail).

    Returns (edges, relation_ids) where edges is list of (head_id, tail_id)
    and relation_ids is a parallel list of relation type indices (0-based,
    assigned in discovery order).

    If ``relation_map`` is provided, it is shared and extended across calls
    so that relation IDs are consistent across train/valid/test splits.
    """
    edges: list[tuple[int, int]] = []
    relation_ids: list[int] = []
    if relation_map is None:
        relation_map = {}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            head, rel, tail = parts[0], parts[1], parts[2]
            if head in entity2id and tail in entity2id:
                edges.append((entity2id[head], entity2id[tail]))
                if rel not in relation_map:
                    relation_map[rel] = len(relation_map)
                relation_ids.append(relation_map[rel])
    return edges, relation_ids


def _discover_wn18rr_protocol() -> LinkDatasetProtocol:
    """Audit the repo for WN18RR link-eval protocol evidence.

    Pure Python — no torch dependency. Returns a LinkDatasetProtocol whose
    ``missing`` list explains every gap preventing evaluation.
    """
    missing: list[str] = []

    # --- entity2id.txt ---
    entity2id: dict[str, int] = {}
    entity2id_path: Path | None = None
    for candidate in [
        REPO_ROOT / "data" / "WN18RR" / "entity2id.txt",
        REPO_ROOT / "data" / "pyg" / "WN18RR" / "entity2id.txt",
    ]:
        if candidate.exists():
            entity2id_path = candidate
            break
    if entity2id_path is None:
        missing.append("entity2id.txt not found in data/WN18RR/ or data/pyg/WN18RR/")
    else:
        entity2id = _read_entity2id_for_link(entity2id_path)
        if not entity2id:
            missing.append("entity2id.txt is empty or unparseable")

    num_entities = len(entity2id)

    # --- split directory (train.txt, test.txt) ---
    split_dir: Path | None = None
    for candidate in [
        REPO_ROOT / "data" / "WN18RR" / "raw",
        REPO_ROOT / "data" / "pyg" / "WN18RR" / "raw",
    ]:
        if (candidate / "train.txt").exists() and (candidate / "test.txt").exists():
            split_dir = candidate
            break
    if split_dir is None:
        missing.append(
            "Split files (train.txt, test.txt) not found in "
            "data/WN18RR/raw/ or data/pyg/WN18RR/raw/"
        )

    # --- load edges (only if both entity2id and splits are available) ---
    train_edges: list[tuple[int, int]] = []
    valid_edges: list[tuple[int, int]] = []
    test_edges: list[tuple[int, int]] = []
    rel_ids_train: list[int] | None = None
    rel_ids_valid: list[int] | None = None
    rel_ids_test: list[int] | None = None
    num_relations: int | None = None

    if entity2id and split_dir is not None:
        # Shared relation_map ensures consistent relation IDs across splits
        shared_relation_map: dict[str, int] = {}
        train_edges, rel_ids_train = _read_split_edges_for_link(
            split_dir / "train.txt", entity2id, shared_relation_map,
        )
        valid_path = split_dir / "valid.txt"
        if valid_path.exists():
            valid_edges, rel_ids_valid = _read_split_edges_for_link(
                valid_path, entity2id, shared_relation_map,
            )
        test_edges, rel_ids_test = _read_split_edges_for_link(
            split_dir / "test.txt", entity2id, shared_relation_map,
        )
        if not test_edges:
            missing.append("test.txt has no parseable edges")
        # Count unique relations across all splits
        all_rel_ids: set[int] = set()
        for rids in (rel_ids_train, rel_ids_valid, rel_ids_test):
            if rids:
                all_rel_ids.update(rids)
        num_relations = len(all_rel_ids) if all_rel_ids else None

    # --- SBERT feature file ---
    sbert_available = (REPO_ROOT / "data" / "wn18rr_sbert.pt").exists()

    return LinkDatasetProtocol(
        dataset_name="wn18rr",
        num_entities=num_entities,
        train_edges=train_edges,
        valid_edges=valid_edges,
        test_edges=test_edges,
        relation_ids_train=rel_ids_train,
        relation_ids_valid=rel_ids_valid,
        relation_ids_test=rel_ids_test,
        num_relations=num_relations,
        sbert_available=sbert_available,
        missing=missing,
        extra={"entity2id_count": num_entities, "split_dir": str(split_dir)},
    )


# Link dataset protocol discovery registry — future link datasets register here
_LINK_DATASET_DISCOVERY: dict[str, callable] = {
    "wn18rr": _discover_wn18rr_protocol,
}


def discover_link_protocol(dataset_name: str) -> LinkDatasetProtocol:
    """Discover link-eval protocol for a dataset by name.

    Raises ValueError if no discovery function is registered.
    """
    normalized = dataset_name.lower()
    if normalized not in _LINK_DATASET_DISCOVERY:
        raise ValueError(
            f"No link protocol discovery registered for dataset {dataset_name!r}. "
            f"Registered: {sorted(_LINK_DATASET_DISCOVERY)}"
        )
    return _LINK_DATASET_DISCOVERY[normalized]()


def _encode_link_graphmae(
    encoder,
    device: str,
    dataset_name: str,
    *,
    feat_pt: str | None = None,
):
    """Load a link dataset graph via GraphMAE loader, encode nodes, return [N, dim] on CPU."""
    with _graphmae_context():
        from graphmae.datasets.data_util import load_dataset

        graph, (_, _) = load_dataset(dataset_name)

    if feat_pt is not None:
        graph.ndata["feat"] = _load_feat_pt(feat_pt, graph.num_nodes())

    graph = graph.to(device)
    features = graph.ndata["feat"].to(device)
    _validate_feature_dim(encoder, features.shape[1], feat_pt is not None, dataset_name)

    with torch.no_grad():
        return encoder(graph, features).detach().cpu()


def _encode_link_bgrl(
    encoder,
    device: str,
    dataset_name: str,
    *,
    feat_pt: str | None = None,
):
    """Load a link dataset graph via BGRL/PyG loader, encode nodes, return [N, dim] on CPU."""
    with _bgrl_context():
        data_module = _load_module_from_path("_gfm_safety_bgrl_data", BGRL_ROOT / "data.py")
        dataset = data_module.Dataset(root=str(REPO_ROOT / "data"), name=dataset_name)[0]

    if feat_pt is not None:
        dataset.x = _load_feat_pt(feat_pt, dataset.num_nodes)

    dataset = dataset.to(device)
    _validate_feature_dim(encoder, dataset.x.shape[1], feat_pt is not None, dataset_name)

    with torch.no_grad():
        return encoder(dataset.x, dataset.edge_index, dataset.edge_attr).detach().cpu()


# Link node encoding dispatch — keyed by (model, dataset_name)
_LINK_ENCODE: dict[str, callable] = {
    "graphmae": _encode_link_graphmae,
    "bgrl": _encode_link_bgrl,
}


def run_link_eval(
    model: str,
    ckpt: str,
    device: str,
    *,
    debug: bool = False,
    feat_pt: str | None = None,
    scorer_name: str = "dot_product",
    ranking_protocol: RankingProtocol | None = None,
    scorer_eval_every: int | None = None,
    scorer_patience: int | None = None,
) -> EvalResult:
    """Link-prediction evaluation via generalized protocol-based runner.

    Execution engine stages:
        1. Dataset protocol discovery / validation
        2. Encoder loading
        3. Feature loading / node encoding
        4. Scorer selection
        5. Ranking / filtered metric computation
        6. Structured result JSON assembly

    The default scorer (dot_product) and ranking protocol (both-corruption,
    full-ranking, filtered) match the current WN18RR experimental behavior.

    Args:
        model:            Model name ("graphmae")
        ckpt:             Path to encoder checkpoint
        device:           Torch device string
        debug:            Enable debug mode (truncate test edges)
        feat_pt:          Path to external features .pt file
        scorer_name:      Link scorer name from SCORER_REGISTRY
        ranking_protocol: Optional RankingProtocol override
    """
    # -- 1. Protocol discovery --
    # Currently only WN18RR is registered; future datasets register in
    # _LINK_DATASET_DISCOVERY and this code runs unchanged.
    adapter = REGISTRY.get_adapter("wn18rr")
    adapter.validate_model(model)
    dataset_name = adapter.dataset_name

    protocol = discover_link_protocol(dataset_name)

    if not protocol.is_runnable:
        return EvalResult(
            model=model,
            dataset=dataset_name,
            task="link",
            status="blocked",
            metric_name=None,
            metric_value=None,
            notes=_structured_notes(
                "link_protocol_incomplete",
                dataset=dataset_name,
                missing_pieces="; ".join(protocol.missing),
            ),
        )

    print(
        f"[link_eval] {dataset_name} protocol OK: {protocol.num_entities} entities, "
        f"train={len(protocol.train_edges)}, valid={len(protocol.valid_edges)}, "
        f"test={len(protocol.test_edges)}, sbert={protocol.sbert_available}, "
        f"num_relations={protocol.num_relations}"
    )

    # -- 2. Load encoder --
    _ensure_eval_deps()
    encoder = load_encoder(model, ckpt, device)

    # -- 3. Encode nodes --
    encode_fn = _LINK_ENCODE.get(model)
    if encode_fn is None:
        raise ValueError(
            _structured_notes(
                "link_model_unsupported",
                model=model,
                dataset=dataset_name,
                detail=(
                    f"Only {', '.join(sorted(_LINK_ENCODE))} supported for "
                    f"link eval on {dataset_name}."
                ),
            )
        )
    node_embeddings = encode_fn(encoder, device, dataset_name, feat_pt=feat_pt)

    # -- 4. Scorer selection --
    scorer_kwargs = {}
    if scorer_name != "dot_product" and protocol.num_relations is not None:
        scorer_kwargs["num_relations"] = protocol.num_relations
        scorer_kwargs["embedding_dim"] = node_embeddings.shape[1]
    scorer = get_scorer(scorer_name, **scorer_kwargs)
    relation_types_ignored = not scorer.info.relation_aware

    # -- 4.5. Build filter sets (needed for both scorer validation and final eval) --
    head_to_tails, tail_to_heads = build_filter_sets(
        protocol.train_edges,
        protocol.valid_edges,
        protocol.test_edges,
    )

    # -- 4.6. Train scorer if it has learnable parameters --
    scorer_train_meta: dict[str, object] | None = None
    if scorer.needs_training:
        scorer_train_epochs = 100
        scorer_max_steps: int | None = None
        if debug:
            scorer_train_epochs = 20
            scorer_max_steps = 10

        print(
            f"[link_eval] training {scorer.info.name} scorer: "
            f"epochs={scorer_train_epochs}, "
            f"num_relations={protocol.num_relations}, "
            f"embedding_dim={node_embeddings.shape[1]}"
        )
        scorer_train_meta = train_link_scorer(
            scorer,
            node_embeddings,
            protocol.train_edges,
            protocol.relation_ids_train,
            num_entities=protocol.num_entities,
            epochs=scorer_train_epochs,
            debug=debug,
            max_train_steps=scorer_max_steps,
            eval_every=scorer_eval_every,
            patience=scorer_patience,
            valid_edges=protocol.valid_edges,
            relation_ids_valid=protocol.relation_ids_valid,
            filter_head_to_tails=head_to_tails,
            filter_tail_to_heads=tail_to_heads,
        )
        if debug:
            print(f"[link_eval] scorer training result: {scorer_train_meta}")

    # -- 5. Select test edges --
    test_edge_list = protocol.test_edges
    test_rel_ids = protocol.relation_ids_test
    if debug and len(test_edge_list) > LINK_DEBUG_MAX_TEST_EDGES:
        print(
            f"[debug] link eval: truncating test edges "
            f"{len(test_edge_list)} \u2192 {LINK_DEBUG_MAX_TEST_EDGES}"
        )
        test_edge_list = test_edge_list[:LINK_DEBUG_MAX_TEST_EDGES]
        if test_rel_ids is not None:
            test_rel_ids = test_rel_ids[:LINK_DEBUG_MAX_TEST_EDGES]

    test_edges_tensor = torch.tensor(test_edge_list, dtype=torch.long)
    relation_ids_tensor = (
        torch.tensor(test_rel_ids, dtype=torch.long)
        if test_rel_ids is not None and scorer.info.relation_aware
        else None
    )

    # -- 6. Compute metrics --
    if ranking_protocol is None:
        ranking_protocol = DEFAULT_RANKING_PROTOCOL

    print(
        f"[link_eval] computing filtered MRR/Hits@K on "
        f"{test_edges_tensor.shape[0]} test edges, {protocol.num_entities} entities, "
        f"scorer={scorer.info.name}, protocol={ranking_protocol.corruption_policy}"
    )
    metrics = compute_link_metrics(
        node_embeddings,
        test_edges_tensor,
        head_to_tails,
        tail_to_heads,
        scorer=scorer,
        protocol=ranking_protocol,
        relation_ids=relation_ids_tensor,
    )

    for k, v in sorted(metrics.items()):
        print(f"[link_eval] {k} = {v:.6f}")

    # -- 7. Build result --
    result_code = (
        "wn18rr_relaware_experimental_link_eval"
        if scorer.info.relation_aware
        else "wn18rr_experimental_link_eval"
    )
    notes_fields: dict[str, object] = {
        "experimental": adapter.experimental,
        "scoring": scorer.info.name,
        "relation_types_ignored": relation_types_ignored,
        "negative_sampling_contract_defined": True,
        "filtering": "standard_filtered" if ranking_protocol.use_filtering else "unfiltered",
        "num_entities": protocol.num_entities,
        "test_edges_total": len(protocol.test_edges),
        "test_edges_evaluated": test_edges_tensor.shape[0],
    }
    if protocol.num_relations is not None:
        notes_fields["num_relations"] = protocol.num_relations
    for k, v in metrics.items():
        if k != "mrr":
            notes_fields[k] = round(v, 6)
    if scorer_train_meta is not None and not scorer_train_meta.get("skipped"):
        notes_fields["scorer_trained"] = True
        notes_fields["scorer_train_steps"] = scorer_train_meta["total_steps"]
        notes_fields["scorer_train_loss"] = scorer_train_meta["final_loss"]
        notes_fields["neg_per_pos"] = scorer_train_meta["neg_per_pos"]
        if "early_stopped" in scorer_train_meta:
            notes_fields["scorer_early_stopped"] = scorer_train_meta["early_stopped"]
            notes_fields["scorer_best_valid_mrr"] = scorer_train_meta["best_valid_mrr"]
            notes_fields["scorer_epochs_at_best_valid"] = scorer_train_meta["epochs_at_best_valid"]
    if debug:
        notes_fields["debug_mode"] = True
        notes_fields["official_metric"] = False
    else:
        notes_fields["debug_mode"] = False
        notes_fields["official_metric"] = True

    return EvalResult(
        model=model,
        dataset=dataset_name,
        task="link",
        status=_eval_status(debug),
        metric_name="mrr",
        metric_value=round(metrics["mrr"], 6),
        notes=_structured_notes(result_code, **notes_fields),
    )
