"""Layer 2 unified dataset / task / evaluator registry.

Centralizes dataset onboarding, task routing, metric specification, and
checkpoint metadata schema for the Layer 2 linear-probe evaluation pipeline.

Architecture:
  Step 1 (training): exports checkpoint {"encoder": state_dict, ...metadata}
  Step 2 (eval):     load_encoder -> freeze -> train task head -> emit metrics

Three task protocols:
  - Node:  encoder(graph, feat) -> node_embeddings -> Linear(H, C) -> accuracy
  - Graph: encoder(graph, feat) -> mean_pool -> Linear(H, C) -> AP
  - Link:  encoder(graph, feat) -> dot(u, v) -> filtered MRR / Hits@K

To onboard a new dataset:
  1. Define a DatasetAdapter with dataset properties
  2. Register it: REGISTRY.register_dataset(adapter)
  3. Implement the eval function in eval/run_lp.py
  4. Add suite target in scripts/layer2_suite_targets.py
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Valid domain values
# ---------------------------------------------------------------------------

VALID_TASK_TYPES = frozenset({"node", "graph", "link"})
VALID_MODELS = frozenset({"graphmae", "bgrl"})

MODEL_BACKENDS: dict[str, str] = {
    "graphmae": "dgl",
    "bgrl": "pyg",
}


# ---------------------------------------------------------------------------
# Checkpoint metadata contract
# ---------------------------------------------------------------------------

CHECKPOINT_METADATA_KEYS: tuple[str, ...] = (
    "model_name",
    "dataset",
    "task_type",
    "hidden_dim",
    "encoder_input_dim",
    "backend",
    "exported_at",
    "feat_pt_used",
)


@dataclass(frozen=True)
class CheckpointMetadataSpec:
    """Standardized metadata schema for Step 1 checkpoint exports.

    Checkpoint .pt files saved by eval/checkpoint.py::export_encoder_checkpoint()
    include these fields alongside the ``encoder`` state_dict key.
    Consumed by eval/load_encoder.py::_extract_checkpoint_metadata().
    """

    model_name: str | None = None
    dataset: str | None = None
    task_type: str | None = None        # "node" | "graph" | "link"
    hidden_dim: int | None = None
    encoder_input_dim: int | None = None
    backend: str | None = None          # "dgl" | "pyg"
    exported_at: str | None = None
    feat_pt_used: bool | None = None

    @classmethod
    def from_dict(cls, d: dict) -> CheckpointMetadataSpec:
        """Create from a checkpoint dict, ignoring unknown keys."""
        kwargs = {k: d.get(k) for k in CHECKPOINT_METADATA_KEYS}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Metric specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    """Describes the primary evaluation metric for a dataset/task."""

    name: str                              # "accuracy" | "ap" | "mrr"
    higher_is_better: bool = True
    additional_metrics: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Readiness / officialization gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadinessGate:
    """Structured readiness gate for Layer 2 officialization decisions.

    Each field captures an evidence-backed requirement or capability.
    Conservative defaults: capabilities default True (met), requirements
    default False (not required).  Override with explicit values for
    datasets with known gaps.

    ``promotion_blockers`` lists reasons preventing official candidate status.
    ``is_promotion_ready`` is True when all gates pass.
    """

    # Alignment / data integrity
    alignment_verified: bool = True
    requires_alignment_audit: bool = False

    # Eval protocol completeness
    eval_protocol_implemented: bool = True
    official_metric_available: bool = True

    # Link-prediction specific (irrelevant for node/graph tasks)
    relation_types_ignored: bool = False
    requires_relation_aware_scoring: bool = False
    requires_negative_sampling_contract: bool = False
    scoring_method: str | None = None
    link_protocol_implemented: bool = True

    # Debug support
    supports_local_debug: bool = True

    @property
    def promotion_blockers(self) -> tuple[str, ...]:
        """Return reasons preventing promotion to official candidate."""
        blockers: list[str] = []
        if self.requires_alignment_audit and not self.alignment_verified:
            blockers.append("alignment_not_verified")
        if not self.eval_protocol_implemented:
            blockers.append("eval_protocol_not_implemented")
        if not self.official_metric_available:
            blockers.append("official_metric_not_available")
        if self.relation_types_ignored:
            blockers.append("relation_types_ignored")
        if self.requires_negative_sampling_contract:
            blockers.append("negative_sampling_contract_undefined")
        if not self.link_protocol_implemented:
            blockers.append("link_protocol_not_implemented")
        return tuple(blockers)

    @property
    def is_promotion_ready(self) -> bool:
        """True if no promotion blockers exist."""
        return len(self.promotion_blockers) == 0

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict for manifest/metadata embedding."""
        return {
            "alignment_verified": self.alignment_verified,
            "requires_alignment_audit": self.requires_alignment_audit,
            "eval_protocol_implemented": self.eval_protocol_implemented,
            "official_metric_available": self.official_metric_available,
            "relation_types_ignored": self.relation_types_ignored,
            "requires_relation_aware_scoring": self.requires_relation_aware_scoring,
            "requires_negative_sampling_contract": self.requires_negative_sampling_contract,
            "scoring_method": self.scoring_method,
            "link_protocol_implemented": self.link_protocol_implemented,
            "supports_local_debug": self.supports_local_debug,
            "is_promotion_ready": self.is_promotion_ready,
            "promotion_blockers": list(self.promotion_blockers),
        }


# ---------------------------------------------------------------------------
# Task runner configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeTaskRunner:
    """Node classification linear-probe protocol.

    Pipeline: encoder -> freeze -> node_embeddings -> Linear(H, C) -> cross_entropy
    Metric:   accuracy (test set, best val checkpoint)
    """

    task_type: str = "node"
    head_class: str = "NodeHead"
    epochs: int = 100
    lr: float = 1e-2
    metric: MetricSpec = MetricSpec(name="accuracy")


@dataclass(frozen=True)
class GraphTaskRunner:
    """Graph classification linear-probe protocol.

    Pipeline: encoder -> freeze -> per-graph mean_pool -> Linear(H, C) -> BCE
    Metric:   AP via OGB evaluator (test set, best val checkpoint)
    """

    task_type: str = "graph"
    head_class: str = "GraphHead"
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 128
    debug_batch_size: int = 4
    metric: MetricSpec = MetricSpec(name="ap")


@dataclass(frozen=True)
class LinkTaskRunner:
    """Link prediction evaluation protocol.

    Pipeline: encoder -> freeze -> node_embeddings -> scorer(u, v) -> filtered ranking
    Metric:   MRR + Hits@{1,3,10}

    The default scorer is dot-product (baseline). Relation-aware scorers
    are available but experimental. See eval/link_protocol.py for details.
    """

    task_type: str = "link"
    head_class: str = "LinkHead"
    debug_max_test_edges: int = 50
    default_scorer: str = "dot_product"
    default_corruption_policy: str = "both"
    default_ranking_mode: str = "full"
    metric: MetricSpec = MetricSpec(
        name="mrr",
        additional_metrics=("hits@1", "hits@3", "hits@10"),
    )


# Canonical runner instances
NODE_RUNNER = NodeTaskRunner()
GRAPH_RUNNER = GraphTaskRunner()
LINK_RUNNER = LinkTaskRunner()

_TASK_RUNNERS: dict[str, NodeTaskRunner | GraphTaskRunner | LinkTaskRunner] = {
    "node": NODE_RUNNER,
    "graph": GRAPH_RUNNER,
    "link": LINK_RUNNER,
}


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetAdapter:
    """Declares a dataset's properties for Layer 2 evaluation onboarding.

    Fields:
        dataset_name:             Canonical name (e.g. "ogbn-arxiv")
        task_type:                "node" | "graph" | "link"
        native_feature_dim:       Dimension of built-in features (None if placeholder)
        supports_external_feat_pt: Whether --feat-pt swaps in external features
        supported_models:         Tuple of model names this dataset works with
        metric:                   Primary metric specification
        experimental:             True if not yet an official candidate
        requires_alignment_audit: True if alignment must be verified before eval
        caveats:                  Tuple of caveat strings for result metadata
        readiness:                Structured readiness gate for officialization
    """

    dataset_name: str
    task_type: str
    native_feature_dim: int | None
    supports_external_feat_pt: bool
    supported_models: tuple[str, ...]
    metric: MetricSpec
    experimental: bool = False
    requires_alignment_audit: bool = False
    caveats: tuple[str, ...] = ()
    readiness: ReadinessGate = ReadinessGate()

    def validate_model(self, model_name: str) -> None:
        """Raise ValueError if model_name is not supported for this dataset."""
        if model_name not in self.supported_models:
            caveat_info = ""
            if self.caveats:
                caveat_info = f" ({'; '.join(self.caveats)})"
            raise ValueError(
                f"Model {model_name!r} is not supported for dataset "
                f"{self.dataset_name!r}. "
                f"Supported: {', '.join(self.supported_models)}.{caveat_info}"
            )

    def get_task_runner(self) -> NodeTaskRunner | GraphTaskRunner | LinkTaskRunner:
        """Return the task runner config for this dataset's task type."""
        return _TASK_RUNNERS[self.task_type]

    @property
    def is_official_candidate(self) -> bool:
        """True if this dataset is eligible for official evaluation results.

        Requires both: not experimental AND readiness gate has no blockers.
        """
        return not self.experimental and self.readiness.is_promotion_ready

    @property
    def structured_caveats(self) -> dict[str, object]:
        """Return gate-derived structured caveats as a dict."""
        caveats: dict[str, object] = {}
        if self.experimental:
            caveats["experimental"] = True
        if self.readiness.relation_types_ignored:
            caveats["relation_types_ignored"] = True
        if not self.readiness.official_metric_available:
            caveats["official_metric"] = False
        if self.readiness.scoring_method is not None:
            caveats["scoring"] = self.readiness.scoring_method
        if self.readiness.promotion_blockers:
            caveats["promotion_blockers"] = list(self.readiness.promotion_blockers)
        return caveats


# ---------------------------------------------------------------------------
# Concrete dataset adapters
# ---------------------------------------------------------------------------

ARXIV_ADAPTER = DatasetAdapter(
    dataset_name="ogbn-arxiv",
    task_type="node",
    native_feature_dim=128,
    supports_external_feat_pt=True,
    supported_models=("graphmae", "bgrl"),
    metric=MetricSpec(name="accuracy"),
)

PCBA_ADAPTER = DatasetAdapter(
    dataset_name="ogbg-molpcba",
    task_type="graph",
    native_feature_dim=9,
    supports_external_feat_pt=False,
    supported_models=("graphmae", "bgrl"),
    metric=MetricSpec(name="ap"),
    caveats=(
        "official_metric=false",
    ),
    readiness=ReadinessGate(
        official_metric_available=False,
    ),
)

WN18RR_ADAPTER = DatasetAdapter(
    dataset_name="wn18rr",
    task_type="link",
    native_feature_dim=None,
    supports_external_feat_pt=True,
    supported_models=("graphmae", "bgrl"),
    metric=MetricSpec(
        name="mrr",
        additional_metrics=("hits@1", "hits@3", "hits@10"),
    ),
    experimental=False,
    requires_alignment_audit=True,
    caveats=(
        "relation_types_ignored=true",
        "scoring=dot_product",
    ),
    readiness=ReadinessGate(
        alignment_verified=True,
        requires_alignment_audit=True,
        eval_protocol_implemented=True,
        official_metric_available=True,
        relation_types_ignored=True,
        requires_relation_aware_scoring=True,
        # Negative-sampling contract is now fully defined:
        #   eval/link_protocol.py:278-354 (NegativeSamplingContract + default instance)
        #   eval/runners.py uses DEFAULT_NEGATIVE_SAMPLING_CONTRACT via train_link_scorer()
        #   Eval uses build_filter_sets(train, valid, test) + DEFAULT_RANKING_PROTOCOL
        requires_negative_sampling_contract=False,
        scoring_method="dot_product",
        link_protocol_implemented=True,
    ),
)


# ---------------------------------------------------------------------------
# Central registry
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """Central registry for dataset adapters and task configurations.

    Usage::

        adapter = REGISTRY.get_adapter("ogbn-arxiv")
        task_type = REGISTRY.route_task("ogbn-arxiv")
        runner = REGISTRY.get_task_runner("node")

    Adding a new dataset::

        new_adapter = DatasetAdapter(
            dataset_name="new-dataset",
            task_type="node",
            native_feature_dim=128,
            supports_external_feat_pt=True,
            supported_models=("graphmae",),
            metric=MetricSpec(name="accuracy"),
        )
        REGISTRY.register_dataset(new_adapter)
    """

    def __init__(self) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}

    def register_dataset(self, adapter: DatasetAdapter) -> None:
        """Register a dataset adapter. Raises if already registered."""
        if adapter.dataset_name in self._adapters:
            raise ValueError(
                f"Dataset {adapter.dataset_name!r} is already registered."
            )
        self._adapters[adapter.dataset_name] = adapter

    def get_adapter(self, dataset_name: str) -> DatasetAdapter:
        """Look up adapter by dataset name (case-insensitive)."""
        normalized = dataset_name.lower()
        if normalized not in self._adapters:
            raise ValueError(
                f"Unsupported dataset: {dataset_name!r}. "
                f"Registered: {sorted(self._adapters)}"
            )
        return self._adapters[normalized]

    def route_task(self, dataset_name: str) -> str:
        """Return the task type for a dataset."""
        return self.get_adapter(dataset_name).task_type

    def get_task_runner(
        self, task_type: str,
    ) -> NodeTaskRunner | GraphTaskRunner | LinkTaskRunner:
        """Look up the task runner config by task type."""
        if task_type not in _TASK_RUNNERS:
            raise ValueError(
                f"Unknown task type: {task_type!r}. "
                f"Valid: {sorted(_TASK_RUNNERS)}"
            )
        return _TASK_RUNNERS[task_type]

    def validate_model_dataset(
        self, model_name: str, dataset_name: str,
    ) -> DatasetAdapter:
        """Validate model/dataset combination; return adapter if valid."""
        adapter = self.get_adapter(dataset_name)
        adapter.validate_model(model_name)
        return adapter

    def list_datasets(self) -> list[str]:
        """Return sorted list of all registered dataset names."""
        return sorted(self._adapters)

    def list_official_datasets(self) -> list[str]:
        """Return sorted list of official candidate dataset names."""
        return sorted(
            name
            for name, adapter in self._adapters.items()
            if adapter.is_official_candidate
        )

    def list_experimental_datasets(self) -> list[str]:
        """Return sorted list of experimental dataset names."""
        return sorted(
            name
            for name, adapter in self._adapters.items()
            if adapter.experimental
        )

    def get_promotion_status(self, dataset_name: str) -> dict[str, object]:
        """Return promotion readiness status for a dataset."""
        adapter = self.get_adapter(dataset_name)
        return {
            "dataset_name": adapter.dataset_name,
            "experimental": adapter.experimental,
            "is_official_candidate": adapter.is_official_candidate,
            "is_promotion_ready": adapter.readiness.is_promotion_ready,
            "promotion_blockers": list(adapter.readiness.promotion_blockers),
            "readiness_gate": adapter.readiness.to_dict(),
        }

    def list_promotion_ready_datasets(self) -> list[str]:
        """Return sorted list of datasets whose readiness gates all pass."""
        return sorted(
            name
            for name, adapter in self._adapters.items()
            if adapter.readiness.is_promotion_ready
        )


# ---------------------------------------------------------------------------
# Singleton instance — import REGISTRY from here
# ---------------------------------------------------------------------------

REGISTRY = DatasetRegistry()
REGISTRY.register_dataset(ARXIV_ADAPTER)
REGISTRY.register_dataset(PCBA_ADAPTER)
REGISTRY.register_dataset(WN18RR_ADAPTER)
