"""eval/link_protocol.py — Generalized link-task scorer and ranking protocol.

Provides:
    LinkScorer          — Abstract base for link scoring strategies
    DotProductScorer    — Baseline dot-product scorer (current behavior)
    RelationAwareScorer — Experimental per-relation bilinear scorer
    RankingProtocol     — Structured negative-sampling / filtered ranking config
    compute_link_metrics — Unified filtered MRR/Hits@K computation

Design goals:
    - Preserve current dot-product WN18RR experimental behavior as the default
    - Make scorer selection explicit and inspectable
    - Generalize ranking protocol so future link datasets don't need ad hoc code
    - Keep torch imports lazy for fast CLI startup
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Ranking protocol configuration
# ---------------------------------------------------------------------------

VALID_CORRUPTION_POLICIES = frozenset({"head", "tail", "both"})
VALID_RANKING_MODES = frozenset({"full", "sampled"})
HITS_K_VALUES = (1, 3, 10)


@dataclass(frozen=True)
class RankingProtocol:
    """Structured specification for link-prediction ranking evaluation.

    Fields:
        corruption_policy: Which end(s) to corrupt ("head", "tail", "both").
        use_filtering:     Whether to filter known-true edges from ranking.
        ranking_mode:      "full" (score all entities) or "sampled" (subset).
        sample_size:       Number of negative samples if ranking_mode="sampled".
        hits_k_values:     Tuple of K values for Hits@K computation.
        metric_names:      Canonical metric names this protocol produces.
    """

    corruption_policy: str = "both"
    use_filtering: bool = True
    ranking_mode: str = "full"
    sample_size: int | None = None
    hits_k_values: tuple[int, ...] = HITS_K_VALUES
    metric_names: tuple[str, ...] = ("mrr", "hits@1", "hits@3", "hits@10")

    def __post_init__(self) -> None:
        if self.corruption_policy not in VALID_CORRUPTION_POLICIES:
            raise ValueError(
                f"Invalid corruption_policy: {self.corruption_policy!r}. "
                f"Valid: {sorted(VALID_CORRUPTION_POLICIES)}"
            )
        if self.ranking_mode not in VALID_RANKING_MODES:
            raise ValueError(
                f"Invalid ranking_mode: {self.ranking_mode!r}. "
                f"Valid: {sorted(VALID_RANKING_MODES)}"
            )
        if self.ranking_mode == "sampled" and (
            self.sample_size is None or self.sample_size <= 0
        ):
            raise ValueError(
                "sample_size must be a positive integer when ranking_mode='sampled'."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "corruption_policy": self.corruption_policy,
            "use_filtering": self.use_filtering,
            "ranking_mode": self.ranking_mode,
            "sample_size": self.sample_size,
            "hits_k_values": list(self.hits_k_values),
            "metric_names": list(self.metric_names),
        }


# Default protocol: matches current WN18RR experimental behavior exactly
DEFAULT_RANKING_PROTOCOL = RankingProtocol()


# ---------------------------------------------------------------------------
# Link scorer abstraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScorerInfo:
    """Metadata about a link scorer for result/artifact embedding."""
    name: str
    experimental: bool
    relation_aware: bool
    description: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "experimental": self.experimental,
            "relation_aware": self.relation_aware,
            "description": self.description,
        }


class LinkScorer(ABC):
    """Abstract base class for link scoring strategies.

    Subclasses must implement:
        - score_tail_corruption(h_emb, all_emb, relation_id) -> scores [N]
        - score_head_corruption(t_emb, all_emb, relation_id) -> scores [N]
        - info -> ScorerInfo metadata
        - needs_training -> whether the scorer has learnable parameters
    """

    @abstractmethod
    def score_tail_corruption(
        self, h_emb, all_emb, relation_id: int | None = None,
    ):
        """Score all candidate tails for a given head.

        Args:
            h_emb: Head entity embedding [dim]
            all_emb: All entity embeddings [N, dim]
            relation_id: Optional relation type index

        Returns:
            Scores tensor [N], higher = more likely
        """

    @abstractmethod
    def score_head_corruption(
        self, t_emb, all_emb, relation_id: int | None = None,
    ):
        """Score all candidate heads for a given tail.

        Args:
            t_emb: Tail entity embedding [dim]
            all_emb: All entity embeddings [N, dim]
            relation_id: Optional relation type index

        Returns:
            Scores tensor [N], higher = more likely
        """

    @property
    @abstractmethod
    def info(self) -> ScorerInfo:
        """Return metadata about this scorer."""

    @property
    def needs_training(self) -> bool:
        """Whether this scorer has learnable parameters requiring training."""
        return False


class DotProductScorer(LinkScorer):
    """Baseline dot-product link scorer.

    Matches the current WN18RR experimental behavior exactly:
        score(u, v) = u . v

    Relation types are ignored (relation_id parameter is unused).
    """

    _INFO = ScorerInfo(
        name="dot_product",
        experimental=False,
        relation_aware=False,
        description="Dot-product scoring: score(u,v) = u . v. Relation types ignored.",
    )

    def score_tail_corruption(self, h_emb, all_emb, relation_id=None):
        return (h_emb * all_emb).sum(dim=-1)

    def score_head_corruption(self, t_emb, all_emb, relation_id=None):
        return (all_emb * t_emb).sum(dim=-1)

    @property
    def info(self) -> ScorerInfo:
        return self._INFO


class RelationAwareScorer(LinkScorer):
    """Experimental per-relation diagonal scorer.

    score(h, r, t) = h . diag(R_r) . t

    where R_r is a learned per-relation diagonal scaling vector.
    This is the smallest relation-aware extension that preserves the
    dot-product structure while adding relation sensitivity.

    EXPERIMENTAL: This scorer has learnable parameters and requires
    training on the link-prediction training set before evaluation.
    """

    _INFO = ScorerInfo(
        name="relation_diagonal",
        experimental=True,
        relation_aware=True,
        description=(
            "Per-relation diagonal scorer: score(h,r,t) = h . diag(R_r) . t. "
            "Experimental; requires training."
        ),
    )

    def __init__(self, num_relations: int, embedding_dim: int):
        # Lazy torch import
        import torch
        from torch import nn

        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        # Per-relation diagonal: [num_relations, dim], initialized to ones
        # so that untrained behavior is equivalent to dot-product
        self.relation_diag = nn.Parameter(
            torch.ones(num_relations, embedding_dim)
        )

    def score_tail_corruption(self, h_emb, all_emb, relation_id=None):
        if relation_id is not None and self.relation_diag is not None:
            r_diag = self.relation_diag[relation_id]
            return (h_emb * r_diag * all_emb).sum(dim=-1)
        return (h_emb * all_emb).sum(dim=-1)

    def score_head_corruption(self, t_emb, all_emb, relation_id=None):
        if relation_id is not None and self.relation_diag is not None:
            r_diag = self.relation_diag[relation_id]
            return (all_emb * r_diag * t_emb).sum(dim=-1)
        return (all_emb * t_emb).sum(dim=-1)

    @property
    def info(self) -> ScorerInfo:
        return self._INFO

    @property
    def needs_training(self) -> bool:
        return True

    def parameters(self):
        """Yield learnable parameters for optimizer construction."""
        yield self.relation_diag


# ---------------------------------------------------------------------------
# Scorer registry
# ---------------------------------------------------------------------------

SCORER_REGISTRY: dict[str, type[LinkScorer]] = {
    "dot_product": DotProductScorer,
    "relation_diagonal": RelationAwareScorer,
}

BASELINE_SCORER_NAME = "dot_product"


def get_scorer(name: str, **kwargs) -> LinkScorer:
    """Instantiate a link scorer by name.

    Args:
        name: Scorer name from SCORER_REGISTRY
        **kwargs: Constructor arguments (e.g. num_relations, embedding_dim)
    """
    if name not in SCORER_REGISTRY:
        raise ValueError(
            f"Unknown link scorer: {name!r}. "
            f"Available: {sorted(SCORER_REGISTRY)}"
        )
    cls = SCORER_REGISTRY[name]
    if cls is DotProductScorer:
        return cls()
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Negative-sampling contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NegativeSamplingContract:
    """Formal negative-sampling specification for link-prediction training/eval.

    Train-time policy:
        For each positive triple (h, r, t), generate negatives by uniformly
        replacing head or tail with a random entity.  No filtering applied
        during training (standard KGE practice).

    Eval-time policy:
        Default: full filtered ranking — for each test triple, score all
        entities as candidate heads/tails, filter known-true edges, compute
        rank.  Sampled ranking is specifiable but NOT yet implemented in
        ``compute_link_metrics``; requests for sampled ranking will raise
        at validation time.

    Filter exclusion sets:
        Union of train + valid + test edges forms the exclusion set for
        filtered evaluation (standard practice).
    """

    train_negatives_per_positive: int = 32
    train_corruption_policy: str = "both"
    eval_corruption_policy: str = "both"
    eval_ranking_mode: str = "full"
    eval_sample_size: int | None = None
    filter_known_edges: bool = True
    filter_sets: tuple[str, ...] = ("train", "valid", "test")

    def __post_init__(self) -> None:
        if self.train_corruption_policy not in VALID_CORRUPTION_POLICIES:
            raise ValueError(
                f"Invalid train_corruption_policy: "
                f"{self.train_corruption_policy!r}. "
                f"Valid: {sorted(VALID_CORRUPTION_POLICIES)}"
            )
        if self.eval_corruption_policy not in VALID_CORRUPTION_POLICIES:
            raise ValueError(
                f"Invalid eval_corruption_policy: "
                f"{self.eval_corruption_policy!r}. "
                f"Valid: {sorted(VALID_CORRUPTION_POLICIES)}"
            )
        if self.eval_ranking_mode not in VALID_RANKING_MODES:
            raise ValueError(
                f"Invalid eval_ranking_mode: {self.eval_ranking_mode!r}. "
                f"Valid: {sorted(VALID_RANKING_MODES)}"
            )
        if self.eval_ranking_mode == "sampled" and (
            self.eval_sample_size is None or self.eval_sample_size <= 0
        ):
            raise ValueError(
                "eval_sample_size must be a positive integer when "
                "eval_ranking_mode='sampled'."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "train_negatives_per_positive": self.train_negatives_per_positive,
            "train_corruption_policy": self.train_corruption_policy,
            "eval_corruption_policy": self.eval_corruption_policy,
            "eval_ranking_mode": self.eval_ranking_mode,
            "eval_sample_size": self.eval_sample_size,
            "filter_known_edges": self.filter_known_edges,
            "filter_sets": list(self.filter_sets),
        }

    def to_ranking_protocol(self) -> RankingProtocol:
        """Derive the eval-time RankingProtocol from this contract."""
        return RankingProtocol(
            corruption_policy=self.eval_corruption_policy,
            use_filtering=self.filter_known_edges,
            ranking_mode=self.eval_ranking_mode,
            sample_size=self.eval_sample_size,
        )


DEFAULT_NEGATIVE_SAMPLING_CONTRACT = NegativeSamplingContract()


# ---------------------------------------------------------------------------
# Filter set construction
# ---------------------------------------------------------------------------

def build_filter_sets(
    *edge_lists: list[tuple[int, int]],
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Build head->{tails} and tail->{heads} dicts for filtered ranking.

    Accepts any number of edge lists (train, valid, test) so that all
    known-true edges are excluded during filtered evaluation.
    """
    head_to_tails: dict[int, set[int]] = {}
    tail_to_heads: dict[int, set[int]] = {}
    for edges in edge_lists:
        for h, t in edges:
            head_to_tails.setdefault(h, set()).add(t)
            tail_to_heads.setdefault(t, set()).add(h)
    return head_to_tails, tail_to_heads


# ---------------------------------------------------------------------------
# Unified filtered ranking metric computation
# ---------------------------------------------------------------------------

def compute_link_metrics(
    node_embeddings,
    test_edges,
    head_to_tails: dict[int, set[int]],
    tail_to_heads: dict[int, set[int]],
    *,
    scorer: LinkScorer | None = None,
    protocol: RankingProtocol | None = None,
    relation_ids=None,
) -> dict[str, float]:
    """Compute filtered MRR and Hits@K using the specified scorer and protocol.

    Args:
        node_embeddings: [N, dim] tensor of entity embeddings
        test_edges: [E, 2] tensor of (head, tail) test edges
        head_to_tails: Filter set mapping head -> known tails
        tail_to_heads: Filter set mapping tail -> known heads
        scorer: LinkScorer instance (defaults to DotProductScorer)
        protocol: RankingProtocol config (defaults to DEFAULT_RANKING_PROTOCOL)
        relation_ids: Optional [E] tensor of relation type indices per test edge

    Returns:
        Dict with keys like "mrr", "hits@1", "hits@3", "hits@10"
    """
    if scorer is None:
        scorer = DotProductScorer()
    if protocol is None:
        protocol = DEFAULT_RANKING_PROTOCOL

    num_test = test_edges.shape[0]
    reciprocal_rank_sum = 0.0
    hits = {k: 0 for k in protocol.hits_k_values}
    total = 0

    do_tail = protocol.corruption_policy in ("tail", "both")
    do_head = protocol.corruption_policy in ("head", "both")

    for i in range(num_test):
        h = int(test_edges[i, 0])
        t = int(test_edges[i, 1])
        rel_id = int(relation_ids[i]) if relation_ids is not None else None

        if do_tail:
            scores_tail = scorer.score_tail_corruption(
                node_embeddings[h], node_embeddings, relation_id=rel_id,
            )
            if protocol.use_filtering:
                for t_true in head_to_tails.get(h, ()):
                    if t_true != t:
                        scores_tail[t_true] = float("-inf")
            rank_tail = int((scores_tail > scores_tail[t]).sum().item()) + 1
            reciprocal_rank_sum += 1.0 / rank_tail
            total += 1
            for k in hits:
                if rank_tail <= k:
                    hits[k] += 1

        if do_head:
            scores_head = scorer.score_head_corruption(
                node_embeddings[t], node_embeddings, relation_id=rel_id,
            )
            if protocol.use_filtering:
                for h_true in tail_to_heads.get(t, ()):
                    if h_true != h:
                        scores_head[h_true] = float("-inf")
            rank_head = int((scores_head > scores_head[h]).sum().item()) + 1
            reciprocal_rank_sum += 1.0 / rank_head
            total += 1
            for k in hits:
                if rank_head <= k:
                    hits[k] += 1

    mrr = reciprocal_rank_sum / total if total > 0 else 0.0
    result: dict[str, float] = {"mrr": mrr}
    for k, count in sorted(hits.items()):
        result[f"hits@{k}"] = count / total if total > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
# Link dataset protocol (data-side contract for any link dataset)
# ---------------------------------------------------------------------------

@dataclass
class LinkDatasetProtocol:
    """Generalized link dataset protocol discovery result.

    Populated by dataset-specific discovery functions. Captures all
    evidence needed to run link evaluation without dataset-specific
    branching in the runner.

    Fields:
        dataset_name:    Canonical dataset identifier
        num_entities:    Total entity count
        train_edges:     List of (head_id, tail_id) training edges
        valid_edges:     List of (head_id, tail_id) validation edges
        test_edges:      List of (head_id, tail_id) test edges
        relation_ids_train: Optional per-edge relation type indices
        relation_ids_valid: Optional per-edge relation type indices
        relation_ids_test:  Optional per-edge relation type indices
        num_relations:   Number of distinct relation types (None if unknown)
        sbert_available: Whether SBERT features exist for this dataset
        missing:         List of strings explaining gaps preventing eval
        extra:           Additional dataset-specific metadata
    """

    dataset_name: str
    num_entities: int
    train_edges: list[tuple[int, int]]
    valid_edges: list[tuple[int, int]]
    test_edges: list[tuple[int, int]]
    relation_ids_train: list[int] | None = None
    relation_ids_valid: list[int] | None = None
    relation_ids_test: list[int] | None = None
    num_relations: int | None = None
    sbert_available: bool = False
    missing: list[str] = field(default_factory=list)
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def is_runnable(self) -> bool:
        return len(self.missing) == 0

    def summary(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "num_entities": self.num_entities,
            "train_edges": len(self.train_edges),
            "valid_edges": len(self.valid_edges),
            "test_edges": len(self.test_edges),
            "num_relations": self.num_relations,
            "has_relation_ids": self.relation_ids_test is not None,
            "sbert_available": self.sbert_available,
            "is_runnable": self.is_runnable,
            "missing": self.missing,
        }


# ---------------------------------------------------------------------------
# Relation-aware scorer training
# ---------------------------------------------------------------------------

def train_link_scorer(
    scorer: LinkScorer,
    node_embeddings,
    train_edges: list[tuple[int, int]],
    relation_ids_train: list[int] | None = None,
    *,
    num_entities: int,
    contract: NegativeSamplingContract | None = None,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 256,
    debug: bool = False,
    max_train_steps: int | None = None,
) -> dict[str, object]:
    """Train a relation-aware link scorer on frozen node embeddings.

    Uses BPR-style pairwise ranking loss::

        L = -log(sigmoid(score_pos - score_neg))

    with uniform negative sampling per the provided contract.
    Node embeddings are NOT updated; only scorer parameters are trained.

    Args:
        scorer:              LinkScorer with needs_training=True
        node_embeddings:     [N, dim] frozen entity embeddings
        train_edges:         List of (head_id, tail_id) training edges
        relation_ids_train:  Parallel list of relation type IDs (0-based,
                             consistent across all splits)
        num_entities:        Total entity count for negative sampling
        contract:            Negative sampling specification
        epochs:              Max training epochs
        lr:                  Learning rate for Adam optimizer
        batch_size:          Mini-batch size
        debug:               Print training progress
        max_train_steps:     Hard cap on gradient steps

    Returns:
        Dict with training metadata (skipped, epochs_completed, loss, etc.)
    """
    import torch

    if contract is None:
        contract = DEFAULT_NEGATIVE_SAMPLING_CONTRACT

    if not scorer.needs_training:
        return {"skipped": True, "reason": "scorer_does_not_need_training"}

    params = list(scorer.parameters())
    if not params:
        return {"skipped": True, "reason": "no_learnable_parameters"}

    # Detach embeddings — only scorer parameters receive gradients
    emb = node_embeddings.detach()
    optimizer = torch.optim.Adam(params, lr=lr)

    edges_t = torch.tensor(train_edges, dtype=torch.long)
    rels_t = (
        torch.tensor(relation_ids_train, dtype=torch.long)
        if relation_ids_train is not None
        else None
    )
    num_train = edges_t.shape[0]
    neg_k = contract.train_negatives_per_positive
    do_tail = contract.train_corruption_policy in ("tail", "both")
    do_head = contract.train_corruption_policy in ("head", "both")

    total_steps = 0
    best_loss = float("inf")
    final_loss = float("inf")

    for epoch in range(epochs):
        perm = torch.randperm(num_train)
        epoch_loss = 0.0
        batch_count = 0

        for start in range(0, num_train, batch_size):
            end = min(start + batch_size, num_train)
            idx = perm[start:end]
            b = idx.shape[0]

            h_ids = edges_t[idx, 0]
            t_ids = edges_t[idx, 1]
            h_emb = emb[h_ids]
            t_emb = emb[t_ids]

            rel_ids = rels_t[idx] if rels_t is not None else None
            r_diag = (
                scorer.relation_diag[rel_ids]
                if rel_ids is not None and hasattr(scorer, "relation_diag")
                else None
            )

            loss = torch.tensor(0.0)

            if do_tail:
                h_sc = h_emb * r_diag if r_diag is not None else h_emb
                pos = (h_sc * t_emb).sum(-1)
                neg_ids = torch.randint(0, num_entities, (b, neg_k))
                neg_emb = emb[neg_ids]
                neg = torch.einsum("bd,bkd->bk", h_sc, neg_emb)
                loss = loss - torch.nn.functional.logsigmoid(
                    pos.unsqueeze(1) - neg
                ).mean()

            if do_head:
                t_sc = t_emb * r_diag if r_diag is not None else t_emb
                pos = (h_emb * t_sc).sum(-1)
                neg_ids = torch.randint(0, num_entities, (b, neg_k))
                neg_emb = emb[neg_ids]
                neg = torch.einsum("bd,bkd->bk", t_sc, neg_emb)
                loss = loss - torch.nn.functional.logsigmoid(
                    pos.unsqueeze(1) - neg
                ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            total_steps += 1

            if max_train_steps is not None and total_steps >= max_train_steps:
                break

        avg_loss = epoch_loss / max(batch_count, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
        final_loss = avg_loss

        if debug and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
            print(
                f"[scorer_train] epoch {epoch}/{epochs}, "
                f"avg_loss={avg_loss:.6f}"
            )

        if max_train_steps is not None and total_steps >= max_train_steps:
            if debug:
                print(
                    f"[scorer_train] max_train_steps={max_train_steps} "
                    f"reached at epoch {epoch}"
                )
            break

    return {
        "skipped": False,
        "epochs_completed": epoch + 1,
        "total_steps": total_steps,
        "best_loss": round(best_loss, 6),
        "final_loss": round(final_loss, 6),
        "num_train_edges": num_train,
        "neg_per_pos": neg_k,
        "lr": lr,
        "batch_size": batch_size,
    }
