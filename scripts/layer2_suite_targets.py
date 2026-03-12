from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import shlex
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.registry import REGISTRY


WN18RR_BLOCKED_MESSAGE = (
    "WN18RR remains isolated in Layer 2 outside the wn18rr_experimental and "
    "wn18rr_experimental_compare target groups."
)


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: Path
    conda_env: str

    def shell_command(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True)
class TargetMetadata:
    target_name: str
    label: str
    model_name: str
    dataset_name: str
    feature_profile: str
    feature_path_rel: str | None
    suite_groups: tuple[str, ...]
    artifact_key: str
    artifact_group: str
    artifact_mode: str
    artifact_profile: str
    artifact_manifest_kind: str
    artifact_fallback_result_label: str | None
    supported_modes: tuple[str, ...]
    supports_regression_profile: bool
    task_type: str
    metric_name: str
    metric_additional_names: tuple[str, ...]
    dataset_experimental: bool
    requires_alignment_audit: bool
    supports_external_feat_pt: bool
    supported_models: tuple[str, ...]
    registry_caveats: tuple[str, ...]
    readiness_gate: dict[str, object]

    @property
    def experimental(self) -> bool:
        return self.dataset_experimental or self.artifact_group == "experimental"

    @property
    def official_candidate(self) -> bool:
        return (
            self.artifact_group == "official_candidate"
            and not self.dataset_experimental
            and not self.readiness_gate.get("promotion_blockers")
        )

    @property
    def debug_local(self) -> bool:
        return self.artifact_group == "local_debug"

    def classification(self, *, status: str | None = None) -> dict[str, bool]:
        return {
            "official_candidate": self.official_candidate,
            "debug_local": self.debug_local,
            "experimental": self.experimental,
            "blocked": status == "blocked",
            "promotion_ready": bool(
                not self.dataset_experimental
                and not self.readiness_gate.get("promotion_blockers")
            ),
        }

    def gate_caveats(self) -> dict[str, object]:
        """Return structured caveats derived from the readiness gate."""
        caveats: dict[str, object] = {}
        gate = self.readiness_gate
        if self.dataset_experimental:
            caveats["experimental"] = True
        if gate.get("relation_types_ignored"):
            caveats["relation_types_ignored"] = True
        if not gate.get("official_metric_available", True):
            caveats["official_metric"] = False
        if gate.get("scoring_method"):
            caveats["scoring"] = gate["scoring_method"]
        return caveats

    def to_manifest_metadata(self) -> dict[str, object]:
        return {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "task": self.task_type,
            "feature_profile": self.feature_profile,
            "artifact_group": self.artifact_group,
            "artifact_classification_base": {
                "official_candidate": self.official_candidate,
                "debug_local": self.debug_local,
                "experimental": self.experimental,
                "promotion_ready": bool(
                    not self.dataset_experimental
                    and not self.readiness_gate.get("promotion_blockers")
                ),
            },
            "target_groups": list(self.suite_groups),
            "supported_modes": list(self.supported_modes),
            "registry_metric_name": self.metric_name,
            "registry_metric_additional_names": list(self.metric_additional_names),
            "registry_dataset_experimental": self.dataset_experimental,
            "registry_official_candidate": not self.dataset_experimental,
            "registry_requires_alignment_audit": self.requires_alignment_audit,
            "registry_supports_external_feat_pt": self.supports_external_feat_pt,
            "registry_supported_models": list(self.supported_models),
            "registry_caveats": list(self.registry_caveats),
            "readiness_gate": dict(self.readiness_gate),
        }


@dataclass(frozen=True)
class ArtifactItem:
    key: str
    label: str
    kind: str
    group: str
    target_name: str
    mode: str | None = None
    profile: str | None = None
    manifest_kind: str | None = None
    fallback_result_label: str | None = None
    allow_missing: bool = False


@dataclass(frozen=True)
class ExpandedTarget:
    target_name: str
    mode: str
    profile: str


@dataclass(frozen=True)
class TargetPlan:
    target_name: str
    mode: str
    profile: str
    metadata: TargetMetadata
    checkpoint_path: Path
    out_json_path: Path
    export: CommandSpec
    eval: CommandSpec


@dataclass(frozen=True)
class AlignmentAuditPlan:
    command: CommandSpec
    out_json_path: Path


def _target_metadata(
    *,
    target_name: str,
    label: str,
    model_name: str,
    dataset_name: str,
    feature_profile: str,
    feature_path_rel: str | None,
    suite_groups: tuple[str, ...],
    artifact_key: str,
    artifact_group: str,
    artifact_mode: str,
    artifact_profile: str,
    artifact_manifest_kind: str,
    artifact_fallback_result_label: str | None,
    supported_modes: tuple[str, ...] = ("debug", "official"),
    supports_regression_profile: bool = True,
    readiness_overrides: dict[str, object] | None = None,
    caveats_override: tuple[str, ...] | None = None,
) -> TargetMetadata:
    adapter = REGISTRY.get_adapter(dataset_name)
    adapter.validate_model(model_name)
    readiness = replace(adapter.readiness, **(readiness_overrides or {}))
    registry_caveats = caveats_override if caveats_override is not None else adapter.caveats

    # Validate: official_candidate artifact group requires gate readiness
    if artifact_group == "official_candidate":
        if adapter.experimental:
            raise ValueError(
                f"Cannot assign official_candidate artifact group to experimental "
                f"dataset {dataset_name!r}."
            )
        if not readiness.is_promotion_ready:
            raise ValueError(
                f"Cannot assign official_candidate artifact group to dataset "
                f"{dataset_name!r}: promotion blockers "
                f"{readiness.promotion_blockers}"
            )

    return TargetMetadata(
        target_name=target_name,
        label=label,
        model_name=model_name,
        dataset_name=dataset_name,
        feature_profile=feature_profile,
        feature_path_rel=feature_path_rel,
        suite_groups=suite_groups,
        artifact_key=artifact_key,
        artifact_group=artifact_group,
        artifact_mode=artifact_mode,
        artifact_profile=artifact_profile,
        artifact_manifest_kind=artifact_manifest_kind,
        artifact_fallback_result_label=artifact_fallback_result_label,
        supported_modes=supported_modes,
        supports_regression_profile=supports_regression_profile,
        task_type=adapter.task_type,
        metric_name=adapter.metric.name,
        metric_additional_names=adapter.metric.additional_metrics,
        dataset_experimental=adapter.experimental,
        requires_alignment_audit=adapter.requires_alignment_audit,
        supports_external_feat_pt=adapter.supports_external_feat_pt,
        supported_models=adapter.supported_models,
        registry_caveats=registry_caveats,
        readiness_gate=readiness.to_dict(),
    )


_TARGET_METADATA_BY_NAME: dict[str, TargetMetadata] = {
    "graphmae_arxiv_sbert_node": _target_metadata(
        target_name="graphmae_arxiv_sbert_node",
        label="GraphMAE arXiv SBERT node",
        model_name="graphmae",
        dataset_name="ogbn-arxiv",
        feature_profile="sbert",
        feature_path_rel="data/arxiv_sbert.pt",
        suite_groups=(
            "official_candidate_arxiv",
            "official_candidate_local",
            "all_proven_local",
        ),
        artifact_key="graphmae_arxiv_official_candidate",
        artifact_group="official_candidate",
        artifact_mode="official",
        artifact_profile="default",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
    ),
    "bgrl_arxiv_sbert_node": _target_metadata(
        target_name="bgrl_arxiv_sbert_node",
        label="BGRL arXiv SBERT node",
        model_name="bgrl",
        dataset_name="ogbn-arxiv",
        feature_profile="sbert",
        feature_path_rel="data/arxiv_sbert.pt",
        suite_groups=(
            "official_candidate_arxiv",
            "official_candidate_local",
            "all_proven_local",
        ),
        artifact_key="bgrl_arxiv_official_candidate",
        artifact_group="official_candidate",
        artifact_mode="official",
        artifact_profile="default",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
    ),
    "graphmae_pcba_native_graph": _target_metadata(
        target_name="graphmae_pcba_native_graph",
        label="GraphMAE PCBA native graph",
        model_name="graphmae",
        dataset_name="ogbg-molpcba",
        feature_profile="native",
        feature_path_rel=None,
        suite_groups=(
            "official_candidate_local",
            "all_proven_local",
        ),
        artifact_key="graphmae_pcba_native_local_debug",
        artifact_group="local_debug",
        artifact_mode="debug",
        artifact_profile="default",
        artifact_manifest_kind="debug",
        artifact_fallback_result_label="pcba_debug_result",
    ),
    "graphmae_wn18rr_sbert_link": _target_metadata(
        target_name="graphmae_wn18rr_sbert_link",
        label="WN18RR experimental link-eval",
        model_name="graphmae",
        dataset_name="wn18rr",
        feature_profile="sbert",
        feature_path_rel="data/wn18rr_sbert.pt",
        suite_groups=("wn18rr_experimental", "wn18rr_experimental_compare"),
        artifact_key="wn18rr_experimental_link_eval",
        artifact_group="experimental",
        artifact_mode="debug",
        artifact_profile="default",
        artifact_manifest_kind="debug",
        artifact_fallback_result_label="wn18rr_debug_result",
        supports_regression_profile=False,
    ),
    "graphmae_wn18rr_sbert_link_relaware": _target_metadata(
        target_name="graphmae_wn18rr_sbert_link_relaware",
        label="WN18RR relation-aware experimental link-eval",
        model_name="graphmae",
        dataset_name="wn18rr",
        feature_profile="sbert",
        feature_path_rel="data/wn18rr_sbert.pt",
        suite_groups=("wn18rr_experimental", "wn18rr_experimental_compare"),
        artifact_key="wn18rr_relaware_experimental_link_eval",
        artifact_group="experimental",
        artifact_mode="debug",
        artifact_profile="default",
        artifact_manifest_kind="debug",
        artifact_fallback_result_label="wn18rr_relaware_debug_result",
        supports_regression_profile=False,
        readiness_overrides={
            "relation_types_ignored": False,
            "scoring_method": "relation_diagonal",
        },
        caveats_override=(
            "experimental=true",
            "scoring=relation_diagonal",
            "official_metric=false",
        ),
    ),
}

_TARGET_NAME_ORDER = tuple(_TARGET_METADATA_BY_NAME)

SINGLE_TARGETS = tuple(
    name
    for name in _TARGET_NAME_ORDER
    if not _TARGET_METADATA_BY_NAME[name].experimental
)

EXPERIMENTAL_TARGETS = tuple(
    name
    for name in _TARGET_NAME_ORDER
    if _TARGET_METADATA_BY_NAME[name].experimental
)

GROUP_TARGETS = (
    "official_candidate_arxiv",
    "official_candidate_local",
    "all_proven_local",
    "pcba_graph_compare",
    "regression_only",
    "wn18rr_experimental",
    "wn18rr_experimental_compare",
)

TARGET_GROUP_MEMBERS: dict[str, tuple[str, ...]] = {
    "official_candidate_arxiv": tuple(
        name
        for name in _TARGET_NAME_ORDER
        if "official_candidate_arxiv" in _TARGET_METADATA_BY_NAME[name].suite_groups
    ),
    "official_candidate_local": tuple(
        name
        for name in _TARGET_NAME_ORDER
        if "official_candidate_local" in _TARGET_METADATA_BY_NAME[name].suite_groups
    ),
    "all_proven_local": tuple(
        name
        for name in _TARGET_NAME_ORDER
        if "all_proven_local" in _TARGET_METADATA_BY_NAME[name].suite_groups
    ),
    "wn18rr_experimental": tuple(
        name
        for name in _TARGET_NAME_ORDER
        if "wn18rr_experimental" in _TARGET_METADATA_BY_NAME[name].suite_groups
    ),
    "wn18rr_experimental_compare": tuple(
        name
        for name in _TARGET_NAME_ORDER
        if "wn18rr_experimental_compare" in _TARGET_METADATA_BY_NAME[name].suite_groups
    ),
}

SUPPORTED_TARGETS = SINGLE_TARGETS + EXPERIMENTAL_TARGETS + GROUP_TARGETS

ARTIFACT_ITEMS: tuple[ArtifactItem, ...] = (
    ArtifactItem(
        key=_TARGET_METADATA_BY_NAME["graphmae_arxiv_sbert_node"].artifact_key,
        label=_TARGET_METADATA_BY_NAME["graphmae_arxiv_sbert_node"].label,
        kind="evaluation",
        group=_TARGET_METADATA_BY_NAME["graphmae_arxiv_sbert_node"].artifact_group,
        target_name="graphmae_arxiv_sbert_node",
        mode=_TARGET_METADATA_BY_NAME["graphmae_arxiv_sbert_node"].artifact_mode,
        profile=_TARGET_METADATA_BY_NAME["graphmae_arxiv_sbert_node"].artifact_profile,
        manifest_kind=_TARGET_METADATA_BY_NAME["graphmae_arxiv_sbert_node"].artifact_manifest_kind,
        fallback_result_label=_TARGET_METADATA_BY_NAME["graphmae_arxiv_sbert_node"].artifact_fallback_result_label,
    ),
    ArtifactItem(
        key=_TARGET_METADATA_BY_NAME["bgrl_arxiv_sbert_node"].artifact_key,
        label=_TARGET_METADATA_BY_NAME["bgrl_arxiv_sbert_node"].label,
        kind="evaluation",
        group=_TARGET_METADATA_BY_NAME["bgrl_arxiv_sbert_node"].artifact_group,
        target_name="bgrl_arxiv_sbert_node",
        mode=_TARGET_METADATA_BY_NAME["bgrl_arxiv_sbert_node"].artifact_mode,
        profile=_TARGET_METADATA_BY_NAME["bgrl_arxiv_sbert_node"].artifact_profile,
        manifest_kind=_TARGET_METADATA_BY_NAME["bgrl_arxiv_sbert_node"].artifact_manifest_kind,
        fallback_result_label=_TARGET_METADATA_BY_NAME["bgrl_arxiv_sbert_node"].artifact_fallback_result_label,
    ),
    ArtifactItem(
        key=_TARGET_METADATA_BY_NAME["graphmae_pcba_native_graph"].artifact_key,
        label=_TARGET_METADATA_BY_NAME["graphmae_pcba_native_graph"].label,
        kind="evaluation",
        group=_TARGET_METADATA_BY_NAME["graphmae_pcba_native_graph"].artifact_group,
        target_name="graphmae_pcba_native_graph",
        mode=_TARGET_METADATA_BY_NAME["graphmae_pcba_native_graph"].artifact_mode,
        profile=_TARGET_METADATA_BY_NAME["graphmae_pcba_native_graph"].artifact_profile,
        manifest_kind=_TARGET_METADATA_BY_NAME["graphmae_pcba_native_graph"].artifact_manifest_kind,
        fallback_result_label=_TARGET_METADATA_BY_NAME["graphmae_pcba_native_graph"].artifact_fallback_result_label,
    ),
    ArtifactItem(
        key="graphmae_pcba_native_full_local_non_debug",
        label="GraphMAE PCBA native graph (full-local non-debug)",
        kind="evaluation",
        group="local_debug",
        target_name="graphmae_pcba_native_graph",
        mode="official",
        profile="full_local_non_debug",
        manifest_kind="official",
        fallback_result_label="pcba_full_local_result",
        allow_missing=True,
    ),
    ArtifactItem(
        key="wn18rr_alignment_audit",
        label="WN18RR alignment audit",
        kind="alignment_audit",
        group="experimental",
        target_name="graphmae_wn18rr_sbert_link",
    ),
    ArtifactItem(
        key=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link"].artifact_key,
        label=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link"].label,
        kind="evaluation",
        group=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link"].artifact_group,
        target_name="graphmae_wn18rr_sbert_link",
        mode=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link"].artifact_mode,
        profile=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link"].artifact_profile,
        manifest_kind=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link"].artifact_manifest_kind,
        fallback_result_label=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link"].artifact_fallback_result_label,
    ),
    ArtifactItem(
        key=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link_relaware"].artifact_key,
        label=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link_relaware"].label,
        kind="evaluation",
        group=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link_relaware"].artifact_group,
        target_name="graphmae_wn18rr_sbert_link_relaware",
        mode=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link_relaware"].artifact_mode,
        profile=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link_relaware"].artifact_profile,
        manifest_kind=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link_relaware"].artifact_manifest_kind,
        fallback_result_label=_TARGET_METADATA_BY_NAME["graphmae_wn18rr_sbert_link_relaware"].artifact_fallback_result_label,
    ),
    # Full-scale WN18RR artifact items (mode=official, all 3134 test edges)
    ArtifactItem(
        key="wn18rr_fullscale_experimental_link_eval",
        label="WN18RR full-scale experimental link-eval",
        kind="evaluation",
        group="experimental",
        target_name="graphmae_wn18rr_sbert_link",
        mode="official",
        profile="default",
        manifest_kind="official",
        fallback_result_label="wn18rr_fullscale_result",
        allow_missing=True,
    ),
    ArtifactItem(
        key="wn18rr_relaware_fullscale_experimental_link_eval",
        label="WN18RR relation-aware full-scale experimental link-eval",
        kind="evaluation",
        group="experimental",
        target_name="graphmae_wn18rr_sbert_link_relaware",
        mode="official",
        profile="default",
        manifest_kind="official",
        fallback_result_label="wn18rr_relaware_fullscale_result",
        allow_missing=True,
    ),
)


def get_target_metadata(target_name: str) -> TargetMetadata:
    normalized = target_name.strip().lower()
    if normalized not in _TARGET_METADATA_BY_NAME:
        raise ValueError(
            f"Unsupported target: {target_name}. "
            f"Expected one of {', '.join(_TARGET_NAME_ORDER)}."
        )
    return _TARGET_METADATA_BY_NAME[normalized]


def list_target_metadata() -> tuple[TargetMetadata, ...]:
    return tuple(_TARGET_METADATA_BY_NAME[name] for name in _TARGET_NAME_ORDER)


def get_artifact_items() -> tuple[ArtifactItem, ...]:
    return ARTIFACT_ITEMS


def _conda_python_command(
    *,
    env_name: str,
    script_name: str,
    script_args: list[str],
    cwd: Path,
) -> CommandSpec:
    return CommandSpec(
        argv=tuple(["conda", "run", "-n", env_name, "python", script_name, *script_args]),
        cwd=cwd,
        conda_env=env_name,
    )


def _project_path(project_root: Path, relative_path: str) -> Path:
    return (project_root / relative_path).resolve()


def _relative_argument_path(cwd: Path, project_root: Path, relative_path: str) -> str:
    return os.path.relpath(_project_path(project_root, relative_path), start=cwd)


def _run_lp_eval_command(
    project_root: Path,
    *,
    metadata: TargetMetadata,
    env_name: str,
    checkpoint_rel: str,
    out_json_rel: str,
    debug: bool,
    extra_args: list[str],
) -> CommandSpec:
    cwd = project_root
    args = [
        "--model", metadata.model_name,
        "--dataset", metadata.dataset_name,
        "--ckpt", checkpoint_rel,
        "--out_json", out_json_rel,
    ]
    if debug:
        args.append("--debug")
    args.extend(extra_args)
    if metadata.feature_path_rel is not None:
        args.extend(["--feat-pt", metadata.feature_path_rel])
    return _conda_python_command(
        env_name=env_name,
        script_name="eval/run_lp.py",
        script_args=args,
        cwd=cwd,
    )


def _graphmae_transductive_export(
    project_root: Path,
    *,
    metadata: TargetMetadata,
    checkpoint_rel: str,
    max_epoch: int,
) -> CommandSpec:
    cwd = _project_path(project_root, "repos/graphmae")
    args = [
        "--seeds", "0",
        "--dataset", metadata.dataset_name,
        "--device", "-1",
        "--max_epoch", str(max_epoch),
        "--max_epoch_f", str(max_epoch),
        "--num_hidden", "512",
        "--num_heads", "4",
        "--num_layers", "2",
        "--lr", "0.005",
        "--weight_decay", "0.0005",
        "--lr_f", "0.001",
        "--weight_decay_f", "0.0",
        "--in_drop", "0.2",
        "--attn_drop", "0.1",
        "--mask_rate", "0.5",
        "--replace_rate", "0.0",
        "--encoder", "gat",
        "--decoder", "gat",
        "--loss_fn", "sce",
        "--alpha_l", "2.0",
        "--optimizer", "adam",
    ]
    if metadata.feature_path_rel is not None:
        args.extend(
            [
                "--feat-pt",
                _relative_argument_path(cwd, project_root, metadata.feature_path_rel),
            ]
        )
    args.extend(["--export-encoder-ckpt", checkpoint_rel])
    return _conda_python_command(
        env_name="graphmae_env",
        script_name="main_transductive.py",
        script_args=args,
        cwd=cwd,
    )


def _graphmae_arxiv_export(
    project_root: Path,
    checkpoint_rel: str,
) -> CommandSpec:
    return _graphmae_transductive_export(
        project_root,
        metadata=get_target_metadata("graphmae_arxiv_sbert_node"),
        checkpoint_rel=checkpoint_rel,
        max_epoch=30,
    )


def _bgrl_arxiv_export(
    project_root: Path,
    *,
    checkpoint_rel: str,
    epochs: int,
    cache_step: int,
    force_cpu: bool,
) -> CommandSpec:
    metadata = get_target_metadata("bgrl_arxiv_sbert_node")
    cwd = _project_path(project_root, "repos/bgrl")
    args = [
        "--name", "arxiv",
        "--root", "../../data",
        "--epochs", str(epochs),
        "--cache-step", str(cache_step),
    ]
    if metadata.feature_path_rel is not None:
        args.extend(
            [
                "--feat-pt",
                _relative_argument_path(cwd, project_root, metadata.feature_path_rel),
            ]
        )
    args.extend(["--export-encoder-ckpt", checkpoint_rel])
    if force_cpu:
        args.extend(["--device", "-1"])
    return _conda_python_command(
        env_name="llm",
        script_name="train.py",
        script_args=args,
        cwd=cwd,
    )


def _graphmae_pcba_export(
    project_root: Path,
    *,
    checkpoint_rel: str,
    max_epoch: int,
    eval_mode: str,
    eval_max_graphs: int | None,
    debug: bool,
    debug_max_graphs: int | None,
    batch_size: int | None,
) -> CommandSpec:
    metadata = get_target_metadata("graphmae_pcba_native_graph")
    cwd = _project_path(project_root, "repos/graphmae")
    args = [
        "--dataset", metadata.dataset_name,
        "--device", "-1",
        "--max_epoch", str(max_epoch),
        "--eval", eval_mode,
    ]
    if eval_max_graphs is not None:
        args.extend(["--eval_max_graphs", str(eval_max_graphs)])
    if debug:
        args.append("--debug")
    if debug_max_graphs is not None:
        args.extend(["--debug_max_graphs", str(debug_max_graphs)])
    if batch_size is not None:
        args.extend(["--batch_size", str(batch_size)])
    args.extend(["--export-encoder-ckpt", checkpoint_rel])
    return _conda_python_command(
        env_name="graphmae_env",
        script_name="main_graph.py",
        script_args=args,
        cwd=cwd,
    )


def _graphmae_arxiv_eval(
    project_root: Path,
    *,
    checkpoint_rel: str,
    out_json_rel: str,
    max_train_steps: int,
) -> CommandSpec:
    return _run_lp_eval_command(
        project_root,
        metadata=get_target_metadata("graphmae_arxiv_sbert_node"),
        env_name="graphmae_env",
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=True,
        extra_args=["--max_train_steps", str(max_train_steps)],
    )


def _bgrl_arxiv_eval(
    project_root: Path,
    *,
    checkpoint_rel: str,
    out_json_rel: str,
    max_train_steps: int,
) -> CommandSpec:
    return _run_lp_eval_command(
        project_root,
        metadata=get_target_metadata("bgrl_arxiv_sbert_node"),
        env_name="llm",
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=True,
        extra_args=["--max_train_steps", str(max_train_steps)],
    )


def _graphmae_pcba_eval(
    project_root: Path,
    *,
    checkpoint_rel: str,
    out_json_rel: str,
    debug_max_graphs: int,
    batch_size: int,
    max_train_steps: int,
    max_eval_batches: int,
) -> CommandSpec:
    return _run_lp_eval_command(
        project_root,
        metadata=get_target_metadata("graphmae_pcba_native_graph"),
        env_name="graphmae_env",
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=True,
        extra_args=[
            "--debug_max_graphs", str(debug_max_graphs),
            "--batch_size", str(batch_size),
            "--num_workers", "0",
            "--max_train_steps", str(max_train_steps),
            "--max_eval_batches", str(max_eval_batches),
        ],
    )


def _graphmae_arxiv_eval_official(
    project_root: Path,
    *,
    checkpoint_rel: str,
    out_json_rel: str,
) -> CommandSpec:
    return _run_lp_eval_command(
        project_root,
        metadata=get_target_metadata("graphmae_arxiv_sbert_node"),
        env_name="graphmae_env",
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=False,
        extra_args=[],
    )


def _bgrl_arxiv_eval_official(
    project_root: Path,
    *,
    checkpoint_rel: str,
    out_json_rel: str,
) -> CommandSpec:
    return _run_lp_eval_command(
        project_root,
        metadata=get_target_metadata("bgrl_arxiv_sbert_node"),
        env_name="llm",
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=False,
        extra_args=[],
    )


def _graphmae_pcba_eval_official(
    project_root: Path,
    *,
    checkpoint_rel: str,
    out_json_rel: str,
    max_train_steps: int | None = None,
) -> CommandSpec:
    extra_args: list[str] = []
    if max_train_steps is not None:
        extra_args.extend(["--max_train_steps", str(max_train_steps)])
    return _run_lp_eval_command(
        project_root,
        metadata=get_target_metadata("graphmae_pcba_native_graph"),
        env_name="graphmae_env",
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=False,
        extra_args=extra_args,
    )


def _graphmae_wn18rr_export(
    project_root: Path,
    checkpoint_rel: str,
    *,
    debug: bool = False,
) -> CommandSpec:
    return _graphmae_transductive_export(
        project_root,
        metadata=get_target_metadata("graphmae_wn18rr_sbert_link"),
        checkpoint_rel=checkpoint_rel,
        max_epoch=5 if debug else 30,
    )


def _graphmae_wn18rr_eval(
    project_root: Path,
    *,
    checkpoint_rel: str,
    out_json_rel: str,
) -> CommandSpec:
    return _run_lp_eval_command(
        project_root,
        metadata=get_target_metadata("graphmae_wn18rr_sbert_link"),
        env_name="graphmae_env",
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=True,
        extra_args=[],
    )


def _graphmae_wn18rr_eval_official(
    project_root: Path,
    *,
    checkpoint_rel: str,
    out_json_rel: str,
) -> CommandSpec:
    return _run_lp_eval_command(
        project_root,
        metadata=get_target_metadata("graphmae_wn18rr_sbert_link"),
        env_name="graphmae_env",
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=False,
        extra_args=[],
    )


def expand_target(target: str, mode: str) -> list[ExpandedTarget]:
    normalized = target.strip().lower()
    if normalized == "pcba_graph_compare":
        if mode != "debug":
            raise ValueError("pcba_graph_compare is a debug-only target group.")
        return [
            ExpandedTarget(
                target_name="graphmae_pcba_native_graph",
                mode="debug",
                profile="default",
            ),
            ExpandedTarget(
                target_name="graphmae_pcba_native_graph",
                mode="official",
                profile="full_local_non_debug",
            ),
        ]
    if normalized in TARGET_GROUP_MEMBERS:
        return [
            ExpandedTarget(target_name=name, mode=mode, profile="default")
            for name in TARGET_GROUP_MEMBERS[normalized]
        ]
    if normalized == "regression_only":
        if mode != "debug":
            raise ValueError("regression_only is a debug-only target group.")
        return [
            ExpandedTarget(target_name=name, mode=mode, profile="regression")
            for name in _TARGET_NAME_ORDER
            if _TARGET_METADATA_BY_NAME[name].supports_regression_profile
        ]
    if normalized in _TARGET_METADATA_BY_NAME:
        if normalized == "graphmae_pcba_native_graph" and mode == "official":
            return [
                ExpandedTarget(
                    target_name=normalized,
                    mode="official",
                    profile="full_local_non_debug",
                )
            ]
        return [ExpandedTarget(target_name=normalized, mode=mode, profile="default")]
    if "wn18rr" in normalized:
        raise ValueError(WN18RR_BLOCKED_MESSAGE)
    raise ValueError(
        f"Unsupported target: {target}. Expected one of {', '.join(SUPPORTED_TARGETS)}."
    )


def build_target_plan(
    *,
    project_root: Path,
    target_name: str,
    mode: str,
    profile: str,
) -> TargetPlan:
    project_root = project_root.resolve()
    metadata = get_target_metadata(target_name)

    if mode not in metadata.supported_modes:
        raise ValueError(
            f"Target {target_name} does not support mode={mode!r}. "
            f"Supported modes: {', '.join(metadata.supported_modes)}."
        )
    if profile not in {"default", "regression", "full_local_non_debug", "official_local"}:
        raise ValueError(
            "Unsupported profile. Expected 'default', 'regression', "
            "'full_local_non_debug', or 'official_local'."
        )
    if profile == "regression" and not metadata.supports_regression_profile:
        raise ValueError(f"Target {target_name} does not define a regression profile.")
    if profile in {"full_local_non_debug", "official_local"} and target_name != "graphmae_pcba_native_graph":
        raise ValueError(
            f"Target {target_name} does not define the {profile} profile."
        )

    if target_name == "graphmae_arxiv_sbert_node":
        if mode == "debug":
            checkpoint_rel = "checkpoints/graphmae_arxiv_debug.pt"
            export = _graphmae_arxiv_export(
                project_root,
                checkpoint_rel="../../checkpoints/graphmae_arxiv_debug.pt",
            )
            eval_steps = 2 if profile == "regression" else 5
            out_json_rel = (
                "results/baseline/graphmae_ogbn-arxiv.regression.json"
                if profile == "regression"
                else "results/baseline/graphmae_ogbn-arxiv.debug.json"
            )
            eval_cmd = _graphmae_arxiv_eval(
                project_root,
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
                max_train_steps=eval_steps,
            )
        else:
            checkpoint_rel = "checkpoints/graphmae_ogbn-arxiv.pt"
            export = _graphmae_arxiv_export(
                project_root,
                checkpoint_rel="../../checkpoints/graphmae_ogbn-arxiv.pt",
            )
            out_json_rel = "results/baseline/graphmae_ogbn-arxiv.json"
            eval_cmd = _graphmae_arxiv_eval_official(
                project_root,
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
            )
        return TargetPlan(
            target_name=target_name,
            mode=mode,
            profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export,
            eval=eval_cmd,
        )

    if target_name == "bgrl_arxiv_sbert_node":
        if mode == "debug":
            checkpoint_rel = "checkpoints/bgrl_arxiv_sbert_debug.pt"
            export = _bgrl_arxiv_export(
                project_root,
                checkpoint_rel="../../checkpoints/bgrl_arxiv_sbert_debug.pt",
                epochs=1,
                cache_step=1,
                force_cpu=True,
            )
            out_json_rel = (
                "results/baseline/bgrl_ogbn-arxiv.regression.json"
                if profile == "regression"
                else "results/baseline/bgrl_ogbn-arxiv.debug.json"
            )
            eval_cmd = _bgrl_arxiv_eval(
                project_root,
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
                max_train_steps=5,
            )
        else:
            checkpoint_rel = "checkpoints/bgrl_ogbn-arxiv.pt"
            export = _bgrl_arxiv_export(
                project_root,
                checkpoint_rel="../../checkpoints/bgrl_ogbn-arxiv.pt",
                epochs=100,
                cache_step=10,
                force_cpu=False,
            )
            out_json_rel = "results/baseline/bgrl_ogbn-arxiv.json"
            eval_cmd = _bgrl_arxiv_eval_official(
                project_root,
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
            )
        return TargetPlan(
            target_name=target_name,
            mode=mode,
            profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export,
            eval=eval_cmd,
        )

    if target_name == "graphmae_pcba_native_graph":
        if mode == "debug":
            checkpoint_rel = "checkpoints/graphmae_pcba_native_debug.pt"
            export = _graphmae_pcba_export(
                project_root,
                checkpoint_rel="../../checkpoints/graphmae_pcba_native_debug.pt",
                max_epoch=1,
                eval_mode="none",
                eval_max_graphs=None,
                debug=True,
                debug_max_graphs=64,
                batch_size=8,
            )
            if profile == "regression":
                out_json_rel = "results/baseline/graphmae_ogbg-molpcba.regression.json"
                eval_cmd = _graphmae_pcba_eval(
                    project_root,
                    checkpoint_rel=checkpoint_rel,
                    out_json_rel=out_json_rel,
                    debug_max_graphs=16,
                    batch_size=2,
                    max_train_steps=2,
                    max_eval_batches=1,
                )
            else:
                out_json_rel = "results/baseline/graphmae_ogbg-molpcba.native.debug.json"
                eval_cmd = _graphmae_pcba_eval(
                    project_root,
                    checkpoint_rel=checkpoint_rel,
                    out_json_rel=out_json_rel,
                    debug_max_graphs=64,
                    batch_size=8,
                    max_train_steps=4,
                    max_eval_batches=8,
                )
        else:
            checkpoint_rel = "checkpoints/graphmae_ogbg-molpcba.official_local.pt"
            export = _graphmae_pcba_export(
                project_root,
                checkpoint_rel="../../checkpoints/graphmae_ogbg-molpcba.official_local.pt",
                max_epoch=1,
                eval_mode="none",
                eval_max_graphs=None,
                debug=False,
                debug_max_graphs=None,
                batch_size=None,
            )
            out_json_rel = "results/baseline/graphmae_ogbg-molpcba.official_local.json"
            eval_cmd = _graphmae_pcba_eval_official(
                project_root,
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
                max_train_steps=32,
            )
        return TargetPlan(
            target_name=target_name,
            mode=mode,
            profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export,
            eval=eval_cmd,
        )

    if target_name == "graphmae_wn18rr_sbert_link":
        if mode == "debug":
            checkpoint_rel = "checkpoints/graphmae_wn18rr_debug.pt"
            export = _graphmae_wn18rr_export(
                project_root,
                checkpoint_rel="../../checkpoints/graphmae_wn18rr_debug.pt",
                debug=True,
            )
            out_json_rel = "results/baseline/graphmae_wn18rr.experimental.debug.json"
            eval_cmd = _graphmae_wn18rr_eval(
                project_root,
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
            )
        else:
            checkpoint_rel = "checkpoints/graphmae_wn18rr.pt"
            export = _graphmae_wn18rr_export(
                project_root,
                checkpoint_rel="../../checkpoints/graphmae_wn18rr.pt",
                debug=False,
            )
            out_json_rel = "results/baseline/graphmae_wn18rr.experimental.json"
            eval_cmd = _graphmae_wn18rr_eval_official(
                project_root,
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
            )
        return TargetPlan(
            target_name=target_name,
            mode=mode,
            profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export,
            eval=eval_cmd,
        )

    if target_name == "graphmae_wn18rr_sbert_link_relaware":
        # Same encoder checkpoint as dot-product target; only eval differs
        if mode == "debug":
            checkpoint_rel = "checkpoints/graphmae_wn18rr_debug.pt"
            export = _graphmae_wn18rr_export(
                project_root,
                checkpoint_rel="../../checkpoints/graphmae_wn18rr_debug.pt",
                debug=True,
            )
            out_json_rel = "results/baseline/graphmae_wn18rr.relaware.experimental.debug.json"
            eval_cmd = _run_lp_eval_command(
                project_root,
                metadata=metadata,
                env_name="graphmae_env",
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
                debug=True,
                extra_args=["--link-scorer", "relation_diagonal"],
            )
        else:
            checkpoint_rel = "checkpoints/graphmae_wn18rr.pt"
            export = _graphmae_wn18rr_export(
                project_root,
                checkpoint_rel="../../checkpoints/graphmae_wn18rr.pt",
                debug=False,
            )
            out_json_rel = "results/baseline/graphmae_wn18rr.relaware.experimental.json"
            eval_cmd = _run_lp_eval_command(
                project_root,
                metadata=metadata,
                env_name="graphmae_env",
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
                debug=False,
                extra_args=["--link-scorer", "relation_diagonal"],
            )
        return TargetPlan(
            target_name=target_name,
            mode=mode,
            profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export,
            eval=eval_cmd,
        )

    raise ValueError(
        f"Unsupported target: {target_name}. "
        f"Expected one of {', '.join(_TARGET_NAME_ORDER)}."
    )


def get_alignment_audit(
    project_root: Path,
    target_name: str,
) -> AlignmentAuditPlan | None:
    """Return an alignment audit plan if the target requires pre-eval verification."""
    metadata = get_target_metadata(target_name)
    if not metadata.requires_alignment_audit:
        return None
    if metadata.dataset_name != "wn18rr":
        raise ValueError(
            f"No alignment audit command is configured for dataset {metadata.dataset_name!r}."
        )

    project_root = project_root.resolve()
    out_json_rel = "results/baseline/wn18rr_alignment_audit.json"
    return AlignmentAuditPlan(
        command=_conda_python_command(
            env_name="graphmae_env",
            script_name="scripts/verify_wn18rr_alignment.py",
            script_args=[
                "--dataset-root", "data/WN18RR",
                "--feat-pt", metadata.feature_path_rel or "data/wn18rr_sbert.pt",
                "--out_json", out_json_rel,
            ],
            cwd=project_root,
        ),
        out_json_path=(project_root / out_json_rel).resolve(),
    )


def get_semantic_alignment_audit(
    project_root: Path,
    target_name: str,
) -> AlignmentAuditPlan | None:
    """Return a semantic alignment audit plan for WN18RR targets."""
    metadata = get_target_metadata(target_name)
    if not metadata.requires_alignment_audit:
        return None
    if metadata.dataset_name != "wn18rr":
        return None

    project_root = project_root.resolve()
    out_json_rel = "results/baseline/wn18rr_semantic_alignment_audit.json"
    return AlignmentAuditPlan(
        command=_conda_python_command(
            env_name="graphmae_env",
            script_name="scripts/verify_wn18rr_semantic_alignment.py",
            script_args=[
                "--dataset-root", "data/WN18RR",
                "--feat-pt", metadata.feature_path_rel or "data/wn18rr_sbert.pt",
                "--out_json", out_json_rel,
            ],
            cwd=project_root,
        ),
        out_json_path=(project_root / out_json_rel).resolve(),
    )


def target_help_text() -> str:
    supported = ", ".join(SUPPORTED_TARGETS)
    experimental_groups = ", ".join(
        group for group, members in TARGET_GROUP_MEMBERS.items() if any("wn18rr" in name for name in members)
    )
    return (
        "Supported targets: "
        f"{supported}. "
        f"WN18RR experimental targets ({experimental_groups}) are isolated from "
        "official_candidate_* and all_proven_local groups. "
        "Direct WN18RR references outside those WN18RR experimental groups remain blocked."
    )
