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


# Unified Conda environment for HPC dispatch.
# Defaults to "gfm_safety" (the single env created during UF HPC migration).
# Override at runtime via:  GFM_CONDA_ENV=other_env python -m scripts.run_layer2_suite ...
_CONDA_ENV: str = os.environ.get("GFM_CONDA_ENV") or "gfm_safety"

# Device index for official-mode export runs.
# Default: "0" (first GPU on HPC nodes with L4/B200).
# Debug-mode paths keep "-1" (CPU) where originally intended.
# Override at runtime via:  GFM_DEVICE=1 python3 scripts/run_layer2_suite.py ...
_DEVICE: str = os.environ.get("GFM_DEVICE") or "0"


def _make_eval_device(device_id: str) -> str:
    """Convert export device index ('0', '-1') to PyTorch device string for eval."""
    if device_id in ("-1", "cpu"):
        return "cpu"
    try:
        return f"cuda:{int(device_id)}"
    except ValueError:
        return device_id


# Eval-side device string — derived from _DEVICE.
# Export scripts accept integer indices ("0", "-1"); eval/run_lp.py accepts
# PyTorch device strings ("cpu", "cuda:0").
_EVAL_DEVICE: str = _make_eval_device(_DEVICE)

WN18RR_BLOCKED_MESSAGE = (
    "Unknown WN18RR target. Recognized WN18RR targets: "
    "graphmae_wn18rr_sbert_link, graphmae_wn18rr_sbert_link_relaware, "
    "bgrl_wn18rr_sbert_link, bgrl_wn18rr_sbert_link_relaware, "
    "graphmae_wn18rr_sbert_link_relaware_longrun, graphmae_wn18rr_sbert_link_longrun, "
    "bgrl_wn18rr_sbert_link_relaware_longrun, bgrl_wn18rr_sbert_link_longrun."
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
        label="WN18RR link-eval",
        model_name="graphmae",
        dataset_name="wn18rr",
        feature_profile="sbert",
        feature_path_rel="data/wn18rr_sbert.pt",
        suite_groups=("wn18rr_experimental", "wn18rr_experimental_compare", "all_proven_local"),
        artifact_key="wn18rr_experimental_link_eval",
        artifact_group="local_debug",
        artifact_mode="debug",
        artifact_profile="default",
        artifact_manifest_kind="debug",
        artifact_fallback_result_label="wn18rr_debug_result",
        supports_regression_profile=False,
    ),
    "graphmae_wn18rr_sbert_link_relaware": _target_metadata(
        target_name="graphmae_wn18rr_sbert_link_relaware",
        label="WN18RR relation-aware link-eval",
        model_name="graphmae",
        dataset_name="wn18rr",
        feature_profile="sbert",
        feature_path_rel="data/wn18rr_sbert.pt",
        suite_groups=("wn18rr_experimental", "wn18rr_experimental_compare", "all_proven_local"),
        artifact_key="wn18rr_relaware_experimental_link_eval",
        artifact_group="local_debug",
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
            "scoring=relation_diagonal",
        ),
    ),
    "bgrl_wn18rr_sbert_link": _target_metadata(
        target_name="bgrl_wn18rr_sbert_link",
        label="BGRL WN18RR link-eval",
        model_name="bgrl",
        dataset_name="wn18rr",
        feature_profile="sbert",
        feature_path_rel="data/wn18rr_sbert.pt",
        suite_groups=("wn18rr_experimental", "wn18rr_experimental_compare", "all_proven_local"),
        artifact_key="bgrl_wn18rr_experimental_link_eval",
        artifact_group="local_debug",
        artifact_mode="debug",
        artifact_profile="default",
        artifact_manifest_kind="debug",
        artifact_fallback_result_label="bgrl_wn18rr_debug_result",
        supports_regression_profile=False,
    ),
    "bgrl_wn18rr_sbert_link_relaware": _target_metadata(
        target_name="bgrl_wn18rr_sbert_link_relaware",
        label="BGRL WN18RR relation-aware link-eval",
        model_name="bgrl",
        dataset_name="wn18rr",
        feature_profile="sbert",
        feature_path_rel="data/wn18rr_sbert.pt",
        suite_groups=("wn18rr_experimental", "wn18rr_experimental_compare", "all_proven_local"),
        artifact_key="bgrl_wn18rr_relaware_experimental_link_eval",
        artifact_group="local_debug",
        artifact_mode="debug",
        artifact_profile="default",
        artifact_manifest_kind="debug",
        artifact_fallback_result_label="bgrl_wn18rr_relaware_debug_result",
        supports_regression_profile=False,
        readiness_overrides={
            "relation_types_ignored": False,
            "scoring_method": "relation_diagonal",
        },
        caveats_override=(
            "scoring=relation_diagonal",
        ),
    ),
    "bgrl_pcba_native_graph": _target_metadata(
        target_name="bgrl_pcba_native_graph",
        label="BGRL PCBA native graph",
        model_name="bgrl",
        dataset_name="ogbg-molpcba",
        feature_profile="native",
        feature_path_rel=None,
        suite_groups=(
            "all_proven_local",
        ),
        artifact_key="bgrl_pcba_native_local_debug",
        artifact_group="local_debug",
        artifact_mode="debug",
        artifact_profile="default",
        artifact_manifest_kind="debug",
        artifact_fallback_result_label="bgrl_pcba_debug_result",
    ),
    # ------------------------------------------------------------------
    # Phase 2 long-run targets
    # ------------------------------------------------------------------
    "graphmae_arxiv_sbert_node_longrun": _target_metadata(
        target_name="graphmae_arxiv_sbert_node_longrun",
        label="GraphMAE arXiv SBERT node (Phase 2 long-run)",
        model_name="graphmae",
        dataset_name="ogbn-arxiv",
        feature_profile="sbert",
        feature_path_rel="data/arxiv_sbert.pt",
        suite_groups=(
            "official_candidate_arxiv_longrun",
            "phase2_primary",
            "phase2_all_variants",
        ),
        artifact_key="graphmae_arxiv_longrun",
        artifact_group="phase2_longrun",
        artifact_mode="official",
        artifact_profile="longrun",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
        supported_modes=("official",),
        supports_regression_profile=False,
    ),
    "graphmae_arxiv_sbert_node_longrun_alt_ft": _target_metadata(
        target_name="graphmae_arxiv_sbert_node_longrun_alt_ft",
        label="GraphMAE arXiv SBERT node (Phase 2 alt finetune/lr)",
        model_name="graphmae",
        dataset_name="ogbn-arxiv",
        feature_profile="sbert",
        feature_path_rel="data/arxiv_sbert.pt",
        suite_groups=(
            "official_candidate_arxiv_longrun",
            "phase2_all_variants",
        ),
        artifact_key="graphmae_arxiv_longrun_alt_ft",
        artifact_group="phase2_longrun",
        artifact_mode="official",
        artifact_profile="longrun_alt_ft",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
        supported_modes=("official",),
        supports_regression_profile=False,
    ),
    "bgrl_arxiv_sbert_node_longrun": _target_metadata(
        target_name="bgrl_arxiv_sbert_node_longrun",
        label="BGRL arXiv SBERT node (Phase 2 long-run)",
        model_name="bgrl",
        dataset_name="ogbn-arxiv",
        feature_profile="sbert",
        feature_path_rel="data/arxiv_sbert.pt",
        suite_groups=(
            "official_candidate_arxiv_longrun",
            "phase2_primary",
            "phase2_all_variants",
        ),
        artifact_key="bgrl_arxiv_longrun",
        artifact_group="phase2_longrun",
        artifact_mode="official",
        artifact_profile="longrun",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
        supported_modes=("official",),
        supports_regression_profile=False,
    ),
    "bgrl_arxiv_sbert_node_longrun_alt_lr": _target_metadata(
        target_name="bgrl_arxiv_sbert_node_longrun_alt_lr",
        label="BGRL arXiv SBERT node (Phase 2 alt lr=0.0005)",
        model_name="bgrl",
        dataset_name="ogbn-arxiv",
        feature_profile="sbert",
        feature_path_rel="data/arxiv_sbert.pt",
        suite_groups=(
            "official_candidate_arxiv_longrun",
            "phase2_all_variants",
        ),
        artifact_key="bgrl_arxiv_longrun_alt_lr",
        artifact_group="phase2_longrun",
        artifact_mode="official",
        artifact_profile="longrun_alt_lr",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
        supported_modes=("official",),
        supports_regression_profile=False,
    ),
    "graphmae_pcba_native_graph_longrun": _target_metadata(
        target_name="graphmae_pcba_native_graph_longrun",
        label="GraphMAE PCBA native graph (Phase 2 long-run)",
        model_name="graphmae",
        dataset_name="ogbg-molpcba",
        feature_profile="native",
        feature_path_rel=None,
        suite_groups=(
            "pcba_longrun",
            "phase2_primary",
            "phase2_all_variants",
        ),
        artifact_key="graphmae_pcba_longrun",
        artifact_group="phase2_longrun",
        artifact_mode="official",
        artifact_profile="longrun",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
        supported_modes=("official",),
        supports_regression_profile=False,
    ),
    "graphmae_pcba_native_graph_longrun_alt_batch512": _target_metadata(
        target_name="graphmae_pcba_native_graph_longrun_alt_batch512",
        label="GraphMAE PCBA native graph (Phase 2 alt batch=512)",
        model_name="graphmae",
        dataset_name="ogbg-molpcba",
        feature_profile="native",
        feature_path_rel=None,
        suite_groups=(
            "pcba_longrun",
            "phase2_all_variants",
        ),
        artifact_key="graphmae_pcba_longrun_alt_batch512",
        artifact_group="phase2_longrun",
        artifact_mode="official",
        artifact_profile="longrun_alt_batch512",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
        supported_modes=("official",),
        supports_regression_profile=False,
    ),
    "bgrl_pcba_native_graph_longrun": _target_metadata(
        target_name="bgrl_pcba_native_graph_longrun",
        label="BGRL PCBA native graph (Phase 2 long-run)",
        model_name="bgrl",
        dataset_name="ogbg-molpcba",
        feature_profile="native",
        feature_path_rel=None,
        suite_groups=(
            "pcba_longrun",
            "phase2_primary",
            "phase2_all_variants",
        ),
        artifact_key="bgrl_pcba_longrun",
        artifact_group="phase2_longrun",
        artifact_mode="official",
        artifact_profile="longrun",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
        supported_modes=("official",),
        supports_regression_profile=False,
    ),
    "graphmae_wn18rr_sbert_link_relaware_longrun": _target_metadata(
        target_name="graphmae_wn18rr_sbert_link_relaware_longrun",
        label="WN18RR relation-aware link-eval (Phase 2 long-run)",
        model_name="graphmae",
        dataset_name="wn18rr",
        feature_profile="sbert",
        feature_path_rel="data/wn18rr_sbert.pt",
        suite_groups=(
            "wn18rr_longrun_compare",
            "phase2_primary",
            "phase2_all_variants",
        ),
        artifact_key="wn18rr_relaware_longrun",
        artifact_group="phase2_longrun",
        artifact_mode="official",
        artifact_profile="longrun",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
        supported_modes=("official",),
        supports_regression_profile=False,
        readiness_overrides={
            "relation_types_ignored": False,
            "scoring_method": "relation_diagonal",
        },
        caveats_override=(
            "scoring=relation_diagonal",
        ),
    ),
    "graphmae_wn18rr_sbert_link_longrun": _target_metadata(
        target_name="graphmae_wn18rr_sbert_link_longrun",
        label="WN18RR dot-product link-eval (Phase 2 long-run compare)",
        model_name="graphmae",
        dataset_name="wn18rr",
        feature_profile="sbert",
        feature_path_rel="data/wn18rr_sbert.pt",
        suite_groups=(
            "wn18rr_longrun_compare",
            "phase2_all_variants",
        ),
        artifact_key="wn18rr_dotprod_longrun",
        artifact_group="phase2_longrun",
        artifact_mode="official",
        artifact_profile="longrun",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
        supported_modes=("official",),
        supports_regression_profile=False,
    ),
    "bgrl_wn18rr_sbert_link_relaware_longrun": _target_metadata(
        target_name="bgrl_wn18rr_sbert_link_relaware_longrun",
        label="BGRL WN18RR relation-aware link-eval (Phase 2 long-run)",
        model_name="bgrl",
        dataset_name="wn18rr",
        feature_profile="sbert",
        feature_path_rel="data/wn18rr_sbert.pt",
        suite_groups=(
            "wn18rr_longrun_compare",
            "phase2_primary",
            "phase2_all_variants",
        ),
        artifact_key="bgrl_wn18rr_relaware_longrun",
        artifact_group="phase2_longrun",
        artifact_mode="official",
        artifact_profile="longrun",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
        supported_modes=("official",),
        supports_regression_profile=False,
        readiness_overrides={
            "relation_types_ignored": False,
            "scoring_method": "relation_diagonal",
        },
        caveats_override=(
            "scoring=relation_diagonal",
        ),
    ),
    "bgrl_wn18rr_sbert_link_longrun": _target_metadata(
        target_name="bgrl_wn18rr_sbert_link_longrun",
        label="BGRL WN18RR dot-product link-eval (Phase 2 long-run compare)",
        model_name="bgrl",
        dataset_name="wn18rr",
        feature_profile="sbert",
        feature_path_rel="data/wn18rr_sbert.pt",
        suite_groups=(
            "wn18rr_longrun_compare",
            "phase2_all_variants",
        ),
        artifact_key="bgrl_wn18rr_dotprod_longrun",
        artifact_group="phase2_longrun",
        artifact_mode="official",
        artifact_profile="longrun",
        artifact_manifest_kind="official",
        artifact_fallback_result_label=None,
        supported_modes=("official",),
        supports_regression_profile=False,
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
    # Phase 2 long-run groups
    "official_candidate_arxiv_longrun",
    "pcba_longrun",
    "wn18rr_longrun_compare",
    "phase2_primary",
    "phase2_all_variants",
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
    "official_candidate_arxiv_longrun": tuple(
        name
        for name in _TARGET_NAME_ORDER
        if "official_candidate_arxiv_longrun" in _TARGET_METADATA_BY_NAME[name].suite_groups
    ),
    "pcba_longrun": tuple(
        name
        for name in _TARGET_NAME_ORDER
        if "pcba_longrun" in _TARGET_METADATA_BY_NAME[name].suite_groups
    ),
    "wn18rr_longrun_compare": tuple(
        name
        for name in _TARGET_NAME_ORDER
        if "wn18rr_longrun_compare" in _TARGET_METADATA_BY_NAME[name].suite_groups
    ),
    "phase2_primary": tuple(
        name
        for name in _TARGET_NAME_ORDER
        if "phase2_primary" in _TARGET_METADATA_BY_NAME[name].suite_groups
    ),
    "phase2_all_variants": tuple(
        name
        for name in _TARGET_NAME_ORDER
        if "phase2_all_variants" in _TARGET_METADATA_BY_NAME[name].suite_groups
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
        key=_TARGET_METADATA_BY_NAME["bgrl_pcba_native_graph"].artifact_key,
        label=_TARGET_METADATA_BY_NAME["bgrl_pcba_native_graph"].label,
        kind="evaluation",
        group=_TARGET_METADATA_BY_NAME["bgrl_pcba_native_graph"].artifact_group,
        target_name="bgrl_pcba_native_graph",
        mode=_TARGET_METADATA_BY_NAME["bgrl_pcba_native_graph"].artifact_mode,
        profile=_TARGET_METADATA_BY_NAME["bgrl_pcba_native_graph"].artifact_profile,
        manifest_kind=_TARGET_METADATA_BY_NAME["bgrl_pcba_native_graph"].artifact_manifest_kind,
        fallback_result_label=_TARGET_METADATA_BY_NAME["bgrl_pcba_native_graph"].artifact_fallback_result_label,
    ),
    ArtifactItem(
        key="wn18rr_alignment_audit",
        label="WN18RR alignment audit",
        kind="alignment_audit",
        group="local_debug",
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
    ArtifactItem(
        key=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link"].artifact_key,
        label=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link"].label,
        kind="evaluation",
        group=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link"].artifact_group,
        target_name="bgrl_wn18rr_sbert_link",
        mode=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link"].artifact_mode,
        profile=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link"].artifact_profile,
        manifest_kind=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link"].artifact_manifest_kind,
        fallback_result_label=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link"].artifact_fallback_result_label,
    ),
    ArtifactItem(
        key=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link_relaware"].artifact_key,
        label=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link_relaware"].label,
        kind="evaluation",
        group=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link_relaware"].artifact_group,
        target_name="bgrl_wn18rr_sbert_link_relaware",
        mode=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link_relaware"].artifact_mode,
        profile=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link_relaware"].artifact_profile,
        manifest_kind=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link_relaware"].artifact_manifest_kind,
        fallback_result_label=_TARGET_METADATA_BY_NAME["bgrl_wn18rr_sbert_link_relaware"].artifact_fallback_result_label,
    ),
    # Full-scale WN18RR artifact items (mode=official, all 3134 test edges)
    ArtifactItem(
        key="wn18rr_fullscale_experimental_link_eval",
        label="WN18RR full-scale link-eval",
        kind="evaluation",
        group="local_debug",
        target_name="graphmae_wn18rr_sbert_link",
        mode="official",
        profile="default",
        manifest_kind="official",
        fallback_result_label="wn18rr_fullscale_result",
        allow_missing=True,
    ),
    ArtifactItem(
        key="wn18rr_relaware_fullscale_experimental_link_eval",
        label="WN18RR relation-aware full-scale link-eval",
        kind="evaluation",
        group="local_debug",
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
    device: str | None = None,
) -> CommandSpec:
    cwd = project_root
    args = [
        "--model", metadata.model_name,
        "--dataset", metadata.dataset_name,
        "--ckpt", checkpoint_rel,
        "--out_json", out_json_rel,
    ]
    if device is not None:
        args.extend(["--device", device])
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
    max_epoch_f: int | None = None,
    lr: float = 0.005,
    lr_f: float = 0.001,
    device: str = _DEVICE,
) -> CommandSpec:
    effective_max_epoch_f = max_epoch_f if max_epoch_f is not None else max_epoch
    cwd = _project_path(project_root, "repos/graphmae")
    args = [
        "--seeds", "0",
        "--dataset", metadata.dataset_name,
        "--device", device,
        "--max_epoch", str(max_epoch),
        "--max_epoch_f", str(effective_max_epoch_f),
        "--num_hidden", "512",
        "--num_heads", "4",
        "--num_layers", "2",
        "--lr", str(lr),
        "--weight_decay", "0.0005",
        "--lr_f", str(lr_f),
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
        env_name=_CONDA_ENV,
        script_name="main_transductive.py",
        script_args=args,
        cwd=cwd,
    )


def _graphmae_arxiv_export(
    project_root: Path,
    checkpoint_rel: str,
    *,
    device: str = _DEVICE,
) -> CommandSpec:
    return _graphmae_transductive_export(
        project_root,
        metadata=get_target_metadata("graphmae_arxiv_sbert_node"),
        checkpoint_rel=checkpoint_rel,
        max_epoch=30,
        device=device,
    )


def _bgrl_arxiv_export(
    project_root: Path,
    *,
    checkpoint_rel: str,
    epochs: int,
    cache_step: int,
    force_cpu: bool,
    lr: float | None = None,
    feat_pt_rel: str | None = None,
) -> CommandSpec:
    metadata = get_target_metadata("bgrl_arxiv_sbert_node")
    cwd = _project_path(project_root, "repos/bgrl")
    args = [
        "--name", "arxiv",
        "--root", "../../data",
        "--epochs", str(epochs),
        "--cache-step", str(cache_step),
    ]
    if lr is not None:
        args.extend(["--lr", str(lr)])
    effective_feat_pt = feat_pt_rel if feat_pt_rel is not None else metadata.feature_path_rel
    if effective_feat_pt is not None:
        args.extend(
            [
                "--feat-pt",
                _relative_argument_path(cwd, project_root, effective_feat_pt),
            ]
        )
    args.extend(["--export-encoder-ckpt", checkpoint_rel])
    if force_cpu:
        args.extend(["--device", "-1"])
    return _conda_python_command(
        env_name=_CONDA_ENV,
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
    lr: float | None = None,
) -> CommandSpec:
    metadata = get_target_metadata("graphmae_pcba_native_graph")
    cwd = _project_path(project_root, "repos/graphmae")
    args = [
        "--dataset", metadata.dataset_name,
        "--device", "-1" if debug else _DEVICE,
        "--max_epoch", str(max_epoch),
        "--eval", eval_mode,
    ]
    if lr is not None:
        args.extend(["--lr", str(lr)])
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
        env_name=_CONDA_ENV,
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
        env_name=_CONDA_ENV,
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
        env_name=_CONDA_ENV,
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
        env_name=_CONDA_ENV,
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
        env_name=_CONDA_ENV,
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=False,
        extra_args=[],
        device=_EVAL_DEVICE,
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
        env_name=_CONDA_ENV,
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=False,
        extra_args=[],
        device=_EVAL_DEVICE,
    )


def _graphmae_pcba_eval_official(
    project_root: Path,
    *,
    checkpoint_rel: str,
    out_json_rel: str,
    max_train_steps: int | None = None,
    ft_epochs: int | None = None,
    graph_eval_every: int | None = None,
) -> CommandSpec:
    extra_args: list[str] = []
    if max_train_steps is not None:
        extra_args.extend(["--max_train_steps", str(max_train_steps)])
    if ft_epochs is not None:
        extra_args.extend(["--ft_epochs", str(ft_epochs)])
    if graph_eval_every is not None:
        extra_args.extend(["--graph_eval_every", str(graph_eval_every)])
    return _run_lp_eval_command(
        project_root,
        metadata=get_target_metadata("graphmae_pcba_native_graph"),
        env_name=_CONDA_ENV,
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=False,
        extra_args=extra_args,
        device=_EVAL_DEVICE,
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
        device="-1" if debug else _DEVICE,
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
        env_name=_CONDA_ENV,
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
        env_name=_CONDA_ENV,
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=False,
        extra_args=[],
        device=_EVAL_DEVICE,
    )


def _bgrl_wn18rr_export(
    project_root: Path,
    *,
    checkpoint_rel: str,
    epochs: int,
    cache_step: int,
    force_cpu: bool,
) -> CommandSpec:
    metadata = get_target_metadata("bgrl_wn18rr_sbert_link")
    cwd = _project_path(project_root, "repos/bgrl")
    args = [
        "--name", "wn18rr",
        "--root", "../../data",
        "--epochs", str(epochs),
        "--cache-step", str(cache_step),
        "--skip-eval",
    ]
    effective_feat_pt = metadata.feature_path_rel
    if effective_feat_pt is not None:
        args.extend(
            [
                "--feat-pt",
                _relative_argument_path(cwd, project_root, effective_feat_pt),
            ]
        )
    args.extend(["--export-encoder-ckpt", checkpoint_rel])
    if force_cpu:
        args.extend(["--device", "-1"])
    return _conda_python_command(
        env_name=_CONDA_ENV,
        script_name="train.py",
        script_args=args,
        cwd=cwd,
    )


def _bgrl_wn18rr_eval(
    project_root: Path,
    *,
    checkpoint_rel: str,
    out_json_rel: str,
    debug: bool = True,
    extra_args: list[str] | None = None,
    device: str | None = None,
) -> CommandSpec:
    return _run_lp_eval_command(
        project_root,
        metadata=get_target_metadata("bgrl_wn18rr_sbert_link"),
        env_name=_CONDA_ENV,
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=debug,
        extra_args=extra_args or [],
        device=device,
    )


def _bgrl_pcba_export(
    project_root: Path,
    *,
    checkpoint_rel: str,
    epochs: int,
    cache_step: int,
    force_cpu: bool,
) -> CommandSpec:
    cwd = _project_path(project_root, "repos/bgrl")
    args = [
        "--name", "pcba",
        "--root", "../../data",
        "--epochs", str(epochs),
        "--cache-step", str(cache_step),
        "--skip-eval",
    ]
    args.extend(["--export-encoder-ckpt", checkpoint_rel])
    if force_cpu:
        args.extend(["--device", "-1"])
    return _conda_python_command(
        env_name=_CONDA_ENV,
        script_name="train.py",
        script_args=args,
        cwd=cwd,
    )


def _bgrl_pcba_eval(
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
        metadata=get_target_metadata("bgrl_pcba_native_graph"),
        env_name=_CONDA_ENV,
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


def _bgrl_pcba_eval_official(
    project_root: Path,
    *,
    checkpoint_rel: str,
    out_json_rel: str,
    max_train_steps: int | None = None,
    ft_epochs: int | None = None,
    graph_eval_every: int | None = None,
) -> CommandSpec:
    extra_args: list[str] = []
    if max_train_steps is not None:
        extra_args.extend(["--max_train_steps", str(max_train_steps)])
    if ft_epochs is not None:
        extra_args.extend(["--ft_epochs", str(ft_epochs)])
    if graph_eval_every is not None:
        extra_args.extend(["--graph_eval_every", str(graph_eval_every)])
    return _run_lp_eval_command(
        project_root,
        metadata=get_target_metadata("bgrl_pcba_native_graph"),
        env_name=_CONDA_ENV,
        checkpoint_rel=checkpoint_rel,
        out_json_rel=out_json_rel,
        debug=False,
        extra_args=extra_args,
        device=_EVAL_DEVICE,
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
                target_name="bgrl_pcba_native_graph",
                mode="debug",
                profile="default",
            ),
        ]
    if normalized in TARGET_GROUP_MEMBERS:
        result = []
        for name in TARGET_GROUP_MEMBERS[normalized]:
            meta = _TARGET_METADATA_BY_NAME[name]
            p = meta.artifact_profile if meta.artifact_group == "phase2_longrun" else "default"
            result.append(ExpandedTarget(target_name=name, mode=mode, profile=p))
        return result
    if normalized == "regression_only":
        if mode != "debug":
            raise ValueError("regression_only is a debug-only target group.")
        return [
            ExpandedTarget(target_name=name, mode=mode, profile="regression")
            for name in _TARGET_NAME_ORDER
            if _TARGET_METADATA_BY_NAME[name].supports_regression_profile
        ]
    if normalized in _TARGET_METADATA_BY_NAME:
        if normalized in ("graphmae_pcba_native_graph", "bgrl_pcba_native_graph") and mode == "official":
            return [
                ExpandedTarget(
                    target_name=normalized,
                    mode="official",
                    profile="full_local_non_debug",
                )
            ]
        meta = _TARGET_METADATA_BY_NAME[normalized]
        effective_profile = meta.artifact_profile if meta.artifact_group == "phase2_longrun" else "default"
        return [ExpandedTarget(target_name=normalized, mode=mode, profile=effective_profile)]
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
    _VALID_PROFILES = {"default", "regression", "full_local_non_debug", "official_local", "longrun",
                        "longrun_alt_ft", "longrun_alt_lr", "longrun_alt_batch512"}
    if profile not in _VALID_PROFILES:
        raise ValueError(
            f"Unsupported profile {profile!r}. Expected one of {sorted(_VALID_PROFILES)}."
        )
    if profile == "regression" and not metadata.supports_regression_profile:
        raise ValueError(f"Target {target_name} does not define a regression profile.")
    if profile in {"full_local_non_debug", "official_local"} and target_name not in ("graphmae_pcba_native_graph", "bgrl_pcba_native_graph"):
        raise ValueError(
            f"Target {target_name} does not define the {profile} profile."
        )

    if target_name == "graphmae_arxiv_sbert_node":
        if mode == "debug":
            checkpoint_rel = "checkpoints/graphmae_arxiv_debug.pt"
            export = _graphmae_arxiv_export(
                project_root,
                checkpoint_rel="../../checkpoints/graphmae_arxiv_debug.pt",
                device="-1",
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

    if target_name == "bgrl_pcba_native_graph":
        if mode == "debug":
            checkpoint_rel = "checkpoints/bgrl_pcba_native_debug.pt"
            export = _bgrl_pcba_export(
                project_root,
                checkpoint_rel="../../checkpoints/bgrl_pcba_native_debug.pt",
                epochs=1,
                cache_step=1,
                force_cpu=True,
            )
            if profile == "regression":
                out_json_rel = "results/baseline/bgrl_ogbg-molpcba.regression.json"
                eval_cmd = _bgrl_pcba_eval(
                    project_root,
                    checkpoint_rel=checkpoint_rel,
                    out_json_rel=out_json_rel,
                    debug_max_graphs=16,
                    batch_size=2,
                    max_train_steps=2,
                    max_eval_batches=1,
                )
            else:
                out_json_rel = "results/baseline/bgrl_ogbg-molpcba.native.debug.json"
                eval_cmd = _bgrl_pcba_eval(
                    project_root,
                    checkpoint_rel=checkpoint_rel,
                    out_json_rel=out_json_rel,
                    debug_max_graphs=64,
                    batch_size=8,
                    max_train_steps=4,
                    max_eval_batches=8,
                )
        else:
            checkpoint_rel = "checkpoints/bgrl_ogbg-molpcba.official_local.pt"
            export = _bgrl_pcba_export(
                project_root,
                checkpoint_rel="../../checkpoints/bgrl_ogbg-molpcba.official_local.pt",
                epochs=100,
                cache_step=10,
                force_cpu=False,
            )
            out_json_rel = "results/baseline/bgrl_ogbg-molpcba.official_local.json"
            eval_cmd = _bgrl_pcba_eval_official(
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
                env_name=_CONDA_ENV,
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
                env_name=_CONDA_ENV,
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
                debug=False,
                extra_args=["--link-scorer", "relation_diagonal"],
                device=_EVAL_DEVICE,
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

    if target_name == "bgrl_wn18rr_sbert_link":
        if mode == "debug":
            checkpoint_rel = "checkpoints/bgrl_wn18rr_debug.pt"
            export = _bgrl_wn18rr_export(
                project_root,
                checkpoint_rel="../../checkpoints/bgrl_wn18rr_debug.pt",
                epochs=5,
                cache_step=5,
                force_cpu=True,
            )
            out_json_rel = "results/baseline/bgrl_wn18rr.experimental.debug.json"
            eval_cmd = _bgrl_wn18rr_eval(
                project_root,
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
                debug=True,
            )
        else:
            checkpoint_rel = "checkpoints/bgrl_wn18rr.pt"
            export = _bgrl_wn18rr_export(
                project_root,
                checkpoint_rel="../../checkpoints/bgrl_wn18rr.pt",
                epochs=100,
                cache_step=10,
                force_cpu=False,
            )
            out_json_rel = "results/baseline/bgrl_wn18rr.experimental.json"
            eval_cmd = _bgrl_wn18rr_eval(
                project_root,
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
                debug=False,
                device=_EVAL_DEVICE,
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

    if target_name == "bgrl_wn18rr_sbert_link_relaware":
        # Same encoder checkpoint as dot-product target; only eval differs
        if mode == "debug":
            checkpoint_rel = "checkpoints/bgrl_wn18rr_debug.pt"
            export = _bgrl_wn18rr_export(
                project_root,
                checkpoint_rel="../../checkpoints/bgrl_wn18rr_debug.pt",
                epochs=5,
                cache_step=5,
                force_cpu=True,
            )
            out_json_rel = "results/baseline/bgrl_wn18rr.relaware.experimental.debug.json"
            eval_cmd = _bgrl_wn18rr_eval(
                project_root,
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
                debug=True,
                extra_args=["--link-scorer", "relation_diagonal"],
            )
        else:
            checkpoint_rel = "checkpoints/bgrl_wn18rr.pt"
            export = _bgrl_wn18rr_export(
                project_root,
                checkpoint_rel="../../checkpoints/bgrl_wn18rr.pt",
                epochs=100,
                cache_step=10,
                force_cpu=False,
            )
            out_json_rel = "results/baseline/bgrl_wn18rr.relaware.experimental.json"
            eval_cmd = _bgrl_wn18rr_eval(
                project_root,
                checkpoint_rel=checkpoint_rel,
                out_json_rel=out_json_rel,
                debug=False,
                extra_args=["--link-scorer", "relation_diagonal"],
                device=_EVAL_DEVICE,
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

    # ------------------------------------------------------------------
    # Phase 2 long-run targets
    # ------------------------------------------------------------------

    # GraphMAE arXiv long-run primary: pretrain=1000, ft=1000, lr=0.001, lr_f=0.001
    if target_name == "graphmae_arxiv_sbert_node_longrun":
        checkpoint_rel = "checkpoints/graphmae_ogbn-arxiv.longrun.pt"
        export = _graphmae_transductive_export(
            project_root,
            metadata=get_target_metadata("graphmae_arxiv_sbert_node"),
            checkpoint_rel="../../checkpoints/graphmae_ogbn-arxiv.longrun.pt",
            max_epoch=1000,
            max_epoch_f=1000,
            lr=0.001,
            lr_f=0.001,
        )
        out_json_rel = "results/baseline/graphmae_ogbn-arxiv.longrun.json"
        eval_cmd = _graphmae_arxiv_eval_official(
            project_root,
            checkpoint_rel=checkpoint_rel,
            out_json_rel=out_json_rel,
        )
        return TargetPlan(
            target_name=target_name, mode=mode, profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export, eval=eval_cmd,
        )

    # GraphMAE arXiv long-run alt: pretrain=1000, ft=500, lr=0.001, lr_f=0.005
    if target_name == "graphmae_arxiv_sbert_node_longrun_alt_ft":
        checkpoint_rel = "checkpoints/graphmae_ogbn-arxiv.longrun_alt_ft.pt"
        export = _graphmae_transductive_export(
            project_root,
            metadata=get_target_metadata("graphmae_arxiv_sbert_node"),
            checkpoint_rel="../../checkpoints/graphmae_ogbn-arxiv.longrun_alt_ft.pt",
            max_epoch=1000,
            max_epoch_f=500,
            lr=0.001,
            lr_f=0.005,
        )
        out_json_rel = "results/baseline/graphmae_ogbn-arxiv.longrun_alt_ft.json"
        eval_cmd = _graphmae_arxiv_eval_official(
            project_root,
            checkpoint_rel=checkpoint_rel,
            out_json_rel=out_json_rel,
        )
        return TargetPlan(
            target_name=target_name, mode=mode, profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export, eval=eval_cmd,
        )

    # BGRL arXiv long-run primary: epochs=1000, lr=0.001
    if target_name == "bgrl_arxiv_sbert_node_longrun":
        checkpoint_rel = "checkpoints/bgrl_ogbn-arxiv.longrun.pt"
        export = _bgrl_arxiv_export(
            project_root,
            checkpoint_rel="../../checkpoints/bgrl_ogbn-arxiv.longrun.pt",
            epochs=1000,
            cache_step=50,
            force_cpu=False,
            lr=0.001,
        )
        out_json_rel = "results/baseline/bgrl_ogbn-arxiv.longrun.json"
        eval_cmd = _bgrl_arxiv_eval_official(
            project_root,
            checkpoint_rel=checkpoint_rel,
            out_json_rel=out_json_rel,
        )
        return TargetPlan(
            target_name=target_name, mode=mode, profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export, eval=eval_cmd,
        )

    # BGRL arXiv long-run alt: epochs=1000, lr=0.0005
    if target_name == "bgrl_arxiv_sbert_node_longrun_alt_lr":
        checkpoint_rel = "checkpoints/bgrl_ogbn-arxiv.longrun_alt_lr.pt"
        export = _bgrl_arxiv_export(
            project_root,
            checkpoint_rel="../../checkpoints/bgrl_ogbn-arxiv.longrun_alt_lr.pt",
            epochs=1000,
            cache_step=50,
            force_cpu=False,
            lr=0.0005,
        )
        out_json_rel = "results/baseline/bgrl_ogbn-arxiv.longrun_alt_lr.json"
        eval_cmd = _bgrl_arxiv_eval_official(
            project_root,
            checkpoint_rel=checkpoint_rel,
            out_json_rel=out_json_rel,
        )
        return TargetPlan(
            target_name=target_name, mode=mode, profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export, eval=eval_cmd,
        )

    # GraphMAE PCBA long-run primary: pretrain=100, batch=256, lr=0.001
    if target_name == "graphmae_pcba_native_graph_longrun":
        checkpoint_rel = "checkpoints/graphmae_ogbg-molpcba.longrun.pt"
        export = _graphmae_pcba_export(
            project_root,
            checkpoint_rel="../../checkpoints/graphmae_ogbg-molpcba.longrun.pt",
            max_epoch=100,
            eval_mode="none",
            eval_max_graphs=None,
            debug=False,
            debug_max_graphs=None,
            batch_size=256,
            lr=0.001,
        )
        out_json_rel = "results/baseline/graphmae_ogbg-molpcba.longrun.json"
        eval_cmd = _graphmae_pcba_eval_official(
            project_root,
            checkpoint_rel=checkpoint_rel,
            out_json_rel=out_json_rel,
            ft_epochs=100,
            graph_eval_every=10,
        )
        return TargetPlan(
            target_name=target_name, mode=mode, profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export, eval=eval_cmd,
        )

    # GraphMAE PCBA long-run alt batch=512: pretrain=100, batch=512, lr=0.001
    if target_name == "graphmae_pcba_native_graph_longrun_alt_batch512":
        checkpoint_rel = "checkpoints/graphmae_ogbg-molpcba.longrun_alt_batch512.pt"
        export = _graphmae_pcba_export(
            project_root,
            checkpoint_rel="../../checkpoints/graphmae_ogbg-molpcba.longrun_alt_batch512.pt",
            max_epoch=100,
            eval_mode="none",
            eval_max_graphs=None,
            debug=False,
            debug_max_graphs=None,
            batch_size=512,
            lr=0.001,
        )
        out_json_rel = "results/baseline/graphmae_ogbg-molpcba.longrun_alt_batch512.json"
        eval_cmd = _graphmae_pcba_eval_official(
            project_root,
            checkpoint_rel=checkpoint_rel,
            out_json_rel=out_json_rel,
            ft_epochs=100,
            graph_eval_every=10,
        )
        return TargetPlan(
            target_name=target_name, mode=mode, profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export, eval=eval_cmd,
        )

    # BGRL PCBA long-run: pretrain=200 (union graph), eval=100 ft epochs
    if target_name == "bgrl_pcba_native_graph_longrun":
        checkpoint_rel = "checkpoints/bgrl_ogbg-molpcba.longrun.pt"
        export = _bgrl_pcba_export(
            project_root,
            checkpoint_rel="../../checkpoints/bgrl_ogbg-molpcba.longrun.pt",
            epochs=200,
            cache_step=20,
            force_cpu=False,
        )
        out_json_rel = "results/baseline/bgrl_ogbg-molpcba.longrun.json"
        eval_cmd = _bgrl_pcba_eval_official(
            project_root,
            checkpoint_rel=checkpoint_rel,
            out_json_rel=out_json_rel,
            ft_epochs=100,
            graph_eval_every=10,
        )
        return TargetPlan(
            target_name=target_name, mode=mode, profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export, eval=eval_cmd,
        )

    # WN18RR relation-aware long-run primary: pretrain=500, lr=0.001
    if target_name == "graphmae_wn18rr_sbert_link_relaware_longrun":
        checkpoint_rel = "checkpoints/graphmae_wn18rr.longrun.pt"
        export = _graphmae_transductive_export(
            project_root,
            metadata=get_target_metadata("graphmae_wn18rr_sbert_link"),
            checkpoint_rel="../../checkpoints/graphmae_wn18rr.longrun.pt",
            max_epoch=500,
            lr=0.001,
            lr_f=0.001,
        )
        out_json_rel = "results/baseline/graphmae_wn18rr.relaware.longrun.json"
        eval_cmd = _run_lp_eval_command(
            project_root,
            metadata=metadata,
            env_name=_CONDA_ENV,
            checkpoint_rel=checkpoint_rel,
            out_json_rel=out_json_rel,
            debug=False,
            extra_args=[
                "--link-scorer", "relation_diagonal",
                "--eval_every", "10",
                "--patience", "50",
            ],
            device=_EVAL_DEVICE,
        )
        return TargetPlan(
            target_name=target_name, mode=mode, profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export, eval=eval_cmd,
        )

    # WN18RR dot-product long-run compare: shares checkpoint with relaware longrun
    if target_name == "graphmae_wn18rr_sbert_link_longrun":
        checkpoint_rel = "checkpoints/graphmae_wn18rr.longrun.pt"
        export = _graphmae_transductive_export(
            project_root,
            metadata=get_target_metadata("graphmae_wn18rr_sbert_link"),
            checkpoint_rel="../../checkpoints/graphmae_wn18rr.longrun.pt",
            max_epoch=500,
            lr=0.001,
            lr_f=0.001,
        )
        out_json_rel = "results/baseline/graphmae_wn18rr.longrun.json"
        eval_cmd = _run_lp_eval_command(
            project_root,
            metadata=metadata,
            env_name=_CONDA_ENV,
            checkpoint_rel=checkpoint_rel,
            out_json_rel=out_json_rel,
            debug=False,
            extra_args=[],
            device=_EVAL_DEVICE,
        )
        return TargetPlan(
            target_name=target_name, mode=mode, profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export, eval=eval_cmd,
        )

    # BGRL WN18RR relation-aware long-run primary: pretrain=500, lr=0.00001
    if target_name == "bgrl_wn18rr_sbert_link_relaware_longrun":
        checkpoint_rel = "checkpoints/bgrl_wn18rr.longrun.pt"
        export = _bgrl_wn18rr_export(
            project_root,
            checkpoint_rel="../../checkpoints/bgrl_wn18rr.longrun.pt",
            epochs=500,
            cache_step=50,
            force_cpu=False,
        )
        out_json_rel = "results/baseline/bgrl_wn18rr.relaware.longrun.json"
        eval_cmd = _bgrl_wn18rr_eval(
            project_root,
            checkpoint_rel=checkpoint_rel,
            out_json_rel=out_json_rel,
            debug=False,
            extra_args=[
                "--link-scorer", "relation_diagonal",
                "--eval_every", "10",
                "--patience", "50",
            ],
            device=_EVAL_DEVICE,
        )
        return TargetPlan(
            target_name=target_name, mode=mode, profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export, eval=eval_cmd,
        )

    # BGRL WN18RR dot-product long-run compare: shares checkpoint with relaware longrun
    if target_name == "bgrl_wn18rr_sbert_link_longrun":
        checkpoint_rel = "checkpoints/bgrl_wn18rr.longrun.pt"
        export = _bgrl_wn18rr_export(
            project_root,
            checkpoint_rel="../../checkpoints/bgrl_wn18rr.longrun.pt",
            epochs=500,
            cache_step=50,
            force_cpu=False,
        )
        out_json_rel = "results/baseline/bgrl_wn18rr.longrun.json"
        eval_cmd = _bgrl_wn18rr_eval(
            project_root,
            checkpoint_rel=checkpoint_rel,
            out_json_rel=out_json_rel,
            debug=False,
            extra_args=[],
            device=_EVAL_DEVICE,
        )
        return TargetPlan(
            target_name=target_name, mode=mode, profile=profile,
            metadata=metadata,
            checkpoint_path=_project_path(project_root, checkpoint_rel),
            out_json_path=_project_path(project_root, out_json_rel),
            export=export, eval=eval_cmd,
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
            env_name=_CONDA_ENV,
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
            env_name=_CONDA_ENV,
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
    wn18rr_groups = ", ".join(
        group for group, members in TARGET_GROUP_MEMBERS.items() if any("wn18rr" in name for name in members)
    )
    phase2_groups = ", ".join(
        group for group in TARGET_GROUP_MEMBERS if group.startswith("phase2_") or group.endswith("_longrun") or group == "wn18rr_longrun_compare"
    )
    return (
        "Supported targets: "
        f"{supported}. "
        f"WN18RR targets are included in all_proven_local and accessible via "
        f"WN18RR-specific groups ({wn18rr_groups}). "
        "Baseline dot-product path retains relation_types_ignored=true caveat. "
        f"Phase 2 long-run groups: {phase2_groups}. "
        "Long-run targets require --mode official."
    )
