from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPHMAE_ROOT = REPO_ROOT / "repos" / "graphmae"
BGRL_ROOT = REPO_ROOT / "repos" / "bgrl"


def _ensure_graphmae_imports() -> None:
    path = str(GRAPHMAE_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_bgrl_models_module():
    module_name = "_gfm_safety_bgrl_models"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, BGRL_ROOT / "models.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load BGRL models module from {BGRL_ROOT / 'models.py'}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _freeze_encoder(encoder: nn.Module, device: str) -> nn.Module:
    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def _looks_like_state_dict(payload: object) -> bool:
    if not isinstance(payload, Mapping) or not payload:
        return False
    return all(isinstance(key, str) for key in payload.keys()) and all(
        torch.is_tensor(value) or isinstance(value, nn.Parameter)
        for value in payload.values()
    )


def _extract_state_dict(payload: object, ckpt_path: Path) -> OrderedDict[str, torch.Tensor]:
    if isinstance(payload, nn.Module):
        return OrderedDict((key, value.detach().clone()) for key, value in payload.state_dict().items())

    if _looks_like_state_dict(payload):
        return OrderedDict(payload.items())

    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "model_state", "encoder"):
            value = payload.get(key)
            if isinstance(value, nn.Module):
                return OrderedDict(
                    (state_key, state_value.detach().clone())
                    for state_key, state_value in value.state_dict().items()
                )
            if _looks_like_state_dict(value):
                return OrderedDict(value.items())

    raise ValueError(
        "Unsupported checkpoint format for "
        f"{ckpt_path}. Expected a state_dict or a dict containing "
        "one of: state_dict, model_state_dict, model, model_state, encoder."
    )


def _extract_checkpoint_metadata(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return {}

    metadata_keys = (
        "model_name",
        "dataset",
        "task_type",
        "hidden_dim",
        "encoder_input_dim",
        "backend",
        "exported_at",
        "feat_pt_used",
    )
    return {key: payload[key] for key in metadata_keys if key in payload}


def _coerce_optional_int(value: object, field_name: str, ckpt_path: Path) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {field_name!r} metadata in checkpoint {ckpt_path}: {value!r}"
        ) from exc


def _validate_checkpoint_metadata(
    metadata: Mapping[str, object],
    *,
    ckpt_path: Path,
    expected_model_name: str,
    inferred_input_dim: int,
    expected_backend: str,
) -> None:
    model_name = metadata.get("model_name")
    if model_name is not None and str(model_name).lower() != expected_model_name:
        raise ValueError(
            f"Checkpoint model_name mismatch for {ckpt_path}: expected "
            f"{expected_model_name!r}, found {model_name!r}"
        )

    backend = metadata.get("backend")
    if backend is not None and str(backend).lower() != expected_backend:
        raise ValueError(
            f"Checkpoint backend mismatch for {ckpt_path}: expected "
            f"{expected_backend!r}, found {backend!r}"
        )

    metadata_input_dim = _coerce_optional_int(
        metadata.get("encoder_input_dim"),
        "encoder_input_dim",
        ckpt_path,
    )
    if metadata_input_dim is not None and metadata_input_dim != inferred_input_dim:
        raise ValueError(
            f"Checkpoint encoder_input_dim mismatch for {ckpt_path}: metadata says "
            f"{metadata_input_dim}, but encoder weights require {inferred_input_dim}."
        )


def _attach_checkpoint_metadata(encoder: nn.Module, metadata: Mapping[str, object]) -> nn.Module:
    encoder.checkpoint_metadata = dict(metadata)
    encoder.checkpoint_dataset = metadata.get("dataset")
    encoder.checkpoint_task_type = metadata.get("task_type")
    encoder.checkpoint_backend = metadata.get("backend")
    encoder.checkpoint_exported_at = metadata.get("exported_at")
    encoder.checkpoint_feat_pt_used = metadata.get("feat_pt_used")
    return encoder


def _strip_prefix_if_present(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> OrderedDict[str, torch.Tensor]:
    subset = OrderedDict(
        (key[len(prefix):], value)
        for key, value in state_dict.items()
        if key.startswith(prefix)
    )
    return subset


def _find_required_key(
    state_dict: Mapping[str, torch.Tensor],
    candidates: list[str],
    ckpt_path: Path,
) -> str:
    for key in candidates:
        if key in state_dict:
            return key
    raise ValueError(
        f"Unsupported checkpoint format for {ckpt_path}: missing expected keys {candidates}"
    )


def _infer_graphmae_encoder_kind(
    encoder_state_dict: Mapping[str, torch.Tensor],
    ckpt_path: Path,
) -> str:
    keys = encoder_state_dict.keys()
    if any(key.startswith("gcn_layers.") for key in keys):
        return "gcn"
    if any(key.startswith("layers.") for key in keys):
        return "gin"
    if any(key.startswith("gat_layers.") for key in keys):
        if any(".attn_l" in key or ".attn_r" in key for key in keys):
            return "gat"
        raise ValueError(
            "Unsupported GraphMAE checkpoint format for "
            f"{ckpt_path}: dotgat-style checkpoints are not supported by this scaffold."
        )
    raise ValueError(
        f"Unsupported GraphMAE checkpoint format for {ckpt_path}: no encoder keys found."
    )


def _infer_graphmae_num_layers(
    encoder_state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> int:
    indices = {
        int(key.split(".", 2)[1])
        for key in encoder_state_dict
        if key.startswith(prefix)
    }
    if not indices:
        raise ValueError(f"Unable to infer GraphMAE num_layers from prefix={prefix!r}")
    return max(indices) + 1


def _infer_graphmae_norm(encoder_state_dict: Mapping[str, torch.Tensor]) -> str | None:
    norm_keys = [key for key in encoder_state_dict if ".norm." in key]
    if not norm_keys:
        return None
    if any(key.endswith("mean_scale") for key in norm_keys):
        return "graphnorm"
    if any(
        key.endswith("running_mean")
        or key.endswith("running_var")
        or key.endswith("num_batches_tracked")
        for key in norm_keys
    ):
        return "batchnorm"
    return "layernorm"


def _infer_graphmae_activation(encoder_state_dict: Mapping[str, torch.Tensor]) -> str:
    if any(
        key.endswith("activation.weight") or key.endswith("act.weight")
        for key in encoder_state_dict
    ):
        return "prelu"
    return "relu"


def _infer_graphmae_residual(encoder_state_dict: Mapping[str, torch.Tensor]) -> bool:
    return any(key.endswith("res_fc.weight") for key in encoder_state_dict)


def _load_graphmae_encoder(ckpt_path: Path, device: str) -> nn.Module:
    _ensure_graphmae_imports()
    from graphmae.models.edcoder import setup_module

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    metadata = _extract_checkpoint_metadata(payload)
    state_dict = _extract_state_dict(payload, ckpt_path)

    encoder_state_dict = _strip_prefix_if_present(state_dict, "encoder.")
    if not encoder_state_dict:
        encoder_state_dict = OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if key.startswith(("gcn_layers.", "gat_layers.", "layers."))
        )
    if not encoder_state_dict:
        raise ValueError(
            "Unsupported GraphMAE checkpoint format for "
            f"{ckpt_path}: expected encoder.* keys or a bare encoder state_dict."
        )

    encoder_kind = _infer_graphmae_encoder_kind(encoder_state_dict, ckpt_path)
    activation = _infer_graphmae_activation(encoder_state_dict)
    norm = _infer_graphmae_norm(encoder_state_dict)
    residual = _infer_graphmae_residual(encoder_state_dict)

    if encoder_kind == "gat":
        weight_key = _find_required_key(
            encoder_state_dict,
            ["gat_layers.0.fc.weight", "gat_layers.0.fc_src.weight"],
            ckpt_path,
        )
        first_weight = encoder_state_dict[weight_key]
        in_dim = int(first_weight.shape[1])
        hidden_dim = int(first_weight.shape[0])
        nhead = int(encoder_state_dict["gat_layers.0.attn_l"].shape[1])
        if hidden_dim % nhead != 0:
            raise ValueError(
                "Unsupported GraphMAE checkpoint format for "
                f"{ckpt_path}: hidden_dim={hidden_dim} is not divisible by nhead={nhead}."
            )
        num_layers = _infer_graphmae_num_layers(encoder_state_dict, "gat_layers.")
        per_head_hidden = hidden_dim // nhead
        encoder = setup_module(
            m_type="gat",
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=per_head_hidden,
            out_dim=per_head_hidden,
            num_layers=num_layers,
            dropout=0.0,
            activation=activation,
            residual=residual,
            norm=norm,
            nhead=nhead,
            nhead_out=nhead,
            attn_drop=0.0,
            negative_slope=0.2,
            concat_out=True,
        )
    elif encoder_kind == "gcn":
        first_weight = encoder_state_dict["gcn_layers.0.fc.weight"]
        in_dim = int(first_weight.shape[1])
        hidden_dim = int(first_weight.shape[0])
        num_layers = _infer_graphmae_num_layers(encoder_state_dict, "gcn_layers.")
        encoder = setup_module(
            m_type="gcn",
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
            activation=activation,
            residual=residual,
            norm=norm,
            nhead=1,
            nhead_out=1,
            attn_drop=0.0,
            negative_slope=0.2,
            concat_out=True,
        )
    else:
        weight_key = _find_required_key(
            encoder_state_dict,
            ["layers.0.apply_func.mlp.linears.0.weight", "layers.0.apply_func.mlp.linear.weight"],
            ckpt_path,
        )
        first_weight = encoder_state_dict[weight_key]
        in_dim = int(first_weight.shape[1])
        hidden_dim = int(first_weight.shape[0])
        num_layers = _infer_graphmae_num_layers(encoder_state_dict, "layers.")
        encoder = setup_module(
            m_type="gin",
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
            activation=activation,
            residual=residual,
            norm=norm,
            nhead=1,
            nhead_out=1,
            attn_drop=0.0,
            negative_slope=0.2,
            concat_out=True,
        )

    try:
        encoder.load_state_dict(encoder_state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Failed to load GraphMAE encoder from {ckpt_path}: {exc}"
        ) from exc

    _validate_checkpoint_metadata(
        metadata,
        ckpt_path=ckpt_path,
        expected_model_name="graphmae",
        inferred_input_dim=in_dim,
        expected_backend="dgl",
    )
    encoder.input_dim = in_dim
    encoder.hidden_dim = hidden_dim
    encoder.backend = "dgl"
    encoder.model_name = "graphmae"
    _attach_checkpoint_metadata(encoder, metadata)
    return _freeze_encoder(encoder, device)


def _extract_bgrl_encoder_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    ckpt_path: Path,
) -> OrderedDict[str, torch.Tensor]:
    student_state = _strip_prefix_if_present(state_dict, "student_encoder.")
    if student_state:
        return student_state

    teacher_state = _strip_prefix_if_present(state_dict, "teacher_encoder.")
    if teacher_state:
        return teacher_state

    bare_state = OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if key.startswith(("conv1.", "bn1.", "prelu1.", "conv2.", "bn2.", "prelu2."))
    )
    if bare_state:
        return bare_state

    raise ValueError(
        "Unsupported BGRL checkpoint format for "
        f"{ckpt_path}: expected student_encoder.*, teacher_encoder.*, or a bare encoder state_dict."
    )


def _load_bgrl_encoder(ckpt_path: Path, device: str) -> nn.Module:
    bgrl_models = _load_bgrl_models_module()

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    metadata = _extract_checkpoint_metadata(payload)
    state_dict = _extract_state_dict(payload, ckpt_path)
    encoder_state_dict = _extract_bgrl_encoder_state_dict(state_dict, ckpt_path)

    conv1_weight_key = _find_required_key(
        encoder_state_dict,
        ["conv1.lin.weight", "conv1.weight"],
        ckpt_path,
    )
    conv2_weight_key = _find_required_key(
        encoder_state_dict,
        ["conv2.lin.weight", "conv2.weight"],
        ckpt_path,
    )
    conv1_weight = encoder_state_dict[conv1_weight_key]
    conv2_weight = encoder_state_dict[conv2_weight_key]

    input_dim = int(conv1_weight.shape[1])
    hidden_dim = int(conv1_weight.shape[0])
    output_dim = int(conv2_weight.shape[0])

    encoder = bgrl_models.Encoder(layer_config=[input_dim, hidden_dim, output_dim])
    try:
        encoder.load_state_dict(encoder_state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Failed to load BGRL encoder from {ckpt_path}: {exc}") from exc

    _validate_checkpoint_metadata(
        metadata,
        ckpt_path=ckpt_path,
        expected_model_name="bgrl",
        inferred_input_dim=input_dim,
        expected_backend="pyg",
    )
    encoder.input_dim = input_dim
    encoder.hidden_dim = output_dim
    encoder.backend = "pyg"
    encoder.model_name = "bgrl"
    _attach_checkpoint_metadata(encoder, metadata)
    return _freeze_encoder(encoder, device)


def load_encoder(model_name: str, ckpt_path: str | Path, device: str) -> nn.Module:
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_name = model_name.lower()
    if model_name == "graphmae":
        return _load_graphmae_encoder(ckpt_path, device)
    if model_name == "bgrl":
        return _load_bgrl_encoder(ckpt_path, device)

    raise ValueError(f"Unsupported model: {model_name}. Expected one of ['graphmae', 'bgrl'].")
