import logging
import contextlib
import io
import numpy as np
import re
import sys
from tqdm import tqdm
import torch

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_dataset
from graphmae.evaluation import node_classification_evaluation
from graphmae.models import build_model


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


_EPOCH_METRIC_PATTERN = re.compile(
    r"TestAcc:\s*([\d.]+),\s*early-stopping-TestAcc:\s*([\d.]+),\s*Best ValAcc:\s*([\d.]+)\s*in epoch\s+(\d+)"
)


class _EpochMetricTee(io.TextIOBase):
    def __init__(self, original_stream, tracker: dict[str, float | int]):
        self.original_stream = original_stream
        self.tracker = tracker
        self._buf = ""

    def write(self, s):
        self.original_stream.write(s)
        if not isinstance(s, str):
            return len(s)

        self._buf = (self._buf + s)[-4096:]
        self._parse_line(self._buf)
        return len(s)

    def flush(self):
        self.original_stream.flush()

    def _parse_line(self, line: str):
        for match in _EPOCH_METRIC_PATTERN.finditer(line):
            # Groups: (1) test_acc, (2) estp_test_acc, (3) best_val_acc, (4) best_epoch
            # The summary line already encodes the best-by-val values directly.
            estp_test_acc = float(match.group(2))
            best_val_acc = float(match.group(3))
            best_epoch = int(match.group(4))

            self.tracker["best_val"] = best_val_acc
            self.tracker["best_test_at_best_val"] = estp_test_acc
            self.tracker["best_epoch"] = best_epoch


def _load_external_features(graph, feat_pt_path: str, dataset_name: str):
    """
    Load external node features from a .pt file and replace graph.ndata["feat"].

    Expects a dict with key "x": Tensor [num_nodes, dim], floating-point.
    Validates node count, dtype (casts to float32), and finite values.
    Returns (graph, new_num_features).

    NOTE: scale_feats() is NOT re-applied to the replacement features.
    SBERT embeddings are already L2-normalised and must NOT be StandardScaled.
    """
    print(f"  [feat_pt] Loading external features: {feat_pt_path!r}")
    payload = torch.load(feat_pt_path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise TypeError(
            f"[feat_pt] Expected a dict, got {type(payload).__name__}. "
            "Use the schema from preprocess_sbert_features.py."
        )
    if "x" not in payload:
        raise KeyError(
            f"[feat_pt] Key 'x' not found in dict.  Keys present: {list(payload.keys())}"
        )

    x_ext = payload["x"]
    encoder = payload.get("encoder", "unknown")
    dim     = payload.get("dim",     "unknown")
    notes   = payload.get("notes",   "")
    print(f"  [feat_pt] dict schema — encoder={encoder}, dim={dim}, notes={notes!r}")

    # Validation
    if x_ext.dim() != 2:
        raise ValueError(f"[feat_pt] Expected 2-D tensor, got {x_ext.dim()}-D.")
    expected_nodes = graph.num_nodes()
    if x_ext.shape[0] != expected_nodes:
        raise ValueError(
            f"[feat_pt] Node count mismatch: feat_pt has {x_ext.shape[0]:,} rows, "
            f"graph has {expected_nodes:,} nodes."
        )
    if not x_ext.is_floating_point():
        raise TypeError(f"[feat_pt] Expected floating-point tensor, got {x_ext.dtype}.")
    if not torch.isfinite(x_ext).all():
        raise ValueError("[feat_pt] Feature tensor contains NaN or Inf values.")

    if x_ext.dtype != torch.float32:
        x_ext = x_ext.float()
        print("  [feat_pt] Cast to float32.")

    orig_shape = tuple(graph.ndata["feat"].shape)
    new_shape  = tuple(x_ext.shape)
    print(
        f"  [feat_pt] Replacing graph.ndata['feat'] for '{dataset_name}': "
        f"original shape={orig_shape} → new shape={new_shape}, dtype={x_ext.dtype}"
    )
    graph.ndata["feat"] = x_ext
    return graph, x_ext.shape[1]


def _clone_module_state_dict(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _restore_encoder_state(model, encoder_state):
    if hasattr(model, "encoder") and hasattr(model.encoder, "head"):
        model.encoder.head = torch.nn.Identity()
        model.encoder.load_state_dict(encoder_state)


def _debug_check_encoder_width(model, graph, feat, num_classes, where: str):
    expected_width = getattr(model, "output_hidden_dim", None)
    model_device = next(model.parameters()).device

    with torch.no_grad():
        enc_feat = model.embed(graph.to(model_device), feat.to(model_device))

    if enc_feat.dim() != 2:
        raise RuntimeError(f"[encoder-width] expected 2-D encoder features at {where}")

    width = enc_feat.shape[1]
    print(f"[debug] {where}: encoder feature width={width}")

    if expected_width is not None and width != expected_width:
        if width == num_classes:
            raise RuntimeError(
                f"[encoder-width] got class-width {width} where hidden-width {expected_width} was expected at {where}"
            )
        raise RuntimeError(
            f"[encoder-width] expected hidden-width {expected_width}, got {width} at {where}"
        )

    return width


def _run_node_classification_eval_preserve_encoder(
    model,
    graph,
    feat,
    num_classes,
    lr_f,
    weight_decay_f,
    max_epoch_f,
    device,
    linear_prob,
    mute=False,
    where="node_classification_evaluation",
):
    encoder_state = _clone_module_state_dict(model.encoder)
    try:
        return node_classification_evaluation(
            model,
            graph,
            feat,
            num_classes,
            lr_f,
            weight_decay_f,
            max_epoch_f,
            device,
            linear_prob,
            mute=mute,
        )
    finally:
        _restore_encoder_state(model, encoder_state)
        _debug_check_encoder_width(model, graph, feat, num_classes, where)


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 200 == 0:
            _run_node_classification_eval_preserve_encoder(
                model,
                graph,
                x,
                num_classes,
                lr_f,
                weight_decay_f,
                max_epoch_f,
                device,
                linear_prob,
                mute=True,
                where=f"pretrain-epoch-{epoch + 1}",
            )

    # return best_model
    _debug_check_encoder_width(model, graph, x, num_classes, "pretrain-return")
    return model


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer 
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_dataset(dataset_name)

    # Optional external feature replacement (--feat-pt).
    # Must happen AFTER load_dataset() (so the graph structure is ready) and
    # BEFORE args.num_features is set (so build_model() sees the correct dim).
    if getattr(args, "feat_pt", None) is not None:
        graph, num_features = _load_external_features(graph, args.feat_pt, dataset_name)

    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        _debug_check_encoder_width(model, graph, x, num_classes, "main-pre-export")

        # Export encoder-only checkpoint for eval/run_lp.py (Layer 2 contract).
        if getattr(args, "export_encoder_ckpt", None) is not None:
            from pathlib import Path as _Path
            from collections import OrderedDict as _OD

            _repo_root = str(_Path(__file__).resolve().parents[2])
            if _repo_root not in sys.path:
                sys.path.insert(0, _repo_root)
            from eval.checkpoint import export_encoder_checkpoint

            _enc_prefix = "encoder."
            _enc_sd = _OD(
                (k[len(_enc_prefix):], v)
                for k, v in model.state_dict().items()
                if k.startswith(_enc_prefix)
            )
            _hidden = num_hidden * args.num_heads if encoder_type == "gat" else num_hidden
            export_encoder_checkpoint(
                _enc_sd,
                args.export_encoder_ckpt,
                model_name="graphmae",
                dataset=dataset_name,
                task_type="node",
                hidden_dim=_hidden,
                encoder_input_dim=num_features,
                backend="dgl",
                extra_metadata={"feat_pt_used": getattr(args, "feat_pt", None) is not None},
            )

        model = model.to(device)
        model.eval()

        best_tracker: dict[str, float | int] = {
            "best_val": -1.0,
            "best_test_at_best_val": -1.0,
            "best_epoch": -1,
            "final_val": -1.0,
            "final_test": -1.0,
            "final_epoch": -1,
        }

        tee_out = _EpochMetricTee(sys.stdout, best_tracker)
        tee_err = _EpochMetricTee(sys.stderr, best_tracker)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            final_acc, estp_acc = _run_node_classification_eval_preserve_encoder(
                model,
                graph,
                x,
                num_classes,
                lr_f,
                weight_decay_f,
                max_epoch_f,
                device,
                linear_prob,
                where="final-node-classification-eval",
            )

        if best_tracker["best_epoch"] >= 0:
            print(
                "[best] best_val={best_val:.6f} "
                "best_test={best_test:.6f} best_epoch={best_epoch}".format(
                    best_val=float(best_tracker["best_val"]),
                    best_test=float(best_tracker["best_test_at_best_val"]),
                    best_epoch=int(best_tracker["best_epoch"]),
                )
            )
        if best_tracker["final_epoch"] >= 0:
            print(
                "[final] final_val_acc={final_val:.6f} "
                "final_test_acc={final_test:.6f} final_epoch={final_epoch}".format(
                    final_val=float(best_tracker["final_val"]),
                    final_test=float(best_tracker["final_test"]),
                    final_epoch=int(best_tracker["final_epoch"]),
                )
            )

        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
