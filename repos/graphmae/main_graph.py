import logging
import sys
from tqdm import tqdm
import numpy as np
from functools import partial
from pathlib import Path

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_graph_classification_dataset
from graphmae.models import build_model

GRAPHMAE_ROOT = Path(__file__).resolve().parent
GRAPHMAE_DATASET_ROOT = GRAPHMAE_ROOT / "dataset"


def _extract_eval_mode_from_argv(default="svm"):
    choices = {"svm", "ogb", "none"}
    argv = sys.argv
    mode = default
    i = 1
    while i < len(argv):
        token = argv[i]
        if token == "--eval":
            if i + 1 >= len(argv):
                raise ValueError("--eval requires a value in {svm, ogb, none}")
            cand = argv[i + 1].strip().lower()
            if cand not in choices:
                raise ValueError(f"Invalid --eval value: {cand!r}. Choose from {sorted(choices)}")
            mode = cand
            del argv[i:i + 2]
            continue
        if token.startswith("--eval="):
            cand = token.split("=", 1)[1].strip().lower()
            if cand not in choices:
                raise ValueError(f"Invalid --eval value: {cand!r}. Choose from {sorted(choices)}")
            mode = cand
            del argv[i]
            continue
        i += 1
    return mode


def _extract_eval_max_graphs_from_argv(default=0):
    argv = sys.argv
    value = int(default)
    i = 1
    while i < len(argv):
        token = argv[i]
        if token == "--eval_max_graphs":
            if i + 1 >= len(argv):
                raise ValueError("--eval_max_graphs requires an integer value")
            cand = int(argv[i + 1])
            if cand < 0:
                raise ValueError("--eval_max_graphs must be >= 0")
            value = cand
            del argv[i:i + 2]
            continue
        if token.startswith("--eval_max_graphs="):
            cand = int(token.split("=", 1)[1])
            if cand < 0:
                raise ValueError("--eval_max_graphs must be >= 0")
            value = cand
            del argv[i]
            continue
        i += 1
    return value


def _extract_bool_flag_from_argv(flag: str, default: bool = False) -> bool:
    argv = sys.argv
    value = bool(default)
    i = 1
    bare_flag = f"--{flag}"
    negated_flag = f"--no-{flag}"
    while i < len(argv):
        token = argv[i]
        if token == bare_flag:
            value = True
            del argv[i]
            continue
        if token == negated_flag:
            value = False
            del argv[i]
            continue
        i += 1
    return value


def _extract_int_flag_from_argv(flag: str, default: int, *, minimum: int | None = None) -> int:
    argv = sys.argv
    value = int(default)
    i = 1
    bare_flag = f"--{flag}"
    while i < len(argv):
        token = argv[i]
        if token == bare_flag:
            if i + 1 >= len(argv):
                raise ValueError(f"{bare_flag} requires an integer value")
            cand = int(argv[i + 1])
            if minimum is not None and cand < minimum:
                raise ValueError(f"{bare_flag} must be >= {minimum}")
            value = cand
            del argv[i:i + 2]
            continue
        if token.startswith(f"{bare_flag}="):
            cand = int(token.split("=", 1)[1])
            if minimum is not None and cand < minimum:
                raise ValueError(f"{bare_flag} must be >= {minimum}")
            value = cand
            del argv[i]
            continue
        i += 1
    return value


def _resolve_script_relative_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((GRAPHMAE_ROOT / path).resolve())


class _LazyPCBADatasetView:
    def __init__(self, dataset, indices):
        self._dataset = dataset
        self._indices = [int(idx) for idx in indices]
        self.full_dataset_size = len(dataset)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        graph_idx = self._indices[idx]
        dgl_graph, label = self._dataset[graph_idx]
        dgl_graph = dgl_graph.remove_self_loop().add_self_loop()
        if "feat" in dgl_graph.ndata:
            node_attr = dgl_graph.ndata["feat"].float()
        else:
            node_attr = torch.ones((dgl_graph.num_nodes(), 1), dtype=torch.float32)
        dgl_graph.ndata["attr"] = node_attr
        return dgl_graph, torch.as_tensor(label).float().view(1, -1), graph_idx


def _load_pcba_debug_dataset(debug_max_graphs: int):
    if debug_max_graphs <= 0:
        raise ValueError("--debug_max_graphs must be > 0 for ogbg-molpcba debug mode.")

    from ogb.graphproppred import DglGraphPropPredDataset

    dataset = DglGraphPropPredDataset(
        name="ogbg-molpcba",
        root=str(GRAPHMAE_DATASET_ROOT),
    )
    split_idx = dataset.get_idx_split()
    train_idx = torch.as_tensor(split_idx["train"], dtype=torch.long)
    subset_idx = train_idx[: min(debug_max_graphs, train_idx.numel())]
    if subset_idx.numel() == 0:
        raise ValueError("ogbg-molpcba train split is empty; cannot build debug subset.")

    sample_graph, sample_label = dataset[int(subset_idx[0])]
    feature_dim = int(sample_graph.ndata["feat"].shape[1]) if "feat" in sample_graph.ndata else 1
    num_classes = int(torch.as_tensor(sample_label).view(1, -1).shape[1])

    print(
        f"[debug] ogbg-molpcba local subset enabled: "
        f"source_split=train, graphs={subset_idx.numel()}, "
        f"feature_dim={feature_dim}, num_classes={num_classes}"
    )
    return _LazyPCBADatasetView(dataset, subset_idx), (feature_dim, num_classes)


def _load_graph_dataset(args):
    dataset_name = args.dataset.lower()
    if dataset_name == "ogbg-molpcba" and getattr(args, "debug", False):
        return _load_pcba_debug_dataset(args.debug_max_graphs)
    return load_graph_classification_dataset(args.dataset, deg4feat=args.deg4feat)


def _valid_svm_labels(labels):
    y = np.asarray(labels)
    if y.ndim != 1:
        return False
    if y.size == 0 or not np.isfinite(y).all():
        return False
    if np.unique(y).size < 2:
        return False
    return True


def _compute_ogb_rocauc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 2:
        raise ValueError("y_true must be 2D for OGB ROC-AUC evaluation")
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred must have the same shape as y_true")

    rocauc_list = []
    tasks_used = 0
    num_tasks = y_true.shape[1]
    for task_id in range(num_tasks):
        y_t = y_true[:, task_id]
        y_p = y_pred[:, task_id]
        finite_mask = np.isfinite(y_t)
        if finite_mask.sum() == 0:
            continue
        y_t = y_t[finite_mask]
        y_p = y_p[finite_mask]
        if np.unique(y_t).size < 2:
            continue
        rocauc_list.append(roc_auc_score(y_t, y_p))
        tasks_used += 1

    if tasks_used == 0:
        return np.nan, 0, num_tasks
    return float(np.mean(rocauc_list)), tasks_used, num_tasks


def evaluate_graph_embeddings_using_ogb_rocauc(embeddings, labels, eval_max_graphs=0):
    from ogb.graphproppred import DglGraphPropPredDataset

    y = np.asarray(labels)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        print("[eval] skipped: invalid labels for ogb")
        return np.nan

    finite = np.isfinite(y)
    finite_vals = y[finite]
    if finite_vals.size == 0 or np.unique(finite_vals).size < 2:
        print("[eval] skipped: invalid labels for ogb")
        return np.nan

    dataset = DglGraphPropPredDataset(name="ogbg-molpcba", root="dataset")
    split_idx = dataset.get_idx_split()
    train_idx = np.asarray(split_idx["train"])
    valid_idx = np.asarray(split_idx["valid"])
    test_idx = np.asarray(split_idx["test"])
    if eval_max_graphs > 0:
        train_idx = train_idx[:eval_max_graphs]
        valid_idx = valid_idx[:eval_max_graphs]
        test_idx = test_idx[:eval_max_graphs]
    print(
        f"[eval][ogb] using subset: train={len(train_idx)} valid={len(valid_idx)} "
        f"test={len(test_idx)} (eval_max_graphs={eval_max_graphs})"
    )

    n, t = y.shape
    if n != len(dataset):
        print("[eval] skipped: invalid labels for ogb")
        return np.nan

    valid_pred = np.full((len(valid_idx), t), np.nan, dtype=np.float64)
    test_pred = np.full((len(test_idx), t), np.nan, dtype=np.float64)
    trained_task_mask = np.zeros(t, dtype=bool)

    for task_id in range(t):
        y_task = y[:, task_id]

        train_mask = np.isfinite(y_task[train_idx])
        if train_mask.sum() == 0:
            continue
        y_train = y_task[train_idx][train_mask]
        if np.unique(y_train).size < 2:
            continue
        x_train = embeddings[train_idx][train_mask]

        clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=0)
        clf.fit(x_train, y_train.astype(np.int64))
        trained_task_mask[task_id] = True

        valid_mask = np.isfinite(y_task[valid_idx])
        if valid_mask.any():
            valid_pred[valid_mask, task_id] = clf.predict_proba(embeddings[valid_idx][valid_mask])[:, 1]

        test_mask = np.isfinite(y_task[test_idx])
        if test_mask.any():
            test_pred[test_mask, task_id] = clf.predict_proba(embeddings[test_idx][test_mask])[:, 1]

    valid_y = y[valid_idx]
    test_y = y[test_idx]

    # --- OGB AP evaluator (ogbg-molpcba official metric) ---
    from ogb.graphproppred import Evaluator as OGBEvaluator
    evaluator = OGBEvaluator(name="ogbg-molpcba")
    print(f"[eval][ogb][molpcba] trained_tasks={int(trained_task_mask.sum())} / total_tasks={t}")

    def _build_eval_inputs(split, y_true_s, y_pred_s):
        y_true_eval = np.array(y_true_s, copy=True)
        y_pred_eval = np.array(y_pred_s, copy=True)
        skipped_task_mask = ~trained_task_mask
        y_true_eval[:, skipped_task_mask] = np.nan
        y_pred_eval[:, skipped_task_mask] = 0.0
        print(f"[eval][ogb][molpcba] {split}_skipped_tasks={int(skipped_task_mask.sum())}")
        return y_true_eval, y_pred_eval

    def _debug_pred_stats(split, y_true_s, y_pred_s):
        """Print NaN diagnostics for one split; raise RuntimeError if y_pred is non-finite at
        labeled positions (i.e. where y_true is finite), so we fail clearly before sklearn crashes.
        Labeled positions are defined the same way the OGB evaluator uses them: np.isfinite(y_true).
        """
        labeled = np.isfinite(y_true_s)
        nan_yt = int(np.sum(~np.isfinite(y_true_s)))
        nan_yp = int(np.sum(~np.isfinite(y_pred_s)))
        nonfinite_labeled = int(np.sum(~np.isfinite(y_pred_s[labeled])))
        print(
            f"[eval][ogb][molpcba][debug] split={split}  "
            f"y_true.shape={y_true_s.shape}  y_pred.shape={y_pred_s.shape}  "
            f"nan_in_y_true={nan_yt}  nan_in_y_pred={nan_yp}  "
            f"nonfinite_y_pred_at_labeled={nonfinite_labeled}"
        )
        if nonfinite_labeled > 0:
            print(
                f"[eval][ogb][molpcba][debug] ABORT split={split}: "
                f"{nonfinite_labeled} non-finite y_pred values at labeled positions — "
                "Evaluator.eval() would crash inside sklearn; raising early for diagnosis."
            )
            raise RuntimeError(
                f"split={split!r}: {nonfinite_labeled} non-finite y_pred values at labeled "
                "positions (where y_true is finite)"
            )

    valid_y_eval, valid_pred_eval = _build_eval_inputs("valid", valid_y, valid_pred)
    # y_true NaN positions are masked by the evaluator via (y_true == y_true);
    # y_pred NaN positions align with y_true NaN positions (filled only where labels are finite).
    _debug_pred_stats("valid", valid_y_eval, valid_pred_eval)
    try:
        valid_ap = evaluator.eval({"y_true": valid_y_eval, "y_pred": valid_pred_eval})["ap"]
    except RuntimeError as e:
        print(f"[eval][ogb][molpcba] valid AP skipped: {e}")
        valid_ap = np.nan
    test_y_eval, test_pred_eval = _build_eval_inputs("test", test_y, test_pred)
    _debug_pred_stats("test", test_y_eval, test_pred_eval)
    try:
        test_ap = evaluator.eval({"y_true": test_y_eval, "y_pred": test_pred_eval})["ap"]
    except RuntimeError as e:
        print(f"[eval][ogb][molpcba] test AP skipped: {e}")
        test_ap = np.nan

    print(f"[eval][ogb][molpcba] valid_ap={valid_ap:.4f}")
    print(f"[eval][ogb][molpcba] test_ap={test_ap:.4f}")
    if np.isnan(test_ap):
        print("[eval] skipped: invalid labels for ogb")
    return test_ap


def _load_external_graph_features(feat_pt_path: str):
    payload = torch.load(feat_pt_path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise TypeError(f"[feat_pt] Expected dict payload, got {type(payload).__name__}")

    x_graph = payload.get("x_graph", payload.get("x", None))
    if x_graph is None:
        raise KeyError(
            f"[feat_pt] Expected key 'x_graph' (or fallback 'x'). Keys: {list(payload.keys())}"
        )
    if x_graph.dim() != 2:
        raise ValueError(f"[feat_pt] Expected 2-D tensor, got {x_graph.dim()}-D")
    if not x_graph.is_floating_point():
        raise TypeError(f"[feat_pt] Expected floating tensor, got {x_graph.dtype}")
    if not torch.isfinite(x_graph).all():
        raise ValueError("[feat_pt] External graph feature tensor contains NaN or Inf")

    if x_graph.dtype != torch.float32:
        x_graph = x_graph.float()

    print(
        f"  [feat_pt] graph-level schema — encoder={payload.get('encoder', 'unknown')}, "
        f"dim={payload.get('dim', x_graph.shape[1])}, rows={x_graph.shape[0]}"
    )
    return x_graph


def graph_classification_evaluation(model, pooler, dataloader, num_classes, lr_f, weight_decay_f, max_epoch_f, device, eval_mode="svm", eval_max_graphs=0, mute=False):
    if eval_mode == "none":
        print("[eval] skipped (mode=none)")
        return np.nan

    model.eval()
    x_list = []
    y_list = []
    with torch.no_grad():
        for i, (batch_g, labels) in enumerate(dataloader):
            batch_g = batch_g.to(device)
            feat = batch_g.ndata["attr"]
            out = model.embed(batch_g, feat)
            out = pooler(batch_g, out)

            y_list.append(labels.numpy())
            x_list.append(out.cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    if eval_mode == "svm":
        if not _valid_svm_labels(y):
            print("[eval] skipped: invalid labels for svm")
            return np.nan
        test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
        print(f"#Test_f1: {test_f1:.4f}±{test_std:.4f}")
        return test_f1
    if eval_mode == "ogb":
        return evaluate_graph_embeddings_using_ogb_rocauc(x, y, eval_max_graphs=eval_max_graphs)
    raise ValueError(f"Unknown eval mode: {eval_mode}")


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std


def pretrain(model, pooler, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob=True, logger=None):
    train_loader, eval_loader = dataloaders

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        for batch in train_loader:
            batch_g, _ = batch
            batch_g = batch_g.to(device)

            feat = batch_g.ndata["attr"]
            model.train()
            loss, loss_dict = model(batch_g, feat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                logger.note(loss_dict, step=epoch)
        if scheduler is not None:
            scheduler.step()
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")

    return model

            
def collate_fn(batch, x_graph=None):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    graph_ids = [x[2] for x in batch] if len(batch) > 0 and len(batch[0]) >= 3 else None
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)

    # Optional graph-level SBERT broadcast: for each graph i in batch,
    # repeat x_graph[graph_id_i] to all nodes of that graph.
    if x_graph is not None:
        if graph_ids is None:
            raise ValueError("[feat_pt] Dataset items must include graph_idx for x_graph lookup.")
        graph_ids = torch.as_tensor(graph_ids, dtype=torch.long)
        gfeat = x_graph[graph_ids]  # [B, D]
        num_nodes_per_graph = batch_g.batch_num_nodes().to(torch.long)  # [B]
        node_feat = torch.repeat_interleave(gfeat, num_nodes_per_graph, dim=0)  # [N, D]
        batch_g.ndata["attr"] = node_feat

    return batch_g, labels


def main(args):
    args.feat_pt = _resolve_script_relative_path(getattr(args, "feat_pt", None))
    args.export_encoder_ckpt = _resolve_script_relative_path(
        getattr(args, "export_encoder_ckpt", None)
    )

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
    pooling = args.pooling
    deg4feat = args.deg4feat
    batch_size = args.batch_size

    if args.debug and dataset_name.lower() == "ogbg-molpcba" and args.eval != "none":
        raise ValueError(
            "ogbg-molpcba debug mode uses a truncated local subset for fast export; "
            "use --eval none for this local debug path."
        )

    graphs, (num_features, num_classes) = _load_graph_dataset(args)

    x_graph_ext = None
    if getattr(args, "feat_pt", None) is not None:
        print(f"  [feat_pt] Loading external graph-level features: {args.feat_pt!r}")
        x_graph_ext = _load_external_graph_features(args.feat_pt)
        expected_graph_rows = getattr(graphs, "full_dataset_size", len(graphs))
        if x_graph_ext.shape[0] != expected_graph_rows:
            raise ValueError(
                f"[feat_pt] row-count mismatch: x_graph has {x_graph_ext.shape[0]:,} rows, "
                f"dataset has {expected_graph_rows:,} graphs"
            )
        num_features = int(x_graph_ext.shape[1])

    args.num_features = num_features
    loader_num_workers = getattr(args, "num_workers", 0)
    pin_memory = not getattr(args, "debug", False)

    collate = partial(collate_fn, x_graph=x_graph_ext)
    if dataset_name.lower() == "ogbg-molpcba" and getattr(args, "debug", False):
        train_loader = GraphDataLoader(
            graphs,
            collate_fn=collate,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=loader_num_workers,
        )
        eval_loader = GraphDataLoader(
            graphs,
            collate_fn=collate,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=loader_num_workers,
        )
    else:
        train_idx = torch.arange(len(graphs))
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = GraphDataLoader(
            graphs,
            sampler=train_sampler,
            collate_fn=collate,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=loader_num_workers,
        )
        eval_loader = GraphDataLoader(
            graphs,
            collate_fn=collate,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=loader_num_workers,
        )

    if pooling == "mean":
        pooler = AvgPooling()
    elif pooling == "max":
        pooler = MaxPooling()
    elif pooling == "sum":
        pooler = SumPooling()
    else:
        raise NotImplementedError

    acc_list = []
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
            
        if not load_model:
            model = pretrain(model, pooler, (train_loader, eval_loader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob,  logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

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
                task_type="graph",
                hidden_dim=_hidden,
                encoder_input_dim=num_features,
                backend="dgl",
                extra_metadata={"feat_pt_used": getattr(args, "feat_pt", None) is not None},
            )

        model = model.to(device)
        model.eval()
        test_f1 = graph_classification_evaluation(
            model,
            pooler,
            eval_loader,
            num_classes,
            lr_f,
            weight_decay_f,
            max_epoch_f,
            device,
            eval_mode=args.eval,
            eval_max_graphs=args.eval_max_graphs,
            mute=False,
        )
        acc_list.append(test_f1)

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")


if __name__ == "__main__":
    eval_mode = _extract_eval_mode_from_argv(default="svm")
    eval_max_graphs = _extract_eval_max_graphs_from_argv(default=0)
    debug = _extract_bool_flag_from_argv("debug", default=False)
    debug_max_graphs = _extract_int_flag_from_argv("debug_max_graphs", default=64, minimum=0)
    num_workers = _extract_int_flag_from_argv("num_workers", default=0, minimum=0)
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    args.eval = eval_mode
    args.eval_max_graphs = eval_max_graphs
    args.debug = debug
    args.debug_max_graphs = debug_max_graphs
    args.num_workers = num_workers
    print(args)
    main(args)
