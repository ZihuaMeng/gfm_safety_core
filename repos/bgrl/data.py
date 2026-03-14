import contextlib
import os

import torch
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected

import os.path as osp

import utils


def download_pyg_data(config):
    """
    Downloads a dataset from the PyTorch Geometric library
    :param config: A dict containing info on the dataset to be downloaded
    :return: A tuple containing (root directory, dataset name, data directory)
    """
    leaf_dir = config["kwargs"]["root"].split("/")[-1].strip()
    data_dir = osp.join(config["kwargs"]["root"], "" if config["name"] == leaf_dir else config["name"])
    dst_path = osp.join(data_dir, "raw", "data.pt")
    if not osp.exists(dst_path):
        DatasetClass = config["class"]
        if config["name"] == "WikiCS":
            dataset = DatasetClass(data_dir, transform=T.NormalizeFeatures())
            std, mean = torch.std_mean(dataset.data.x, dim=0, unbiased=False)
            dataset.data.x = (dataset.data.x - mean) / std
            dataset.data.edge_index = to_undirected(dataset.data.edge_index)
            utils.create_masks(data=dataset.data)
        elif config["name"] == "WN18RR":
            # WN18RR has no node features (x) or node labels (y).
            # Skip NormalizeFeatures (nothing to normalise) and create_masks (no y).
            dataset = DatasetClass(**config["kwargs"])
            graph_data = dataset.data

            # BGRL compatibility: ensure key fields exist even though WN18RR is
            # a link-prediction dataset and node-classification eval is skipped.
            if graph_data.x is None:
                graph_data.x = torch.ones((graph_data.num_nodes, 1), dtype=torch.float32)
            if graph_data.edge_attr is None:
                graph_data.edge_attr = torch.ones(graph_data.edge_index.shape[1], dtype=torch.float32)
            if graph_data.y is None:
                graph_data.y = torch.zeros(graph_data.num_nodes, dtype=torch.long)

            # PyG WN18RR provides edge-level split masks; create smoke-only
            # node masks so downstream code sees the expected attributes.
            all_nodes = torch.arange(graph_data.num_nodes, dtype=torch.long)
            node_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
            node_mask[all_nodes] = True
            graph_data.train_mask = node_mask.unsqueeze(0).expand(20, -1).clone()
            graph_data.val_mask = node_mask.unsqueeze(0).expand(20, -1).clone()
            graph_data.test_mask = node_mask.unsqueeze(0).expand(20, -1).clone()

            print(f"  [WN18RR] Loaded: num_nodes={dataset.data.num_nodes:,}, "
                  f"num_edges={dataset.data.edge_index.shape[1]:,}")
            has_et = hasattr(dataset.data, "edge_type")
            print(f"  [WN18RR] edge_type present: {has_et}"
                  + (f", shape={tuple(dataset.data.edge_type.shape)}" if has_et else ""))
            # No create_masks: link-prediction dataset, no node-classification labels.
        else:
            dataset = DatasetClass(**config["kwargs"], transform=T.NormalizeFeatures())
            utils.create_masks(data=dataset.data)
        torch.save((dataset.data, dataset.slices), dst_path)
    
    return config["kwargs"]["root"], config["name"], data_dir


def _is_weights_only_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "weights_only" in msg
        or "weightsunpickler" in msg
        or "weights only" in msg
    )


@contextlib.contextmanager
def _torch_load_weights_only_false():
    """
    Temporarily patch torch.load to default weights_only=False, scoped to
    the body of this context manager only.  The original torch.load is
    restored in the finally block regardless of exceptions.

    Use ONLY for trusted local OGB processed-cache files — do NOT copy
    this pattern into training or inference code blindly.
    """
    _orig = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _patched
    try:
        yield
    finally:
        torch.load = _orig


def download_ogb_data(config):
    """
    Prepare an OGB node-property-prediction dataset for use with BGRL's Dataset class.

    Downloads via OGB's PygNodePropPredDataset (which stores files under
    <root>/ogbn_arxiv/), then normalises the Data object and writes a BGRL-
    compatible raw cache to <data_dir>/raw/data.pt, where
        data_dir = <root>/bgrl_<name>   (e.g. data/ogb/bgrl_ogbn-arxiv/)

    Keeping data_dir separate from OGB's native directory avoids file-layout
    conflicts between OGB's own raw/processed structure and BGRL's.

    Normalisation applied:
      - data.y squeezed from [N, 1] → [N]  (OGB stores labels as 2-D)
      - data.edge_attr set to ones([E]) if absent  (BGRL augmentation expects it)
      - train_mask / val_mask / test_mask created from OGB's official split,
        replicated 20 times along dim-0 to match BGRL's evaluate() loop shape
        [20, N]

    The saved tuple (data, None) is the format expected by Dataset.process().
    """
    root = config["kwargs"]["root"]       # e.g. "data/ogb"
    name = config["kwargs"]["name"]       # e.g. "ogbn-arxiv"

    # BGRL's own cache dir — separate from OGB's <root>/ogbn_arxiv/
    data_dir = osp.join(root, "bgrl_" + name)
    dst_path = osp.join(data_dir, "raw", "data.pt")

    if not osp.exists(dst_path):
        print(f"  [OGB] Building BGRL raw cache for {name!r} → {dst_path!r} ...")
        from ogb.nodeproppred import PygNodePropPredDataset  # lazy import
        try:
            dataset = PygNodePropPredDataset(name=name, root=root)
            graph_data = dataset[0]
        except Exception as exc:
            if not _is_weights_only_error(exc):
                raise
            print(
                "[WARN] OGB PyG cache load failed under PyTorch ≥2.6 "
                "weights_only=True default.\n"
                "       Retrying with weights_only=False scoped to this "
                "trusted local OGB cache only.\n"
                "       Do NOT copy this workaround into training code."
            )
            with _torch_load_weights_only_false():
                dataset = PygNodePropPredDataset(name=name, root=root)
                graph_data = dataset[0]

        # OGB labels: [N, 1] → [N]
        if graph_data.y is not None and graph_data.y.dim() == 2:
            graph_data.y = graph_data.y.squeeze(1)

        # ogbn-arxiv has no edge attributes; BGRL augmentation requires them
        if graph_data.edge_attr is None:
            graph_data.edge_attr = torch.ones(graph_data.edge_index.shape[1])

        # Convert OGB split indices to boolean masks of shape [20, N].
        # BGRL's evaluate() iterates 20 splits (see train.py); we replicate
        # OGB's official single split 20 times so the shape contract is met.
        split_idx = dataset.get_idx_split()
        n = graph_data.num_nodes
        for mask_attr, idx_key in [
            ("train_mask", "train"),
            ("val_mask",   "valid"),
            ("test_mask",  "test"),
        ]:
            mask = torch.zeros(n, dtype=torch.bool)
            mask[split_idx[idx_key]] = True
            # .clone() materialises the expanded tensor into contiguous storage
            setattr(graph_data, mask_attr,
                    mask.unsqueeze(0).expand(20, -1).clone())

        utils.create_dirs([osp.join(data_dir, "raw")])
        # Format: (data, _) — Dataset.process() does: data, _ = torch.load(path)
        torch.save((graph_data, None), dst_path)
        print(f"  [OGB] Saved BGRL raw cache ({graph_data.num_nodes:,} nodes).")

    return root, name, data_dir


_PCBA_MAX_TRAIN_GRAPHS = int(os.environ.get("BGRL_PCBA_MAX_GRAPHS", "10000"))


def download_ogb_graph_data(config):
    """Prepare an OGB graph-property-prediction dataset for BGRL's single-graph training.

    BGRL operates on a single large graph.  For graph-level datasets like
    ogbg-molpcba, this function creates a **union graph** by concatenating a
    configurable number of training-split molecular graphs into one
    disconnected graph (block-diagonal adjacency).

    The node features are the native OGB atom features (9-dim for PCBA).
    Edge attributes are replaced with ones (1-D) because BGRL's GCNConv
    expects scalar edge weights, not multi-dimensional bond features.

    Dummy node labels and masks are created so BGRL's Dataset.process()
    contract is satisfied.  BGRL's built-in logreg evaluation is skipped
    for graph datasets (--skip-eval).

    Max molecules: controlled by BGRL_PCBA_MAX_GRAPHS env var (default 10000).
    """
    root = config["kwargs"]["root"]
    name = config["kwargs"]["name"]

    max_graphs = _PCBA_MAX_TRAIN_GRAPHS
    data_dir = osp.join(root, f"bgrl_{name}_n{max_graphs}")
    dst_path = osp.join(data_dir, "raw", "data.pt")

    if not osp.exists(dst_path):
        print(f"  [OGB-graph] Building BGRL union-graph cache for {name!r} "
              f"(max_graphs={max_graphs}) -> {dst_path!r} ...")
        from ogb.graphproppred import PygGraphPropPredDataset

        try:
            dataset = PygGraphPropPredDataset(name=name, root=root)
        except Exception as exc:
            if not _is_weights_only_error(exc):
                raise
            print("[WARN] OGB cache load failed (weights_only); retrying with patch.")
            with _torch_load_weights_only_false():
                dataset = PygGraphPropPredDataset(name=name, root=root)

        split_idx = dataset.get_idx_split()
        train_indices = split_idx["train"]

        # Subsample training molecules if needed
        if len(train_indices) > max_graphs:
            perm = torch.randperm(len(train_indices))[:max_graphs]
            train_indices = train_indices[perm].sort().values

        # Build union graph from selected training molecules
        all_x = []
        all_edge_index = []
        node_offset = 0

        for i, idx in enumerate(train_indices):
            mol = dataset[int(idx)]
            n_nodes = mol.x.shape[0]
            all_x.append(mol.x.float())
            if mol.edge_index is not None and mol.edge_index.numel() > 0:
                all_edge_index.append(mol.edge_index + node_offset)
            node_offset += n_nodes
            if (i + 1) % 2000 == 0:
                print(f"  [OGB-graph]   processed {i + 1}/{len(train_indices)} molecules "
                      f"({node_offset:,} nodes so far)")

        union_x = torch.cat(all_x, dim=0)
        union_edge_index = (
            torch.cat(all_edge_index, dim=1) if all_edge_index
            else torch.zeros((2, 0), dtype=torch.long)
        )
        # 1-D edge weights (BGRL GCNConv contract)
        union_edge_attr = torch.ones(union_edge_index.shape[1], dtype=torch.float32)

        # Dummy labels and masks for BGRL compatibility
        num_nodes = union_x.shape[0]
        union_y = torch.zeros(num_nodes, dtype=torch.long)
        node_mask = torch.ones(num_nodes, dtype=torch.bool)
        masks_2d = node_mask.unsqueeze(0).expand(20, -1).clone()

        graph_data = Data(
            x=union_x,
            edge_index=union_edge_index,
            edge_attr=union_edge_attr,
            y=union_y,
            train_mask=masks_2d,
            val_mask=masks_2d.clone(),
            test_mask=masks_2d.clone(),
            num_nodes=num_nodes,
        )

        utils.create_dirs([osp.join(data_dir, "raw")])
        torch.save((graph_data, None), dst_path)
        print(f"  [OGB-graph] Saved union graph: {num_nodes:,} nodes, "
              f"{union_edge_index.shape[1]:,} edges, feat_dim={union_x.shape[1]}, "
              f"molecules={len(train_indices)}")

    return root, name, data_dir


def download_data(root, name):
    """
    Download data from different repositories.
    Supports PyTorch Geometric datasets (src='pyg'), OGB node datasets (src='ogb'),
    and OGB graph datasets (src='ogb_graph').
    :param root: The root directory of the dataset
    :param name: The name of the dataset
    :return:
    """
    config = utils.decide_config(root=root, name=name)
    if config["src"] == "pyg":
        return download_pyg_data(config)
    elif config["src"] == "ogb":
        return download_ogb_data(config)
    elif config["src"] == "ogb_graph":
        return download_ogb_graph_data(config)


class Dataset(InMemoryDataset):

    """
    A PyTorch InMemoryDataset to build multi-view dataset through graph data augmentation
    """

    def __init__(self, root="data", name='cora', num_parts=1, final_parts=1, augumentation=None, transform=None,
                 pre_transform=None):
        self.num_parts = num_parts
        self.final_parts = final_parts
        self.augumentation = augumentation
        self.root, self.name, self.data_dir = download_data(root=root, name=name)
        utils.create_dirs(self.dirs)
        super().__init__(root=self.data_dir, transform=transform, pre_transform=pre_transform)
        path = osp.join(self.data_dir, "processed", self.processed_file_names[0])
        # weights_only=False: BGRL processed files contain PyG Data objects
        # (non-tensor content); incompatible with PyTorch ≥2.6 default.
        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        return ["data.pt"]

    @property
    def processed_file_names(self):
        if self.num_parts == 1:
            return [f'byg.data.aug.pt']
        else:
            return [f'byg.data.aug.ip.{self.num_parts}.fp.{self.final_parts}.pt']

    @property
    def raw_dir(self):
        return osp.join(self.data_dir, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.data_dir, "processed")

    @property
    def model_dir(self):
        return osp.join(self.data_dir, "model")

    @property
    def result_dir(self):
        return osp.join(self.data_dir, "result")

    @property
    def dirs(self):
        return [self.raw_dir, self.processed_dir, self.model_dir, self.result_dir]


    def process_full_batch_data(self, data):
        """
        Augmented view data generation using the full-batch data.
        :param view1data:
        :return:
        """
        print("Processing full batch data")

        data = Data(edge_index=data.edge_index, edge_attr= data.edge_attr, 
                    x = data.x, y = data.y, 
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                    num_nodes=data.num_nodes)
        return [data]

    def download(self):
        pass

    def process(self):
        """
        Process either a full batch or cluster data.
        :return:
        """
        processed_path = osp.join(self.processed_dir, self.processed_file_names[0])
        if not osp.exists(processed_path):
            path = osp.join(self.raw_dir, self.raw_file_names[0])
            # weights_only=False: raw/data.pt contains PyG Data objects.
            data, _ = torch.load(path, weights_only=False)

            # Backfill compatibility fields (important for cached WN18RR raws
            # created before wn18rr support updates).
            if data.x is None:
                data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)
            if data.y is None:
                data.y = torch.zeros(data.num_nodes, dtype=torch.long)

            edge_attr = data.edge_attr
            edge_attr = torch.ones(data.edge_index.shape[1]) if edge_attr is None else edge_attr
            data.edge_attr = edge_attr

            need_node_masks = (
                (not hasattr(data, "train_mask"))
                or data.train_mask is None
                or data.train_mask.dim() != 2
                or data.train_mask.shape[-1] != data.num_nodes
            )
            if need_node_masks:
                node_mask = torch.ones(data.num_nodes, dtype=torch.bool)
                data.train_mask = node_mask.unsqueeze(0).expand(20, -1).clone()
                data.val_mask = node_mask.unsqueeze(0).expand(20, -1).clone()
                data.test_mask = node_mask.unsqueeze(0).expand(20, -1).clone()

            data_list = self.process_full_batch_data(data)
            data, slices = self.collate(data_list)
            torch.save((data, slices), processed_path)