
from collections import namedtuple, Counter
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
    CoraGraphDataset, 
    CiteseerGraphDataset, 
    PubmedGraphDataset
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import DglGraphPropPredDataset

from sklearn.preprocessing import StandardScaler


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}


def _read_wn18rr_entity2id(path: Path):
    entity2id = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            entity = " ".join(parts[:-1])
            idx = int(parts[-1])
            entity2id[entity] = idx
    return entity2id


def _read_wn18rr_split_edges(path: Path, entity2id: dict, rel2id: dict):
    src, dst, rel = [], [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            h_id = entity2id[h]
            t_id = entity2id[t]
            r_id = rel2id.setdefault(r, len(rel2id))
            src.append(h_id)
            dst.append(t_id)
            rel.append(r_id)
    return src, dst, rel


def _load_wn18rr_graph(data_root="data"):
    module_root = Path(__file__).resolve().parents[4]  # <workspace>/
    data_root_candidates = [
        Path(data_root),
        module_root / "data",
    ]

    base_candidates = []
    for root in data_root_candidates:
        base_candidates.extend([
            root / "WN18RR",
            root / "pyg" / "WN18RR",
        ])
    base = None
    for candidate in base_candidates:
        if (candidate / "entity2id.txt").exists() and (candidate / "raw").exists():
            base = candidate
            break
    if base is None:
        raise FileNotFoundError(
            "Could not find local WN18RR files. Expected one of: "
            "data/WN18RR or data/pyg/WN18RR with entity2id.txt and raw/*.txt"
        )

    entity2id = _read_wn18rr_entity2id(base / "entity2id.txt")
    rel2id = {}

    train_src, train_dst, train_rel = _read_wn18rr_split_edges(base / "raw" / "train.txt", entity2id, rel2id)
    valid_src, valid_dst, valid_rel = _read_wn18rr_split_edges(base / "raw" / "valid.txt", entity2id, rel2id)
    test_src, test_dst, test_rel = _read_wn18rr_split_edges(base / "raw" / "test.txt", entity2id, rel2id)

    src = train_src + valid_src + test_src
    dst = train_dst + valid_dst + test_dst
    rel = train_rel + valid_rel + test_rel

    num_nodes = len(entity2id)
    graph = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=num_nodes)
    graph.edata["etype"] = torch.tensor(rel, dtype=torch.long)

    # Node masks (for GraphMAE's node-classification evaluation path).
    # WN18RR is a link-prediction dataset; these node masks are smoke-only.
    train_nodes = torch.zeros(num_nodes, dtype=torch.bool)
    val_nodes = torch.zeros(num_nodes, dtype=torch.bool)
    test_nodes = torch.zeros(num_nodes, dtype=torch.bool)

    if train_src or train_dst:
        train_nodes[torch.unique(torch.tensor(train_src + train_dst, dtype=torch.long))] = True
    if valid_src or valid_dst:
        val_nodes[torch.unique(torch.tensor(valid_src + valid_dst, dtype=torch.long))] = True
    if test_src or test_dst:
        test_nodes[torch.unique(torch.tensor(test_src + test_dst, dtype=torch.long))] = True

    if not torch.any(val_nodes):
        val_nodes = train_nodes.clone()
    if not torch.any(test_nodes):
        test_nodes = train_nodes.clone()

    graph.ndata["train_mask"] = train_nodes
    graph.ndata["val_mask"] = val_nodes
    graph.ndata["test_mask"] = test_nodes

    # Placeholder single-class labels for compatibility with GraphMAE evaluation.
    graph.ndata["label"] = torch.zeros(num_nodes, dtype=torch.long)

    # Default placeholder features; will be replaced by --feat-pt in this workflow.
    graph.ndata["feat"] = torch.ones(num_nodes, 1, dtype=torch.float32)

    num_classes = 1
    return graph, (graph.ndata["feat"].shape[1], num_classes)


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name):
    if dataset_name == "wn18rr":
        graph, (num_features, num_classes) = _load_wn18rr_graph("data")
        graph = graph.remove_self_loop().add_self_loop()
        graph.create_formats_()
        return graph, (num_features, num_classes)

    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 2
        # define loss function
        # create the dataset
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]
        
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



def load_graph_classification_dataset(dataset_name, deg4feat=False):
    if dataset_name.lower() == "ogbg-molpcba":
        dataset = DglGraphPropPredDataset(name="ogbg-molpcba", root="dataset")

        graph_label_list = []
        feature_dim = None
        num_classes = None

        for graph_idx in range(len(dataset)):
            dgl_graph, label = dataset[graph_idx]
            dgl_graph = dgl_graph.remove_self_loop().add_self_loop()

            if "feat" in dgl_graph.ndata:
                node_attr = dgl_graph.ndata["feat"].float()
            else:
                node_attr = torch.ones((dgl_graph.num_nodes(), 1), dtype=torch.float32)
            dgl_graph.ndata["attr"] = node_attr

            label = torch.as_tensor(label).float().view(1, -1)
            if num_classes is None:
                num_classes = int(label.shape[1])
            if feature_dim is None:
                feature_dim = int(node_attr.shape[1])

            # Preserve original OGB graph index for external x_graph lookup.
            graph_label_list.append((dgl_graph, label, graph_idx))

        print(
            f"******** # Num Graphs: {len(graph_label_list)}, "
            f"# Num Feat: {feature_dim}, # Num Classes: {num_classes} ********"
        )
        return graph_label_list, (feature_dim, num_classes)

    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
            
            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)
