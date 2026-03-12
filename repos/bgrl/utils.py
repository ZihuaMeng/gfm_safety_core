from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS, WordNet18RR
from torch_geometric.utils import dropout_adj

import os.path as osp
import os

import argparse

import numpy as np

import torch

"""
The Following code is borrowed from SelfGNN
"""
class Augmentation:

    def __init__(self, p_f1 = 0.2, p_f2 = 0.1, p_e1 = 0.2, p_e2 = 0.3):
        """
        two simple graph augmentation functions --> "Node feature masking" and "Edge masking"
        Random binary node feature mask following Bernoulli distribution with parameter p_f
        Random binary edge mask following Bernoulli distribution with parameter p_e
        """
        self.p_f1 = p_f1
        self.p_f2 = p_f2
        self.p_e1 = p_e1
        self.p_e2 = p_e2
        self.method = "BGRL"
    
    def _feature_masking(self, data, device):
        feat_mask1 = torch.FloatTensor(data.x.shape[1]).uniform_() > self.p_f1
        feat_mask2 = torch.FloatTensor(data.x.shape[1]).uniform_() > self.p_f2
        feat_mask1, feat_mask2 = feat_mask1.to(device), feat_mask2.to(device)
        x1, x2 = data.x.clone(), data.x.clone()
        x1, x2 = x1 * feat_mask1, x2 * feat_mask2

        edge_index1, edge_attr1 = dropout_adj(data.edge_index, data.edge_attr, p = self.p_e1)
        edge_index2, edge_attr2 = dropout_adj(data.edge_index, data.edge_attr, p = self.p_e2)

        new_data1, new_data2 = data.clone(), data.clone()
        new_data1.x, new_data2.x = x1, x2
        new_data1.edge_index, new_data2.edge_index = edge_index1, edge_index2
        new_data1.edge_attr , new_data2.edge_attr = edge_attr1, edge_attr2

        return new_data1, new_data2

    def __call__(self, data):
        
        return self._feature_masking(data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="data",
                        help="Path to data directory, where all the datasets will be placed. Default is 'data'")
    parser.add_argument("--name", "-n",type=str, default="WikiCS",
                        help="Name of the dataset. Supported names are: cora, citeseer, pubmed, photo, computers, cs, and physics")
    parser.add_argument("--layers", "-l", nargs="+", default=[
                        512, 256], help="The number of units of each layer of the GNN. Default is [512, 128]")
    parser.add_argument("--pred_hid", '-ph', type=int,
                        default=512, help="The number of hidden units of layer of the predictor. Default is 512")
    parser.add_argument("--init-parts", "-ip", type=int, default=1,
                        help="The number of initial partitions. Default is 1. Applicable for ClusterSelfGNN")
    parser.add_argument("--final-parts", "-fp", type=int, default=1,
                        help="The number of final partitions. Default is 1. Applicable for ClusterSelfGNN")
    parser.add_argument("--aug_params", "-p", nargs="+", default=[
                        0.3, 0.4, 0.3, 0.2], help="Hyperparameters for augmentation (p_f1, p_f2, p_e1, p_e2). Default is [0.2, 0.1, 0.2, 0.3]")
    parser.add_argument("--lr", '-lr', type=float, default=0.00001,
                        help="Learning rate. Default is 0.0001.")
    parser.add_argument("--dropout", "-do", type=float,
                        default=0.0, help="Dropout rate. Default is 0.2")
    parser.add_argument("--cache-step", '-cs', type=int, default=10,
                        help="The step size to cache the model, that is, every cache_step the model is persisted. Default is 100.")
    parser.add_argument("--epochs", '-e', type=int,
                        default=20, help="The number of epochs")
    parser.add_argument("--device", '-d', type=int,
                        default=0, help="GPU to use")
    parser.add_argument("--feat-pt", type=str, default=None,
                        dest="feat_pt",
                        help=(
                            "Optional path to a .pt file containing external node features "
                            "(e.g. SBERT embeddings from preprocess_sbert_features.py). "
                            "Accepted formats: a dict with key 'x' (preferred), or a raw Tensor. "
                            "When provided, data.x is replaced after dataset loading and validated "
                            "(node count, floating dtype, finite values) before use. "
                            "Default: None (use original dataset features)."
                        ))
    parser.add_argument("--export-encoder-ckpt", type=str, default=None,
                        dest="export_encoder_ckpt",
                        help=(
                            "Path to save an encoder-only checkpoint after training. "
                            "The saved file follows the contract in eval/checkpoint.py "
                            "and is loadable by eval/load_encoder.py. "
                            "Default: None (no export)."
                        ))
    parser.add_argument("--skip-eval", action="store_true", default=False,
                        dest="skip_eval",
                        help=(
                            "Skip BGRL's built-in logistic-regression evaluation during training. "
                            "Useful for fast Layer 2 checkpoint export/smoke runs. "
                            "Default: False."
                        ))
    return parser.parse_args()


def decide_config(root, name):
    """
    Create a configuration to download datasets
    :param root: A path to a root directory where data will be stored
    :param name: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    name = name.lower()
    if name == 'cora' or name == 'citeseer' or name == "pubmed":
        root = osp.join(root, "pyg", "planetoid")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Planetoid, "src": "pyg"}
    elif name == "computers":
        name = "Computers"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Amazon, "src": "pyg"}        
    elif name == "photo":
        name = "Photo"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Amazon, "src": "pyg"}
    elif name == "cs" :
        name = "CS"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Coauthor, "src": "pyg"}
    elif name == "physics":
        name = "Physics"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "class": Coauthor, "src": "pyg"}
    elif name == "wikics":
        name = "WikiCS"
        root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root},
                  "name": name, "class": WikiCS, "src": "pyg"}
    elif name == "wn18rr":
        # WN18RR knowledge graph — link-prediction dataset, no node features or labels.
        # PyG downloads to root/pyg/WN18RR/; BGRL cache lives in the same tree.
        name = "WN18RR"
        root = osp.join(root, "pyg", "WN18RR")
        params = {"kwargs": {"root": root},
                  "name": name, "class": WordNet18RR, "src": "pyg"}
    elif name in ("arxiv", "ogbn-arxiv"):
        # Route ogbn-arxiv through OGB's PygNodePropPredDataset.
        # OGB files live under root/ogb/ogbn_arxiv/; BGRL's own raw/processed
        # cache lives under root/ogb/bgrl_ogbn-arxiv/ (kept separate to avoid
        # conflicts with OGB's native directory layout).
        name = "ogbn-arxiv"
        root = osp.join(root, "ogb")
        params = {"kwargs": {"root": root, "name": name},
                  "name": name, "src": "ogb"}
    else:
        raise Exception(
            f"Unknown dataset name {name}, name has to be one of the following "
            f"'cora', 'citeseer', 'pubmed', 'photo', 'computers', 'cs', 'physics', "
            f"'wn18rr', 'arxiv' (ogbn-arxiv)")
    return params


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = [sub_dir for sub_dir in dir_tree.split("/") if sub_dir]
        path = "/" if osp.isabs(dir_tree) else ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


def create_masks(data):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place
    :param data: Data object
    :return: The modified data
    """
    if not hasattr(data, "val_mask"):

        data.train_mask = data.dev_mask = data.test_mask = None
        
        for i in range(20):
            labels = data.y.numpy()
            dev_size = int(labels.shape[0] * 0.1)
            test_size = int(labels.shape[0] * 0.8)

            perm = np.random.permutation(labels.shape[0])
            test_index = perm[:test_size]
            dev_index = perm[test_size:test_size+dev_size]
            
            data_index = np.arange(labels.shape[0])
            test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
            dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
            train_mask = ~(dev_mask + test_mask)
            test_mask = test_mask.reshape(1, -1)
            dev_mask = dev_mask.reshape(1, -1)
            train_mask = train_mask.reshape(1, -1)
            
            if data.train_mask is None :
                data.train_mask = train_mask
                data.val_mask = dev_mask
                data.test_mask = test_mask
            else :
                data.train_mask = torch.cat((data.train_mask, train_mask), dim = 0)
                data.val_mask = torch.cat((data.val_mask, dev_mask), dim = 0)
                data.test_mask = torch.cat((data.test_mask, test_mask), dim = 0)
    
    else :
        data.train_mask = data.train_mask.T
        data.val_mask = data.val_mask.T
    
    return data
