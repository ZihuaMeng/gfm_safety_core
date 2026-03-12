"""
audit_datasets.py
-----------------
Load each of the five pretrain datasets and print a structured audit summary,
then optionally write the results to notes/dataset_audit.md.

What this script does:
  - Tries known loader candidates for each dataset inside try/except blocks
  - Probes the loaded graph object for: num_nodes, num_edges, feature field
    names/shapes/dtypes, label shape, and any text-like attributes
  - Scans data_root for text-format files that might be the text source
  - Scans repo_root (repos/) for Python files that reference each dataset,
    helping identify which fork-specific loader the project actually uses
  - Prints a structured stdout summary and (by default) writes markdown to
    notes/dataset_audit.md

CONFIRMED from upstream source inspection (against upstream repos, not the
project fork — forks may differ):
  [confirmed]   BGRL (PyG): features live in data.x
  [confirmed]   GraphMAE (DGL): features live in graph.ndata["feat"]
  [confirmed]   GraphMAE applies scale_feats() — must skip for SBERT features

NOT CONFIRMED until this script runs against the actual cloned repos:
  [not confirmed]  Exact loader used in the project fork for each dataset
  [not confirmed]  Which datasets the project fork actually supports
  [must-verify]    Node ordering alignment between graph nodes and text lists

Recommended run order (cheapest / safest first):
    1. arxiv        — moderate size (~169k nodes); good first test
    2. wn18rr       — small (~41k entities); loader unknown, useful to probe early
    3. roman-empire — small (~23k nodes); loader unknown
    4. pcba         — graph-level; use --no-write-report until strategy is decided
    5. products     — run LAST; ~2.4M nodes, may need 16+ GB RAM (risky on WSL)

Run:
    python src/audit_datasets.py --dataset arxiv
    python src/audit_datasets.py --dataset wn18rr
    python src/audit_datasets.py --dataset roman-empire
    python src/audit_datasets.py --dataset pcba         --no-write-report
    python src/audit_datasets.py --dataset products     --data-root data/ --repo-root repos/
    python src/audit_datasets.py --dataset all          --data-root data/ --repo-root repos/
"""

import argparse
import contextlib
import datetime
import importlib
import os
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# File extensions treated as potential text sources
_TEXT_EXTS = frozenset({".tsv", ".csv", ".txt", ".json", ".jsonl", ".npz", ".npy"})

# Attribute names that suggest text content on a dataset/graph object
_TEXT_ATTR_NAMES = frozenset({
    "raw_texts", "texts", "text", "node_text", "raw_text", "node_texts",
    "sentences", "titles", "abstracts", "descriptions", "gloss",
    "smiles", "smiles_list", "raw_smiles",
})

_SEP = "=" * 64


# ---------------------------------------------------------------------------
# Report class — stdout + markdown in one place
# ---------------------------------------------------------------------------

class Report:
    """
    Accumulates human-readable output for stdout and markdown simultaneously.

    Every write method prints to stdout immediately and appends a markdown
    representation to an internal buffer.  Call to_markdown() at the end.
    """

    def __init__(self) -> None:
        self._md: list[str] = []

    # -- Structural --------------------------------------------------------

    def section(self, title: str) -> None:
        print(f"\n{_SEP}\n  {title}\n{_SEP}")
        self._md.append(f"\n---\n\n## {title}\n\n")

    def subsection(self, title: str) -> None:
        print(f"\n  -- {title} --")
        self._md.append(f"\n### {title}\n\n")

    # -- Content -----------------------------------------------------------

    def field(self, name: str, value: object, status: str = "unconfirmed") -> None:
        """Print a key-value pair with a confidence tag."""
        tag = f"[{status}]"
        print(f"  {name:<38} {tag:<16} {value}")
        self._md.append(f"- **{name}**: `{value}`  _{tag}_\n")

    def info(self, msg: str) -> None:
        print(f"  {msg}")
        self._md.append(f"  {msg}\n")

    def todo(self, msg: str) -> None:
        """Explicit alignment / verification TODO that must not be removed prematurely."""
        print(f"  [TODO] {msg}")
        self._md.append(f"- **[TODO]** {msg}\n")

    def warn(self, msg: str) -> None:
        print(f"  [WARN] {msg}", file=sys.stderr)
        self._md.append(f"- **[WARN]** {msg}\n")

    def error(self, msg: str) -> None:
        print(f"  [ERROR] {msg}", file=sys.stderr)
        self._md.append(f"- **[ERROR]** {msg}\n")

    # -- Output ------------------------------------------------------------

    def to_markdown(self) -> str:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        header = (
            "# Dataset Audit — Auto-generated\n\n"
            f"> Generated by `src/audit_datasets.py` on {ts}\n"
            "> To update, re-run the audit script.  "
            "Manual notes belong in a separate `notes/` file.\n\n"
        )
        return header + "".join(self._md)


# ---------------------------------------------------------------------------
# Generic probing helpers
# ---------------------------------------------------------------------------

def _try_import(dotted_path: str) -> tuple[bool, Any]:
    """
    Safely import a class/function by dotted path.
    Returns (True, obj) on success, (False, None) on any failure.

    Example: _try_import("ogb.nodeproppred.DglNodePropPredDataset")
    """
    parts = dotted_path.rsplit(".", 1)
    module_path = parts[0]
    attr = parts[1] if len(parts) == 2 else None
    try:
        module = importlib.import_module(module_path)
        if attr:
            return True, getattr(module, attr)
        return True, module
    except (ImportError, AttributeError, ModuleNotFoundError):
        return False, None


def _probe_dgl_graph(graph: Any, report: Report) -> dict:
    """
    Extract stats from a DGL graph object (homogeneous or heterogeneous).
    All values printed are tagged [confirmed] — they come from the live object.
    """
    stats: dict = {}
    try:
        # Heterogeneous graph
        ntypes = getattr(graph, "ntypes", [])
        if len(ntypes) > 1:
            report.field("Graph type", "heterogeneous DGL", status="confirmed")
            stats["graph_type"] = "heterogeneous"
            for ntype in ntypes:
                n = graph.num_nodes(ntype)
                report.field(f"  num_nodes[{ntype}]", n, status="confirmed")
                for key, val in graph.ndata[ntype].items() if hasattr(graph.ndata, "__getitem__") else []:
                    report.field(f"  ndata[{ntype}]['{key}'] shape", tuple(val.shape), status="confirmed")
                    report.field(f"  ndata[{ntype}]['{key}'] dtype", str(val.dtype), status="confirmed")
        else:
            stats["num_nodes"] = graph.num_nodes()
            stats["num_edges"] = graph.num_edges()
            report.field("num_nodes", stats["num_nodes"], status="confirmed")
            report.field("num_edges", stats["num_edges"], status="confirmed")

            ndata_info: dict = {}
            for key in graph.ndata:
                val = graph.ndata[key]
                shape = tuple(val.shape)
                dtype = str(val.dtype)
                ndata_info[key] = {"shape": shape, "dtype": dtype}
                report.field(f"ndata['{key}'] shape", shape, status="confirmed")
                report.field(f"ndata['{key}'] dtype", dtype, status="confirmed")
            stats["ndata"] = ndata_info

    except Exception as exc:
        report.error(f"Error probing DGL graph: {exc}")

    return stats


def _probe_pyg_data(data: Any, report: Report) -> dict:
    """
    Extract stats from a PyG Data or HeteroData object.
    All values tagged [confirmed] — they come from the live object.
    """
    stats: dict = {}
    try:
        # HeteroData check (duck typing — avoid importing torch_geometric)
        if hasattr(data, "node_types"):
            report.field("Graph type", "heterogeneous PyG (HeteroData)", status="confirmed")
            stats["graph_type"] = "heterogeneous"
            for ntype in data.node_types:
                node_store = data[ntype]
                if hasattr(node_store, "x") and node_store.x is not None:
                    report.field(f"  x[{ntype}] shape", tuple(node_store.x.shape), status="confirmed")
                    report.field(f"  x[{ntype}] dtype", str(node_store.x.dtype), status="confirmed")
            return stats

        # Homogeneous Data
        stats["num_nodes"] = data.num_nodes
        report.field("num_nodes", stats["num_nodes"], status="confirmed")

        if hasattr(data, "edge_index") and data.edge_index is not None:
            stats["num_edges"] = data.edge_index.shape[1]
            report.field("num_edges", stats["num_edges"], status="confirmed")

        if hasattr(data, "x") and data.x is not None:
            stats["feat_shape"] = tuple(data.x.shape)
            stats["feat_dtype"] = str(data.x.dtype)
            report.field("x shape", stats["feat_shape"], status="confirmed")
            report.field("x dtype", stats["feat_dtype"], status="confirmed")
        else:
            report.field("x", "None — no default node features", status="confirmed")

        if hasattr(data, "y") and data.y is not None:
            stats["label_shape"] = tuple(data.y.shape)
            report.field("y shape", stats["label_shape"], status="confirmed")

        # All keys present on the object
        try:
            all_keys = list(data.keys()) if hasattr(data, "keys") and callable(data.keys) else []
            if all_keys:
                stats["all_keys"] = all_keys
                report.field("All data.keys()", all_keys, status="confirmed")
        except Exception:
            pass

    except Exception as exc:
        report.error(f"Error probing PyG Data: {exc}")

    return stats


def _probe_ogb_dict_graph(graph: dict, labels: Any, report: Report) -> dict:
    """
    Extract stats from a backend-agnostic OGB dict graph as returned by
    NodePropPredDataset[0].  Keys accessed:
      graph["num_nodes"]  — int
      graph["edge_index"] — numpy array shape [2, E]
      graph["node_feat"]  — numpy array shape [N, D], or None
    All reported values are tagged [confirmed] — they come from the live object.
    """
    stats: dict = {}

    try:
        stats["num_nodes"] = int(graph["num_nodes"])
        report.field("num_nodes", stats["num_nodes"], status="confirmed")
    except (KeyError, TypeError) as exc:
        report.error(f"Could not read graph['num_nodes']: {exc}")

    try:
        edge_index = graph["edge_index"]
        stats["num_edges"] = int(edge_index.shape[1])
        report.field("num_edges", stats["num_edges"], status="confirmed")
    except (KeyError, AttributeError, IndexError) as exc:
        report.error(f"Could not read graph['edge_index'].shape[1]: {exc}")

    try:
        node_feat = graph["node_feat"]
        if node_feat is not None:
            stats["feat_shape"] = tuple(node_feat.shape)
            stats["feat_dtype"] = str(node_feat.dtype)
            report.field("node_feat shape", stats["feat_shape"], status="confirmed")
            report.field("node_feat dtype", stats["feat_dtype"], status="confirmed")
        else:
            report.field("node_feat", "None — no default node features", status="confirmed")
    except (KeyError, AttributeError) as exc:
        report.error(f"Could not read graph['node_feat']: {exc}")

    try:
        if labels is not None:
            stats["label_shape"] = tuple(labels.shape)
            report.field("labels shape", stats["label_shape"], status="confirmed")
    except (AttributeError, TypeError) as exc:
        report.error(f"Could not read labels.shape: {exc}")

    return stats


def _probe_object_text_attrs(obj: Any) -> list[str]:
    """
    Return non-callable attribute names on obj that likely contain text data.
    Checks against known text field names and any name containing 'text' or 'smiles'.
    """
    found: list[str] = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        name_lower = name.lower()
        if name_lower not in _TEXT_ATTR_NAMES and "text" not in name_lower and "smiles" not in name_lower:
            continue
        try:
            val = getattr(obj, name)
            if val is not None and not callable(val):
                found.append(name)
        except Exception:
            pass
    return found


def _scan_text_files(root: str, keywords: list[str], max_results: int = 20) -> list[str]:
    """
    Scan data_root recursively for files whose name contains any keyword and
    whose extension is in _TEXT_EXTS.  Returns a sorted list of matching paths.
    """
    found: list[str] = []
    if not os.path.isdir(root):
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _TEXT_EXTS:
                continue
            if any(kw.lower() in fname.lower() for kw in keywords):
                found.append(os.path.join(dirpath, fname))
                if len(found) >= max_results:
                    return found
    return found


def _scan_repo_for_loader(repo_root: str, keywords: list[str], max_results: int = 12) -> list[str]:
    """
    Scan Python files under repo_root for lines that mention any of the keywords.
    Returns a list of "filepath:lineno: line" strings.
    Useful for finding which file in the project fork loads a given dataset.
    """
    matches: list[str] = []
    if not os.path.isdir(repo_root):
        return matches
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and d not in {"__pycache__", "node_modules", ".git"}
        )
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if any(kw.lower() in line.lower() for kw in keywords):
                            matches.append(f"{fpath}:{lineno}: {line.rstrip()}")
                            if len(matches) >= max_results:
                                return matches
            except Exception:
                pass
    return matches


# ---------------------------------------------------------------------------
# PyTorch 2.6+ / OGB cache compatibility helpers
# ---------------------------------------------------------------------------
# PyTorch 2.6 changed torch.load's weights_only default from False to True.
# OGB's processed cache files contain complex pickled objects that are
# incompatible with weights_only=True, causing a WeightsUnpickler error.
# These helpers add a single retry with weights_only=False, scoped ONLY to
# trusted local OGB cache reads during this audit script.
# DO NOT copy this pattern into training or preprocessing code.
# ---------------------------------------------------------------------------

_WEIGHTS_ONLY_COMPAT_WARNING = (
    "PyTorch 2.6+ changed weights_only default to True, which breaks OGB's "
    "processed cache format.  Retrying with weights_only=False for this trusted "
    "local OGB cache file.  AUDIT-ONLY — do NOT apply to training code."
)


def _is_weights_only_error(exc: Exception) -> bool:
    """Return True if exc looks like a PyTorch 2.6+ weights_only incompatibility."""
    msg = str(exc).lower()
    return "weights_only" in msg or "weightsunpickler" in msg or "weights only" in msg


@contextlib.contextmanager
def _ogb_load_compat():
    """
    Context manager: temporarily patches torch.load so that calls which do
    not explicitly pass weights_only get weights_only=False injected.

    Scope: wrap only trusted local OGB cache reads (NodePropPredDataset /
    PygNodePropPredDataset construction and __getitem__) inside this audit
    script.  Restore the original torch.load on exit regardless of outcome.
    """
    import torch as _torch
    _original = _torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original(*args, **kwargs)

    _torch.load = _patched
    try:
        yield
    finally:
        _torch.load = _original


# ---------------------------------------------------------------------------
# Per-dataset audit functions
# ---------------------------------------------------------------------------
# Each function:
#   - Accepts (data_root, repo_root, report)
#   - Tries known loader candidates in try/except; never crashes the whole run
#   - Uses _probe_* helpers to extract live stats
#   - Scans for text files and repo references
#   - Adds explicit alignment TODOs — never claims alignment is confirmed
#   - Returns a result dict with at minimum: dataset, status, loader_used
# ---------------------------------------------------------------------------

def audit_arxiv(data_root: str, repo_root: str, report: Report) -> dict:
    """
    ogbn-arxiv audit.

    Candidate loaders (both tried):
      - ogb.nodeproppred.DglNodePropPredDataset  (used by upstream GraphMAE)
      - ogb.nodeproppred.PygNodePropPredDataset  (used by upstream BGRL path)

    Text source: titleabs.tsv via OGB nodeidx2paperid mapping (NOT CONFIRMED for fork).
    """
    result: dict = {"dataset": "arxiv", "status": "not_loaded", "loader_used": None}
    report.section("Arxiv (ogbn-arxiv)")

    # Scan repos/ for any fork-specific loader references
    refs = _scan_repo_for_loader(repo_root, ["ogbn-arxiv", "arxiv"])
    if refs:
        report.subsection("Loader references found in repos/")
        for ref in refs[:8]:
            report.info(ref)
    else:
        report.warn("No loader references found in repos/ — may not be cloned yet")

    graph_dict = labels = dataset = None

    # --- Attempt 1: backend-agnostic OGB loader (preferred for audit) ---
    # NodePropPredDataset does not require DGL or PyG; returns a plain dict graph.
    ok, NodeDataset = _try_import("ogb.nodeproppred.NodePropPredDataset")
    if ok:
        try:
            try:
                dataset = NodeDataset(name="ogbn-arxiv", root=data_root)
                graph_dict, labels = dataset[0]
            except Exception as exc:
                if not _is_weights_only_error(exc):
                    raise
                report.warn(_WEIGHTS_ONLY_COMPAT_WARNING)
                with _ogb_load_compat():
                    dataset = NodeDataset(name="ogbn-arxiv", root=data_root)
                    graph_dict, labels = dataset[0]
            result["loader_used"] = "ogb.nodeproppred.NodePropPredDataset"
            result["status"] = "loaded"
            report.field("Loader", result["loader_used"], status="confirmed")
            stats = _probe_ogb_dict_graph(graph_dict, labels, report)
            result.update(stats)
        except Exception as exc:
            report.warn(f"NodePropPredDataset raised: {exc}")
    else:
        report.warn("ogb.nodeproppred.NodePropPredDataset not importable (ogb not installed?)")

    # --- Attempt 2: PyG (upstream BGRL path) — only if attempt 1 failed ---
    if graph_dict is None:
        ok, PygDataset = _try_import("ogb.nodeproppred.PygNodePropPredDataset")
        if ok:
            try:
                try:
                    dataset = PygDataset("ogbn-arxiv", root=data_root)
                    pyg_data = dataset[0]
                except Exception as exc:
                    if not _is_weights_only_error(exc):
                        raise
                    report.warn(_WEIGHTS_ONLY_COMPAT_WARNING)
                    with _ogb_load_compat():
                        dataset = PygDataset("ogbn-arxiv", root=data_root)
                        pyg_data = dataset[0]
                result["loader_used"] = "ogb.nodeproppred.PygNodePropPredDataset"
                result["status"] = "loaded"
                report.field("Loader", result["loader_used"], status="confirmed")
                stats = _probe_pyg_data(pyg_data, report)
                result.update(stats)
                graph_dict = pyg_data  # mark as loaded
            except Exception as exc:
                report.warn(f"PyG loader raised: {exc}")

    if result["status"] == "loaded" and dataset is not None:
        text_attrs = _probe_object_text_attrs(dataset)
        if text_attrs:
            report.field("Text attrs on dataset object", text_attrs, status="confirmed")
        else:
            report.info("No text attributes found on dataset object")
        report.info(
            "NOTE: DGL/PyG backend selection for BGRL and GraphMAE integration "
            "is NOT confirmed — must be determined from the cloned project repos."
        )

    if result["status"] == "not_loaded":
        report.warn("Dataset could not be loaded (ogb not installed or data not downloaded)")
        report.field("Expected num_nodes (public doc)", "~169,343", status="unconfirmed")
        report.field("Expected num_edges (public doc)", "~1,166,243", status="unconfirmed")
        report.field("Expected feat shape (public doc)", "[N, 128] float32", status="unconfirmed")

    # Scan for text files regardless of loader status
    text_files = _scan_text_files(data_root, ["arxiv", "titleabs", "nodeidx", "abstract"])
    if text_files:
        report.subsection("Candidate text files found under data_root")
        for f in text_files:
            report.info(f)
    else:
        report.info("No candidate text files found under data_root for arxiv")

    report.field("Text source (public doc)", "titleabs.tsv via OGB nodeidx2paperid", status="unconfirmed")
    report.todo(
        "Verify that text[i] in titleabs.tsv corresponds exactly to node i in the loaded graph. "
        "Alignment is NOT confirmed — must check nodeidx2paperid mapping manually."
    )
    report.todo(
        "Decide handling for papers with empty abstracts (use title only, or skip?)."
    )
    report.todo(
        "Confirm whether project fork uses DGL or PyG loader, and which path "
        "BGRL and GraphMAE integration should target."
    )

    return result


def audit_products(data_root: str, repo_root: str, report: Report) -> dict:
    """
    ogbn-products audit.

    NOTE: ~2.4M nodes; loading may take several minutes.
    """
    result: dict = {"dataset": "products", "status": "not_loaded", "loader_used": None}
    report.section("Products (ogbn-products)")
    report.warn("Dataset has ~2.4M nodes. Loading may be slow and memory-intensive.")
    report.warn(
        "WSL environments typically share the host's RAM pool via a memory cap "
        "(default 50% of physical RAM or 8 GB, whichever is lower). "
        "The full ogbn-products graph requires ~16+ GB. "
        "Run this dataset last and consider increasing the WSL memory limit in "
        ".wslconfig before loading."
    )

    refs = _scan_repo_for_loader(repo_root, ["ogbn-products", "products"])
    if refs:
        report.subsection("Loader references found in repos/")
        for ref in refs[:8]:
            report.info(ref)
    else:
        report.warn("No loader references found in repos/ — may not be cloned yet")

    graph_dict = labels = dataset = None

    # --- Attempt 1: backend-agnostic OGB loader (preferred for audit) ---
    ok, NodeDataset = _try_import("ogb.nodeproppred.NodePropPredDataset")
    if ok:
        try:
            try:
                dataset = NodeDataset(name="ogbn-products", root=data_root)
                graph_dict, labels = dataset[0]
            except Exception as exc:
                if not _is_weights_only_error(exc):
                    raise
                report.warn(_WEIGHTS_ONLY_COMPAT_WARNING)
                with _ogb_load_compat():
                    dataset = NodeDataset(name="ogbn-products", root=data_root)
                    graph_dict, labels = dataset[0]
            result["loader_used"] = "ogb.nodeproppred.NodePropPredDataset"
            result["status"] = "loaded"
            report.field("Loader", result["loader_used"], status="confirmed")
            stats = _probe_ogb_dict_graph(graph_dict, labels, report)
            result.update(stats)
        except Exception as exc:
            report.warn(f"NodePropPredDataset raised: {exc}")
    else:
        report.warn("ogb.nodeproppred.NodePropPredDataset not importable (ogb not installed?)")

    # --- Attempt 2: PyG — only if attempt 1 failed ---
    if graph_dict is None:
        ok, PygDataset = _try_import("ogb.nodeproppred.PygNodePropPredDataset")
        if ok:
            try:
                try:
                    dataset = PygDataset("ogbn-products", root=data_root)
                    pyg_data = dataset[0]
                except Exception as exc:
                    if not _is_weights_only_error(exc):
                        raise
                    report.warn(_WEIGHTS_ONLY_COMPAT_WARNING)
                    with _ogb_load_compat():
                        dataset = PygDataset("ogbn-products", root=data_root)
                        pyg_data = dataset[0]
                result["loader_used"] = "ogb.nodeproppred.PygNodePropPredDataset"
                result["status"] = "loaded"
                report.field("Loader", result["loader_used"], status="confirmed")
                stats = _probe_pyg_data(pyg_data, report)
                result.update(stats)
                graph_dict = pyg_data
            except Exception as exc:
                report.warn(f"PyG loader raised: {exc}")

    if result["status"] == "loaded" and dataset is not None:
        text_attrs = _probe_object_text_attrs(dataset)
        if text_attrs:
            report.field("Text attrs on dataset object", text_attrs, status="confirmed")
        report.info(
            "NOTE: DGL/PyG backend selection for BGRL and GraphMAE integration "
            "is NOT confirmed — must be determined from the cloned project repos."
        )

    if result["status"] == "not_loaded":
        report.warn("Dataset could not be loaded")
        report.field("Expected num_nodes (public doc)", "~2,449,029", status="unconfirmed")
        report.field("Expected num_edges (public doc)", "~61,859,140", status="unconfirmed")
        report.field("Expected feat shape (public doc)", "[N, 100] float32", status="unconfirmed")

    text_files = _scan_text_files(data_root, ["products", "amazon", "title", "description"])
    if text_files:
        report.subsection("Candidate text files found under data_root")
        for f in text_files:
            report.info(f)

    report.field("Text source (public doc)", "Amazon product title+description (OGB text file)", status="unconfirmed")
    report.todo(
        "Verify that text[i] in the products text file corresponds exactly to node i. "
        "Alignment is NOT confirmed."
    )
    report.todo(
        "Confirm text file path. OGB products text is separate from the graph download."
    )
    report.todo("Plan memory budget: 2.4M nodes × 768 × 4 bytes ≈ 7.4 GB output tensor.")

    return result


def audit_wn18rr(data_root: str, repo_root: str, report: Report) -> dict:
    """
    WN18RR audit.

    Loader is NOT present in default BGRL or GraphMAE repos.
    Candidate loaders tried here:
      - torch_geometric.datasets.WordNet18RR
    If that fails, report what the project fork uses (via repo scan).
    """
    result: dict = {"dataset": "wn18rr", "status": "not_loaded", "loader_used": None}
    report.section("WN18RR")

    refs = _scan_repo_for_loader(repo_root, ["wn18rr", "WN18RR", "wordnet"])
    if refs:
        report.subsection("Loader references found in repos/")
        for ref in refs[:8]:
            report.info(ref)
        report.todo(
            "Identify which of the above files is the canonical loader. "
            "Read it before filling in preprocess_sbert_features.py."
        )
    else:
        report.warn(
            "No loader references found in repos/. "
            "WN18RR is NOT in default BGRL or GraphMAE — loader must come from the project fork."
        )

    graph = dataset = None

    # --- Attempt: PyG WordNet18RR ---
    ok, WordNet18RR = _try_import("torch_geometric.datasets.WordNet18RR")
    if ok:
        try:
            dataset = WordNet18RR(root=os.path.join(data_root, "wn18rr"))
            graph = dataset[0]
            result["loader_used"] = "torch_geometric.datasets.WordNet18RR"
            result["status"] = "loaded"
            report.field("Loader", result["loader_used"], status="confirmed")
            stats = _probe_pyg_data(graph, report)
            result.update(stats)
        except Exception as exc:
            report.warn(f"PyG WordNet18RR raised: {exc}")
    else:
        report.info("torch_geometric.datasets.WordNet18RR not importable")

    if result["status"] == "not_loaded":
        report.warn("Could not load WN18RR with any candidate loader")
        report.field("Expected num_entities (public doc)", "~40,943", status="unconfirmed")
        report.field("Expected num_triples (public doc)", "~93,003", status="unconfirmed")
        report.field("Feat shape", "UNKNOWN — likely none or ID embeddings", status="unconfirmed")

    if graph is not None and dataset is not None:
        text_attrs = _probe_object_text_attrs(dataset)
        if text_attrs:
            report.field("Text attrs on dataset object", text_attrs, status="confirmed")
        else:
            report.info("No text attributes found on dataset object")

    text_files = _scan_text_files(data_root, ["wn18rr", "wn18", "wordnet", "entity", "gloss"])
    if text_files:
        report.subsection("Candidate text files found under data_root")
        for f in text_files:
            report.info(f)

    report.field("Text source (public doc)", "WordNet entity name + gloss definition", status="unconfirmed")
    report.todo(
        "Confirm entity-to-index mapping. The loader may remap entity IDs. "
        "Verify that texts[i] corresponds to entity_to_idx[i]. Alignment NOT confirmed."
    )
    report.todo(
        "Determine whether the project loader returns a homogeneous graph or a "
        "multi-relational heterogeneous graph (affects how node features are accessed)."
    )
    report.todo(
        "Identify the exact text file path once the project fork is cloned."
    )

    return result


def audit_roman_empire(data_root: str, repo_root: str, report: Report) -> dict:
    """
    Roman-Empire audit.

    Candidate loaders tried here:
      - torch_geometric.datasets.HeterophilousGraphDataset (requires PyG >= 2.3)
    Also checks for raw .npz files under data_root.
    """
    result: dict = {"dataset": "roman-empire", "status": "not_loaded", "loader_used": None}
    report.section("Roman-Empire")

    refs = _scan_repo_for_loader(repo_root, ["roman", "roman_empire", "roman-empire"])
    if refs:
        report.subsection("Loader references found in repos/")
        for ref in refs[:8]:
            report.info(ref)
        report.todo("Identify which file is the canonical loader for Roman-Empire.")
    else:
        report.warn(
            "No loader references found in repos/. "
            "Roman-Empire is NOT in default BGRL or GraphMAE — must come from project fork."
        )

    graph = dataset = None

    # --- Attempt: PyG HeterophilousGraphDataset ---
    ok, HetGraph = _try_import("torch_geometric.datasets.HeterophilousGraphDataset")
    if ok:
        try:
            dataset = HetGraph(root=os.path.join(data_root, "roman_empire"), name="Roman-empire")
            graph = dataset[0]
            result["loader_used"] = "torch_geometric.datasets.HeterophilousGraphDataset"
            result["status"] = "loaded"
            report.field("Loader", result["loader_used"], status="confirmed")
            stats = _probe_pyg_data(graph, report)
            result.update(stats)
        except Exception as exc:
            report.warn(f"PyG HeterophilousGraphDataset raised: {exc}")
    else:
        report.info("torch_geometric.datasets.HeterophilousGraphDataset not importable")

    if result["status"] == "not_loaded":
        report.warn("Could not load Roman-Empire with any candidate loader")
        report.field("Expected num_nodes (public doc)", "~22,662", status="unconfirmed")
        report.field("Expected num_edges (public doc)", "~32,927", status="unconfirmed")
        report.field("Expected feat shape (public doc)", "[N, 300] float32 (FastText)", status="unconfirmed")

    if graph is not None and dataset is not None:
        text_attrs = _probe_object_text_attrs(dataset)
        text_attrs += _probe_object_text_attrs(graph)
        text_attrs = sorted(set(text_attrs))
        if text_attrs:
            report.field("Text attrs on dataset/graph object", text_attrs, status="confirmed")
        else:
            report.info("No text attributes found on dataset or graph object")

    # Check for .npz files (Platonov et al. raw format)
    npz_files = _scan_text_files(data_root, ["roman", "empire"])
    if npz_files:
        report.subsection("Candidate data files found under data_root")
        for f in npz_files:
            report.info(f)
        report.todo(
            "If a .npz file was found, inspect its keys with "
            "`np.load(path, allow_pickle=True).files` to check for a raw_texts field."
        )

    report.field("Text source (public doc)", "Wikipedia sentences (one per node)", status="unconfirmed")
    report.todo(
        "Verify that sentence index in the raw_texts field (or text file) matches node index. "
        "Alignment NOT confirmed."
    )
    report.todo(
        "Check whether raw_texts is exposed on the dataset object after loading, "
        "or must be loaded from a separate .npz / text file."
    )

    return result


def audit_pcba(data_root: str, repo_root: str, report: Report) -> dict:
    """
    PCBA (ogbg-molpcba) audit.

    This is a GRAPH-LEVEL classification dataset.  Each graph = one molecule.
    Nodes = atoms.  There is no per-node natural-language text.

    This audit probes:
      - First molecule's atom feature shape and dtype
      - Whether SMILES strings are accessible via the dataset object
      - How many graphs total

    Strategy decision (raw passthrough / SMILES→SBERT / zero-pad) is pending PI input.
    """
    result: dict = {"dataset": "pcba", "status": "not_loaded", "loader_used": None}
    report.section("PCBA (ogbg-molpcba)")
    report.warn(
        "PCBA is a graph-level dataset.  "
        "Whether/how it is used in BGRL/GraphMAE pretrain is NOT confirmed."
    )

    refs = _scan_repo_for_loader(repo_root, ["ogbg-molpcba", "pcba", "molpcba"])
    if refs:
        report.subsection("Loader references found in repos/")
        for ref in refs[:8]:
            report.info(ref)
    else:
        report.warn("No loader references found in repos/ for PCBA")

    dataset = None

    # --- Attempt 1: PyG ---
    ok, PygGraph = _try_import("ogb.graphproppred.PygGraphPropPredDataset")
    if ok:
        try:
            dataset = PygGraph("ogbg-molpcba", root=data_root)
            result["loader_used"] = "ogb.graphproppred.PygGraphPropPredDataset"
            result["status"] = "loaded"
            report.field("Loader", result["loader_used"], status="confirmed")
            report.field("num_graphs", len(dataset), status="confirmed")
            result["num_graphs"] = len(dataset)
            # Probe first graph only — do NOT iterate all 437k
            g0 = dataset[0]
            report.subsection("First molecule (graph index 0)")
            stats = _probe_pyg_data(g0, report)
            result.update(stats)
        except Exception as exc:
            report.warn(f"PyG loader raised: {exc}")

    # --- Attempt 2: DGL ---
    if dataset is None:
        ok, DglGraph = _try_import("ogb.graphproppred.DglGraphPropPredDataset")
        if ok:
            try:
                dataset = DglGraph("ogbg-molpcba", root=data_root)
                result["loader_used"] = "ogb.graphproppred.DglGraphPropPredDataset"
                result["status"] = "loaded"
                report.field("Loader", result["loader_used"], status="confirmed")
                report.field("num_graphs", len(dataset), status="confirmed")
                result["num_graphs"] = len(dataset)
                g0, _ = dataset[0]
                report.subsection("First molecule (graph index 0)")
                stats = _probe_dgl_graph(g0, report)
                result.update(stats)
            except Exception as exc:
                report.warn(f"DGL loader raised: {exc}")

    if result["status"] == "not_loaded":
        report.warn("Could not load PCBA with any candidate loader")
        report.field("Expected num_graphs (public doc)", "~437,929", status="unconfirmed")
        report.field("Avg atoms per molecule (public doc)", "~26", status="unconfirmed")
        report.field("Node feat shape per graph (public doc)", "[num_atoms, 9] int64", status="unconfirmed")

    # Check SMILES availability
    if dataset is not None:
        smiles_attrs = _probe_object_text_attrs(dataset)
        if smiles_attrs:
            report.field("SMILES/text attrs on dataset", smiles_attrs, status="confirmed")
        else:
            report.field("SMILES attrs on dataset", "None found", status="confirmed")

    report.field("Per-node NL text", "None — molecule atoms have no text description", status="unconfirmed")
    report.todo(
        "Get PI decision on PCBA feature strategy before implementing: "
        "(a) raw passthrough — keep 9-dim atom features, "
        "(b) SMILES→SBERT at graph level (one embedding per graph, not per atom), "
        "(c) zero 768-dim placeholder (not recommended)."
    )
    report.todo(
        "Confirm whether the project pretrain loop expects PCBA node features "
        "to be 768-dim or whether PCBA is treated as a special case."
    )

    return result


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

DATASET_AUDITORS = {
    "arxiv":        audit_arxiv,
    "products":     audit_products,
    "wn18rr":       audit_wn18rr,
    "roman-empire": audit_roman_empire,
    "pcba":         audit_pcba,
}


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_markdown_report(report: Report, out_path: str) -> None:
    """Write the accumulated markdown report to out_path (overwrites)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    content = report.to_markdown()
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"\n  Report written to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit pretrain datasets — loads each dataset and reports stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python src/audit_datasets.py --dataset all\n"
            "  python src/audit_datasets.py --dataset arxiv --data-root /data\n"
            "  python src/audit_datasets.py --dataset pcba  --no-write-report\n"
        ),
    )
    parser.add_argument(
        "--dataset", "-d",
        default="all",
        choices=list(DATASET_AUDITORS.keys()) + ["all"],
        help="Which dataset to audit (default: all)",
    )
    parser.add_argument(
        "--data-root", "-r",
        default="data",
        help="Root directory where datasets are stored / downloaded (default: data/)",
    )
    parser.add_argument(
        "--repo-root",
        default="repos",
        help=(
            "Root directory containing cloned model repos (default: repos/). "
            "Used to scan for fork-specific loader references."
        ),
    )
    parser.add_argument(
        "--write-report",
        default=True,
        action="store_true",
        help="Write markdown report to notes/dataset_audit.md (default: on)",
    )
    parser.add_argument(
        "--no-write-report",
        dest="write_report",
        action="store_false",
        help="Skip writing the markdown report",
    )
    parser.add_argument(
        "--report-path",
        default="notes/dataset_audit.md",
        help="Output path for the markdown report (default: notes/dataset_audit.md)",
    )
    args = parser.parse_args()

    datasets = list(DATASET_AUDITORS.keys()) if args.dataset == "all" else [args.dataset]

    report = Report()
    results: dict = {}

    for name in datasets:
        try:
            results[name] = DATASET_AUDITORS[name](args.data_root, args.repo_root, report)
        except Exception as exc:
            # Should not reach here (each auditor uses internal try/except),
            # but guard anyway to prevent one dataset from killing the whole run.
            print(f"\n  [ERROR] Unhandled exception in audit_{name}: {exc}", file=sys.stderr)
            results[name] = {"dataset": name, "status": "ERROR", "error": str(exc)}

    # Summary table
    print(f"\n{_SEP}")
    print("  SUMMARY")
    print(_SEP)
    print(f"  {'Dataset':<20} {'Status':<20} {'Loader'}")
    print(f"  {'-'*20} {'-'*20} {'-'*30}")
    for name, res in results.items():
        status = res.get("status", "?")
        loader = res.get("loader_used") or "—"
        print(f"  {name:<20} {status:<20} {loader}")

    print(f"\n  data_root:  {args.data_root}")
    print(f"  repo_root:  {args.repo_root}")

    print("\nNext steps:")
    print("  1. Fill in TODO items above, starting with alignment verification.")
    print("  2. If any dataset shows status=not_loaded, install dependencies or")
    print("     download the data, then re-run.")
    print("  3. Once loaders are confirmed, implement extract_texts_* functions")
    print("     in src/preprocess_sbert_features.py.")

    if args.write_report:
        write_markdown_report(report, args.report_path)
    else:
        print("\n  (--no-write-report: markdown not written)")


if __name__ == "__main__":
    main()
