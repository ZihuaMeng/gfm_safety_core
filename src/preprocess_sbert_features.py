"""
preprocess_sbert_features.py
-----------------------------
Encode node text descriptions with SBERT and save unified .pt feature files.

Unified saved schema (one file per dataset):
    {
        "x":          torch.Tensor  # float32, shape [num_nodes, 768]
        "dataset":    str           # canonical dataset name, e.g. "ogbn-arxiv"
        "encoder":    str           # SBERT model name used
        "dim":        int           # embedding dimension (768)
        "num_nodes":  int           # number of nodes
        "created_at": str           # ISO-8601 UTC timestamp
        "script_hash": str          # git commit hash of this script (or "unknown")
        "notes":      str           # any dataset-specific remarks
    }

Validation checks applied before saving:
    1. shape[0] == expected num_nodes (if known from audit)
    2. shape[1] == 768
    3. dtype == torch.float32
    4. no NaN values
    5. no Inf values

CONFIRMED from upstream source inspection:
    - BGRL reads data.x directly; replacing it is sufficient
    - GraphMAE reads graph.ndata["feat"] then num_features = feat.shape[1];
      replacing the tensor and returning the new dim is sufficient
    - GraphMAE applies scale_feats() to raw features — this must be skipped
      when loading SBERT features (they are already L2-normalised by SBERT)

IMPLEMENTED:
    - arxiv:        title+abstract via nodeidx2paperid.csv.gz + titleabs.tsv
                    (alignment NOT runtime-confirmed end-to-end)
    - roman-empire: Wikipedia sentence per node via PyG HeterophilousGraphDataset
                    or npz probe or plain text file (alignment NOT CONFIRMED)
    - wn18rr:       entity description via entity2id.txt + entity2text.txt
                    (alignment NOT CONFIRMED; requires entity2id.txt)
    - products:     product title+description via row-order text file or ASIN mapping
                    (alignment NOT CONFIRMED; OGB nodeidx2asin spec assumed)

NOT CONFIRMED:
    - PCBA strategy (no per-node text available)

Run:
    python src/preprocess_sbert_features.py --dataset arxiv        --data-root data/ --out-dir data/
    python src/preprocess_sbert_features.py --dataset products     --data-root data/ --out-dir data/
    python src/preprocess_sbert_features.py --dataset wn18rr       --data-root data/ --out-dir data/
    python src/preprocess_sbert_features.py --dataset roman-empire --data-root data/ --out-dir data/
    python src/preprocess_sbert_features.py --dataset all-text     --data-root data/ --out-dir data/
    python src/preprocess_sbert_features.py --dataset all          --data-root data/ --out-dir data/
    # all-text: arxiv, products, wn18rr, roman-empire (text available)
    # all:      all-text + pcba (pcba will raise until strategy is decided)

Dry-run (validate text extraction without encoding or saving):
    python src/preprocess_sbert_features.py --dataset arxiv        --data-root data/ --dry-run
    python src/preprocess_sbert_features.py --dataset roman-empire --data-root data/ --dry-run
    python src/preprocess_sbert_features.py --dataset wn18rr       --data-root data/ --dry-run
    python src/preprocess_sbert_features.py --dataset products     --data-root data/ --dry-run
"""

import argparse
import contextlib
import datetime
import os
import subprocess
import sys

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SBERT_MODEL_DEFAULT = "all-mpnet-base-v2"   # produces 768d embeddings; must match EMBED_DIM
                                            # NOTE: all-MiniLM-L6-v2 produces 384d and is NOT compatible
EMBED_DIM = 768
BATCH_SIZE = 256                            # adjust based on GPU memory


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    """Return the current git commit hash, or 'unknown' if not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def encode_texts(texts: list, model_name: str, batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """
    Encode a list of strings with SBERT.

    Returns a float32 tensor of shape [len(texts), 768].
    Raises if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[ERROR] sentence-transformers not installed.  Run: pip install sentence-transformers")
        sys.exit(1)

    print(f"  Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  Encoding {len(texts):,} texts (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,   # L2 normalise; consistent with SBERT best-practice
    )
    # Fail early if the encoder does not produce the expected dimension.
    # This catches mistakes like passing all-MiniLM-L6-v2 (384d) when 768d is required.
    actual_dim = embeddings.shape[1]
    if actual_dim != EMBED_DIM:
        raise ValueError(
            f"Encoder '{model_name}' produced {actual_dim}-dim embeddings, "
            f"but EMBED_DIM={EMBED_DIM}.  Use a 768-dim model "
            f"(e.g. all-mpnet-base-v2) or update EMBED_DIM explicitly."
        )
    # Ensure CPU float32
    return embeddings.cpu().float()
    
def validate_features(x: torch.Tensor, dataset_name: str, expected_num_nodes: int | None = None) -> None:
    """
    Run all validation checks on the output embedding tensor.
    Raises AssertionError on any failure (fail fast — never save corrupt data).
    """
    print(f"  Validating features for {dataset_name}...")

    assert x.dim() == 2, f"Expected 2-D tensor, got {x.dim()}-D"
    assert x.shape[1] == EMBED_DIM, (
        f"Expected embedding dim {EMBED_DIM}, got {x.shape[1]}"
    )
    assert x.dtype == torch.float32, (
        f"Expected float32, got {x.dtype}"
    )
    assert not torch.isnan(x).any(), "NaN values found in embeddings"
    assert not torch.isinf(x).any(), "Inf values found in embeddings"

    if expected_num_nodes is not None:
        assert x.shape[0] == expected_num_nodes, (
            f"Node count mismatch: embedding has {x.shape[0]}, "
            f"expected {expected_num_nodes}"
        )

    print(f"  [OK] shape={tuple(x.shape)}, dtype={x.dtype}, "
          f"min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")


def save_features(
    x: torch.Tensor,
    out_path: str,
    dataset_name: str,
    encoder: str,
    notes: str = "",
) -> None:
    """Save features plus metadata to a .pt file."""
    payload = {
        "x":           x,
        "dataset":     dataset_name,
        "encoder":     encoder,
        "dim":         x.shape[1],
        "num_nodes":   x.shape[0],
        "created_at":  _now_iso(),
        "script_hash": _git_hash(),
        "notes":       notes,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(payload, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Per-dataset text extraction functions
# ---------------------------------------------------------------------------
# Each function must:
#   1. Load the dataset from disk (via its native loader)
#   2. Extract a Python list of strings: one entry per node, in node-index order
#   3. Return (texts, expected_num_nodes, notes)
#
# IMPLEMENTED:  arxiv        (helpers + extract_texts_arxiv)
#               products     (helpers + extract_texts_products)
#               wn18rr       (helpers + extract_texts_wn18rr)
#               roman-empire (helpers + extract_texts_roman_empire)
# PLACEHOLDER:  pcba (raises NotImplementedError until strategy decided)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OGB processed-cache compatibility (PyTorch 2.6+)
# ---------------------------------------------------------------------------
# PyTorch 2.6 changed the default of torch.load to weights_only=True.
# OGB's cached processed files contain non-tensor objects (numpy arrays,
# dicts, etc.) that cannot be loaded under this restriction.  The helpers
# below provide a narrowly-scoped workaround for loading TRUSTED LOCAL
# OGB cache files only.  Do NOT copy this pattern into training code.
# ---------------------------------------------------------------------------

_OGB_WEIGHTS_ONLY_WARN = (
    "[WARN] OGB processed-cache load failed under PyTorch 2.6+ "
    "weights_only=True default.\n"
    "       Retrying with weights_only=False scoped to this trusted local "
    "OGB cache load only.\n"
    "       This is a local compatibility workaround — "
    "do NOT copy into training code blindly."
)


def _is_weights_only_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "weights_only" in msg
        or "weightsunpickler" in msg
        or "weights only" in msg
    )


@contextlib.contextmanager
def _ogb_load_compat():
    """
    Temporarily patch torch.load to inject weights_only=False, scoped to
    the body of this context manager only.  The original torch.load is
    restored on exit regardless of exceptions.

    Use ONLY for trusted local OGB processed-cache files.
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


# ---------------------------------------------------------------------------
# Arxiv text extraction helpers
# ---------------------------------------------------------------------------
# CONFIRMED (from audit_datasets.py run 2026-02-23):
#   - Loader: ogb.nodeproppred.NodePropPredDataset("ogbn-arxiv") succeeds
#   - num_nodes = 169,343
#   - node_feat shape = (169343, 128), dtype float32
#
# NOT CONFIRMED (must verify at runtime before trusting output):
#   - Column layout of nodeidx2paperid.csv.gz (assumed: [node_idx, paper_id])
#   - Format of titleabs.tsv (assumed: [paper_id, title, abstract], tab-sep, no header)
#   - That MAG paper IDs in nodeidx2paperid match those in titleabs.tsv
#   - End-to-end alignment: texts[i] corresponds to node i
# ---------------------------------------------------------------------------

def _find_file_candidates(candidates: list, description: str) -> str:
    """
    Try each path in candidates; return the first that exists on disk.
    Raises FileNotFoundError listing all searched paths if none is found.
    """
    for path in candidates:
        if os.path.isfile(path):
            return path
    searched = "\n    ".join(candidates)
    raise FileNotFoundError(
        f"Could not find {description}.  Searched:\n    {searched}\n"
        f"Download or place the file at one of the paths above."
    )


def _load_nodeidx2paperid(mapping_path: str) -> dict:
    """
    Load OGB's nodeidx2paperid mapping (CSV or CSV.gz).
    Returns dict: node_index (int) -> MAG paper_id (int).

    Assumed format (per OGB spec; NOT runtime-verified):
        node idx,paper id
        0,2982570
        1,1234567
        ...

    Column positions are used positionally; the header row is printed for inspection.
    """
    import csv
    import gzip as _gzip

    opener = _gzip.open if mapping_path.endswith(".gz") else open
    node_to_paper: dict = {}
    with opener(mapping_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        print(f"  [nodeidx2paperid] header row: {header}")
        for row in reader:
            if len(row) < 2:
                continue
            try:
                node_to_paper[int(row[0])] = int(row[1])
            except ValueError:
                continue   # skip any non-integer rows
    return node_to_paper


def _load_titleabs(text_path: str) -> dict:
    """
    Load the title+abstract file for ogbn-arxiv (TSV or TSV.gz).
    Returns dict: MAG paper_id (int) -> (title str, abstract str).

    Assumed format (NOT runtime-verified):
        [paper_id]\\t[title]\\t[abstract]   (no header line)

    Lines whose first column is not a valid integer are silently skipped
    (covers any accidental header lines).  Empty fields are kept as "".
    """
    import gzip as _gzip

    opener = _gzip.open if text_path.endswith(".gz") else open
    paper_to_ta: dict = {}
    with opener(text_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            try:
                paper_id = int(parts[0])
            except (ValueError, IndexError):
                continue   # skip header-like or malformed lines
            title    = parts[1].strip() if len(parts) > 1 else ""
            abstract = parts[2].strip() if len(parts) > 2 else ""
            paper_to_ta[paper_id] = (title, abstract)
    return paper_to_ta


def _build_arxiv_text_list(
    node_to_paper: dict,
    paper_to_ta: dict,
    num_nodes: int,
) -> tuple:
    """
    Build the final list of text strings, one per node, in node-index order.

    Text formatting policy (consistent with TAPE / GIANT precedents):
      - title AND abstract present: "Title: {title}\\nAbstract: {abstract}"
      - title only:                  title string (no prefix)
      - neither:                     "" (empty string; counted and reported)

    Returns (texts: list[str], stats: dict).
    stats keys: title_and_abs, title_only, empty, missing_mapping
    """
    texts: list = []
    stats: dict = {
        "title_and_abs":   0,
        "title_only":      0,
        "empty":           0,
        "missing_mapping": 0,
    }

    for i in range(num_nodes):
        paper_id = node_to_paper.get(i)
        if paper_id is None:
            texts.append("")
            stats["missing_mapping"] += 1
            stats["empty"] += 1
            continue
        title, abstract = paper_to_ta.get(paper_id, ("", ""))
        title    = title.strip()
        abstract = abstract.strip()
        if title and abstract:
            texts.append(f"Title: {title}\nAbstract: {abstract}")
            stats["title_and_abs"] += 1
        elif title:
            texts.append(title)
            stats["title_only"] += 1
        else:
            texts.append("")
            stats["empty"] += 1

    return texts, stats


def _print_arxiv_sample(texts: list, stats: dict, n: int = 3) -> None:
    """Print extraction counts summary and the first n non-empty example texts."""
    print(f"\n  --- arxiv text extraction summary ---")
    print(f"  Total texts:       {len(texts):>8,}")
    print(f"  title + abstract:  {stats['title_and_abs']:>8,}")
    print(f"  title only:        {stats['title_only']:>8,}")
    print(f"  empty:             {stats['empty']:>8,}")
    print(f"  missing mapping:   {stats['missing_mapping']:>8,}")
    shown = 0
    if n > 0:
        print(f"\n  First {n} non-empty examples (truncated to 300 chars each):")
        for idx, t in enumerate(texts):
            if t:
                preview = t[:300].replace("\n", "  //  ")
                print(f"    [node {idx}] {preview}")
                shown += 1
                if shown >= n:
                    break
    if shown == 0:
        print("  [WARN] No non-empty texts found — check mapping and text file paths.")


def _print_extraction_sample(dataset_name: str, texts: list, stats: dict, n: int = 3) -> None:
    """Generic extraction summary printer: stat counts + first n non-empty example texts."""
    print(f"\n  --- {dataset_name} text extraction summary ---")
    print(f"  Total texts:             {len(texts):>8,}")
    for key, val in stats.items():
        print(f"  {key:<24} {val:>8,}")
    shown = 0
    if n > 0:
        print(f"\n  First {n} non-empty examples (truncated to 300 chars each):")
        for idx, t in enumerate(texts):
            if t:
                preview = t[:300].replace("\n", "  //  ")
                print(f"    [node {idx}] {preview}")
                shown += 1
                if shown >= n:
                    break
    if shown == 0:
        print("  [WARN] No non-empty texts found — check mapping and text file paths.")


def extract_texts_arxiv(data_root: str) -> tuple[list[str], int | None, str]:
    """
    ogbn-arxiv: title + abstract concatenation via nodeidx2paperid + titleabs.tsv.

    CONFIRMED (from audit_datasets.py run 2026-02-23):
      - Loader: ogb.nodeproppred.NodePropPredDataset("ogbn-arxiv") succeeds
      - num_nodes = 169,343
      - node_feat shape = (169343, 128), dtype float32

    NOT CONFIRMED (alignment must be verified end-to-end before production use):
      - nodeidx2paperid.csv.gz column order matches OGB spec
      - titleabs.tsv MAG paper IDs match those in nodeidx2paperid
      - texts[i] corresponds to node i after mapping lookup

    NOTE: titleabs.tsv is NOT included in the standard OGB download.
    Download it separately:
        wget https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
    and place it under data/ogbn_arxiv/ (or data/) before calling this function.

    Raises:
        FileNotFoundError: if nodeidx2paperid or titleabs files are not found.
        AssertionError:    if len(texts) != num_nodes (alignment check failed).
        TypeError:         if any entry in texts is not a str.
    """
    from ogb.nodeproppred import NodePropPredDataset

    # Step 1: load dataset to get num_nodes and dataset root
    # Wrapped with a weights_only retry for PyTorch 2.6+ OGB cache compat.
    print(f"  Loading ogbn-arxiv from {data_root!r} ...")
    try:
        dataset = NodePropPredDataset(name="ogbn-arxiv", root=data_root)
        graph_dict, _ = dataset[0]
    except Exception as exc:
        if not _is_weights_only_error(exc):
            raise
        print(_OGB_WEIGHTS_ONLY_WARN)
        with _ogb_load_compat():
            dataset = NodePropPredDataset(name="ogbn-arxiv", root=data_root)
            graph_dict, _ = dataset[0]
    num_nodes = int(graph_dict["num_nodes"])
    dataset_root = dataset.root   # e.g., <data_root>/ogbn_arxiv
    print(f"  num_nodes = {num_nodes:,}   dataset_root = {dataset_root!r}")

    # Step 2: find nodeidx2paperid mapping (part of standard OGB download)
    mapping_candidates = [
        os.path.join(dataset_root, "mapping", "nodeidx2paperid.csv.gz"),
        os.path.join(dataset_root, "mapping", "nodeidx2paperid.csv"),
    ]
    mapping_path = _find_file_candidates(mapping_candidates, "nodeidx2paperid mapping")
    print(f"  mapping file: {mapping_path!r}")

    # Step 3: find titleabs text file (NOT in standard OGB download — place manually)
    text_candidates = [
        os.path.join(dataset_root, "titleabs.tsv.gz"),
        os.path.join(dataset_root, "titleabs.tsv"),
        os.path.join(dataset_root, "raw", "titleabs.tsv.gz"),
        os.path.join(dataset_root, "raw", "titleabs.tsv"),
        os.path.join(data_root, "titleabs.tsv.gz"),
        os.path.join(data_root, "titleabs.tsv"),
    ]
    text_path = _find_file_candidates(
        text_candidates,
        "titleabs text file "
        "(download: https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz)",
    )
    print(f"  text file:    {text_path!r}")

    # Step 4: load files
    print("  Loading nodeidx2paperid ...")
    node_to_paper = _load_nodeidx2paperid(mapping_path)
    print(f"  Loaded {len(node_to_paper):,} node→paper entries.")

    print("  Loading titleabs ...")
    paper_to_ta = _load_titleabs(text_path)
    print(f"  Loaded {len(paper_to_ta):,} paper text entries.")

    # Step 5: build text list in node-index order
    print("  Building text list ...")
    texts, stats = _build_arxiv_text_list(node_to_paper, paper_to_ta, num_nodes)

    # Step 6: strict validation
    assert len(texts) == num_nodes, (
        f"ALIGNMENT ERROR: len(texts)={len(texts):,} != num_nodes={num_nodes:,}"
    )
    bad_types = [i for i, t in enumerate(texts) if not isinstance(t, str)]
    if bad_types:
        raise TypeError(
            f"Non-string entries at node indices {bad_types[:10]} "
            f"(first 10 of {len(bad_types)} total)"
        )
    if stats["missing_mapping"] > 0:
        print(
            f"  [WARN] {stats['missing_mapping']:,} nodes have no nodeidx2paperid entry "
            f"— their texts will be empty strings."
        )
    if stats["empty"] > 0:
        print(f"  [WARN] {stats['empty']:,} nodes will have empty text strings.")

    # Step 7: print sample for manual inspection
    _print_arxiv_sample(texts, stats)

    notes = (
        f"title_and_abs={stats['title_and_abs']} "
        f"title_only={stats['title_only']} "
        f"empty={stats['empty']} "
        f"missing_mapping={stats['missing_mapping']}"
    )
    return texts, num_nodes, notes


# ---------------------------------------------------------------------------
# ogbn-products text extraction helpers
# ---------------------------------------------------------------------------
# NOT CONFIRMED (OGB nodeidx2asin spec assumed):
#   - That row order in Approach-A text files matches node indices
#   - That ASIN order in nodeidx2asin.csv.gz matches GNN loader node indices
#   Verify against the specific GNN repo's data loader before production use.
# ---------------------------------------------------------------------------

def _load_nodeidx2asin(mapping_path: str) -> list:
    """
    Load OGB's nodeidx2asin mapping (CSV or CSV.gz).
    Returns list: asin_list[node_index] = asin_str.

    Assumed format (OGB spec):
        node idx,asin
        0,B000Q3NXOO
        1,B000H4MH4O
        ...
    """
    import csv
    import gzip as _gzip

    opener = _gzip.open if mapping_path.endswith(".gz") else open
    rows: list = []
    with opener(mapping_path, "rt", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        print(f"  [products] nodeidx2asin header: {header}")
        for row in reader:
            if len(row) < 2:
                continue
            try:
                idx = int(row[0])
                asin = row[1].strip()
                while len(rows) <= idx:
                    rows.append("")
                rows[idx] = asin
            except (ValueError, IndexError):
                continue
    return rows


def _load_product_text_mapped(csv_path: str) -> dict:
    """
    Load a product text CSV keyed by ASIN (Approach B: ASIN-mapped file).
    Returns dict: asin (str) -> text (str).

    Assumed format:
        asin,title,description   (or asin,text)
    If both title and description columns are present, concatenates them.
    """
    import csv
    import gzip as _gzip

    opener = _gzip.open if csv_path.endswith(".gz") else open
    asin_to_text: dict = {}
    with opener(csv_path, "rt", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        print(f"  [products] text file columns: {header}")
        for row in reader:
            asin = (row.get("asin") or "").strip()
            if not asin:
                continue
            title = (row.get("title") or "").strip()
            desc  = (row.get("description") or row.get("text") or "").strip()
            if title and desc:
                asin_to_text[asin] = f"{title}: {desc}"
            elif title:
                asin_to_text[asin] = title
            elif desc:
                asin_to_text[asin] = desc
    return asin_to_text


def extract_texts_products(data_root: str) -> tuple[list[str], int | None, str]:
    """
    ogbn-products: product title + description.

    Step 1: Load OGB NodePropPredDataset to obtain num_nodes (with weights_only retry).
    Step 2a (Approach A): Row-order text file where row i = node i's text.
    Step 2b (Approach B): nodeidx2asin.csv.gz + ASIN-keyed text file.

    NOT CONFIRMED (alignment assumed from OGB spec):
      Row order in Approach-A files or nodeidx2asin ASIN order assumed to
      match GNN loader node indices.  Verify before production use.

    Raises:
        FileNotFoundError: if no text source is found after both approaches.
        AssertionError:    if len(texts) != num_nodes.
    """
    from ogb.nodeproppred import NodePropPredDataset

    # Step 1: load OGB for num_nodes (with weights_only retry for PyTorch 2.6+)
    print(f"  [products] Loading ogbn-products from {data_root!r} ...")
    try:
        dataset = NodePropPredDataset(name="ogbn-products", root=data_root)
        graph_dict, _ = dataset[0]
    except Exception as exc:
        if not _is_weights_only_error(exc):
            raise
        print(_OGB_WEIGHTS_ONLY_WARN)
        with _ogb_load_compat():
            dataset = NodePropPredDataset(name="ogbn-products", root=data_root)
            graph_dict, _ = dataset[0]
    num_nodes = int(graph_dict["num_nodes"])
    dataset_root = dataset.root   # e.g., <data_root>/ogbn_products
    print(f"  [products] num_nodes = {num_nodes:,}   dataset_root = {dataset_root!r}")

    # Pre-define all candidate paths (needed for error reporting)
    row_order_candidates = [
        os.path.join(dataset_root, "raw", "node-feat", "text", "product_text.csv"),
        os.path.join(dataset_root, "raw", "node-feat", "text", "titlecat.csv"),
        os.path.join(dataset_root, "product_text.csv"),
        os.path.join(data_root, "product_text.csv"),
        os.path.join(data_root, "ogbn_products_text.csv"),
        os.path.join(dataset_root, "raw", "Amazon_product.csv"),
    ]
    mapping_candidates = [
        os.path.join(dataset_root, "mapping", "nodeidx2asin.csv.gz"),
        os.path.join(dataset_root, "mapping", "nodeidx2asin.csv"),
    ]
    asin_text_candidates = [
        os.path.join(dataset_root, "raw", "Amazon_product.csv"),
        os.path.join(data_root, "Amazon_product.csv"),
        os.path.join(dataset_root, "raw", "Amazon_product.csv.gz"),
    ]

    texts = None
    approach_used = "none"

    # --- Approach A: row-order text file ---
    for path in row_order_candidates:
        if os.path.isfile(path):
            print(f"  [products] Approach A: row-order text file {path!r}")
            import csv as _csv
            import gzip as _gzip
            opener = _gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
                reader = _csv.reader(fh)
                header = next(reader, None)
                print(f"  [products] row-order file header: {header}")
                raw_rows = list(reader)
            if raw_rows and len(raw_rows[0]) >= 2:
                texts = [", ".join(c.strip() for c in row[1:] if c.strip())
                         for row in raw_rows]
            elif raw_rows:
                texts = [row[0].strip() for row in raw_rows]
            if texts is not None:
                approach_used = f"row-order:{os.path.basename(path)}"
            break

    # --- Approach B: nodeidx2asin + ASIN-keyed file ---
    if texts is None:
        mapping_path = None
        for p in mapping_candidates:
            if os.path.isfile(p):
                mapping_path = p
                break
        asin_text_path = None
        for p in asin_text_candidates:
            if os.path.isfile(p):
                asin_text_path = p
                break

        if mapping_path is not None and asin_text_path is not None:
            print(f"  [products] Approach B: ASIN mapping {mapping_path!r}")
            print(f"             text file:  {asin_text_path!r}")
            asin_list = _load_nodeidx2asin(mapping_path)
            asin_to_text = _load_product_text_mapped(asin_text_path)
            texts = [""] * num_nodes
            for i, asin in enumerate(asin_list):
                if i >= num_nodes:
                    break
                texts[i] = asin_to_text.get(asin, "")
            approach_used = f"asin-mapped:{os.path.basename(asin_text_path)}"
        else:
            missing = []
            if mapping_path is None:
                missing.append("nodeidx2asin not found")
            if asin_text_path is None:
                missing.append("ASIN text file not found")
            print(f"  [products] Approach B skipped: {'; '.join(missing)}")

    if texts is None:
        all_searched = row_order_candidates + mapping_candidates + asin_text_candidates
        searched_str = "\n    ".join(all_searched)
        raise FileNotFoundError(
            "ogbn-products: no text source found.\n"
            "  Tried Approach A (row-order text files) and "
            "Approach B (nodeidx2asin + ASIN-keyed text).\n"
            f"  Searched:\n    {searched_str}\n"
            "  Place a product text file at one of the paths above."
        )

    # Validation
    assert len(texts) == num_nodes, (
        f"ALIGNMENT ERROR: len(texts)={len(texts):,} != num_nodes={num_nodes:,}"
    )
    bad_types = [i for i, t in enumerate(texts) if not isinstance(t, str)]
    if bad_types:
        raise TypeError(
            f"Non-string entries at node indices {bad_types[:10]} "
            f"(first 10 of {len(bad_types)} total)"
        )
    empty_count = sum(1 for t in texts if not t)
    if empty_count:
        print(f"  [WARN] {empty_count:,} nodes have empty text strings.")

    stats = {"non-empty": num_nodes - empty_count, "empty": empty_count}
    _print_extraction_sample("products", texts, stats)

    notes = (
        f"product text via {approach_used}; alignment NOT CONFIRMED "
        f"(OGB nodeidx2asin spec assumed); empty={empty_count}"
    )
    return texts, num_nodes, notes


# ---------------------------------------------------------------------------
# WN18RR text extraction helpers
# ---------------------------------------------------------------------------
# NOT CONFIRMED (alignment assumed via entity2id.txt ordering):
#   - That entity2id.txt indices match the GNN loader's node indices
#   - That entity2text.txt descriptions cover every entity in entity2id
#   CRITICAL: entity2id.txt is REQUIRED; function raises FileNotFoundError
#             rather than guessing alignment silently.
# ---------------------------------------------------------------------------

def _load_wn18rr_entity2id(path: str) -> dict:
    """
    Load entity2id.txt (standard KG dataset convention).
    Returns dict: entity_name (str) -> node_index (int).

    Assumed format:
        <entity_name>\\t<node_index>   (or whitespace-separated)
    First line may be the total entity count (integer alone) — skipped if so.
    """
    entity2id: dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                # Possibly a count-only header line; skip it
                try:
                    int(parts[0])
                    continue
                except ValueError:
                    pass
            if len(parts) < 2:
                continue
            try:
                node_idx = int(parts[-1])
            except ValueError:
                continue
            entity_name = " ".join(parts[:-1])
            entity2id[entity_name] = node_idx
    return entity2id


def _load_wn18rr_entity_texts(path: str) -> dict:
    """
    Load a WN18RR entity text file (entity2text.txt or similar).
    Returns dict: entity_name (str) -> description (str).

    Assumed format:
        <entity_name>\\t<text description>
    """
    entity_texts: dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "\t" in line:
                entity_name, _, text = line.partition("\t")
                entity_texts[entity_name.strip()] = text.strip()
            else:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    entity_texts[parts[0]] = parts[1].strip()
    return entity_texts


def extract_texts_wn18rr(data_root: str) -> tuple[list[str], int | None, str]:
    """
    WN18RR: entity name + description text.

    REQUIRES entity2id.txt to prove node-index alignment.
    Raises FileNotFoundError if entity2id.txt is not found — safe failure,
    never silently guesses ordering.

    NOT CONFIRMED (alignment assumed):
      entity2id.txt indices are assumed to match the GNN loader's node order.
      Verify against the specific GNN repo's data loader before production use.

    Text format per node:  "<entity_name>: <description>"
    If no text file is available: entity_name only (with a WARN).

    Raises:
        FileNotFoundError: entity2id.txt not found (required for safe alignment).
        AssertionError:    len(texts) != num_nodes.
    """
    # --- Find entity2id.txt (required) ---
    entity2id_candidates = [
        os.path.join(data_root, "WN18RR", "entity2id.txt"),
        os.path.join(data_root, "wn18rr", "entity2id.txt"),
        os.path.join(data_root, "entity2id.txt"),
        os.path.join(data_root, "WN18RR", "raw", "entity2id.txt"),
    ]
    entity2id_path = None
    for p in entity2id_candidates:
        if os.path.isfile(p):
            entity2id_path = p
            break
    if entity2id_path is None:
        searched = "\n    ".join(entity2id_candidates)
        raise FileNotFoundError(
            "WN18RR: entity2id.txt not found (required for safe node alignment).\n"
            f"  Searched:\n    {searched}\n"
            "  Place entity2id.txt at one of the paths above before calling this function.\n"
            "  NEVER guess entity ordering without a confirmed index mapping."
        )
    print(f"  [wn18rr] entity2id file: {entity2id_path!r}")
    entity2id = _load_wn18rr_entity2id(entity2id_path)
    num_nodes = len(entity2id)
    print(f"  [wn18rr] Loaded {num_nodes:,} entity→id mappings.")

    # --- Find entity text file (optional but strongly preferred) ---
    text_candidates = [
        os.path.join(data_root, "WN18RR", "entity2text.txt"),
        os.path.join(data_root, "wn18rr", "entity2text.txt"),
        os.path.join(data_root, "entity2text.txt"),
        os.path.join(data_root, "WN18RR", "raw", "entity2text.txt"),
        os.path.join(data_root, "WN18RR", "entity_text.txt"),
        os.path.join(data_root, "WN18RR", "entity_gloss.txt"),
    ]
    entity_text_path = None
    for p in text_candidates:
        if os.path.isfile(p):
            entity_text_path = p
            break
    if entity_text_path is not None:
        print(f"  [wn18rr] entity text file: {entity_text_path!r}")
        entity_texts = _load_wn18rr_entity_texts(entity_text_path)
        print(f"  [wn18rr] Loaded {len(entity_texts):,} entity text entries.")
    else:
        print("  [WARN] No entity text file found; using entity names only.")
        entity_texts = {}

    # --- Build text list in node-index order ---
    texts: list = [""] * num_nodes
    missing_text = 0
    for entity_name, node_idx in entity2id.items():
        if node_idx >= num_nodes:
            continue
        description = entity_texts.get(entity_name, "")
        if description:
            texts[node_idx] = f"{entity_name}: {description}"
        else:
            texts[node_idx] = entity_name
            missing_text += 1

    # Validation
    assert len(texts) == num_nodes, (
        f"ALIGNMENT ERROR: len(texts)={len(texts):,} != num_nodes={num_nodes:,}"
    )
    bad_types = [i for i, t in enumerate(texts) if not isinstance(t, str)]
    if bad_types:
        raise TypeError(
            f"Non-string entries at node indices {bad_types[:10]} "
            f"(first 10 of {len(bad_types)} total)"
        )
    empty_count = sum(1 for t in texts if not t)
    if missing_text:
        print(f"  [WARN] {missing_text:,} entities had no text description (entity name used).")
    if empty_count:
        print(f"  [WARN] {empty_count:,} nodes have empty text strings.")

    stats = {
        "with description": num_nodes - missing_text - empty_count,
        "entity name only": missing_text,
        "empty":            empty_count,
    }
    _print_extraction_sample("wn18rr", texts, stats)

    notes = (
        "entity name + description; alignment NOT CONFIRMED "
        f"(entity2id.txt ordering assumed); "
        f"entity_name_only={missing_text} empty={empty_count}"
    )
    return texts, num_nodes, notes


# ---------------------------------------------------------------------------
# Roman-Empire text extraction helpers
# ---------------------------------------------------------------------------
# NOT CONFIRMED (assumed node-index order):
#   - Whether HeterophilousGraphDataset exposes data.raw_texts in all versions
#   - Whether the downloaded .npz contains a usable text field
#   - Whether a separate plain-text file is provided in node-index order
# ---------------------------------------------------------------------------

def _load_roman_empire_texts_from_npz(npz_path: str) -> list | None:
    """
    Try to load node text strings from a Roman-Empire .npz file.
    Returns a list of strings if a text field is found, else None.

    Probed keys (in priority order): raw_texts, texts, node_text, node_texts, text
    """
    import numpy as np
    data = np.load(npz_path, allow_pickle=True)
    for key in ("raw_texts", "texts", "node_text", "node_texts", "text"):
        if key in data:
            arr = data[key]
            texts = [str(t) for t in arr]
            print(f"  [roman-empire] Found text field {key!r} in {os.path.basename(npz_path)!r}")
            return texts
    print(f"  [roman-empire] No text field found in {os.path.basename(npz_path)!r}.")
    print(f"  Available keys: {list(data.keys())}")
    return None


def extract_texts_roman_empire(data_root: str) -> tuple[list[str], int | None, str]:
    """
    Roman-Empire: one Wikipedia sentence per node.

    Three-attempt fallback strategy:
      1. PyG HeterophilousGraphDataset → data.raw_texts (list of strings)
      2. Probe .npz files under data_root for a raw_texts/texts field
      3. Plain text file with one sentence per line in node-index order

    NOT CONFIRMED (alignment assumed — NOT verified end-to-end):
      The dataset stores node sentences in the same order as node indices.

    Raises:
        RuntimeError: if all three attempts fail to find any text data.
    """
    texts = None
    num_nodes = None

    # --- Attempt 1: PyG HeterophilousGraphDataset ---
    print("  [roman-empire] Attempt 1: PyG HeterophilousGraphDataset ...")
    try:
        from torch_geometric.datasets import HeterophilousGraphDataset
        dataset = HeterophilousGraphDataset(root=data_root, name="Roman-empire")
        graph_data = dataset[0]
        num_nodes = graph_data.num_nodes
        if hasattr(graph_data, "raw_texts") and graph_data.raw_texts is not None:
            texts = list(graph_data.raw_texts)
            print(f"  [roman-empire] data.raw_texts found: {len(texts):,} entries.")
        else:
            print("  [roman-empire] data.raw_texts not available in this PyG version.")
    except Exception as exc:
        print(f"  [roman-empire] Attempt 1 failed: {exc}")

    # --- Attempt 2: probe .npz files ---
    if texts is None:
        print("  [roman-empire] Attempt 2: probing .npz files under data_root ...")
        npz_candidates = []
        for dirpath, _, filenames in os.walk(data_root):
            for fn in filenames:
                if fn.endswith(".npz") and "roman" in fn.lower():
                    npz_candidates.append(os.path.join(dirpath, fn))
        # Also check common download paths explicitly
        for extra in [
            os.path.join(data_root, "Roman-empire", "raw", "roman_empire.npz"),
            os.path.join(data_root, "Roman-empire", "raw", "data.npz"),
            os.path.join(data_root, "roman_empire.npz"),
        ]:
            if extra not in npz_candidates and os.path.isfile(extra):
                npz_candidates.append(extra)
        for npz_path in npz_candidates:
            print(f"  [roman-empire] Probing: {npz_path!r}")
            result = _load_roman_empire_texts_from_npz(npz_path)
            if result is not None:
                texts = result
                if num_nodes is None:
                    num_nodes = len(texts)
                break
        if texts is None:
            print(f"  [roman-empire] No usable .npz found (searched {len(npz_candidates)} files).")

    # --- Attempt 3: plain text file (one sentence per line) ---
    if texts is None:
        print("  [roman-empire] Attempt 3: looking for plain text file ...")
        txt_candidates = [
            os.path.join(data_root, "Roman-empire", "raw", "node_text.txt"),
            os.path.join(data_root, "Roman-empire", "raw", "raw_texts.txt"),
            os.path.join(data_root, "roman_empire_texts.txt"),
            os.path.join(data_root, "node_text.txt"),
        ]
        for path in txt_candidates:
            if os.path.isfile(path):
                print(f"  [roman-empire] Reading: {path!r}")
                with open(path, "r", encoding="utf-8") as fh:
                    texts = [line.rstrip("\n") for line in fh]
                if num_nodes is None:
                    num_nodes = len(texts)
                break
        if texts is None:
            searched = "\n    ".join(txt_candidates)
            raise RuntimeError(
                "roman-empire: all three extraction attempts failed.\n"
                "  Attempt 1: HeterophilousGraphDataset.raw_texts unavailable\n"
                "  Attempt 2: no usable .npz file found\n"
                "  Attempt 3: no plain text file found at:\n"
                f"    {searched}\n"
                "  Place a text file with one sentence per line in node-index order "
                "at one of the paths above."
            )

    # Validation
    assert len(texts) > 0, "roman-empire: extracted zero texts"
    bad_types = [i for i, t in enumerate(texts) if not isinstance(t, str)]
    if bad_types:
        raise TypeError(
            f"Non-string entries at node indices {bad_types[:10]} "
            f"(first 10 of {len(bad_types)} total)"
        )
    if num_nodes is not None:
        assert len(texts) == num_nodes, (
            f"ALIGNMENT ERROR: len(texts)={len(texts):,} != num_nodes={num_nodes:,}"
        )
    empty_count = sum(1 for t in texts if not t)
    if empty_count:
        print(f"  [WARN] {empty_count:,} nodes have empty text strings.")

    stats = {"non-empty": len(texts) - empty_count, "empty": empty_count}
    _print_extraction_sample("roman-empire", texts, stats)

    notes = (
        "Wikipedia sentence per node; alignment NOT CONFIRMED "
        f"(assumed node-index order); empty={empty_count}"
    )
    return texts, num_nodes, notes


def extract_texts_pcba(data_root: str) -> tuple[list[str], int | None, str]:
    """
    PCBA: NO natural-language text available per atom node.

    Strategy is undecided.  This function is a placeholder that raises until
    a strategy is approved by the PI.

    Options:
      - raw passthrough: skip SBERT entirely, keep original 9-dim atom features
      - SMILES → SBERT:  one embedding per graph (not per atom); needs design decision
      - zero 768-dim:    not recommended (zero features break pretraining)

    Do NOT implement until a strategy is confirmed.
    """
    raise NotImplementedError(
        "PCBA text extraction is not applicable — no per-node text exists.\n"
        "Decide on a strategy (raw passthrough / SMILES-level / other) before "
        "calling this function.  See notes/dataset_audit.md PCBA section."
    )


# ---------------------------------------------------------------------------
# Dataset dispatch table
# ---------------------------------------------------------------------------

# Text-available datasets — safe to pass to SBERT.
TEXT_DATASETS = ["arxiv", "products", "wn18rr", "roman-empire"]

# Full extractor map.  pcba raises NotImplementedError until strategy is decided.
EXTRACTORS = {
    "arxiv":        extract_texts_arxiv,
    "products":     extract_texts_products,
    "wn18rr":       extract_texts_wn18rr,
    "roman-empire": extract_texts_roman_empire,
    "pcba":         extract_texts_pcba,    # raises until strategy is decided
}

# Canonical dataset names for metadata
CANONICAL_NAMES = {
    "arxiv":        "ogbn-arxiv",
    "products":     "ogbn-products",
    "wn18rr":       "WN18RR",
    "roman-empire": "Roman-Empire",
    "pcba":         "ogbg-molpcba",
}


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_key: str,
    data_root: str,
    out_dir: str,
    encoder: str,
    force: bool = False,
) -> None:
    """Run the full pipeline for one dataset: extract → encode → validate → save."""
    canonical = CANONICAL_NAMES[dataset_key]
    out_path = os.path.join(out_dir, f"{dataset_key}_sbert.pt")

    print(f"\n{'=' * 60}")
    print(f"  Processing: {canonical}")
    print(f"{'=' * 60}")

    if os.path.exists(out_path) and not force:
        print(f"  [SKIP] {out_path} already exists.  Use --force to overwrite.")
        return

    # Step 1: extract texts
    print(f"  Extracting texts from {data_root} ...")
    extractor = EXTRACTORS[dataset_key]
    texts, expected_num_nodes, notes = extractor(data_root)
    print(f"  Extracted {len(texts):,} texts.")

    # Step 2: encode
    x = encode_texts(texts, model_name=encoder)

    # Step 3: validate
    validate_features(x, canonical, expected_num_nodes)

    # Step 4: save
    save_features(x, out_path, canonical, encoder, notes)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess SBERT features for GFM Safety datasets"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="all-text",
        choices=list(EXTRACTORS.keys()) + ["all-text", "all"],
        help=(
            "Dataset to process.  "
            "'all-text' runs the four text-available datasets (arxiv, products, wn18rr, roman-empire).  "
            "'all' additionally includes pcba (which will raise until its strategy is decided).  "
            "Default: all-text"
        ),
    )
    parser.add_argument(
        "--data-root", "-r",
        default="data",
        help="Root directory where datasets are stored (default: data/)",
    )
    parser.add_argument(
        "--out-dir", "-o",
        default="data",
        help="Output directory for .pt files (default: data/)",
    )
    parser.add_argument(
        "--encoder", "-e",
        default=SBERT_MODEL_DEFAULT,
        help=f"SBERT model name (default: {SBERT_MODEL_DEFAULT})",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing .pt files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate text extraction without encoding or saving.  "
            "Supported for all text datasets: arxiv, products, wn18rr, roman-empire."
        ),
    )
    args = parser.parse_args()

    if args.dry_run:
        if args.dataset not in TEXT_DATASETS:
            print(
                f"[ERROR] --dry-run supports text datasets only: {TEXT_DATASETS}\n"
                f"        Got: {args.dataset!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"\n=== DRY RUN: {args.dataset} — text extraction only (no encoding or saving) ===\n")
        extractor = EXTRACTORS[args.dataset]
        texts, num_nodes, notes = extractor(args.data_root)
        print(f"\n[OK] Dry run complete.")
        print(f"     len(texts) = {len(texts):,}   num_nodes = {num_nodes}")
        print(f"     notes: {notes}")
        return

    if args.dataset == "all":
        datasets = list(EXTRACTORS.keys())       # all five, pcba will raise
    elif args.dataset == "all-text":
        datasets = list(TEXT_DATASETS)           # only the four text-available datasets
    else:
        datasets = [args.dataset]

    errors = []
    for name in datasets:
        try:
            process_dataset(
                name,
                data_root=args.data_root,
                out_dir=args.out_dir,
                encoder=args.encoder,
                force=args.force,
            )
        except NotImplementedError as exc:
            print(f"\n  [SKIP] {name}: {exc}", file=sys.stderr)
            errors.append((name, "NotImplementedError"))
        except Exception as exc:
            print(f"\n  [ERROR] {name}: {exc}", file=sys.stderr)
            errors.append((name, str(exc)))

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}")
    if errors:
        print("  Datasets with errors or skips:")
        for name, reason in errors:
            print(f"    {name:<20} {reason}")
    else:
        print("  All datasets processed successfully.")


if __name__ == "__main__":
    main()
