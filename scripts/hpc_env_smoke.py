"""hpc_env_smoke.py — Quick import-level health check for the gfm_safety conda env.

Usage:
    python scripts/hpc_env_smoke.py

Checks critical runtime modules without performing any training, dataset download,
or GPU computation.  Exit code 0 = all pass, 1 = one or more failures.
"""
from __future__ import annotations

import importlib
import sys


# ---------------------------------------------------------------------------
# Registry of checks
# Each entry: (display_label, module_to_import, optional_version_attr)
# ---------------------------------------------------------------------------
_CHECKS: list[tuple[str, str, str | None]] = [
    ("torch",           "torch",            "__version__"),
    ("dgl",             "dgl",              "__version__"),
    ("torch_geometric", "torch_geometric",  "__version__"),
    ("ogb",             "ogb",              "__version__"),
    ("tensorboardX",    "tensorboardX",     "__version__"),
    ("tensorboard",     "tensorboard",      "__version__"),
    ("sklearn",         "sklearn",          "__version__"),
    ("numpy",           "numpy",            "__version__"),
    ("scipy",           "scipy",            "__version__"),
    ("pandas",          "pandas",           "__version__"),
    ("networkx",        "networkx",         "__version__"),
    ("tqdm",            "tqdm",             "__version__"),
    ("yaml (pyyaml)",   "yaml",             "__version__"),
]

# ---------------------------------------------------------------------------
# Additional spot-checks that go one level deeper
# ---------------------------------------------------------------------------
_SUBMODULE_CHECKS: list[tuple[str, str]] = [
    ("ogb.nodeproppred",         "ogb.nodeproppred"),
    ("ogb.graphproppred",        "ogb.graphproppred"),
    ("tensorboardX.SummaryWriter", "tensorboardX"),
    ("dgl.dataloading",          "dgl.dataloading"),
    ("torch_geometric.nn",       "torch_geometric.nn"),
    ("sklearn.linear_model",     "sklearn.linear_model"),
]


def _check(label: str, module: str, version_attr: str | None) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, version_attr, "?") if version_attr else ""
        return True, str(version)
    except Exception as exc:
        return False, str(exc)


def _subcheck(label: str, module: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    width = max(len(label) for label, *_ in _CHECKS + _SUBMODULE_CHECKS) + 2
    failures = 0

    print("=" * 60)
    print("  GFM-Safety HPC Environment Smoke Test")
    print("=" * 60)

    print("\n[core modules]")
    for label, module, version_attr in _CHECKS:
        ok, info = _check(label, module, version_attr)
        status = "PASS" if ok else "FAIL"
        version_str = f"  ({info})" if ok and info and info != "?" else ""
        error_str = f"  ← {info}" if not ok else ""
        print(f"  {status}  {label:<{width}}{version_str}{error_str}")
        if not ok:
            failures += 1

    print("\n[submodule spot-checks]")
    for label, module in _SUBMODULE_CHECKS:
        ok, info = _subcheck(label, module)
        status = "PASS" if ok else "FAIL"
        error_str = f"  ← {info}" if not ok else ""
        print(f"  {status}  {label:<{width}}{error_str}")
        if not ok:
            failures += 1

    print("\n[torch device probe]")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        print(f"  PASS  cuda_available={cuda_available}  devices={device_count}  name={device_name!r}")
    except Exception as exc:
        print(f"  FAIL  torch device probe: {exc}")
        failures += 1

    print("\n" + "=" * 60)
    if failures == 0:
        print("  RESULT: ALL PASS")
    else:
        print(f"  RESULT: {failures} FAILURE(S)")
    print("=" * 60)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
