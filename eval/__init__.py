from .registry import REGISTRY, DatasetAdapter, MetricSpec, ReadinessGate

__all__ = [
    "DatasetAdapter",
    "GraphHead",
    "LinkHead",
    "MetricSpec",
    "NodeHead",
    "REGISTRY",
    "ReadinessGate",
    "export_encoder_checkpoint",
    "load_encoder",
]


def __getattr__(name: str):
    if name == "export_encoder_checkpoint":
        from .checkpoint import export_encoder_checkpoint

        return export_encoder_checkpoint
    if name == "load_encoder":
        from .load_encoder import load_encoder

        return load_encoder
    if name in {"GraphHead", "LinkHead", "NodeHead"}:
        from .heads import GraphHead, LinkHead, NodeHead

        return {
            "GraphHead": GraphHead,
            "LinkHead": LinkHead,
            "NodeHead": NodeHead,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
