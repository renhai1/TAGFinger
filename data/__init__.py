"""Lazy dataset exports; legacy loaders do not block the TAGFinger path."""
from importlib import import_module

_EXPORTS = {"induced_graphs": "induced_graph", "split_induced_graphs": "induced_graph",
            "graph_split": "graph_split", "GraphDataset": "Dataset"}
_EXPORTS.update({name: "tag_data" for name in ("load_tag", "mix_tag_datasets", "TAG_DATASETS")})
_EXPORTS.update({name: "load4data" for name in (
    "load4graph", "load4link_prediction_single_graph", "load4node",
    "load4link_prediction_multi_graph", "NodePretrain", "node_sample_and_save",
    "graph_sample_and_save")})
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
