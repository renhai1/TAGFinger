"""Load only the model requested; optional backbones need not block imports."""
from importlib import import_module

_EXPORTS = {name: name for name in ("GAT", "GCN", "GCov", "GIN", "GraphSAGE",
                                   "GraphTransformer", "NodeClassifier")}
_EXPORTS.update(GraphClassifier="GraphHead", RGCN="RobustGCN")
_EXPORTS.update({name: "SAKA" for name in ("BlackBoxGNN", "GAGGenerator",
                                         "dac_loss", "dvc_loss", "train_saka")})
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
