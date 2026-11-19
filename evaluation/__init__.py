from importlib import import_module

_EXPORTS = {"GpromptEva": "GpromptEva", "GNNGraphEva": "GNNEva", "GNNNodeEva": "GNNEva",
            "GPFEva": "GPFEva", "AllInOneEva": "AllInOneEva", "GPPTEva": "GPPTEva",
            "GPPTGraphEva": "GPPTEva", "MultiGpromptEva": "MultiGpromptEva"}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
