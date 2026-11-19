from importlib import import_module

_EXPORTS = {"GPF": "GPF", "GPF_plus": "GPF", "TaskUnifiedPrompt": "TUP",
            "GPPTPrompt": "GPPTPrompt", "Gprompt": "GPrompt"}
_EXPORTS.update({name: "AllInOnePrompt" for name in ("HeavyPrompt", "FrontAndHead", "LightPrompt")})
_EXPORTS.update({name: "SUPT" for name in ("DiffPoolPrompt", "SAGPoolPrompt")})
_EXPORTS.update({name: "MultiGprompt" for name in (
    "downprompt", "DGI", "GraphCL", "Lp", "AvgReadout", "Discriminator", "LogReg",
    "Lpprompt", "DGIprompt", "GraphCLprompt", "weighted_feature", "weighted_prompt",
    "downstreamprompt", "featureprompt", "GcnLayers")})
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


