from importlib import import_module

_EXPORTS = {name: name for name in ("act", "mkdir", "edge_index_to_sparse_matrix",
    "constraint", "NegativeEdge", "prepare_structured_data", "get_args")}
_EXPORTS.update({name: "perturbation" for name in ("graph_views", "drop_nodes", "mask_nodes", "permute_edges")})
_EXPORTS.update({name: "center_embedding" for name in ("center_embedding", "distance2center")})
_EXPORTS.update({name: "loss" for name in ("Gprompt_tuning_loss", "Gprompt_link_loss")})
_EXPORTS.update(seed_everything="seed", print_model_parameters="print_para")
_EXPORTS.update({name: "contrast" for name in ("generate_random_model_output", "contrastive_loss", "generate_corrupted_graph")})
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
