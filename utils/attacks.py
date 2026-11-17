"""Adaptive attack strategies against dataset ownership verification (RQ3).

The attacker sanitizes the graph to weaken fingerprint signals:
  - Prune: removes edges whose endpoints are least similar in the attribute
    space; the pruning ratio is swept up to 80% (Figure 5);
  - OD: outlier detection defense of DPGBA (Zhang et al., KDD 2024) — nodes
    whose attributes deviate most from their neighborhood are flagged as
    poisoned and removed;
  - RIGBD: robust influence-based graph backdoor defense of SPEAR (Ding et
    al., WWW 2025) — edges connecting dissimilar, low-degree endpoints carry
    the highest suspicious influence and are removed first;
  - MAD: multidimensional sanitization (Zhao et al., IJCAI 2025) — combines
    attribute smoothing, edge pruning, and outlier-node removal.

All attacks return a new Data object; the protected dataset itself is never
modified by TAGFinger (non-intrusive verification).
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def _remove_nodes(data, keep_mask):
    keep = keep_mask.nonzero(as_tuple=False).view(-1)
    edge_index, _ = subgraph(keep, data.edge_index, relabel_nodes=True,
                             num_nodes=data.num_nodes)
    out = Data(x=data.x[keep], edge_index=edge_index, y=data.y[keep])
    for key in ("train_mask", "val_mask", "test_mask", "owner_mask"):
        if hasattr(data, key) and getattr(data, key) is not None:
            setattr(out, key, getattr(data, key)[keep])
    if hasattr(data, "raw_texts") and data.raw_texts is not None:
        out.raw_texts = [data.raw_texts[int(i)] for i in keep]
    for key in ("num_classes", "owner_num_classes", "dataset_name"):
        if hasattr(data, key):
            setattr(out, key, getattr(data, key))
    return out


def _edge_similarity(data):
    src, dst = data.edge_index[0], data.edge_index[1]
    return F.cosine_similarity(data.x[src], data.x[dst], dim=1)


def _prune_edges(data, ratio):
    num_edges = data.edge_index.size(1)
    num_keep = max(1, int((1.0 - ratio) * num_edges))
    sim = _edge_similarity(data)
    keep = sim.topk(num_keep).indices
    out = data.clone()
    out.edge_index = data.edge_index[:, keep]
    return out


def prune(data, ratio=0.2):
    """Prune the least attribute-similar edges."""
    return _prune_edges(data, ratio)


def od(data, ratio=0.1):
    """Outlier detection: remove nodes with the largest attribute deviation
    from their one-hop neighborhood mean."""
    src, dst = data.edge_index[0], data.edge_index[1]
    neigh_sum = torch.zeros_like(data.x).index_add(0, dst, data.x[src])
    deg = torch.zeros(data.num_nodes, device=data.x.device).index_add(
        0, dst, torch.ones(src.numel(), device=data.x.device))
    neigh_mean = neigh_sum / deg.clamp(min=1).unsqueeze(1)
    anomaly = (data.x - neigh_mean).norm(dim=1)
    num_remove = int(ratio * data.num_nodes)
    if num_remove == 0:
        return data.clone()
    remove = anomaly.topk(num_remove).indices
    keep_mask = torch.ones(data.num_nodes, dtype=torch.bool,
                           device=data.x.device)
    keep_mask[remove] = False
    return _remove_nodes(data, keep_mask)


def rigbd(data, ratio=0.1):
    """Influence-based edge filtering: edges between dissimilar, low-degree
    endpoints are the most suspicious and are removed first."""
    src, dst = data.edge_index[0], data.edge_index[1]
    deg = torch.zeros(data.num_nodes, device=data.x.device).index_add(
        0, dst, torch.ones(src.numel(), device=data.x.device)).clamp(min=1)
    sim = _edge_similarity(data)
    influence = (1.0 - sim) / (deg[src] * deg[dst]).sqrt()
    num_edges = data.edge_index.size(1)
    num_remove = int(ratio * num_edges)
    if num_remove == 0:
        return data.clone()
    remove = influence.topk(num_remove).indices
    keep_mask = torch.ones(num_edges, dtype=torch.bool,
                           device=data.x.device)
    keep_mask[remove] = False
    out = data.clone()
    out.edge_index = data.edge_index[:, keep_mask]
    return out


def mad(data, ratio=0.1):
    """Multidimensional sanitization: attribute smoothing + edge pruning +
    outlier-node removal."""
    src, dst = data.edge_index[0], data.edge_index[1]
    neigh_sum = torch.zeros_like(data.x).index_add(0, dst, data.x[src])
    deg = torch.zeros(data.num_nodes, device=data.x.device).index_add(
        0, dst, torch.ones(src.numel(), device=data.x.device))
    neigh_mean = neigh_sum / deg.clamp(min=1).unsqueeze(1)
    smoothed = data.clone()
    smoothed.x = (1.0 - ratio) * data.x + ratio * neigh_mean
    pruned = _prune_edges(smoothed, ratio)
    return od(pruned, ratio)


ATTACKS = {
    "none": lambda data, ratio: data.clone(),
    "prune": prune,
    "od": od,
    "rigbd": rigbd,
    "mad": mad,
}


def apply_attack(data, name, ratio=0.2):
    if name not in ATTACKS:
        raise ValueError(f"unknown attack {name}; choose from {list(ATTACKS)}")
    return ATTACKS[name](data, ratio)