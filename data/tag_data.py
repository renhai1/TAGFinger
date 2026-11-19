"""Loading and mixing of the five text-attribute graph (TAG) datasets.

Datasets (Appendix E): Cora-ML, CiteSeer-ML, ArXiv-HepPh, ArXiv-CS, Reddit.
Each dataset is expected under ``<root>/<name>/`` with:
  - features.pt   : FloatTensor [N, d], node features encoded by the dataset's
                    text encoder (GCN_LLMEmb / ENGINE / TAPE, see Appendix E);
  - edge_index.pt : LongTensor [2, E];
  - labels.pt     : LongTensor [N];
  - texts.jsonl   : one {"text": ...} (or {"title": ..., "abstract": ...})
                    record per node, the raw textual attributes X;
  - split.pt      : optional dict with train_mask / val_mask / test_mask.
  - graph_instances.pt : original graph identities, class labels and graph
                         splits; required when the graph task is requested.

Use ``clean_data..py`` to prepare these files from raw TAG sources with a
shared text encoder.

Mixed datasets (Appendix D): texts from all sources are encoded with the same
language model so that node features share one dimension; labels from
different sources are assigned distinct class indices (13 classes for
Cora-ML/CiteSeer-ML, 48 classes for Cora-ML/Reddit). The fingerprint target is
selected from the protected source's classes.
"""

import json
import os

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from .graph_instances import attach_original_graphs, require_original_graphs

TAG_DATASETS = {
    'Cora-ML': 7,
    'CiteSeer-ML': 6,
    'ArXiv-HepPh': 3,
    'ArXiv-CS': 40,
    'Reddit': 41,
}


def _load_texts(path, num_nodes):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "text" in obj:
                texts.append(obj["text"])
            else:
                texts.append(f"{obj.get('title', '')}. {obj.get('abstract', '')}")
    if len(texts) != num_nodes:
        raise ValueError(
            f"{path}: expected {num_nodes} text records, got {len(texts)}")
    return texts


def load_tag(name, root="data/TAG", seed=0, num_classes=None):
    if name not in TAG_DATASETS:
        raise ValueError(
            f"unsupported TAG dataset {name}; choose from {list(TAG_DATASETS)}")
    folder = os.path.join(root, name)
    x = torch.load(os.path.join(folder, "features.pt"), map_location="cpu",
                   weights_only=True)
    edge_index = torch.load(os.path.join(folder, "edge_index.pt"),
                            map_location="cpu", weights_only=True)
    y = torch.load(os.path.join(folder, "labels.pt"), map_location="cpu",
                   weights_only=True)
    raw_texts = _load_texts(os.path.join(folder, "texts.jsonl"), x.size(0))
    classes = TAG_DATASETS[name] if num_classes is None else num_classes
    if (x.ndim != 2 or x.size(0) == 0 or not bool(torch.isfinite(x).all())
            or y.dtype != torch.long or y.ndim != 1 or y.numel() != x.size(0)):
        raise ValueError("Prepared features and node labels have invalid shapes/dtypes")
    if (classes < 2 or bool((y < 0).any()) or bool((y >= classes).any())):
        raise ValueError("Node labels are outside the declared task class space")
    if (edge_index.dtype != torch.long or edge_index.ndim != 2
            or edge_index.size(0) != 2 or (edge_index.numel() and
            (int(edge_index.min()) < 0 or int(edge_index.max()) >= x.size(0)))):
        raise ValueError("Prepared edge_index has invalid shape or node IDs")

    split_path = os.path.join(folder, "split.pt")
    if os.path.exists(split_path):
        split = torch.load(split_path, map_location="cpu", weights_only=True)
        train_mask = split["train_mask"]
        val_mask = split["val_mask"]
        test_mask = split["test_mask"]
    else:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(x.size(0), generator=generator)
        n_train = int(0.6 * x.size(0))
        n_val = int(0.2 * x.size(0))
        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True

    masks = (train_mask, val_mask, test_mask)
    if any(m.dtype != torch.bool or m.ndim != 1 or m.numel() != x.size(0) for m in masks):
        raise ValueError("Node split masks must contain one boolean per node")
    if bool((sum(m.long() for m in masks) != 1).any()):
        raise ValueError("Node train/val/test splits must be disjoint and complete")
    if not bool(train_mask.any()) or not bool(val_mask.any()):
        raise ValueError("Nonempty training and validation splits are required")
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.raw_texts = raw_texts
    data.num_classes = classes
    data.dataset_name = name
    graph_path = os.path.join(folder, "graph_instances.pt")
    if os.path.isfile(graph_path):
        attach_original_graphs(data, torch.load(
            graph_path, map_location="cpu", weights_only=True))
    return data


def mix_tag_datasets(owner, other, owner_ratio, seed=0, preserve_graphs=False):
    """Single-source (same domain) or multi-source (different domains) mixture.

    Samples an ``owner_ratio`` fraction of the protected dataset and a
    ``1 - owner_ratio`` fraction of the other dataset, assigns disjoint class
    indices to the two sources, and takes their disjoint union. The returned
    ``owner_mask`` marks the protected source's nodes, from which fingerprint
    targets are selected. With ``preserve_graphs=True``, sampling fractions
    apply to original graph counts, not nodes: whole original graphs are
    retained and their graph-class labels use a separate source-wise offset.
    """
    if owner.x.size(1) != other.x.size(1):
        raise ValueError(
            "mixed sources must be encoded with the same language model so "
            "that node features have the same dimension")
    generator = torch.Generator().manual_seed(seed)
    if not 0 < owner_ratio < 1:
        raise ValueError("Mixture ratio must be strictly between zero and one")

    def sample_source(data, ratio):
        if preserve_graphs:
            require_original_graphs(data)
            count = data.graph_y.numel()
            ids = torch.randperm(count, generator=generator)[
                :max(1, int(ratio * count))].sort().values
            idx = torch.isin(data.graph_index, ids).nonzero(
                as_tuple=False).view(-1)
            ei, _ = subgraph(idx, data.edge_index, relabel_nodes=True,
                             num_nodes=data.num_nodes)
            return idx, ei, ids
        n = max(1, int(ratio * data.num_nodes))
        idx = torch.randperm(data.num_nodes, generator=generator)[:n]
        idx = idx.sort().values
        ei, _ = subgraph(idx, data.edge_index, relabel_nodes=True,
                         num_nodes=data.num_nodes)
        return idx, ei, None

    owner_idx, owner_ei, owner_graphs = sample_source(owner, owner_ratio)
    other_idx, other_ei, other_graphs = sample_source(other, 1.0 - owner_ratio)
    n_owner = owner_idx.numel()

    x = torch.cat([owner.x[owner_idx], other.x[other_idx]], dim=0)
    y = torch.cat([owner.y[owner_idx],
                   other.y[other_idx] + owner.num_classes], dim=0)
    edge_index = torch.cat([owner_ei, other_ei + n_owner], dim=1)
    raw_texts = [owner.raw_texts[int(i)] for i in owner_idx] + \
                [other.raw_texts[int(i)] for i in other_idx]

    train_mask = torch.cat([owner.train_mask[owner_idx],
                            other.train_mask[other_idx]], dim=0)
    val_mask = torch.cat([owner.val_mask[owner_idx],
                          other.val_mask[other_idx]], dim=0)
    test_mask = torch.cat([owner.test_mask[owner_idx],
                           other.test_mask[other_idx]], dim=0)

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.raw_texts = raw_texts
    data.num_classes = owner.num_classes + other.num_classes
    data.owner_mask = torch.cat([
        torch.ones(n_owner, dtype=torch.bool),
        torch.zeros(other_idx.numel(), dtype=torch.bool)], dim=0)
    data.owner_num_classes = owner.num_classes
    data.dataset_name = f"{owner.dataset_name}/{other.dataset_name}"
    if preserve_graphs:
        graph_index = torch.cat([
            torch.searchsorted(owner_graphs, owner.graph_index[owner_idx]),
            torch.searchsorted(other_graphs, other.graph_index[other_idx])
            + owner_graphs.numel()])
        metadata = {
            "graph_index": graph_index,
            "labels": torch.cat([owner.graph_y[owner_graphs],
                                 other.graph_y[other_graphs]
                                 + owner.graph_num_classes]),
            "num_classes": owner.graph_num_classes + other.graph_num_classes,
        }
        for split in ("train", "val", "test"):
            metadata[split + "_mask"] = torch.cat([
                getattr(owner, "graph_" + split + "_mask")[owner_graphs],
                getattr(other, "graph_" + split + "_mask")[other_graphs]])
        attach_original_graphs(data, metadata)
    return data
