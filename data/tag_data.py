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


def load_tag(name, root="data/TAG", seed=0):
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

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.raw_texts = raw_texts
    data.num_classes = TAG_DATASETS[name]
    data.dataset_name = name
    return data


def mix_tag_datasets(owner, other, owner_ratio, seed=0):
    """Single-source (same domain) or multi-source (different domains) mixture.

    Samples an ``owner_ratio`` fraction of the protected dataset and a
    ``1 - owner_ratio`` fraction of the other dataset, assigns disjoint class
    indices to the two sources, and takes their disjoint union. The returned
    ``owner_mask`` marks the protected source's nodes, from which fingerprint
    targets are selected.
    """
    if owner.x.size(1) != other.x.size(1):
        raise ValueError(
            "mixed sources must be encoded with the same language model so "
            "that node features have the same dimension")
    generator = torch.Generator().manual_seed(seed)

    def sample_source(data, ratio):
        n = max(1, int(ratio * data.num_nodes))
        idx = torch.randperm(data.num_nodes, generator=generator)[:n]
        idx = idx.sort().values
        ei, _ = subgraph(idx, data.edge_index, relabel_nodes=True,
                         num_nodes=data.num_nodes)
        return idx, ei

    owner_idx, owner_ei = sample_source(owner, owner_ratio)
    other_idx, other_ei = sample_source(other, 1.0 - owner_ratio)
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
    test_mask = torch.cat([owner.test_mask[other_idx],
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
    return data