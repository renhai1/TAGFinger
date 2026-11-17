"""Prepare the TAG datasets used in the paper.

Encodes the raw textual attributes of a TAG with a shared text encoder and
saves the files expected by ``data/tag_data.py`` under ``<out_root>/<name>/``:
features.pt, edge_index.pt, labels.pt, texts.jsonl, split.pt.

Raw input format (--raw_file, one JSON record per node, in node order):
  {"text": "...", "label": 0, "edges": [1, 5, 9]}
or {"title": "...", "abstract": "...", "label": 0, "edges": [...]}.

Encoders reported in Appendix E: GCN_LLMEmb for Cora-ML / CiteSeer-ML,
ENGINE for ArXiv-HepPh / ArXiv-CS, TAPE for Reddit. When mixing datasets,
encode all sources with the SAME language model so that node features share
one dimension (Appendix D); --text_encoder accepts any local HuggingFace
encoder checkpoint.

Example:
  python clean_data..py --name Cora-ML --raw_file raw/cora_ml.jsonl \
      --text_encoder /path/to/encoder --out_root data/TAG
"""

import argparse
import json
import os

import torch

from data.tag_data import TAG_DATASETS
from utils.text_encoder import TextEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        choices=list(TAG_DATASETS))
    parser.add_argument('--raw_file', type=str, required=True,
                        help='JSONL with one record per node')
    parser.add_argument('--text_encoder', type=str, required=True,
                        help='Local text encoder checkpoint')
    parser.add_argument('--out_root', type=str, default='data/TAG')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    texts, labels, adjacency = [], [], []
    with open(args.raw_file, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if 'text' in obj:
                texts.append(obj['text'])
            else:
                texts.append(f"{obj.get('title', '')}. "
                             f"{obj.get('abstract', '')}")
            labels.append(int(obj['label']))
            adjacency.append([int(u) for u in obj.get('edges', [])])

    num_nodes = len(texts)
    src, dst = [], []
    for v, neighbors in enumerate(adjacency):
        for u in neighbors:
            src.append(v)
            dst.append(u)
            src.append(u)
            dst.append(v)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    if int(y.max()) + 1 > TAG_DATASETS[args.name]:
        raise ValueError(
            f"{args.name} expects {TAG_DATASETS[args.name]} classes, "
            f"found {int(y.max()) + 1}")

    encoder = TextEncoder(args.text_encoder, device=args.device)
    x = encoder.encode(texts)

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(num_nodes, generator=generator)
    n_train = int(0.6 * num_nodes)
    n_val = int(0.2 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    folder = os.path.join(args.out_root, args.name)
    os.makedirs(folder, exist_ok=True)
    torch.save(x, os.path.join(folder, 'features.pt'))
    torch.save(edge_index, os.path.join(folder, 'edge_index.pt'))
    torch.save(y, os.path.join(folder, 'labels.pt'))
    torch.save({'train_mask': train_mask, 'val_mask': val_mask,
                'test_mask': test_mask}, os.path.join(folder, 'split.pt'))
    with open(os.path.join(folder, 'texts.jsonl'), 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
    print(f"[Prep] {args.name}: {num_nodes} nodes, {edge_index.size(1)} "
          f"directed edges, feature dim {x.size(1)} -> {folder}")


if __name__ == '__main__':
    main()