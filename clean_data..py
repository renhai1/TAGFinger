import argparse

import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Flickr
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid

from NodeClassifier import NodeClassifier

from model import GCN, GAT, GIN, GraphSAGE

from prompt_graph.MyPrompt import MyPrompt

import os
import logging
from datetime import datetime


def init_logger(log_dir="logs", prefix="experiment_clean"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{timestamp}.log")
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return log_path


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
                    default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GIN', help='model',
                    choices=['GCN', 'GAT', 'GraphSage', 'GIN'])
parser.add_argument('--dataset', type=str, default='Flickr',
                    help='Dataset',
                    choices=['Cora', 'PubMed', 'Citeseer', 'Flickr'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--num_layer', type=int, default=3,
                    help='Number of num_layer')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train GNN model.')
parser.add_argument('--device_id', type=int, default=0, help="GPU device ID")

args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)
log_file = init_logger()

transform = T.Compose([T.NormalizeFeatures()])
if args.dataset in ['Cora', 'Citeseer', 'PubMed']:
    dataset = Planetoid(root='./data/Planetoid', name=args.dataset, transform=transform)
elif args.dataset == 'Flickr':
    dataset = Flickr(root='./data/Flickr/', transform=transform)
data = dataset[0].to(device)
num_classes = dataset.num_classes
loss_fn = nn.CrossEntropyLoss()

all_idx = torch.randperm(data.num_nodes)
n_total = data.num_nodes
n_train = int(0.7 * n_total)
n_val = int(0.2 * n_total)
n_test = n_total - n_train - n_val

train_idx = all_idx[:n_train]
val_idx = all_idx[n_train:n_train + n_val]
test_idx = all_idx[n_train + n_val:]

if args.model == 'GCN':
    legalGNN = GCN(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer, drop_ratio=args.dropout).to(
        device)
elif args.model == 'GAT':
    legalGNN = GAT(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer, drop_ratio=args.dropout).to(
        device)
elif args.model == 'GraphSage':
    legalGNN = GraphSAGE(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer,
                         drop_ratio=args.dropout).to(device)
elif args.model == 'GIN':
    legalGNN = GIN(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer, drop_ratio=args.dropout).to(
        device)

prompt = MyPrompt(args.hidden).to(device)
legalCls = NodeClassifier(hid_dim=args.hidden, num_classes=num_classes, dropout=args.dropout, inner_dim=args.hidden).to(
    device)

legalGNN_optimizer = torch.optim.Adam(legalGNN.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
prompt_optimizer = torch.optim.Adam(prompt.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
legalCls_optimizer = torch.optim.Adam(legalCls.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)


for epoch in range(1, args.epochs + 1):
    legalGNN.train()
    legalCls.train()
    legalGNN_optimizer.zero_grad()
    legalCls_optimizer.zero_grad()

    h = legalGNN(data.x, data.edge_index)
    out, _ = legalCls(h)
    loss = loss_fn(out[train_idx], data.y[train_idx])
    loss.backward()
    legalGNN_optimizer.step()
    legalCls_optimizer.step()

    legalGNN.eval()
    legalCls.eval()
    with torch.no_grad():
        logits, _ = legalCls(legalGNN(data.x, data.edge_index))
        pred = logits.argmax(dim=1)
        acc_val = (pred[val_idx] == data.y[val_idx]).float().mean().item()
        acc_test = (pred[test_idx] == data.y[test_idx]).float().mean().item()
    print(f"[Stage1] Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {acc_val:.4f} | Test Acc: {acc_test:.4f}")

for p in legalGNN.parameters():
    p.requires_grad = False
for p in legalCls.parameters():
    p.requires_grad = False

test_acc_list = []

for epoch in range(1, args.epochs + 1):
    prompt.train()
    prompt_optimizer.zero_grad()

    h = legalGNN(data.x, data.edge_index)
    h = prompt(h)
    out, _ = legalCls(h)
    loss = loss_fn(out[train_idx], data.y[train_idx])
    loss.backward()
    prompt_optimizer.step()

    prompt.eval()
    with torch.no_grad():
        logits, _ = legalCls(prompt(legalGNN(data.x, data.edge_index)))
        pred = logits.argmax(dim=1)
        acc_val = (pred[val_idx] == data.y[val_idx]).float().mean().item()
        acc_test = (pred[test_idx] == data.y[test_idx]).float().mean().item()
        test_acc_list.append(acc_test)

    print(
        f"[Stage2] Epoch {epoch:03d} | Prompt Loss: {loss.item():.4f} | Val Acc: {acc_val:.4f} | Test Acc: {acc_test:.4f}")


