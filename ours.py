import argparse
import pickle

import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Flickr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from collections import defaultdict
from NodeClassifier import NodeClassifier

from model import GCN, GAT, GIN, GraphSAGE

from prompt_graph.MyPrompt import MyPrompt

import os
import logging
from datetime import datetime


def init_logger(log_dir="logs", prefix="experiment"):
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
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN', 'GAT', 'GraphSage', 'GIN'])
parser.add_argument('--dataset', type=str, default='Cora',
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

# parser.add_argument('--thrd', type=float, default=0.5)

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--n_trials', type=int, default=5,
                    help='n_trials')

parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train GNN model.')

parser.add_argument('--trigger_epochs', type=int, default=200, help='Number of epochs to train trigger generator.')

# trigger setting
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')

parser.add_argument('--total_select', type=int, default=80,
                    help="number of poisoning nodes total_select")

parser.add_argument('--cosine_loss', type=float, default=0.01,
                    help="cosine_loss")

parser.add_argument('--dist_loss', type=float, default=0.01,
                    help="dist_loss")
# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")

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
if (args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'PubMed'):
    dataset = Planetoid(root='./data/Planetoid',
                        name=args.dataset,
                        transform=transform)
elif (args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/',
                     transform=transform)
data = dataset[0]
data = data.to(device)


def find_consistently_misclassified_nodes(data, num_classes, train_idx, test_idx):
    misclass_record = defaultdict(lambda: [0] * data.num_nodes)
    acc_list = []
    for trial in range(args.n_trials):
        model = GCN(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer,
                    drop_ratio=args.dropout).to(device)
        classifier = NodeClassifier(hid_dim=args.hidden, num_classes=num_classes, dropout=args.dropout,
                                    inner_dim=args.hidden).to(device)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.train_lr,
                                                weight_decay=args.weight_decay)
        model.train()
        classifier.train()
        for epoch in range(args.epochs):
            model_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            node_embding = model(data.x, data.edge_index)
            out, _ = classifier(node_embding)
            loss = F.cross_entropy(out[train_idx], data.y[train_idx])
            loss.backward()
            model_optimizer.step()
            classifier_optimizer.step()

        model.eval()
        classifier.eval()
        with torch.no_grad():
            logits, _ = classifier(model(data.x, data.edge_index))
            preds = logits.argmax(dim=1)
            acc = (preds[test_idx] == data.y[test_idx]).float().mean().item()
            acc_list.append(acc)
            print(f"[Trial {trial + 1}] Accuracy = {acc:.4f}")
            for i in range(data.num_nodes):
                if preds[i] != data.y[i]:
                    misclass_record[trial][i] = 1


    total_errors = np.sum([misclass_record[t] for t in range(args.n_trials)], axis=0)
    always_wrong = np.where(total_errors == args.n_trials)[0]

    base = args.total_select // num_classes
    remainder = args.total_select % num_classes
    per_class_allocation = [base + (1 if i < remainder else 0) for i in range(num_classes)]

    selected_nodes_per_class = {}
    for c in range(num_classes):
        candidates = [i for i in always_wrong if data.y[i].item() == c]
        if len(candidates) > per_class_allocation[c]:
            selected = np.random.choice(candidates, per_class_allocation[c], replace=False)
        else:
            selected = candidates
        selected_nodes_per_class[c] = selected

    save_path = './saved_indices/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f'{args.dataset}_{args.n_trials}_{args.total_select}_selected_nodes.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(selected_nodes_per_class, f)
    return selected_nodes_per_class

def load_selected_nodes(save_path='./saved_indices/'):
    file_path = os.path.join(save_path, f'{args.dataset}_{args.n_trials}_{args.total_select}_selected_nodes.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            selected_nodes_per_class = pickle.load(f)
        return selected_nodes_per_class
    else:
        return None



def inject_trigger_and_optimize(data, target_nodes):
    from scipy.stats import wasserstein_distance
    orig_x = data.x.clone().detach()
    orig_y = data.y.clone().detach()
    orig_edge_index = data.edge_index.clone().detach()
    num_orig_nodes = data.num_nodes
    feature_dim = orig_x.size(1)
    total_triggers = len(target_nodes) * args.trigger_size
    trigger_features = torch.randn((total_triggers, feature_dim), requires_grad=True, device=data.x.device)
    trigger_labels = []
    for node in target_nodes:
        label = orig_y[node].item()
        trigger_labels.extend([label] * args.trigger_size)
    trigger_labels = torch.tensor(trigger_labels, device=data.x.device)
    new_edges = []
    for idx, node in enumerate(target_nodes):
        base = idx * args.trigger_size + num_orig_nodes
        for i in range(args.trigger_size):
            for j in range(i + 1, args.trigger_size):
                new_edges.append([base + i, base + j])
                new_edges.append([base + j, base + i])
        for i in range(args.trigger_size):
            new_edges.append([base + i, node])
            new_edges.append([node, base + i])
    new_edges = torch.tensor(new_edges, dtype=torch.long, device=data.x.device).t()
    new_edge_index = to_undirected(torch.cat([orig_edge_index, new_edges], dim=1))
    optimizer = torch.optim.Adam([trigger_features], lr=args.train_lr)
    for curepoch in range(args.trigger_epochs):
        # for curepoch in range(10):
        optimizer.zero_grad()
        cosine_loss = 0
        for idx, node in enumerate(target_nodes):
            base = idx * args.trigger_size
            target_feat = orig_x[node]
            trig_feats = trigger_features[base:base + args.trigger_size]
            cos_sim = F.cosine_similarity(trig_feats, target_feat.unsqueeze(0).expand_as(trig_feats), dim=1)
            cosine_loss += (1 - cos_sim).mean()
        cosine_loss /= len(target_nodes)
        trigger_flat = trigger_features.view(-1).detach().cpu().numpy()
        orig_flat = orig_x.view(-1).detach().cpu().numpy()
        dist_loss = wasserstein_distance(trigger_flat, orig_flat)
        dist_loss = torch.tensor(dist_loss, dtype=torch.float, device=data.x.device)
        loss = args.cosine_loss * cosine_loss + args.dist_loss * dist_loss
        print(
            f"[Epoch {curepoch}] Cosine Loss: {cosine_loss:.4f}, Dist Loss: {dist_loss:.4f}, Total Loss: {loss:.4f}")
        loss.backward()
        optimizer.step()
    new_x = torch.cat([orig_x, trigger_features.detach()], dim=0)
    new_y = torch.cat([orig_y, trigger_labels], dim=0)
    return Data(x=new_x, edge_index=new_edge_index, y=new_y)


train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)
num_classes = dataset.num_classes

selected_nodes = load_selected_nodes()
if selected_nodes is None:
    selected_nodes = find_consistently_misclassified_nodes(data, num_classes, train_idx, test_idx)

selected_all_nodes = []
for nodes in selected_nodes.values():
    selected_all_nodes.extend(nodes)
selected_all_nodes = torch.tensor(selected_all_nodes, dtype=torch.long, device=device)

print(selected_all_nodes)

new_data = inject_trigger_and_optimize(data=data, target_nodes=selected_all_nodes)


trigger_nodes = torch.arange(data.num_nodes, new_data.num_nodes, device=device)


all_idx = torch.arange(new_data.num_nodes, device=device)


non_trigger_mask = ~torch.isin(all_idx, trigger_nodes)


attach_mask = torch.zeros(new_data.num_nodes, dtype=torch.bool, device=device)
attach_mask[selected_all_nodes] = True

idx_pool = all_idx[non_trigger_mask & ~attach_mask]


idx_train_add, idx_temp = train_test_split(idx_pool.cpu(), test_size=0.4, random_state=42)


idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)


idx_train_add = idx_train_add.to(device)

idx_train = torch.cat([selected_all_nodes, idx_train_add])

loss_fn = nn.CrossEntropyLoss()
new_data = new_data.to(device)

if (args.model == 'GCN'):
    legalGNN = GCN(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer,
                   drop_ratio=args.dropout).to(device)
elif (args.model == 'GAT'):
    legalGNN = GAT(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer,
                   drop_ratio=args.dropout).to(device)
elif (args.model == 'GraphSage'):
    legalGNN = GraphSAGE(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer,
                         drop_ratio=args.dropout).to(device)
elif (args.model == 'GIN'):
    legalGNN = GIN(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer,
                   drop_ratio=args.dropout).to(device)
prompt = MyPrompt(args.hidden).to(device)

legalCls = NodeClassifier(hid_dim=args.hidden, num_classes=num_classes, dropout=args.dropout,
                          inner_dim=args.hidden).to(device)

legalGNN_optimizer = torch.optim.Adam(legalGNN.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
prompt_optimizer = torch.optim.Adam(prompt.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
legalCls_optimizer = torch.optim.Adam(legalCls.parameters(), lr=args.train_lr,
                                      weight_decay=args.weight_decay)


val_acc_list = []
test_acc_list = []

for epoch in range(1, args.epochs + 1):
    legalGNN.train()
    legalCls.train()
    legalGNN_optimizer.zero_grad()
    legalCls_optimizer.zero_grad()

    h = legalGNN(new_data.x, new_data.edge_index)
    out, _ = legalCls(h)
    loss = loss_fn(out[idx_train], new_data.y[idx_train])
    loss.backward()
    legalGNN_optimizer.step()
    legalCls_optimizer.step()

    legalGNN.eval()
    legalCls.eval()
    with torch.no_grad():
        logits, _ = legalCls(legalGNN(new_data.x, new_data.edge_index))
        pred = logits.argmax(dim=1)
        acc_val = (pred[idx_val] == new_data.y[idx_val]).float().mean().item()
        acc_test = (pred[idx_test] == new_data.y[idx_test]).float().mean().item()
    print(f"[Stage1] Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {acc_val:.4f} | Test Acc: {acc_test:.4f}")



for param in legalGNN.parameters():
    param.requires_grad = False
for param in legalCls.parameters():
    param.requires_grad = False

all_idx = torch.arange(new_data.num_nodes, device=new_data.x.device)
non_attach_idx = all_idx[~attach_mask]
sampled_non_attach = non_attach_idx[torch.randperm(non_attach_idx.size(0))[:selected_all_nodes.size(0)]]
idx_prompt_train = torch.cat([selected_all_nodes, sampled_non_attach])

idx_eval_attach = selected_all_nodes
idx_eval_nonattach = non_attach_idx[torch.randperm(non_attach_idx.size(0))[:selected_all_nodes.size(0)]]

acc_attach_list = []
acc_nonattach_list = []

for epoch in range(1, args.epochs + 1):
    prompt.train()
    prompt_optimizer.zero_grad()

    h = legalGNN(new_data.x, new_data.edge_index)
    h = prompt(h, attach_mask)
    out, _ = legalCls(h)

    loss = loss_fn(out[idx_prompt_train], new_data.y[idx_prompt_train])
    loss.backward()
    prompt_optimizer.step()

    prompt.eval()
    with torch.no_grad():
        logits, _ = legalCls(prompt(legalGNN(new_data.x, new_data.edge_index), attach_mask))
        pred = logits.argmax(dim=1)

        acc_attach = (pred[idx_eval_attach] == new_data.y[idx_eval_attach]).float().mean().item()
        acc_nonattach = (pred[idx_eval_nonattach] == new_data.y[idx_eval_nonattach]).float().mean().item()

        acc_attach_list.append(acc_attach)
        acc_nonattach_list.append(acc_nonattach)

    print(f"[Stage2] Epoch {epoch:03d} | Prompt Loss: {loss.item():.4f} "
          f"| Attach Acc: {acc_attach:.4f} | Non-Attach Acc: {acc_nonattach:.4f}")

avg_attach_acc = sum(acc_attach_list) / len(acc_attach_list)
avg_nonattach_acc = sum(acc_nonattach_list) / len(acc_nonattach_list)

print(f"\n[Stage2] Average Accuracy on Attach Nodes     : {avg_attach_acc:.4f}")
print(f"[Stage2] Average Accuracy on Non-Attach Nodes : {avg_nonattach_acc:.4f}")

if (args.model == 'GCN'):
    illegalGNN = GCN(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer,
                     drop_ratio=args.dropout).to(device)
elif (args.model == 'GAT'):
    illegalGNN = GAT(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer,
                     drop_ratio=args.dropout).to(device)
elif (args.model == 'GraphSage'):
    illegalGNN = GraphSAGE(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer,
                           drop_ratio=args.dropout).to(device)
elif (args.model == 'GIN'):
    illegalGNN = GIN(input_dim=data.x.size(1), hid_dim=args.hidden, num_layer=args.num_layer,
                     drop_ratio=args.dropout).to(device)

illegalGNN_optimizer = torch.optim.Adam(illegalGNN.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
illegalCls = NodeClassifier(hid_dim=args.hidden, num_classes=num_classes, dropout=args.dropout,
                            inner_dim=args.hidden).to(device)
illegalCls_optimizer = torch.optim.Adam(illegalCls.parameters(), lr=args.train_lr,
                                        weight_decay=args.weight_decay)

val_acc_list_illegal = []
test_acc_list_illegal = []

for epoch in range(1, args.epochs):
    illegalGNN.train()
    illegalCls.train()
    illegalGNN_optimizer.zero_grad()
    illegalCls_optimizer.zero_grad()

    out, _ = illegalCls(illegalGNN(new_data.x, new_data.edge_index))
    loss = loss_fn(out[idx_train], new_data.y[idx_train])
    loss.backward()
    illegalGNN_optimizer.step()
    illegalCls_optimizer.step()

    illegalGNN.eval()
    illegalCls.eval()
    with torch.no_grad():
        logits, _ = illegalCls(illegalGNN(new_data.x, new_data.edge_index))
        pred = logits.argmax(dim=1)
        acc_val = (pred[idx_val] == new_data.y[idx_val]).float().mean().item()
        acc_test = (pred[idx_test] == new_data.y[idx_test]).float().mean().item()

        val_acc_list_illegal.append(acc_val)
        test_acc_list_illegal.append(acc_test)

    print(f"[Illegal] Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {acc_val:.4f} | Test Acc: {acc_test:.4f}")

avg_val_acc_illegal = sum(val_acc_list_illegal) / len(val_acc_list_illegal)
avg_test_acc_illegal = sum(test_acc_list_illegal) / len(test_acc_list_illegal)
print(f"\n[Illegal] Average Val Accuracy: {avg_val_acc_illegal:.4f}")
print(f"[Illegal] Average Test Accuracy: {avg_test_acc_illegal:.4f}")

from sklearn.metrics.pairwise import pairwise_kernels


def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    XX = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
    YY = pairwise_kernels(Y, Y, metric=kernel, gamma=gamma)
    XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd


def permutation_test_mmd(x, y, num_perm=1000):
    combined = np.vstack([x, y])
    n = len(x)
    observed = compute_mmd(x, y)
    count = 0
    for _ in range(num_perm):
        np.random.shuffle(combined)
        x_perm = combined[:n]
        y_perm = combined[n:]
        if compute_mmd(x_perm, y_perm) >= observed:
            count += 1
    p_value = count / num_perm
    return observed, p_value


def verify_ownership(legalGNN, prompt, legalCls, illegalGNN, illegalCls, new_data, selected_all_nodes, attach_mask):
    legalGNN.eval()
    prompt.eval()
    legalCls.eval()
    illegalGNN.eval()
    illegalCls.eval()

    with torch.no_grad():
        legal_logits, _ = legalCls(prompt(legalGNN(new_data.x, new_data.edge_index), attach_mask))
        # legal_logits, _ = legalCls(legalGNN(new_data.x, new_data.edge_index))
        legal_probs = F.softmax(legal_logits[selected_all_nodes], dim=1)
        legal_preds = legal_logits[selected_all_nodes].argmax(dim=1)

        illegal_logits, _ = illegalCls(prompt(illegalGNN(new_data.x, new_data.edge_index), attach_mask))
        # illegal_logits, _ = illegalCls(illegalGNN(new_data.x, new_data.edge_index))
        illegal_probs = F.softmax(illegal_logits[selected_all_nodes], dim=1)
        illegal_preds = illegal_logits[selected_all_nodes].argmax(dim=1)

    label_consistency = (legal_preds == illegal_preds).float().mean().item()
    legal_np = legal_probs.cpu().numpy()
    illegal_np = illegal_probs.cpu().numpy()

    mmd_val = compute_mmd(legal_np, illegal_np)
    perm_mmd, p_value = permutation_test_mmd(legal_np, illegal_np)


verify_ownership(legalGNN, prompt, legalCls, illegalGNN, illegalCls, new_data, selected_all_nodes, attach_mask)
