"""TAGFinger: Semantic-Prompted Structural-Resilient Fingerprinting for
Universal Ownership Verification of Text-Attribute Graphs.

Pipeline (fully black-box interaction with the suspected GNN):
  1. Sensitivity-guided adversarial knowledge alignment (SAKA) trains a
     surrogate GNN that mimics the decision boundary of the suspected model.
  2. Prompt-amplified stable perturbation (PASP) construction anchors
     LLM-guided perturbations in structurally resilient regions and
     jointly optimizes them with task-unified prompts (TUP):
        max_{delta_i, {p_tau}} sum_tau q_tau(y*) + lambda * Delta_i.
  3. Evidence-aggregated transferable ownership verification collects the
     tri-valued fingerprint evidence r_tau in {1, 0, -1} and aggregates it
     over the task set with threshold theta to output
     violated / unviolated / uncertain.
"""

import argparse
import os
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.utils import subgraph

from model import GCN, GAT, GIN, GraphSAGE, NodeClassifier
from model.SAKA import BlackBoxGNN, GAGGenerator, train_saka
from prompt.TUP import TaskUnifiedPrompt
from utils.pasp import (stability_scores, select_stable_region,
                        structural_description, distribution_drift)


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
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN',
                    help='architecture of the suspected GNN',
                    choices=['GCN', 'GAT', 'GraphSage', 'GIN'])
parser.add_argument('--surrogate_model', type=str, default='GCN',
                    help='architecture of the surrogate GNN',
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
                    help='Number of GNN layers.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train GNN models.')

# SAKA settings
parser.add_argument('--saka_epochs', type=int, default=100,
                    help='Number of alternating min-max epochs for SAKA.')
parser.add_argument('--x_budget', type=float, default=0.05,
                    help='Attribute perturbation budget of the GAG generator.')
parser.add_argument('--edge_budget', type=float, default=0.01,
                    help='Structural perturbation budget of the GAG generator.')

# PASP settings
parser.add_argument('--total_select', type=int, default=80,
                    help='Number of fingerprint target nodes.')
parser.add_argument('--topk_stable', type=int, default=5,
                    help='Top-k stable neighbors forming the stable region V_i.')
parser.add_argument('--delta_budget', type=float, default=0.1,
                    help='Budget of the fingerprint perturbation delta_i.')
parser.add_argument('--lam', type=float, default=1.0,
                    help='Trade-off coefficient lambda of the drift term.')
parser.add_argument('--pasp_epochs', type=int, default=100,
                    help='Epochs for jointly optimizing delta_i and prompts.')
parser.add_argument('--pasp_lr', type=float, default=0.01,
                    help='Learning rate for delta_i and prompt optimization.')
parser.add_argument('--llm_path', type=str, default=None,
                    help='Path of the local LLM (e.g., Qwen3-8B) used to '
                         'generate prompt-amplified stable perturbations.')

# TUP settings
parser.add_argument('--prompt_tokens', type=int, default=10,
                    help='Number of tokens in each task prompt graph.')
parser.add_argument('--num_hops', type=int, default=2,
                    help='Number of hops of the induced subgraph in Phi_tau.')
parser.add_argument('--cross_prune', type=float, default=0.1,
                    help='Cross-edge pruning threshold of the prompt graph.')
parser.add_argument('--inner_prune', type=float, default=0.01,
                    help='Inner-edge pruning threshold of the prompt graph.')

# verification settings
parser.add_argument('--theta', type=int, default=2,
                    help='Evidence threshold theta in the multi-source setting '
                         '(majority voting over the three tasks).')

# GPU setting
parser.add_argument('--device_id', type=int, default=0, help='GPU id.')

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(
    ('cuda:{}' if args.cuda else 'cpu').format(args.device_id))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)
log_file = init_logger()
log = logging.info


transform = T.Compose([T.NormalizeFeatures()])
if args.dataset in ('Cora', 'Citeseer', 'PubMed'):
    dataset = Planetoid(root='./data/Planetoid', name=args.dataset,
                        transform=transform)
elif args.dataset == 'Flickr':
    dataset = Flickr(root='./data/Flickr/', transform=transform)
data = dataset[0].to(device)
num_classes = dataset.num_classes
raw_texts = getattr(data, 'raw_texts', None)


def build_gnn(name):
    gnn_cls = {'GCN': GCN, 'GAT': GAT, 'GraphSage': GraphSAGE, 'GIN': GIN}[name]
    return gnn_cls(input_dim=data.x.size(1), hid_dim=args.hidden,
                   num_layer=args.num_layer, drop_ratio=args.dropout).to(device)


def train_blackbox(train_data, tag):
    """Train a GNN on the given graph; only its probability vectors will be
    exposed to the verification pipeline through BlackBoxGNN.query."""
    gnn = build_gnn(args.model)
    classifier = NodeClassifier(hid_dim=args.hidden, num_classes=num_classes,
                                dropout=args.dropout,
                                inner_dim=args.hidden).to(device)
    gnn_opt = torch.optim.Adam(gnn.parameters(), lr=args.train_lr,
                               weight_decay=args.weight_decay)
    cls_opt = torch.optim.Adam(classifier.parameters(), lr=args.train_lr,
                               weight_decay=args.weight_decay)
    train_idx = train_data.train_mask.nonzero(as_tuple=False).view(-1)
    test_idx = train_data.test_mask.nonzero(as_tuple=False).view(-1)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        gnn.train()
        classifier.train()
        gnn_opt.zero_grad()
        cls_opt.zero_grad()
        out, _ = classifier(gnn(train_data.x, train_data.edge_index))
        loss = loss_fn(out[train_idx], train_data.y[train_idx])
        loss.backward()
        gnn_opt.step()
        cls_opt.step()
    gnn.eval()
    classifier.eval()
    with torch.no_grad():
        logits, _ = classifier(gnn(train_data.x, train_data.edge_index))
        acc = (logits.argmax(dim=1)[test_idx] ==
               train_data.y[test_idx]).float().mean().item()
    log(f"[{tag}] test accuracy: {acc:.4f}")
    return BlackBoxGNN(gnn, classifier)


def make_independent_data(original):
    """A GNN trained on this shifted variant plays the innocent model that
    never used the protected TAG dataset."""
    perm = torch.randperm(original.num_nodes, device=device)
    x = original.x[perm]
    num_edges = original.edge_index.size(1)
    rewired = original.edge_index[:, torch.randperm(num_edges, device=device)]
    rewired = torch.stack([original.edge_index[0], rewired[1]], dim=0)
    return Data(x=x, edge_index=rewired, y=original.y,
                train_mask=original.train_mask,
                test_mask=original.test_mask).to(device)


# ---------------------------------------------------------------------------
# Step 0: suspected model (unauthorized usage) and independent model
# ---------------------------------------------------------------------------
suspected = train_blackbox(data, "Suspected")
independent = train_blackbox(make_independent_data(data), "Independent")


# ---------------------------------------------------------------------------
# Step 1: SAKA — surrogate GNN aligned with the suspected decision boundary
# ---------------------------------------------------------------------------
surrogate = build_gnn(args.surrogate_model)
surrogate_cls = NodeClassifier(hid_dim=args.hidden, num_classes=num_classes,
                               dropout=args.dropout,
                               inner_dim=args.hidden).to(device)
generator = GAGGenerator(feat_dim=data.x.size(1), hid_dim=args.hidden,
                         x_budget=args.x_budget,
                         edge_budget=args.edge_budget).to(device)
train_saka(surrogate, surrogate_cls, generator, suspected, data,
           epochs=args.saka_epochs, lr=args.train_lr,
           weight_decay=args.weight_decay, log_fn=log)
surrogate.eval()
surrogate_cls.eval()


def surrogate_probs(x, edge_index):
    logits, _ = surrogate_cls(surrogate(x, edge_index))
    return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Step 2: PASP — fingerprints anchored in structurally resilient regions
# ---------------------------------------------------------------------------
with torch.no_grad():
    probs_clean = surrogate_probs(data.x, data.edge_index)
scores = stability_scores(probs_clean, data.edge_index, data.num_nodes)

base = args.total_select // num_classes
remainder = args.total_select % num_classes
fingerprint_nodes = []
for c in range(num_classes):
    quota = base + (1 if c < remainder else 0)
    candidates = (data.y == c).nonzero(as_tuple=False).view(-1)
    if candidates.numel() == 0:
        continue
    top = scores[candidates].topk(min(quota, candidates.numel())).indices
    fingerprint_nodes.extend(candidates[top].tolist())
fingerprint_nodes = torch.tensor(fingerprint_nodes, dtype=torch.long,
                                 device=device)
log(f"[PASP] selected {fingerprint_nodes.numel()} fingerprint nodes "
    f"in structurally resilient regions")

regions, region_edge_indices, edge_partners, target_labels = [], [], [], []
clean_preds = probs_clean.argmax(dim=1)
for v in fingerprint_nodes.tolist():
    region = select_stable_region(v, scores, data.edge_index, args.topk_stable)
    regions.append(region)
    region_ei, _ = subgraph(region, data.edge_index, relabel_nodes=True,
                            num_nodes=data.num_nodes)
    region_edge_indices.append(region_ei)
    edge_partners.append(int(region[1]) if region.numel() > 1 else v)
    # target label y*: the runner-up class of the surrogate prediction
    target_labels.append(int(probs_clean[v].topk(2).indices[1]))
target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)

llm_perturbator = None
if args.llm_path is not None:
    from llm.generator import LLMPerturbator
    llm_perturbator = LLMPerturbator(args.llm_path, device=str(device))

delta_texts = []
if llm_perturbator is not None and raw_texts is not None:
    for i, v in enumerate(fingerprint_nodes.tolist()):
        subgraph_texts = [(int(u), raw_texts[int(u)]) for u in regions[i]]
        gamma = structural_description(v, regions[i], data.edge_index)
        delta_texts.append(llm_perturbator.generate_pasp(
            subgraph_texts, gamma, str(int(target_labels[i]))))
    log(f"[PASP] LLM generated {len(delta_texts)} text-level perturbations")

# feature-space counterpart of delta_i, jointly refined with the prompts
delta = nn.Parameter(0.01 * torch.randn(
    fingerprint_nodes.numel(), data.x.size(1), device=device))


def perturbed_features(base_x):
    x_pert = base_x.clone()
    x_pert[fingerprint_nodes] = base_x[fingerprint_nodes] + \
        args.delta_budget * torch.tanh(delta)
    return x_pert


# ---------------------------------------------------------------------------
# Step 3: TUP — joint optimization  max sum_tau q_tau(y*) + lambda * Delta_i
# ---------------------------------------------------------------------------
tup = TaskUnifiedPrompt(token_dim=data.x.size(1),
                        token_num=args.prompt_tokens,
                        cross_prune=args.cross_prune,
                        inner_prune=args.inner_prune,
                        num_hops=args.num_hops).to(device)
pasp_opt = torch.optim.Adam([delta] + list(tup.parameters()), lr=args.pasp_lr)


def task_instances(i, x_source):
    v = int(fingerprint_nodes[i])
    return {
        'node': v,
        'edge': (v, edge_partners[i]),
        'graph': Data(x=x_source[regions[i]],
                      edge_index=region_edge_indices[i]),
    }


for epoch in range(1, args.pasp_epochs + 1):
    tup.train()
    pasp_opt.zero_grad()
    x_pert = perturbed_features(data.x)
    probs_pert = surrogate_probs(x_pert, data.edge_index)
    objective = 0.0
    for i in range(fingerprint_nodes.numel()):
        drift = distribution_drift(probs_clean, probs_pert, regions[i])
        q_sum = 0.0
        for tau, instance in task_instances(i, x_pert).items():
            q_tau = tup.task_probability(tau, instance, x_pert,
                                         data.edge_index, surrogate_probs)
            q_sum = q_sum + q_tau[target_labels[i]]
        objective = objective + q_sum + args.lam * drift
    loss = -objective / fingerprint_nodes.numel()
    loss.backward()
    pasp_opt.step()
    log(f"[PASP-TUP] Epoch {epoch:03d} | objective: {-loss.item():.4f}")


# ---------------------------------------------------------------------------
# Step 4: evidence-aggregated transferable ownership verification
# ---------------------------------------------------------------------------
def fingerprint_evidence(pred_pert, pred_clean, y_star):
    """r_tau(v_i): 1 = drift toward y*, 0 = no drift, -1 = inconsistent drift."""
    if pred_pert == y_star:
        return 1
    if pred_pert == pred_clean:
        return 0
    return -1


def ownership_decision(evidence, theta):
    if sum(1 for r in evidence if r == 1) >= theta:
        return 'violated'
    if sum(1 for r in evidence if r == -1) >= theta:
        return 'unviolated'
    return 'uncertain'


@torch.no_grad()
def verify_ownership(blackbox, tag):
    """Query the suspected GNN on the fingerprinted graph appended with each
    task-unified prompt graph, and aggregate cross-task evidence."""
    tup.eval()
    x_pert = perturbed_features(data.x)
    probs_fn = blackbox.query

    decisions_single, decisions_multi = [], []
    for i in range(fingerprint_nodes.numel()):
        y_star = int(target_labels[i])
        evidence = {}
        clean_instances = task_instances(i, data.x)
        pert_instances = task_instances(i, x_pert)
        for tau in TaskUnifiedPrompt.TASKS:
            q_clean = tup.task_probability(tau, clean_instances[tau], data.x,
                                           data.edge_index, probs_fn)
            q_pert = tup.task_probability(tau, pert_instances[tau], x_pert,
                                          data.edge_index, probs_fn)
            evidence[tau] = fingerprint_evidence(
                int(q_pert.argmax()), int(q_clean.argmax()), y_star)
        # single-source: T' = {node} with theta = 1
        decisions_single.append(ownership_decision([evidence['node']], 1))
        # multi-source: T' = all tasks with theta set by majority voting
        decisions_multi.append(
            ownership_decision(list(evidence.values()), args.theta))

    vsr_single = np.mean([d == 'violated' for d in decisions_single])
    vsr_multi = np.mean([d == 'violated' for d in decisions_multi])
    log(f"[Verify-{tag}] single-source (T'={{node}}, theta=1) "
        f"violated rate: {vsr_single:.4f}")
    log(f"[Verify-{tag}] multi-source (T'=all, theta={args.theta}) "
        f"violated rate: {vsr_multi:.4f}")
    return vsr_single, vsr_multi


vsr_sus_single, vsr_sus_multi = verify_ownership(suspected, "Suspected")
fpr_ind_single, fpr_ind_multi = verify_ownership(independent, "Independent")

log(f"[Result] VSR on suspected GNN  | single: {vsr_sus_single:.4f} "
    f"| multi: {vsr_sus_multi:.4f}")
log(f"[Result] FPR on independent GNN | single: {fpr_ind_single:.4f} "
    f"| multi: {fpr_ind_multi:.4f}")
