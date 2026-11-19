"""TAGFinger: Semantic-Prompted Structural-Resilient Fingerprinting for
Universal Ownership Verification of Text-Attribute Graphs.

Pipeline (fully black-box interaction with the suspected GNN):
  1. Sensitivity-guided adversarial knowledge alignment (SAKA) trains a
     surrogate GNN that mimics the decision boundary of the suspected model
     through the DAC (KL) and DVC (deviation) constrainers, at both node
     level and graph/subgraph level (Eqs. 1-4).
  2. Prompt-amplified stable perturbation (PASP) construction anchors
     LLM-guided text edits in structurally resilient regions. Edited texts
     are re-encoded with the same text encoder; delta_i is the encoding
     difference. Candidate edits are scored on the surrogate with
        max_{delta_i, {p_tau}} sum_tau q_tau(y*) + lambda * Delta_i,
     retained by discrete selection, and alternated with gradient ascent on
     the task-unified prompt (TUP) features. No gradients pass through text
     generation, and neither step queries the suspected model (Eqs. 5-9).
  3. Evidence-aggregated transferable ownership verification collects the
     tri-valued fingerprint evidence r_tau in {1, 0, -1} and aggregates it
     over the task set with threshold theta to output
     violated / unviolated / uncertain (Eqs. 10-11). VSR is reported under
     FPR constraints as S@1 / S@5.

Default hyperparameters follow Appendix D: k=2, lambda=0.5, m=200
fingerprint perturbations, hidden dimension 128, 2-layer surrogate, GAG
generator hidden 256, SAKA 500 epochs (early stopping patience 50), surrogate
lr 1e-3, batch 128 (node) / 32 (graph), theta=2 (majority voting). Results in
the paper are averaged over seeds {0, 1, 2, 3, 4}.
"""

import argparse
import json
import os
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import negative_sampling

from model import (GCN, GAT, GIN, RGCN, NodeClassifier, GraphClassifier,
                   BlackBoxGNN, GAGGenerator, train_saka)
from prompt.TUP import TaskUnifiedPrompt, reformulate
from utils.pasp import (stability_scores, select_stable_region,
                        select_fingerprint_nodes, structural_description,
                        distribution_drift)
from utils.attacks import apply_attack
from data.tag_data import load_tag, mix_tag_datasets, TAG_DATASETS
from data.graph_instances import (require_original_graphs, original_graph,
                                 sample_original_graphs, copy_graph_annotations)
from evaluation.metrics import s_at_k, reconstruction_analysis


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
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed (paper averages over seeds 0-4).')
parser.add_argument('--model', type=str, default='GCN',
                    help='architecture of the suspected GNN',
                    choices=['GCN', 'GAT', 'GIN', 'RGCN'])
parser.add_argument('--surrogate_model', type=str, default='GCN',
                    help='architecture of the surrogate GNN',
                    choices=['GCN', 'GAT', 'GIN', 'RGCN'])
parser.add_argument('--dataset', type=str, default='Cora-ML',
                    help='Protected TAG dataset',
                    choices=list(TAG_DATASETS))
parser.add_argument('--mix_dataset', type=str, default=None,
                    help='Second source for mixed-dataset training '
                         '(single-source: Cora-ML/CiteSeer-ML; multi-source: '
                         'Cora-ML/Reddit).',
                    choices=list(TAG_DATASETS))
parser.add_argument('--mix_ratio', type=float, default=0.8,
                    help='Fraction of the protected source in the mixture '
                         '(0.8 / 0.6 / 0.4 / 0.2 in Tables 2-3).')
parser.add_argument('--data_root', type=str, default='data/TAG',
                    help='Root of the prepared TAG datasets.')
parser.add_argument('--tasks', nargs='+', choices=['node', 'edge', 'graph'],
                    default=['node', 'edge', 'graph'],
                    help='Verification tasks. The graph task requires original '
                         'graph labels in graph_instances.pt; node is required '
                         'for regional fingerprint construction.')
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Learning rate for training the suspected GNN.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='GNN hidden dimension (Appendix D: 128).')
parser.add_argument('--num_layer', type=int, default=2,
                    help='Number of GNN layers (Appendix D: 2-layer surrogate).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train the suspected GNN.')

# SAKA settings (Appendix D)
parser.add_argument('--saka_epochs', type=int, default=500,
                    help='Number of alternating min-max epochs for SAKA.')
parser.add_argument('--saka_lr', type=float, default=1e-3,
                    help='Surrogate learning rate (Appendix D: 1e-3).')
parser.add_argument('--patience', type=int, default=50,
                    help='Early stopping patience of SAKA.')
parser.add_argument('--gen_hidden', type=int, default=256,
                    help='Hidden dimension of the 2-layer MLP GAG generator.')
parser.add_argument('--x_budget', type=float, default=0.05,
                    help='Attribute perturbation budget of the GAG generator.')
parser.add_argument('--edge_budget', type=float, default=0.01,
                    help='Structural perturbation budget of the GAG generator.')
parser.add_argument('--batch_node', type=int, default=128,
                    help='Batch size for node-level alignment.')
parser.add_argument('--batch_graph', type=int, default=32,
                    help='Batch size for graph-level tasks.')

# PASP settings (Appendix D)
parser.add_argument('--total_select', type=int, default=200,
                    help='Number m of fingerprint perturbations (Fig. 6: '
                         'performance saturates at 200).')
parser.add_argument('--topk_stable', type=int, default=2,
                    help='Stable region size k (Appendix D: k=2).')
parser.add_argument('--delta_budget', type=float, default=0.1,
                    help='Budget of the fingerprint perturbation delta_i.')
parser.add_argument('--lam', type=float, default=0.5,
                    help='Trade-off coefficient lambda (Appendix D: 0.5).')
parser.add_argument('--pasp_rounds', type=int, default=5,
                    help='Alternating rounds of edit selection and prompt '
                         'updates.')
parser.add_argument('--num_candidates', type=int, default=4,
                    help='Number of LLM candidate edits per stable region.')
parser.add_argument('--prompt_steps', type=int, default=10,
                    help='Gradient ascent steps on the prompts per round.')
parser.add_argument('--pasp_lr', type=float, default=1e-3,
                    help='Learning rate for prompt gradient ascent.')
parser.add_argument('--llm_path', type=str, default=None,
                    help='Path of the fine-tuned Qwen3-8B used to generate '
                         'prompt-amplified stable perturbations.')
parser.add_argument('--text_encoder', type=str, default=None,
                    help='Path of the text encoder used for the node '
                         'features; edited texts are re-encoded with it.')
parser.add_argument('--pert_cache', type=str, default=None,
                    help='Optional JSON cache of LLM text edits '
                         '{node_id: {edited_node_id: text}}.')
parser.add_argument('--llm_mode', type=str, default=None,
                    choices=['single', 'multi'],
                    help='User prompt template (Appendix C); defaults to '
                         'multi when --mix_dataset is set, else single.')

# TUP settings
parser.add_argument('--prompt_tokens', type=int, default=10,
                    help='Number of nodes in each task prompt graph.')
parser.add_argument('--num_hops', type=int, default=2,
                    help='Number of hops of the induced subgraph in Phi_tau.')
parser.add_argument('--cross_prune', type=float, default=0.1,
                    help='Cross-edge pruning threshold of the prompt graph.')
parser.add_argument('--inner_prune', type=float, default=0.01,
                    help='Inner-edge pruning threshold of the prompt graph.')

# verification settings
parser.add_argument('--theta', type=int, default=None,
                    help='Evidence threshold theta in the multi-source '
                         'setting (majority voting: theta=2 for |T|=3).')

# ablation (Table 5)
parser.add_argument('--no_dac', action='store_true',
                    help='Disable the distribution-aware constrainer.')
parser.add_argument('--no_dvc', action='store_true',
                    help='Disable the deviation-aware constrainer.')
parser.add_argument('--no_tup', action='store_true',
                    help='Disable the task-unified prompts.')

# robustness (RQ3, Figures 4-5)
parser.add_argument('--attack', type=str, default='none',
                    choices=['none', 'prune', 'od', 'rigbd', 'mad'],
                    help='Adaptive attack applied by the attacker to the '
                         'training graph of the suspected GNN.')
parser.add_argument('--attack_ratio', type=float, default=0.2,
                    help='Attack ratio (pruning swept up to 0.8 in Fig. 5).')

# analysis (Figure 6(c,d))
parser.add_argument('--recon_analysis', action='store_true',
                    help='Compare reconstruction losses of perturbations '
                         'and normal data.')

# GPU setting
parser.add_argument('--device_id', type=int, default=0, help='GPU id.')

args = parser.parse_known_args()[0]
TASKS = tuple(dict.fromkeys(args.tasks))
if 'node' not in TASKS:
    parser.error('--tasks must include node for regional surrogate predictions')
if args.theta is None:
    args.theta = len(TASKS) // 2 + 1
if not 1 <= args.theta <= len(TASKS):
    parser.error('--theta must be between 1 and the number of selected tasks')
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(
    ('cuda:{}' if args.cuda else 'cpu').format(args.device_id))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)
log_file = init_logger()
log = logging.info


# ---------------------------------------------------------------------------
# Data: protected TAG dataset, optionally mixed with a second source
# ---------------------------------------------------------------------------
data = load_tag(args.dataset, root=args.data_root, seed=args.seed)
if 'graph' in TASKS:
    require_original_graphs(data)
if args.mix_dataset is not None:
    other = load_tag(args.mix_dataset, root=args.data_root, seed=args.seed)
    data = mix_tag_datasets(data, other, args.mix_ratio, seed=args.seed,
                            preserve_graphs='graph' in TASKS)
    log(f"[Data] mixed {args.dataset}/{args.mix_dataset} "
        f"({args.mix_ratio:.0%}/{1 - args.mix_ratio:.0%}), "
        f"{data.num_classes} classes")
    if 'graph' in TASKS:
        log('[Data] graph-task mixture samples whole original graphs; '
            'the mixture ratio applies to graph counts, not node counts')
data = data.to(device)
num_classes = data.num_classes
raw_texts = data.raw_texts
owner_mask = getattr(data, 'owner_mask', None)
llm_mode = args.llm_mode or ('multi' if args.mix_dataset else 'single')

GNN_REGISTRY = {'GCN': GCN, 'GAT': GAT, 'GIN': GIN, 'RGCN': RGCN}
TASK_CLASSES = {'node': num_classes, 'edge': 2}
if 'graph' in TASKS:
    TASK_CLASSES['graph'] = data.graph_num_classes


def build_gnn(name):
    return GNN_REGISTRY[name](input_dim=data.x.size(1), hid_dim=args.hidden,
                              num_layer=args.num_layer,
                              drop_ratio=args.dropout).to(device)


def build_heads():
    classifier = NodeClassifier(hid_dim=args.hidden, num_classes=num_classes,
                                dropout=args.dropout,
                                inner_dim=args.hidden).to(device)
    graph_heads = {
        tau: GraphClassifier(hid_dim=args.hidden,
                             num_classes=TASK_CLASSES[tau],
                             inner_dim=args.hidden,
                             dropout=args.dropout).to(device)
        for tau in TASKS
    }
    return classifier, graph_heads


def sample_graph_batch(tau, source, num_instances):
    """Reformulated graph/subgraph classification instances (Sec. 4.3):
    node -> node-centered subgraph with the node label; edge -> endpoint-
    centered subgraph with the link-existence label; graph -> the entire
    original graph with its source-provided graph label."""
    if tau == 'graph':
        graphs = sample_original_graphs(source, num_instances)
        batch = Batch.from_data_list(graphs).to(device)
        return batch, batch.y.view(-1)
    train_idx = source.train_mask.nonzero(as_tuple=False).view(-1)
    graphs, labels = [], []
    for j in range(num_instances):
        if tau == 'node':
            v = int(train_idx[torch.randint(train_idx.numel(), (1,))].item())
            g = reformulate('node', v, source.x, source.edge_index,
                            args.num_hops)
            label = int(source.y[v])
        elif tau == 'edge':
            if j % 2 == 0:
                e = int(torch.randint(source.edge_index.size(1), (1,)).item())
                u, v = source.edge_index[:, e].tolist()
                label = 1
            else:
                neg = negative_sampling(
                    source.edge_index, num_nodes=source.num_nodes,
                    num_neg_samples=1)
                u, v = neg[:, 0].tolist()
                label = 0
            g = reformulate('edge', (u, v), source.x, source.edge_index,
                            args.num_hops)
        else:
            raise ValueError(f'Unknown training task: {tau}')
        graphs.append(g)
        labels.append(label)
    batch = Batch.from_data_list(graphs).to(device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return batch, labels


def train_blackbox(train_data, tag):
    """Train the attacker's GNN on the given graph. The queried model
    supports the graph/subgraph classification formulation: its graph-level
    readout and task classifiers remain internal, and only classification
    probability vectors are exposed through BlackBoxGNN."""
    gnn = build_gnn(args.model)
    classifier, graph_heads = build_heads()
    params = list(gnn.parameters()) + list(classifier.parameters())
    for head in graph_heads.values():
        params += list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.train_lr,
                           weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train_idx = train_data.train_mask.nonzero(as_tuple=False).view(-1)
    test_idx = train_data.test_mask.nonzero(as_tuple=False).view(-1)

    for epoch in range(1, args.epochs + 1):
        gnn.train()
        classifier.train()
        for head in graph_heads.values():
            head.train()
        opt.zero_grad()
        # node-level objective (batch size 128)
        h = gnn(train_data.x, train_data.edge_index)
        logits, _ = classifier(h)
        batch_nodes = train_idx[torch.randint(train_idx.numel(),
                                              (args.batch_node,))]
        loss = loss_fn(logits[batch_nodes], train_data.y[batch_nodes])
        # graph-level objectives on reformulated instances (batch size 32)
        for tau in TASKS:
            batch, labels = sample_graph_batch(tau, train_data,
                                               args.batch_graph)
            h_g = gnn(batch.x, batch.edge_index, batch=batch.batch)
            g_logits, _ = graph_heads[tau](
                h_g, batch=torch.arange(labels.numel(), device=device))
            loss = loss + loss_fn(g_logits, labels)
        loss.backward()
        opt.step()

    gnn.eval()
    classifier.eval()
    for head in graph_heads.values():
        head.eval()
    with torch.no_grad():
        logits, _ = classifier(gnn(train_data.x, train_data.edge_index))
        acc = (logits.argmax(dim=1)[test_idx] ==
               train_data.y[test_idx]).float().mean().item()
    log(f"[{tag}] test accuracy: {acc:.4f}")
    return BlackBoxGNN(gnn, classifier, graph_heads)


def make_independent_data(original):
    """A GNN trained on this shifted variant plays the innocent model that
    never used the protected TAG dataset."""
    perm = torch.randperm(original.num_nodes, device=device)
    if 'graph' in TASKS:
        # Preserve original graph boundaries when making this shifted control.
        perm = torch.arange(original.num_nodes, device=device)
        for graph_id in range(original.graph_y.numel()):
            nodes = (original.graph_index == graph_id).nonzero(
                as_tuple=False).view(-1)
            perm[nodes] = nodes[torch.randperm(nodes.numel(), device=device)]
    x = original.x[perm]
    num_edges = original.edge_index.size(1)
    rewired = original.edge_index[:, torch.randperm(num_edges, device=device)]
    rewired = torch.stack([original.edge_index[0], rewired[1]], dim=0)
    if 'graph' in TASKS:
        rewired = original.edge_index.clone()
        for graph_id in range(original.graph_y.numel()):
            mask = original.graph_index[original.edge_index[0]] == graph_id
            destinations = original.edge_index[1, mask]
            rewired[1, mask] = destinations[torch.randperm(
                destinations.numel(), device=device)]
    out = Data(x=x, edge_index=rewired, y=original.y,
               train_mask=original.train_mask,
               test_mask=original.test_mask).to(device)
    out.raw_texts = original.raw_texts
    out.num_classes = original.num_classes
    out.dataset_name = original.dataset_name
    if hasattr(original, 'val_mask'):
        out.val_mask = original.val_mask
    if hasattr(original, 'owner_mask'):
        out.owner_mask = original.owner_mask
    return copy_graph_annotations(original, out)


# ---------------------------------------------------------------------------
# Step 0: suspected model (unauthorized usage, possibly under attack) and
# independent model
# ---------------------------------------------------------------------------
suspected_train_data = apply_attack(data, args.attack, args.attack_ratio)
if args.attack != 'none':
    log(f"[Attack] attacker sanitizes the training graph with "
        f"{args.attack} (ratio={args.attack_ratio})")
suspected = train_blackbox(suspected_train_data, "Suspected")
independent = train_blackbox(make_independent_data(data), "Independent")


# ---------------------------------------------------------------------------
# Step 1: SAKA — surrogate GNN aligned with the suspected decision boundary
# ---------------------------------------------------------------------------
surrogate = build_gnn(args.surrogate_model)
surrogate_cls, surrogate_heads = build_heads()
generator = GAGGenerator(feat_dim=data.x.size(1), hid_dim=args.gen_hidden,
                         x_budget=args.x_budget,
                         edge_budget=args.edge_budget).to(device)
train_saka(surrogate, surrogate_cls, generator, suspected, data,
           graph_heads=surrogate_heads, epochs=args.saka_epochs,
           lr=args.saka_lr, weight_decay=args.weight_decay,
           patience=args.patience, batch_nodes=args.batch_node,
           batch_graphs=args.batch_graph, num_hops=args.num_hops,
           use_dac=not args.no_dac, use_dvc=not args.no_dvc, log_fn=log)
surrogate.eval()
surrogate_cls.eval()
for head in surrogate_heads.values():
    head.eval()
for param in list(surrogate.parameters()) + list(surrogate_cls.parameters()):
    param.requires_grad_(False)
for head in surrogate_heads.values():
    for param in head.parameters():
        param.requires_grad_(False)


def surrogate_probs(x, edge_index):
    logits, _ = surrogate_cls(surrogate(x, edge_index))
    return F.softmax(logits, dim=1)


def surrogate_query_graph(x, edge_index, tau):
    """f_{sur,tau}^{graph}: graph-level readout + task classifier (Eq. 8)."""
    logits, _ = surrogate_heads[tau](surrogate(x, edge_index))
    return F.softmax(logits, dim=1).squeeze(0)


# ---------------------------------------------------------------------------
# Step 2: PASP — fingerprints anchored in structurally resilient regions
# ---------------------------------------------------------------------------
with torch.no_grad():
    probs_clean = surrogate_probs(data.x, data.edge_index)
scores = stability_scores(probs_clean, data.edge_index, data.num_nodes)
if owner_mask is not None:
    # the fingerprint target is selected from the protected source's classes
    scores = scores.masked_fill(~owner_mask, float('-inf'))
fingerprint_nodes = select_fingerprint_nodes(
    scores, data.edge_index, data.num_nodes, args.total_select,
    args.topk_stable)
log(f"[PASP] selected {fingerprint_nodes.numel()} fingerprint nodes with "
    f"pairwise disjoint stable regions (k={args.topk_stable})")

regions, edge_partners = [], []
for v in fingerprint_nodes.tolist():
    region = select_stable_region(v, scores, data.edge_index, args.topk_stable)
    regions.append(region)
    if region.numel() > 1:
        partner = int(region[1])
    else:
        neighbors = data.edge_index[0, data.edge_index[1] == v]
        partner = int(neighbors[0]) if neighbors.numel() > 0 else v
    edge_partners.append(partner)


def task_instances(i, x_source):
    v = int(fingerprint_nodes[i])
    instances = {'node': v}
    if 'edge' in TASKS:
        instances['edge'] = (v, edge_partners[i])
    if 'graph' in TASKS:
        instances['graph'] = original_graph(
            data, int(data.graph_index[v]), x=x_source)
    return instances


tup = TaskUnifiedPrompt(token_dim=data.x.size(1),
                        token_num=args.prompt_tokens,
                        cross_prune=args.cross_prune,
                        inner_prune=args.inner_prune,
                        num_hops=args.num_hops,
                        use_prompt=not args.no_tup).to(device)

# task-local target labels y*: the runner-up class of the clean surrogate
# prediction in each task's own label space, so that positive evidence
# requires a change from a non-target prediction to the target label
target_labels = []
with torch.no_grad():
    for i in range(fingerprint_nodes.numel()):
        labels_i = {}
        for tau, instance in task_instances(i, data.x).items():
            q_clean = tup.task_probability(tau, instance, data.x,
                                           data.edge_index,
                                           surrogate_query_graph)
            labels_i[tau] = int(q_clean.topk(2).indices[1])
        target_labels.append(labels_i)

# --- LLM text edits, re-encoded with the same text encoder -----------------
perturbator = None
encoder = None
if args.llm_path is not None:
    from llm.generator import LLMPerturbator
    perturbator = LLMPerturbator(args.llm_path, device=str(device))
    log(f"[PASP] LLM: {args.llm_path} ({llm_mode}-source user prompt)")
if args.text_encoder is not None:
    from utils.text_encoder import TextEncoder
    encoder = TextEncoder(args.text_encoder, device=str(device))

pert_cache = None
if args.pert_cache is not None:
    with open(args.pert_cache, 'r', encoding='utf-8') as f:
        pert_cache = json.load(f)

if perturbator is None and pert_cache is None:
    raise RuntimeError(
        "PASP requires LLM-generated text edits: provide --llm_path (the "
        "fine-tuned Qwen3-8B) together with --text_encoder, or a cached "
        "edit file via --pert_cache")
if encoder is None:
    raise ValueError("--text_encoder is required: edited texts must be "
                     "re-encoded with the same encoder used for the node "
                     "features, and delta_i is the encoding difference")


def candidate_delta(edits):
    """delta_i = difference between the original and updated encodings of
    the edited texts, clipped to the perturbation budget."""
    delta = torch.zeros_like(data.x)
    if not edits:
        return delta
    node_ids = [int(u) for u in edits]
    new_enc = encoder.encode([edits[u] for u in edits]).to(device)
    for j, u in enumerate(node_ids):
        d = new_enc[j] - data.x[u]
        norm = d.norm()
        if norm > args.delta_budget:
            d = d * (args.delta_budget / norm)
        delta[u] = d
    return delta


def load_cached_edits(i):
    v = str(int(fingerprint_nodes[i]))
    entry = pert_cache.get(v, {})
    return {int(u): text for u, text in entry.items()}


def score_candidate(delta_cand, i):
    """Score an edit on the surrogate: sum_tau q_tau(y*) + lambda * Delta_i.
    Both regional drift and task probabilities use only the surrogate."""
    x_cand = data.x + delta_cand
    probs_pert = surrogate_probs(x_cand, data.edge_index)
    drift = distribution_drift(probs_clean, probs_pert, regions[i])
    q_sum = 0.0
    for tau, instance in task_instances(i, x_cand).items():
        q_tau = tup.task_probability(tau, instance, x_cand,
                                     data.edge_index, surrogate_query_graph)
        q_sum = q_sum + q_tau[target_labels[i][tau]]
    return float(q_sum + args.lam * drift)


def select_edits():
    """With the prompts fixed, evaluate candidate edits on the surrogate and
    retain the one with the highest score (no gradients through text)."""
    deltas = torch.zeros_like(data.x)
    for i in range(fingerprint_nodes.numel()):
        v = int(fingerprint_nodes[i])
        if perturbator is not None:
            neighbor_texts = [(int(u), raw_texts[int(u)])
                              for u in regions[i][1:]]
            gamma = structural_description(v, regions[i], data.edge_index)
            candidates = perturbator.generate_candidates(
                v, raw_texts[v], neighbor_texts, gamma,
                str(target_labels[i]['node']),
                num_candidates=args.num_candidates, mode=llm_mode)
        else:
            candidates = [load_cached_edits(i)]
        # delta_i is supported on the stable region V_i (Appendix B)
        region_set = set(int(u) for u in regions[i])
        candidates = [{u: t for u, t in edits.items() if u in region_set}
                      for edits in candidates]
        best_delta, best_score = torch.zeros_like(data.x), float('-inf')
        with torch.no_grad():
            for edits in candidates:
                delta_cand = candidate_delta(edits)
                score = score_candidate(delta_cand, i)
                if score > best_score:
                    best_score, best_delta = score, delta_cand
        deltas = deltas + best_delta
    return deltas


# --- alternating optimization: edit selection <-> prompt gradient ascent ---
prompt_opt = torch.optim.Adam(tup.parameters(), lr=args.pasp_lr)
delta = torch.zeros_like(data.x)

for round_idx in range(1, args.pasp_rounds + 1):
    tup.eval()
    delta = select_edits()
    log(f"[PASP] round {round_idx}: best text edits selected on the surrogate")

    if args.no_tup:
        continue

    # with the selected edits fixed, update the prompt features by gradient
    # ascent on sum_tau q_tau(y*) + lambda * Delta_i (surrogate only)
    tup.train()
    x_pert = data.x + delta
    for step in range(args.prompt_steps):
        prompt_opt.zero_grad()
        probs_pert = surrogate_probs(x_pert, data.edge_index)
        objective = 0.0
        for i in range(fingerprint_nodes.numel()):
            drift = distribution_drift(probs_clean, probs_pert, regions[i])
            q_sum = 0.0
            for tau, instance in task_instances(i, x_pert).items():
                q_tau = tup.task_probability(tau, instance, x_pert,
                                             data.edge_index,
                                             surrogate_query_graph)
                q_sum = q_sum + q_tau[target_labels[i][tau]]
            objective = objective + q_sum + args.lam * drift
        loss = -objective / fingerprint_nodes.numel()
        loss.backward()
        prompt_opt.step()
    log(f"[PASP-TUP] round {round_idx} | objective: {-loss.item():.4f}")

x_pert = data.x + delta
log(f"[PASP] final perturbation norm: {delta.norm().item():.4f}")


# ---------------------------------------------------------------------------
# Step 3: evidence-aggregated transferable ownership verification
# ---------------------------------------------------------------------------
def fingerprint_evidence(pred_pert, pred_clean, y_star):
    """r_tau(v_i) (Eq. 10): 1 = drift from a non-target prediction to y*,
    0 = no drift (including when both equal y*), -1 = inconsistent drift."""
    if pred_pert == y_star and pred_clean != y_star:
        return 1
    if pred_pert == pred_clean:
        return 0
    return -1


def ownership_decision(evidence, theta):
    """D(v_i) (Eq. 11): violated / unviolated / uncertain."""
    if sum(1 for r in evidence if r == 1) >= theta:
        return 'violated'
    if sum(1 for r in evidence if r == -1) >= theta:
        return 'unviolated'
    return 'uncertain'


@torch.no_grad()
def verify_ownership(blackbox, tag):
    """Query the suspected GNN on the fingerprinted graph appended with each
    task-unified prompt graph, and aggregate cross-task evidence. Clean
    predictions are queried with the same prompt (Algorithm 1, Stage IV)."""
    tup.eval()
    query_graph_fn = blackbox.query_graph

    decisions_single, decisions_multi = [], []
    pos_frac_single, pos_frac_multi = [], []
    for i in range(fingerprint_nodes.numel()):
        evidence = {}
        clean_instances = task_instances(i, data.x)
        pert_instances = task_instances(i, x_pert)
        for tau in TASKS:
            q_clean = tup.task_probability(tau, clean_instances[tau], data.x,
                                           data.edge_index, query_graph_fn)
            q_pert = tup.task_probability(tau, pert_instances[tau], x_pert,
                                          data.edge_index, query_graph_fn)
            evidence[tau] = fingerprint_evidence(
                int(q_pert.argmax()), int(q_clean.argmax()),
                target_labels[i][tau])
        # single-source: T' = {node} with theta = 1
        decisions_single.append(ownership_decision([evidence['node']], 1))
        pos_frac_single.append(float(evidence['node'] == 1))
        # multi-source: T' = all tasks with theta set by majority voting
        decisions_multi.append(
            ownership_decision(list(evidence.values()), args.theta))
        pos_frac_multi.append(
            sum(1 for r in evidence.values() if r == 1) / len(TASKS))

    vsr_single = np.mean([d == 'violated' for d in decisions_single])
    vsr_multi = np.mean([d == 'violated' for d in decisions_multi])
    log(f"[Verify-{tag}] single-source (T'={{node}}, theta=1) "
        f"violated rate: {vsr_single:.4f}")
    log(f"[Verify-{tag}] multi-source (T'=all, theta={args.theta}) "
        f"violated rate: {vsr_multi:.4f}")
    return (vsr_single, vsr_multi,
            np.array(pos_frac_single), np.array(pos_frac_multi))


(vsr_sus_single, vsr_sus_multi,
 pos_sus_single, pos_sus_multi) = verify_ownership(suspected, "Suspected")
(fpr_ind_single, fpr_ind_multi,
 pos_ind_single, pos_ind_multi) = verify_ownership(independent, "Independent")

# S@K: VSR under FPR <= K% (Sec. 5.1), sweeping the decision threshold
s1_single, s5_single = s_at_k(pos_sus_single, pos_ind_single)
s1_multi, s5_multi = s_at_k(pos_sus_multi, pos_ind_multi)

log(f"[Result] VSR on suspected GNN  | single: {vsr_sus_single:.4f} "
    f"| multi: {vsr_sus_multi:.4f}")
log(f"[Result] FPR on independent GNN | single: {fpr_ind_single:.4f} "
    f"| multi: {fpr_ind_multi:.4f}")
log(f"[Result] single-source S@1: {s1_single:.4f} | S@5: {s5_single:.4f}")
log(f"[Result] multi-source  S@1: {s1_multi:.4f} | S@5: {s5_multi:.4f}")

if args.recon_analysis:
    normal_errors, pert_errors = reconstruction_analysis(
        data.x, x_pert, fingerprint_nodes, device)
    log(f"[Recon] normal data loss: {normal_errors.mean():.6f} "
        f"(std {normal_errors.std():.6f}) | perturbed fingerprint loss: "
        f"{pert_errors.mean():.6f} (std {pert_errors.std():.6f})")
