"""Sensitivity-Guided Adversarial Knowledge Alignment (SAKA).

Implements the surrogate training of TAGFinger (Sec. 4.2):
  - a black-box wrapper of the suspected GNN that only exposes
    classification probability vectors (Eq. 1), at both node level and
    graph/subgraph level;
  - an MLP-based generative adversarial graph (GAG) generator f_g that
    produces attribute and structural perturbations (X~ = X + dX, A~ = A + dA)
    while keeping G~ close to G through the perturbation budgets;
  - the distribution-aware constrainer (DAC, Eq. 2) and the
    deviation-aware constrainer (DVC, Eq. 3);
  - the alternating min-max optimization of L_SAKA = L_dis + L_dev (Eq. 4).

For tasks reformulated as graph/subgraph classification, the same alignment
objectives are evaluated on the returned graph-level posteriors, replacing V
with the set of queried task instances and C with the class count C_tau.

Implementation details follow Appendix D: a 2-layer MLP generator with hidden
dimension 256, 500 alignment epochs with early stopping (patience 50), batch
size 128 for node-level and 32 for graph-level alignment, surrogate learning
rate 1e-3.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, subgraph


class BlackBoxGNN:
    """Black-box view of the suspected model: only classification
    probability vectors are returned, never parameters or embeddings."""

    def __init__(self, gnn, classifier, graph_heads=None):
        self.gnn = gnn
        self.classifier = classifier
        self.graph_heads = graph_heads if graph_heads is not None else {}

    @torch.no_grad()
    def query(self, x, edge_index):
        self.gnn.eval()
        self.classifier.eval()
        logits, _ = self.classifier(self.gnn(x, edge_index))
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def query_graph(self, x, edge_index, tau):
        """f_{sus,tau}^{graph}: graph-level readout + task classifier."""
        self.gnn.eval()
        head = self.graph_heads[tau]
        head.eval()
        logits, _ = head(self.gnn(x, edge_index))
        return F.softmax(logits, dim=1).squeeze(0)


class GAGGenerator(nn.Module):
    """MLP-based GAG generator f_g: (dX, dA) ~ f_g(delta | X, A).

    A 2-layer MLP with hidden dimension 256 (Appendix D). The tanh output and
    the edge budget keep the generated graph G~ close to G.
    """

    def __init__(self, feat_dim, hid_dim=256, x_budget=0.05, edge_budget=0.01):
        super().__init__()
        self.x_budget = x_budget
        self.edge_budget = edge_budget
        self.attr_mlp = nn.Sequential(
            nn.Linear(feat_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, feat_dim)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * feat_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, 1)
        )

    def forward(self, x, edge_index):
        delta_x = self.x_budget * torch.tanh(self.attr_mlp(x))
        x_tilde = x + delta_x

        num_nodes = x.size(0)
        num_cand = max(1, int(self.edge_budget * edge_index.size(1)))
        src = torch.randint(0, num_nodes, (num_cand,), device=x.device)
        dst = torch.randint(0, num_nodes, (num_cand,), device=x.device)
        scores = torch.sigmoid(
            self.edge_mlp(torch.cat([x[src], x[dst]], dim=1))
        ).squeeze(-1)
        keep = scores > 0.5
        edge_index_tilde = edge_index
        if keep.any():
            new_src, new_dst = src[keep], dst[keep]
            # score-weighted feature relay: lets gradients of the structural
            # perturbation dA reach edge_mlp although added edges are discrete
            relay = scores[keep].unsqueeze(1) * self.x_budget * x_tilde[new_src].detach()
            x_tilde = x_tilde.index_add(0, new_dst, relay)
            delta_a = torch.stack(
                [torch.cat([new_src, new_dst]), torch.cat([new_dst, new_src])], dim=0
            )
            edge_index_tilde = torch.cat([edge_index, delta_a], dim=1)
        return x_tilde, edge_index_tilde


def dac_loss(p_sur, p_sus, eps=1e-12):
    """L_dis = (1/|V|) * sum_v KL(p_sus_v || p_sur_v)."""
    return F.kl_div(torch.log(p_sur + eps), p_sus, reduction='batchmean')


def dvc_loss(p_sur_tilde, p_sur, p_sus_tilde, p_sus):
    """L_dev = (1/|V|) * sum_v ||(p~_sur - p_sur) - (p~_sus - p_sus)||_2^2."""
    dev = (p_sur_tilde - p_sur) - (p_sus_tilde - p_sus)
    return dev.pow(2).sum(dim=1).mean()


def sample_alignment_instances(data, num_hops, num_instances):
    """Task instances for graph-level alignment: node-centered subgraphs,
    endpoint-centered subgraphs, and larger graph-level subgraphs."""
    num_nodes = data.num_nodes
    instances = []
    tasks = ('node', 'edge', 'graph')
    for j in range(num_instances):
        tau = tasks[j % len(tasks)]
        if tau == 'edge' and data.edge_index.size(1) > 0:
            e = int(torch.randint(data.edge_index.size(1), (1,)).item())
            centers = data.edge_index[:, e].tolist()
            hops = num_hops
        else:
            tau = 'node' if tau == 'edge' else tau
            centers = [int(torch.randint(num_nodes, (1,)).item())]
            hops = num_hops if tau == 'node' else num_hops + 1
        subset, sub_ei, _, _ = k_hop_subgraph(
            centers, hops, data.edge_index,
            relabel_nodes=True, num_nodes=num_nodes)
        instances.append((tau, subset, sub_ei))
    return instances


def train_saka(surrogate, classifier, generator, blackbox, data,
               graph_heads=None, epochs=500, lr=1e-3, weight_decay=5e-4,
               patience=50, batch_nodes=128, batch_graphs=32,
               num_hops=2, use_dac=True, use_dvc=True, log_fn=print):
    """Alternating min-max optimization: min_{f_sur} max_{f_g} L_dis + L_dev.

    DAC/DVC can be disabled independently for the ablation study (Table 5).
    Early stopping monitors the alignment loss on held-out validation nodes
    and task instances (patience 50, Appendix D).
    """
    graph_heads = graph_heads if graph_heads is not None else {}
    sur_params = list(surrogate.parameters()) + list(classifier.parameters())
    for head in graph_heads.values():
        sur_params += list(head.parameters())
    sur_opt = torch.optim.Adam(sur_params, lr=lr, weight_decay=weight_decay)
    gen_opt = torch.optim.Adam(generator.parameters(), lr=lr)

    def sur_node_probs(x, edge_index):
        logits, _ = classifier(surrogate(x, edge_index))
        return F.softmax(logits, dim=1)

    def sur_graph_probs(x, edge_index, tau):
        logits, _ = graph_heads[tau](surrogate(x, edge_index))
        return F.softmax(logits, dim=1).squeeze(0)

    def alignment_loss(x_tilde, ei_tilde, nodes, instances, clean_grad):
        p_sus_tilde = blackbox.query(x_tilde.detach(), ei_tilde)
        p_sur_tilde = sur_node_probs(x_tilde, ei_tilde)
        if clean_grad:
            p_sur_clean = sur_node_probs(data.x, data.edge_index)
        else:
            with torch.no_grad():
                p_sur_clean = sur_node_probs(data.x, data.edge_index)
        # zero-valued tensor terms keep the graph connected when both
        # constrainers are disabled in the ablation (Table 5)
        node_loss = 0.0 * p_sur_tilde.sum()
        if use_dac:
            node_loss = node_loss + dac_loss(p_sur_tilde[nodes], p_sus_tilde[nodes])
        if use_dvc:
            node_loss = node_loss + dvc_loss(p_sur_tilde[nodes], p_sur_clean[nodes],
                                             p_sus_tilde[nodes], p_sus_clean[nodes])
        # the same objectives on graph-level posteriors (C_tau classes)
        graph_loss = 0.0
        num_valid = 0
        for tau, subset, sub_ei in instances:
            if tau not in graph_heads:
                continue
            sub_x_t = x_tilde[subset]
            sub_ei_t, _ = subgraph(subset, ei_tilde, relabel_nodes=True,
                                   num_nodes=data.num_nodes)
            q_sus_t = blackbox.query_graph(sub_x_t.detach(), sub_ei_t, tau)
            q_sur_t = sur_graph_probs(sub_x_t, sub_ei_t, tau)
            sub_x_c = data.x[subset]
            q_sus_c = blackbox.query_graph(sub_x_c, sub_ei, tau)
            if clean_grad:
                q_sur_c = sur_graph_probs(sub_x_c, sub_ei, tau)
            else:
                with torch.no_grad():
                    q_sur_c = sur_graph_probs(sub_x_c, sub_ei, tau)
            instance_loss = 0.0 * q_sur_t.sum()
            if use_dac:
                instance_loss = instance_loss + dac_loss(
                    q_sur_t.unsqueeze(0), q_sus_t.unsqueeze(0))
            if use_dvc:
                instance_loss = instance_loss + dvc_loss(
                    q_sur_t.unsqueeze(0), q_sur_c.unsqueeze(0),
                    q_sus_t.unsqueeze(0), q_sus_c.unsqueeze(0))
            graph_loss = graph_loss + instance_loss
            num_valid += 1
        if num_valid > 0:
            return node_loss + graph_loss / num_valid
        return node_loss

    # fixed held-out batches for early stopping
    val_nodes = torch.randperm(data.num_nodes)[:batch_nodes]
    val_instances = sample_alignment_instances(data, num_hops, batch_graphs)

    best_val = float('inf')
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        nodes = torch.randperm(data.num_nodes)[:batch_nodes]
        instances = sample_alignment_instances(data, num_hops, batch_graphs)

        # generator step: f_g maximizes L_SAKA to probe unaligned regions
        surrogate.eval()
        classifier.eval()
        for head in graph_heads.values():
            head.eval()
        generator.train()
        gen_opt.zero_grad()
        x_tilde, ei_tilde = generator(data.x, data.edge_index)
        loss_gen = alignment_loss(x_tilde, ei_tilde, nodes, instances,
                                  clean_grad=False)
        (-loss_gen).backward()
        gen_opt.step()

        # surrogate step: f_sur minimizes L_SAKA to absorb exposed discrepancies
        surrogate.train()
        classifier.train()
        for head in graph_heads.values():
            head.train()
        generator.eval()
        sur_opt.zero_grad()
        with torch.no_grad():
            x_tilde, ei_tilde = generator(data.x, data.edge_index)
        loss_sur = alignment_loss(x_tilde, ei_tilde, nodes, instances,
                                  clean_grad=True)
        loss_sur.backward()
        sur_opt.step()

        # early stopping on the validation alignment loss
        surrogate.eval()
        classifier.eval()
        generator.eval()
        with torch.no_grad():
            x_val, ei_val = generator(data.x, data.edge_index)
            val_loss = alignment_loss(x_val, ei_val, val_nodes, val_instances,
                                      clean_grad=False).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            stale = 0
            best_state = {
                'surrogate': deepcopy(surrogate.state_dict()),
                'classifier': deepcopy(classifier.state_dict()),
                'generator': deepcopy(generator.state_dict()),
                'graph_heads': {t: deepcopy(h.state_dict())
                                for t, h in graph_heads.items()},
            }
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == 1:
            log_fn(f"[SAKA] Epoch {epoch:03d} | L_gen(max): {loss_gen.item():.4f} "
                   f"| L_sur(min): {loss_sur.item():.4f} | val: {val_loss:.4f}")
        if stale >= patience:
            log_fn(f"[SAKA] early stopping at epoch {epoch} "
                   f"(no validation improvement for {patience} epochs)")
            break

    if best_state is not None:
        surrogate.load_state_dict(best_state['surrogate'])
        classifier.load_state_dict(best_state['classifier'])
        generator.load_state_dict(best_state['generator'])
        for tau, head in graph_heads.items():
            head.load_state_dict(best_state['graph_heads'][tau])

    return surrogate, classifier