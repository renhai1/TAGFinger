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
from data.graph_instances import original_graph_nodes, require_original_graphs
from prompt.TUP import remove_query_edge


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

    def forward(self, x, edge_index, graph_index=None):
        delta_x = self.x_budget * torch.tanh(self.attr_mlp(x))
        x_tilde = x + delta_x

        num_nodes = x.size(0)
        # edge_budget counts added COO entries, including both directions.
        max_pairs = int(self.edge_budget * edge_index.size(1)) // 2
        if max_pairs <= 0 or num_nodes < 2:
            return x_tilde, edge_index
        num_cand = max_pairs * 4
        src = torch.randint(0, num_nodes, (num_cand,), device=x.device)
        dst = torch.randint(0, num_nodes, (num_cand,), device=x.device)
        valid = src != dst
        if graph_index is not None:
            valid &= graph_index[src] == graph_index[dst]
        lo, hi = torch.minimum(src, dst), torch.maximum(src, dst)
        pair_ids = torch.unique((lo * num_nodes + hi)[valid])
        old_ids = torch.unique(torch.minimum(edge_index[0], edge_index[1])
                               * num_nodes + torch.maximum(edge_index[0], edge_index[1]))
        pair_ids = pair_ids[~torch.isin(pair_ids, old_ids)]
        if pair_ids.numel() == 0:
            return x_tilde, edge_index
        src, dst = pair_ids // num_nodes, pair_ids % num_nodes
        scores = torch.sigmoid(
            self.edge_mlp(torch.cat([x[src], x[dst]], dim=1))
        ).squeeze(-1)
        keep = (scores > 0.5).nonzero(as_tuple=False).view(-1)[:max_pairs]
        edge_index_tilde = edge_index
        if keep.numel():
            new_src, new_dst = src[keep], dst[keep]
            # score-weighted feature relay: lets gradients of the structural
            # perturbation dA reach edge_mlp although added edges are discrete
            relay = scores[keep].unsqueeze(1) * self.x_budget * x_tilde[new_src].detach()
            x_tilde = x_tilde.index_add(0, new_dst, relay)
            # Relay is only a surrogate gradient signal; the discrete edge
            # selection itself is not differentiable. Bound the final feature
            # change, including relay, by the declared coordinate-wise budget.
            x_tilde = x + (x_tilde - x).clamp(-self.x_budget, self.x_budget)
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


def sample_alignment_instances(data, num_hops, num_instances,
                               tasks=('node', 'edge', 'graph'),
                               node_pool=None, graph_pool=None):
    """Task instances for graph-level alignment: node-centered subgraphs,
    endpoint-centered subgraphs, and entire original graph instances."""
    num_nodes = data.num_nodes
    instances = []
    if not tasks:
        return instances
    if 'graph' in tasks:
        require_original_graphs(data)
    node_pool = (torch.arange(num_nodes, device=data.x.device)
                 if node_pool is None else node_pool)
    if node_pool.numel() == 0:
        raise ValueError("Alignment node pool is empty")
    in_pool = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    in_pool[node_pool] = True
    allowed_edges = (in_pool[data.edge_index[0]] & in_pool[data.edge_index[1]]
                     & (data.edge_index[0] != data.edge_index[1]))
    edge_ids = allowed_edges.nonzero(as_tuple=False).view(-1)
    for j in range(num_instances):
        tau = tasks[j % len(tasks)]
        query_edge = None
        if tau == 'graph':
            ids = (torch.unique(data.graph_index) if graph_pool is None else graph_pool)
            if not ids.numel():
                raise ValueError("Alignment graph pool is empty")
            graph_id = int(ids[torch.randint(ids.numel(), (1,), device=ids.device)].item())
            subset = original_graph_nodes(data, graph_id)
            sub_ei, _ = subgraph(subset, data.edge_index, relabel_nodes=True,
                                 num_nodes=num_nodes)
            instances.append((tau, subset, sub_ei, None))
            continue
        if tau == 'edge':
            if not edge_ids.numel():
                continue  # Do not fabricate an edge or a different task.
            e = int(edge_ids[torch.randint(edge_ids.numel(), (1,), device=edge_ids.device)].item())
            centers = data.edge_index[:, e].tolist()
            query_edge = tuple(centers)
            hops = num_hops
        else:
            centers = [int(node_pool[torch.randint(node_pool.numel(), (1,),
                                                  device=node_pool.device)].item())]
            hops = num_hops
        subset, sub_ei, _, _ = k_hop_subgraph(
            centers, hops, data.edge_index,
            relabel_nodes=True, num_nodes=num_nodes)
        sub_ei = remove_query_edge(sub_ei, subset, query_edge)
        instances.append((tau, subset, sub_ei, query_edge))
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
    alignment_tasks = tuple(graph_heads)
    if not use_dac and not use_dvc:
        raise ValueError("SAKA needs at least one of DAC or DVC")
    sur_params = list(surrogate.parameters()) + list(classifier.parameters())
    for head in graph_heads.values():
        sur_params += list(head.parameters())
    sur_params = list({id(p): p for p in sur_params}.values())
    sur_opt = torch.optim.Adam(sur_params, lr=lr, weight_decay=weight_decay)
    gen_opt = torch.optim.Adam(generator.parameters(), lr=lr)

    def sur_node_probs(x, edge_index):
        logits, _ = classifier(surrogate(x, edge_index))
        return F.softmax(logits, dim=1)

    def sur_graph_probs(x, edge_index, tau):
        logits, _ = graph_heads[tau](surrogate(x, edge_index))
        return F.softmax(logits, dim=1).squeeze(0)

    def alignment_loss(x_tilde, ei_tilde, nodes, instances, clean_grad):
        if graph_heads and not instances:
            raise ValueError("No valid task instances in this alignment batch")
        node_loss = 0.0 * x_tilde.sum()
        if not graph_heads:
            # Legacy node-posterior interface only. The task-unified runner
            # supplies graph heads and uses query_graph exclusively below.
            p_sus_clean = blackbox.query(data.x, data.edge_index)
            p_sus_tilde = blackbox.query(x_tilde.detach(), ei_tilde)
            p_sur_tilde = sur_node_probs(x_tilde, ei_tilde)
            if clean_grad:
                p_sur_clean = sur_node_probs(data.x, data.edge_index)
            else:
                with torch.no_grad():
                    p_sur_clean = sur_node_probs(data.x, data.edge_index)
            node_loss = 0.0 * p_sur_tilde.sum()
            if use_dac:
                node_loss = node_loss + dac_loss(p_sur_tilde[nodes], p_sus_tilde[nodes])
            if use_dvc:
                node_loss = node_loss + dvc_loss(p_sur_tilde[nodes], p_sur_clean[nodes],
                                                 p_sus_tilde[nodes], p_sus_clean[nodes])
        # the same objectives on graph-level posteriors (C_tau classes)
        graph_loss = 0.0
        num_valid = 0
        for tau, subset, sub_ei, query_edge in instances:
            if tau not in graph_heads:
                continue
            sub_x_t = x_tilde[subset]
            sub_ei_t, _ = subgraph(subset, ei_tilde, relabel_nodes=True,
                                   num_nodes=data.num_nodes)
            sub_ei_t = remove_query_edge(sub_ei_t, subset, query_edge)
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

    # Centers and original graph IDs are held out, not randomly relabeled
    # training samples. Message passing may still use transductive context.
    train_mask, val_mask = data.train_mask.clone(), data.val_mask.clone()
    if 'graph' in alignment_tasks:
        require_original_graphs(data)
        train_mask &= data.graph_train_mask[data.graph_index]
        val_mask &= data.graph_val_mask[data.graph_index]
    train_pool = train_mask.nonzero(as_tuple=False).view(-1)
    val_pool = val_mask.nonzero(as_tuple=False).view(-1)
    if not train_pool.numel() or not val_pool.numel():
        raise ValueError("SAKA requires nonempty disjoint train/val node splits")
    if bool((data.train_mask & data.val_mask).any()):
        raise ValueError("SAKA train and validation centers must be disjoint")
    train_graphs = val_graphs = None
    if 'graph' in alignment_tasks:
        train_graphs = data.graph_train_mask.nonzero(as_tuple=False).view(-1)
        val_graphs = data.graph_val_mask.nonzero(as_tuple=False).view(-1)
        if not train_graphs.numel() or not val_graphs.numel():
            raise ValueError("Graph alignment requires original train/val graphs")
    val_nodes = val_pool[:batch_nodes]
    val_instances = sample_alignment_instances(data, num_hops, batch_graphs,
                                               tasks=alignment_tasks,
                                               node_pool=val_pool, graph_pool=val_graphs)

    best_val = float('inf')
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        nodes = train_pool[torch.randperm(train_pool.numel(), device=train_pool.device)[:batch_nodes]]
        instances = sample_alignment_instances(data, num_hops, batch_graphs,
                                               tasks=alignment_tasks,
                                               node_pool=train_pool, graph_pool=train_graphs)

        # generator step: f_g maximizes L_SAKA to probe unaligned regions
        surrogate.eval()
        classifier.eval()
        for head in graph_heads.values():
            head.eval()
        generator.train()
        gen_opt.zero_grad()
        x_tilde, ei_tilde = generator(data.x, data.edge_index,
                                      getattr(data, 'graph_index', None))
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
            x_tilde, ei_tilde = generator(data.x, data.edge_index,
                                          getattr(data, 'graph_index', None))
        loss_sur = alignment_loss(x_tilde, ei_tilde, nodes, instances,
                                  clean_grad=True)
        loss_sur.backward()
        sur_opt.step()

        # early stopping on the validation alignment loss
        surrogate.eval()
        classifier.eval()
        for head in graph_heads.values():
            head.eval()
        generator.eval()
        with torch.no_grad():
            x_val, ei_val = generator(data.x, data.edge_index,
                                      getattr(data, 'graph_index', None))
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
