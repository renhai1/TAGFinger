"""Task-Unified Prompts (TUP).

Unifies node-, edge-, and graph-level verification into a shared
graph-level inference template:
  - Phi_tau maps each task instance into a subgraph query: a node v to its
    induced subgraph G_v, an edge (u, v) to the pair (G_u, G_v), and a
    graph to itself;
  - a learnable prompt graph p_tau per task is appended to the input via
    the operator "circled-plus";
  - q_tau = MeanPool(f(Phi_tau(G) circled-plus p_tau)) mean-pools the
    returned node-level probability vectors into the task-level
    probability vector (Eq. 8).
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


class TaskUnifiedPrompt(torch.nn.Module):
    TASKS = ('node', 'edge', 'graph')

    def __init__(self, token_dim, token_num=10, cross_prune=0.1,
                 inner_prune=0.01, num_hops=2):
        super().__init__()
        self.cross_prune = cross_prune
        self.inner_prune = inner_prune
        self.num_hops = num_hops
        self.tokens = torch.nn.ParameterDict({
            tau: torch.nn.Parameter(torch.empty(token_num, token_dim))
            for tau in self.TASKS
        })
        for token in self.tokens.values():
            torch.nn.init.kaiming_uniform_(
                token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)

    def _induced_subgraph(self, node, x, edge_index):
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            [int(node)], self.num_hops, edge_index,
            relabel_nodes=True, num_nodes=x.size(0))
        return Data(x=x[subset], edge_index=sub_edge_index)

    def phi(self, tau, instance, x, edge_index):
        """Phi_tau: node -> G_v, edge (u, v) -> (G_u, G_v), graph -> itself."""
        if tau == 'node':
            return self._induced_subgraph(instance, x, edge_index)
        if tau == 'edge':
            g_u = self._induced_subgraph(instance[0], x, edge_index)
            g_v = self._induced_subgraph(instance[1], x, edge_index)
            pair_x = torch.cat([g_u.x, g_v.x], dim=0)
            pair_edge_index = torch.cat(
                [g_u.edge_index, g_v.edge_index + g_u.x.size(0)], dim=1)
            return Data(x=pair_x, edge_index=pair_edge_index)
        if tau == 'graph':
            if isinstance(instance, Data):
                return instance
            return Data(x=x, edge_index=edge_index)
        raise ValueError(f"unknown task: {tau}")

    def append_prompt(self, tau, graph):
        """G circled-plus p_tau: append the learnable prompt graph to the input."""
        tokens = self.tokens[tau]
        token_num = tokens.size(0)

        inner_sim = torch.sigmoid(tokens @ tokens.t())
        inner_adj = torch.where(
            inner_sim < self.inner_prune, torch.zeros_like(inner_sim), inner_sim)
        inner_edge_index = inner_adj.nonzero().t().contiguous()

        cross_sim = torch.sigmoid(tokens @ graph.x.t())
        cross_adj = torch.where(
            cross_sim < self.cross_prune, torch.zeros_like(cross_sim), cross_sim)
        cross_edge_index = cross_adj.nonzero().t().contiguous()
        cross_edge_index[1] = cross_edge_index[1] + token_num

        x = torch.cat([tokens, graph.x], dim=0)
        edge_index = torch.cat(
            [inner_edge_index, graph.edge_index + token_num, cross_edge_index],
            dim=1)
        return Data(x=x, edge_index=edge_index)

    def task_probability(self, tau, instance, x, edge_index, probs_fn):
        """q_tau = MeanPool(probs_fn(Phi_tau(G) circled-plus p_tau))."""
        query_graph = self.phi(tau, instance, x, edge_index)
        prompted = self.append_prompt(tau, query_graph)
        probs = probs_fn(prompted.x, prompted.edge_index)
        return probs.mean(dim=0)
