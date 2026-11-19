"""Task-Unified Prompts (TUP).

Unifies node-, edge-, and graph-level verification into a shared
graph/subgraph classification formulation (Sec. 4.3):
  - Phi_tau maps each prediction instance to a graph/subgraph input: a node v
    to its node-centered induced subgraph G_v (label = node label), an edge
    (u, v) to the endpoint-centered subgraph formed by G_u and G_v (label =
    edge label, i.e., link existence for link prediction), and a graph-level
    instance to its own graph and label;
  - a learnable prompt graph p_tau per task is appended to the input via the
    operator "circled-plus" (randomly initialized, Appendix D);
  - q_tau = f_tau^{graph}(Phi_tau(G) circled-plus p_tau) is returned by the
    queried model's graph-level readout followed by its task classifier
    (Eq. 8). The auditor observes only the classification probabilities.

Task unification concerns the input formulation and verification procedure,
not averaging node-classification probabilities to obtain edge or graph
predictions.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


def remove_query_edge(edge_index, node_ids, query_edge):
    """Remove both directions of the classified edge from an induced input."""
    if query_edge is None:
        return edge_index
    u, v = query_edge
    pu = (node_ids == int(u)).nonzero(as_tuple=False).view(-1)
    pv = (node_ids == int(v)).nonzero(as_tuple=False).view(-1)
    if pu.numel() != 1 or pv.numel() != 1:
        raise ValueError("Edge endpoints must be present in the induced input")
    keep = ~(((edge_index[0] == pu[0]) & (edge_index[1] == pv[0])) |
             ((edge_index[0] == pv[0]) & (edge_index[1] == pu[0])))
    return edge_index[:, keep]


def reformulate(tau, instance, x, edge_index, num_hops=2):
    """Phi_tau: node -> G_v, edge (u, v) -> endpoint-centered subgraph,
    graph -> itself. Each task retains its own label semantics."""
    if tau == 'node':
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            [int(instance)], num_hops, edge_index,
            relabel_nodes=True, num_nodes=x.size(0))
        return Data(x=x[subset], edge_index=sub_edge_index, n_id=subset)
    if tau == 'edge':
        u, v = int(instance[0]), int(instance[1])
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            [u, v], num_hops, edge_index,
            relabel_nodes=True, num_nodes=x.size(0))
        # the query edge itself is the classification target (link
        # existence), so it is excluded from the input subgraph
        sub_edge_index = remove_query_edge(sub_edge_index, subset, (u, v))
        return Data(x=x[subset], edge_index=sub_edge_index, n_id=subset)
    if tau == 'graph':
        if not isinstance(instance, Data):
            raise ValueError("The graph task requires an explicit original graph instance")
        if getattr(instance, 'y', None) is None or instance.y.numel() != 1:
            raise ValueError("The graph instance must retain its original graph label")
        return instance
    raise ValueError(f"unknown task: {tau}")


class TaskUnifiedPrompt(torch.nn.Module):
    TASKS = ('node', 'edge', 'graph')

    def __init__(self, token_dim, token_num=10, cross_prune=0.1,
                 inner_prune=0.01, num_hops=2, use_prompt=True):
        super().__init__()
        self.cross_prune = cross_prune
        self.inner_prune = inner_prune
        self.num_hops = num_hops
        self.use_prompt = use_prompt
        self.tokens = torch.nn.ParameterDict({
            tau: torch.nn.Parameter(torch.empty(token_num, token_dim))
            for tau in self.TASKS
        })
        for token in self.tokens.values():
            torch.nn.init.kaiming_uniform_(
                token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)

    def phi(self, tau, instance, x, edge_index):
        return reformulate(tau, instance, x, edge_index, self.num_hops)

    def append_prompt(self, tau, graph):
        """G circled-plus p_tau: append the learnable prompt graph to the input."""
        if not self.use_prompt:
            return graph
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

    def task_probability(self, tau, instance, x, edge_index, query_graph_fn):
        """q_tau = f_tau^{graph}(Phi_tau(G) circled-plus p_tau).

        query_graph_fn(x, edge_index, tau) returns the queried model's
        graph/subgraph classification probability vector in [0, 1]^{C_tau};
        only these probabilities are observable, not internal representations
        or head parameters.
        """
        query_graph = self.phi(tau, instance, x, edge_index)
        prompted = self.append_prompt(tau, query_graph)
        return query_graph_fn(prompted.x, prompted.edge_index, tau)
