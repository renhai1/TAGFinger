"""Prompt-Amplified Stable Perturbation (PASP) utilities.

Implements the structural components of PASP construction in TAGFinger:
  - stability score S(v) over the one-hop neighborhood (Eq. 5);
  - stable region selection V_i = {v_i} U TopK_{u in N(v_i)} S(u);
  - semantic abstraction Gamma(G_i) of the induced subgraph (input of Eq. 6);
  - distribution drift Delta_i = 1 - sim(z_i, z~_i) over the stable region (Eq. 7).
"""

import torch
import torch.nn.functional as F


def stability_scores(probs, edge_index, num_nodes):
    """S(v) = (1/|N(v)|) * sum_{u in N(v)} cos(p_v, p_u)."""
    src, dst = edge_index[0], edge_index[1]
    cos = F.cosine_similarity(probs[src], probs[dst], dim=1)
    score = torch.zeros(num_nodes, device=probs.device).index_add(0, dst, cos)
    deg = torch.zeros(num_nodes, device=probs.device).index_add(
        0, dst, torch.ones_like(cos))
    return score / deg.clamp(min=1)


def select_stable_region(node, scores, edge_index, k):
    """V_i = {v_i} U TopK_{u in N(v_i)} S(u)."""
    node = int(node)
    neighbors = edge_index[0, edge_index[1] == node]
    neighbors = torch.unique(neighbors[neighbors != node])
    if neighbors.numel() > k:
        top = scores[neighbors].topk(k).indices
        neighbors = neighbors[top]
    center = torch.tensor([node], dtype=torch.long, device=edge_index.device)
    return torch.cat([center, neighbors])


def structural_description(node, region, edge_index):
    """Gamma(G_i): natural-language semantic abstraction of the stable subgraph."""
    node = int(node)
    region_set = set(int(v) for v in region.tolist())
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    edges = sorted({(s, d) for s, d in zip(src, dst)
                    if s in region_set and d in region_set and s < d})
    center_degree = sum(1 for s, d in edges if s == node or d == node)
    return (
        f"The target node {node} is anchored in a structurally resilient region "
        f"containing {len(region_set)} nodes and {len(edges)} internal edges. "
        f"The target node is directly connected to {center_degree} stable neighbors. "
        f"Region nodes: {sorted(region_set)}. Internal edges: {edges}."
    )


def distribution_drift(probs_clean, probs_pert, region):
    """Delta_i = 1 - cos(z_i, z~_i), z_i = (1/|V_i|) * sum_{v in V_i} p_v."""
    z = probs_clean[region].mean(dim=0)
    z_tilde = probs_pert[region].mean(dim=0)
    return 1.0 - F.cosine_similarity(z, z_tilde, dim=0)
