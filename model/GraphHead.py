"""Graph-level readout followed by a task classifier (Eq. 8).

q_tau = f_{sur,tau}^{graph}(Phi_tau(G~) circled-plus p_tau), where
f^{graph} denotes the graph-level readout followed by the classifier for
task tau. Task unification concerns the input formulation and verification
procedure, NOT averaging node-classification probabilities: the readout
pools node embeddings, and a per-task classifier maps the pooled embedding
to the task's own label space (C_tau classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class GraphClassifier(nn.Module):
    """Mean readout of node embeddings + two-layer MLP classifier."""

    def __init__(self, hid_dim, num_classes, inner_dim=128, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(hid_dim, inner_dim)
        self.lin2 = nn.Linear(inner_dim, num_classes)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, h, batch=None):
        if batch is None:
            g = h.mean(dim=0, keepdim=True)
        else:
            g = global_mean_pool(h, batch.long())
        emb = F.relu(self.lin1(g))
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        logits = self.lin2(emb)
        return logits, emb