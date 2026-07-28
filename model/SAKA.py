"""Sensitivity-Guided Adversarial Knowledge Alignment (SAKA).

Implements the surrogate training of TAGFinger:
  - a black-box wrapper of the suspected GNN that only exposes
    classification probability vectors (Eq. 1);
  - an MLP-based generative adversarial graph (GAG) generator f_g that
    produces attribute and structural perturbations (X~ = X + dX, A~ = A + dA);
  - the distribution-aware constrainer (DAC, Eq. 2) and the
    deviation-aware constrainer (DVC, Eq. 3);
  - the alternating min-max optimization of L_SAKA (Eq. 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlackBoxGNN:
    """Black-box view of the suspected model: only probability vectors are returned."""

    def __init__(self, gnn, classifier):
        self.gnn = gnn
        self.classifier = classifier

    @torch.no_grad()
    def query(self, x, edge_index):
        self.gnn.eval()
        self.classifier.eval()
        logits, _ = self.classifier(self.gnn(x, edge_index))
        return F.softmax(logits, dim=1)


class GAGGenerator(nn.Module):
    """MLP-based GAG generator f_g: (dX, dA) ~ f_g(delta | X, A)."""

    def __init__(self, feat_dim, hid_dim=128, x_budget=0.05, edge_budget=0.01):
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


def train_saka(surrogate, classifier, generator, blackbox, data,
               epochs=100, lr=0.01, weight_decay=5e-4, log_fn=print):
    """Alternating min-max optimization: min_{f_sur} max_{f_g} L_dis + L_dev."""
    sur_opt = torch.optim.Adam(
        list(surrogate.parameters()) + list(classifier.parameters()),
        lr=lr, weight_decay=weight_decay)
    gen_opt = torch.optim.Adam(generator.parameters(), lr=lr)

    def sur_probs(x, edge_index):
        logits, _ = classifier(surrogate(x, edge_index))
        return F.softmax(logits, dim=1)

    p_sus_clean = blackbox.query(data.x, data.edge_index)

    for epoch in range(1, epochs + 1):
        # generator step: f_g maximizes L_SAKA to probe unaligned regions
        surrogate.eval()
        classifier.eval()
        generator.train()
        gen_opt.zero_grad()
        x_tilde, ei_tilde = generator(data.x, data.edge_index)
        p_sus_tilde = blackbox.query(x_tilde.detach(), ei_tilde)
        p_sur_tilde = sur_probs(x_tilde, ei_tilde)
        with torch.no_grad():
            p_sur_clean = sur_probs(data.x, data.edge_index)
        loss_gen = dac_loss(p_sur_tilde, p_sus_tilde) + \
            dvc_loss(p_sur_tilde, p_sur_clean, p_sus_tilde, p_sus_clean)
        (-loss_gen).backward()
        gen_opt.step()

        # surrogate step: f_sur minimizes L_SAKA to absorb exposed discrepancies
        surrogate.train()
        classifier.train()
        generator.eval()
        sur_opt.zero_grad()
        with torch.no_grad():
            x_tilde, ei_tilde = generator(data.x, data.edge_index)
            p_sus_tilde = blackbox.query(x_tilde, ei_tilde)
        p_sur_tilde = sur_probs(x_tilde, ei_tilde)
        p_sur_clean = sur_probs(data.x, data.edge_index)
        loss_sur = dac_loss(p_sur_tilde, p_sus_tilde) + \
            dvc_loss(p_sur_tilde, p_sur_clean, p_sus_tilde, p_sus_clean)
        loss_sur.backward()
        sur_opt.step()

        log_fn(f"[SAKA] Epoch {epoch:03d} | L_gen(max): {loss_gen.item():.4f} "
               f"| L_sur(min): {loss_sur.item():.4f}")

    return surrogate, classifier
