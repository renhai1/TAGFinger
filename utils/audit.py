"""Reusable, model-specific fingerprint construction and verification.

The same protected inputs and configuration are audited against each model.
Each model gets its own SAKA surrogate and prompt/text optimization. Only
the query/query_graph methods of that model are used by the auditor.
"""
import torch
import torch.nn.functional as F

from data.graph_instances import original_graph
from model.SAKA import GAGGenerator, train_saka
from prompt.TUP import TaskUnifiedPrompt
from utils.pasp import (stability_scores, select_fingerprint_nodes,
                        select_stable_region, structural_description,
                        distribution_drift)
from utils.protocol import fingerprint_evidence, ownership_decision, decision_rates


def make_task_instances(source, tasks, node, edge_partner, x_source):
    instances = {"node": int(node)}
    if "edge" in tasks:
        if edge_partner is None:
            raise ValueError("Edge verification needs distinct valid endpoints")
        instances["edge"] = (int(node), int(edge_partner))
    if "graph" in tasks:
        instances["graph"] = original_graph(
            source, int(source.graph_index[node]), x=x_source)
    return instances


def text_candidate_delta(source, encoder, edits, budget, device, validated_nodes=None):
    """Exact re-encoding difference. Over-budget candidates are rejected.

    No interpolated or clipped embeddings are substituted for edited texts.
    None signals an invalid/over-budget candidate; caller must not silently
    replace all rejected candidates with an identity perturbation.
    """
    if not edits:
        return None
    ids = [int(node) for node in edits]
    if any(node < 0 or node >= source.num_nodes for node in ids):
        raise ValueError("Text edit references an unknown node")
    unseen = [node for node in ids if validated_nodes is None or node not in validated_nodes]
    if unseen:
        encoded_clean = encoder.encode([source.raw_texts[n] for n in unseen]).to(device)
        reference = source.x[unseen]
        if encoded_clean.shape != reference.shape or not torch.allclose(
                encoded_clean, reference, rtol=1e-4, atol=1e-6):
            raise ValueError("--text_encoder does not reproduce the prepared clean features")
        if validated_nodes is not None:
            validated_nodes.update(unseen)
    encoded = encoder.encode([edits[node] for node in ids]).to(device)
    if encoded.shape != source.x[ids].shape or not bool(torch.isfinite(encoded).all()):
        raise ValueError("Edited-text encoder output has invalid dimensions/values")
    change = encoded - source.x[ids]
    if float(change.norm()) > budget or float(change.norm()) == 0:
        return None
    delta = torch.zeros_like(source.x)
    delta[ids] = change
    return delta


class OwnershipAudit:
    def __init__(self, source, args, tasks, device, encoder, perturbator=None,
                 cached_candidates=None, log_fn=print):
        self.data, self.args, self.tasks, self.device = source, args, tasks, device
        self.encoder, self.perturbator = encoder, perturbator
        if encoder is None or (perturbator is None and cached_candidates is None):
            raise ValueError("Text fingerprint construction requires an encoder and candidate source")
        self.cache, self.log = cached_candidates, log_fn
        self.mode = args.llm_mode or ("multi" if args.mix_dataset else "single")
        self.validated_nodes = set()

    def node_probs(self, x, edge_index):
        logits, _ = self.classifier(self.surrogate(x, edge_index))
        return F.softmax(logits, dim=1)

    def graph_probs(self, x, edge_index, tau):
        logits, _ = self.heads[tau](self.surrogate(x, edge_index))
        return F.softmax(logits, dim=1).squeeze(0)

    def instances(self, i, x):
        return make_task_instances(self.data, self.tasks, int(self.nodes[i]),
                                   self.partners[i], x)

    def align(self, blackbox, surrogate, classifier, heads):
        self.surrogate, self.classifier, self.heads = surrogate, classifier, heads
        args, data = self.args, self.data
        generator = GAGGenerator(data.x.size(1), args.gen_hidden,
                                 args.x_budget, args.edge_budget).to(self.device)
        train_saka(surrogate, classifier, generator, blackbox, data,
                   graph_heads=heads, epochs=args.saka_epochs,
                   lr=args.saka_lr, weight_decay=args.weight_decay,
                   patience=args.patience, batch_nodes=args.batch_node,
                   batch_graphs=args.batch_graph, num_hops=args.num_hops,
                   use_dac=not args.no_dac, use_dvc=not args.no_dvc,
                   log_fn=self.log)
        for module in (surrogate, classifier, *heads.values()):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def select_regions(self):
        data, args = self.data, self.args
        with torch.no_grad():
            self.clean_probs = self.node_probs(data.x, data.edge_index)
        scores = stability_scores(self.clean_probs, data.edge_index, data.num_nodes)
        allowed = getattr(data, "owner_mask", None)
        if allowed is None:
            allowed = torch.ones(data.num_nodes, dtype=torch.bool, device=self.device)
        if "edge" in self.tasks:
            src, dst = data.edge_index
            has_neighbor = torch.zeros(data.num_nodes, dtype=torch.bool, device=self.device)
            has_neighbor[dst[src != dst]] = True
            allowed = allowed & has_neighbor
        scores = scores.masked_fill(~allowed, float("-inf"))
        self.nodes = select_fingerprint_nodes(
            scores, data.edge_index, data.num_nodes, args.total_select,
            args.topk_stable, allowed_mask=allowed)
        if not self.nodes.numel():
            raise ValueError("No eligible protected-source fingerprint regions")
        self.regions, self.partners = [], []
        for node in self.nodes.tolist():
            region = select_stable_region(node, scores, data.edge_index, args.topk_stable)
            self.regions.append(region)
            neighbors = data.edge_index[0, data.edge_index[1] == node]
            neighbors = neighbors[neighbors != node]
            self.partners.append(int(neighbors[0]) if neighbors.numel() else None)
        self.log(f"[PASP] selected {self.nodes.numel()} of {args.total_select} requested fingerprints")
        self.prompts = TaskUnifiedPrompt(
            token_dim=data.x.size(1), token_num=args.prompt_tokens,
            cross_prune=args.cross_prune, inner_prune=args.inner_prune,
            num_hops=args.num_hops, use_prompt=not args.no_tup).to(self.device)
        self.targets = []
        with torch.no_grad():
            for i in range(self.nodes.numel()):
                targets = {}
                for tau, instance in self.instances(i, data.x).items():
                    probs = self.prompts.task_probability(
                        tau, instance, data.x, data.edge_index, self.graph_probs)
                    if probs.numel() < 2:
                        raise ValueError("Each verification task needs at least two classes")
                    targets[tau] = int(probs.topk(2).indices[1])
                self.targets.append(targets)

    def proposals(self, i):
        node = int(self.nodes[i])
        if self.perturbator is not None:
            neighbors = [(int(u), self.data.raw_texts[int(u)]) for u in self.regions[i][1:]]
            return self.perturbator.generate_candidates(
                node, self.data.raw_texts[node], neighbors,
                structural_description(node, self.regions[i], self.data.edge_index),
                str(self.targets[i]["node"]), num_candidates=self.args.num_candidates,
                mode=self.mode)
        entry = self.cache.get(str(node), [])
        # Old single-proposal cache files remain readable. Cached texts are
        # proposals; they are scored independently for every audited model.
        entries = [entry] if isinstance(entry, dict) else entry
        if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
            raise ValueError("Each cache entry must be an edit dictionary or list of dictionaries")
        if any(not isinstance(text, str) or not text.strip()
               for edits in entries for text in edits.values()):
            raise ValueError("Candidate cache contains empty or non-text edits")
        return [{int(node): text for node, text in edits.items()} for edits in entries]

    def objective(self, x, i, regional_probs):
        result = self.args.lam * distribution_drift(
            self.clean_probs, regional_probs, self.regions[i])
        for tau, instance in self.instances(i, x).items():
            probs = self.prompts.task_probability(
                tau, instance, x, self.data.edge_index, self.graph_probs)
            result = result + probs[self.targets[i][tau]]
        return result

    def construct(self):
        data, args = self.data, self.args
        optimizer = torch.optim.Adam(self.prompts.parameters(), lr=args.pasp_lr)
        self.delta = torch.zeros_like(data.x)
        for round_id in range(args.pasp_rounds):
            self.prompts.eval()
            combined = torch.zeros_like(data.x)
            selected_edits = {}
            for i in range(self.nodes.numel()):
                allowed = set(int(u) for u in self.regions[i])
                best = None
                with torch.no_grad():
                    for edits in self.proposals(i):
                        if not edits or not set(edits) <= allowed:
                            continue
                        candidate = text_candidate_delta(data, self.encoder, edits,
                            args.delta_budget, self.device, self.validated_nodes)
                        if candidate is None:
                            continue
                        x = data.x + candidate
                        value = float(self.objective(x, i, self.node_probs(x, data.edge_index)))
                        if best is None or value > best[0]:
                            best = (value, candidate, edits)
                if best is None:
                    raise ValueError(f"No valid nonzero text candidate within budget for node {int(self.nodes[i])}; "
                                     "provide more candidates or check the documented budget")
                combined += best[1]
                selected_edits.update(best[2])
            self.delta, self.edits = combined, selected_edits
            if not args.no_tup:
                for _ in range(args.prompt_steps):
                    optimizer.zero_grad()
                    x = data.x + self.delta
                    regional_probs = self.node_probs(x, data.edge_index)
                    loss = -sum(self.objective(x, i, regional_probs)
                                for i in range(self.nodes.numel())) / self.nodes.numel()
                    loss.backward()
                    optimizer.step()
            self.log(f"[PASP] round {round_id + 1}: text edits selected; prompts updated")
        self.prompts.eval()

    @torch.no_grad()
    def verify(self, blackbox):
        decisions = {"single": [], "multi": []}
        by_task = {tau: [] for tau in self.tasks}
        x_pert = self.data.x + self.delta
        for i in range(self.nodes.numel()):
            evidence = {}
            clean, perturbed = self.instances(i, self.data.x), self.instances(i, x_pert)
            for tau in self.tasks:
                q_clean = self.prompts.task_probability(tau, clean[tau], self.data.x,
                                    self.data.edge_index, blackbox.query_graph)
                q_pert = self.prompts.task_probability(tau, perturbed[tau], x_pert,
                                    self.data.edge_index, blackbox.query_graph)
                evidence[tau] = fingerprint_evidence(int(q_pert.argmax()),
                                     int(q_clean.argmax()), self.targets[i][tau])
                by_task[tau].append(ownership_decision([evidence[tau]], 1))
            decisions["single"].append(ownership_decision([evidence["node"]], 1))
            decisions["multi"].append(ownership_decision(evidence.values(), self.args.theta))
        return {"decisions": decisions, "task_decisions": by_task,
                "rates": {name: decision_rates(values) for name, values in decisions.items()},
                "fingerprint_count": self.nodes.numel()}
