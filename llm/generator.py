"""LLM perturbation generator P_psi for PASP construction (Eq. 6).

The LLM proposes edits to the texts of nodes in the selected stable region.
Each edited text is encoded again with the same text encoder to update the
node features; the feature perturbation is the difference between the
original and updated encodings. No gradients are passed through text
generation: candidate edits are sampled, scored on the surrogate, and the
best one is retained (Sec. 4.3).
"""

import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from .prompts import build_pasp_prompt, PERTURB_PROMPTS
except ImportError:
    from prompts import build_pasp_prompt, PERTURB_PROMPTS


def parse_edits(output, target_node):
    """Parse 'Node <id>: <edited text>' lines into {node_id: text}."""
    edits = {}
    for line in output.splitlines():
        match = re.match(r"\s*Node\s+(\d+)\s*[:：]\s*(.+)", line)
        if match:
            edits[int(match.group(1))] = match.group(2).strip()
    if not edits and output.strip():
        edits[int(target_node)] = output.strip()
    return edits


class LLMPerturbator:
    def __init__(self, model_path, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        self.device = device

    @torch.no_grad()
    def _generate(self, prompt, max_new_tokens):
        inputs = self.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True
        ).strip()

    def generate_candidates(self, node_id, target_text, neighbor_texts,
                            structure, target_label, num_candidates=4,
                            mode="single", max_new_tokens=512):
        """Sample candidate text edits for the stable region of node_id.

        Returns a list of edit dicts {node_id: edited_text}; each candidate
        is later re-encoded and scored on the surrogate with the objective
        sum_tau q_tau(y*) + lambda * Delta_i.
        """
        prompt = build_pasp_prompt(
            node_id, target_text, neighbor_texts, structure, target_label,
            mode=mode)
        candidates = []
        for _ in range(num_candidates):
            output = self._generate(prompt, max_new_tokens)
            edits = parse_edits(output, node_id)
            if edits:
                candidates.append(edits)
        return candidates

    def generate_pasp(self, subgraph_texts, structure, target_label,
                      mode="single", max_new_tokens=512):
        """delta_i ~ P_psi(delta | X_i, Gamma(G_i), y*): a single edit."""
        node_id, target_text = subgraph_texts[0]
        candidates = self.generate_candidates(
            node_id, target_text, subgraph_texts[1:], structure, target_label,
            num_candidates=1, mode=mode, max_new_tokens=max_new_tokens)
        return candidates[0] if candidates else {}

    def perturb(self, text, perturb_type="lexical", max_new_tokens=128):
        prompt = PERTURB_PROMPTS[perturb_type].format(text=text)
        return self._generate(prompt, max_new_tokens)