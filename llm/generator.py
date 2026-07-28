from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

try:
    from .prompts import PASP_PROMPT, PERTURB_PROMPTS
except ImportError:
    from prompts import PASP_PROMPT, PERTURB_PROMPTS


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

    def generate_pasp(self, subgraph_texts, structure, target_label,
                      max_new_tokens=256):
        """delta_i ~ P_psi(delta | X_i, Gamma(G_i), y*)."""
        texts = "\n".join(
            f"- Node {node_id}: {text}" for node_id, text in subgraph_texts
        )
        prompt = PASP_PROMPT.format(
            texts=texts, structure=structure, target_label=target_label
        )
        return self._generate(prompt, max_new_tokens)

    def perturb(self, text, perturb_type="lexical", max_new_tokens=128):
        prompt = PERTURB_PROMPTS[perturb_type].format(text=text)
        return self._generate(prompt, max_new_tokens)
