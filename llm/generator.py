

from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import PERTURB_PROMPTS
import torch


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
    def perturb(self, text, perturb_type="lexical", max_new_tokens=128):
        prompt = PERTURB_PROMPTS[perturb_type].format(text=text)

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

        gen_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

        return gen_text.split("Text:")[-1].strip()
