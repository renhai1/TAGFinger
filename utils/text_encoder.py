"""The shared text encoder of TAGFinger.

Node features are produced by encoding the textual attributes with a language
model (Appendix E: GCN_LLMEmb for Cora-ML/CiteSeer-ML, ENGINE for the ArXiv
datasets, TAPE for Reddit). During PASP construction, each edited text is
encoded again with the SAME text encoder to update the node features, and the
feature perturbation delta_i is the difference between the original and
updated encodings (Sec. 4.3).
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class TextEncoder:
    def __init__(self, model_path, device="cpu", batch_size=64,
                 row_normalize=True):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True).to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        # matches T.NormalizeFeatures applied to the prepared node features
        self.row_normalize = row_normalize

    @torch.no_grad()
    def encode(self, texts):
        embeddings = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(dim=1) / \
                mask.sum(dim=1).clamp(min=1e-9)
            embeddings.append(emb.cpu())
        x = torch.cat(embeddings, dim=0)
        if self.row_normalize:
            x = F.normalize(x, p=1, dim=1)
        return x