

import json
import random
from prompts import PERTURB_PROMPTS


def build_llm_finetune_dataset(
    raw_data,
    output_path,
    perturb_types=("lexical", "syntactic", "discourse")
):
    """
    raw_data: List[{"text": str}]
    output_path: jsonl for SFT
    """

    with open(output_path, "w", encoding="utf-8") as f:
        for item in raw_data:
            text = item["text"]
            ptype = random.choice(perturb_types)

            record = {
                "instruction": "You are a text perturbation generator.",
                "input": PERTURB_PROMPTS[ptype].format(text=text),
                "output": item.get("perturbed_text", "")
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_raw_text(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append({"text": obj["text"]})
    return data
