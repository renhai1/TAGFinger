"""Build graph-structured text data for fine-tuning the LLM generator.

Each SFT record pairs the user prompt template of Appendix C (instruction)
and the user content describing a stable region (input) with the perturbed
node texts (output), so that the fine-tuned Qwen3-8B proposes edits
conditioned on (X_i, Gamma(G_i), y*).
"""

import json

try:
    from .prompts import (USER_PROMPT_SINGLE, USER_PROMPT_MULTI,
                          USER_CONTENT_TEMPLATE)
except ImportError:
    from prompts import (USER_PROMPT_SINGLE, USER_PROMPT_MULTI,
                         USER_CONTENT_TEMPLATE)


def build_llm_finetune_dataset(raw_data, output_path):
    """raw_data: List[{
        "node_id": int, "text": str,
        "neighbors": [{"node_id": int, "text": str}, ...],
        "structure": str, "target_label": str,
        "perturbed_text": str, "mode": "single" | "multi" (optional)
    }]
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for item in raw_data:
            mode = item.get("mode", "single")
            instruction = (USER_PROMPT_MULTI if mode == "multi"
                           else USER_PROMPT_SINGLE)
            neighbors = "\n".join(
                f"- Node {n['node_id']}: {n['text']}"
                for n in item.get("neighbors", [])
            )
            content = USER_CONTENT_TEMPLATE.format(
                node_id=item["node_id"], target_text=item["text"],
                neighbor_texts=neighbors, structure=item["structure"],
                target_label=item["target_label"])
            record = {
                "instruction": instruction,
                "input": content,
                "output": f"Node {item['node_id']}: {item['perturbed_text']}"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_raw_text(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data