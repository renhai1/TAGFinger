"""LoRA fine-tuning of the LLM perturbation generator P_psi.

Appendix D: the fine-tuned Qwen3-8B model is used as the local LLM. It is
fine-tuned for 50 epochs on graph-structured text data using the Adam
optimizer with learning rate 1e-5 and weight decay 0.005.
"""

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset


MODEL_NAME = "Qwen/Qwen3-8B"
DATA_PATH = "perturb_data.jsonl"
OUTPUT_DIR = "perturb_llm_lora"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": "cuda:0" if torch.cuda.is_available() else "cpu"},
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=DATA_PATH)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_example(example):
        text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
        return tokenizer(text, truncation=True, max_length=1024)

    tokenized = dataset["train"].map(
        tokenize_example, remove_columns=dataset["train"].column_names)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=1e-5, weight_decay=0.005)
    trainer = Trainer(
        model=model,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        optimizers=(optimizer, None),
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            weight_decay=0.005,
            num_train_epochs=50,
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available()
        )
    )

    # The explicit optimizer follows the manuscript's Adam configuration.
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
