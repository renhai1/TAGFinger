"""LoRA fine-tuning of the LLM perturbation generator P_psi.

Appendix D: the fine-tuned Qwen3-8B model is used as the local LLM. It is
fine-tuned for 50 epochs on graph-structured text data using the Adam
optimizer with learning rate 1e-5 and weight decay 0.005.
"""

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset


MODEL_NAME = "Qwen/Qwen3-8B"
DATA_PATH = "perturb_data.jsonl"
OUTPUT_DIR = "perturb_llm_lora"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
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

    def format_example(example):
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        formatting_func=format_example,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            weight_decay=0.005,
            optim="adamw_torch",
            num_train_epochs=50,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True
        )
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()