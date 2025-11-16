import argparse
import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


def build_prompt_text(example: dict) -> str:
    """
    Build a simple instruction-following text from one JSONL row with keys:
    - input: the instruction/context
    - output: the target answer
    - task: optional label, not used during SFT

    We avoid any KG-specific handling (no stru_ids tensors, no custom tokenization).
    """
    user = example.get("input", "").strip()
    assistant = example.get("output", "").strip()
    # A lightweight chat-style template compatible with Qwen chat formatting tokens.
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant for next POI recommendation."
        "<|im_end|>\n"
        "<|im_start|>user\n" + user + "<|im_end|>\n"
        "<|im_start|>assistant\n" + assistant + "<|im_end|>"
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 on JSONL prompts (LoRA, SFT)")
    parser.add_argument("--location", default="CA", choices=["CA", "NYC", "TKY"], help="Dataset location tag used in file naming")
    parser.add_argument("--prompts_path", default=None, help="Path to generated prompts JSONL; defaults to generated_prompts_{LOCATION}.jsonl")
    parser.add_argument("--model_name", default="unsloth/Qwen3-4B-unsloth-bnb-4bit", help="Base model to fine-tune")
    parser.add_argument("--output_dir", default=None, help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--per_device_batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="Fraction of data for eval split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length for training")
    args = parser.parse_args()

    location = args.location
    prompts_path = args.prompts_path or f"generated_prompts_{location}.jsonl"
    output_dir = args.output_dir or f"./kgtb_llm_{location}_finetuned"

    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompt file not found at {prompts_path}. Run generate_prompts.py first.")

    print("Loading dataset from:", prompts_path)
    # Load JSONL as a Hugging Face Dataset without materializing into pandas first
    dataset = load_dataset("json", data_files=prompts_path, split="train")
    # Build a plain text field for SFTTrainer; no explicit tokenization here
    dataset = dataset.map(lambda ex: {"text": build_prompt_text(ex)}, desc="Formatting prompts")

    # Split into train/eval
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Dataset ready. Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    print(f"Initializing model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # let Unsloth pick
        load_in_4bit=True,
        trust_remote_code=True,
    )

    print("Enabling LoRA (r=16) for parameter-efficient fine-tuning…")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    print("Setting up trainer…")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=f"outputs_{location}",
            eval_strategy="steps",
            eval_steps=200,
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="tensorboard",
            max_grad_norm=0.3,
        ),
    )

    print("Starting training…")
    trainer.train()

    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
