"""
train.py — QLoRA Fine-Tuning Script
====================================
This script fine-tunes any HuggingFace causal LM using QLoRA.
Controlled entirely by a YAML config file.

Run:
    python scripts/train.py --config configs/phase1_mistral_dolly.yaml

Concepts covered:
    - 4-bit quantization with bitsandbytes
    - LoRA adapter injection with PEFT
    - Instruction formatting with chat templates
    - Supervised fine-tuning with TRL's SFTTrainer
    - Saving and pushing LoRA adapter to HuggingFace Hub
"""

import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


# ─────────────────────────────────────────────
# 1. Load Config
# ─────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 2. Build BitsAndBytes Quantization Config
# ─────────────────────────────────────────────

def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    """
    BitsAndBytesConfig tells the transformers library HOW to load
    the model weights — in 4-bit NF4 format instead of float32/float16.

    This is what makes QLoRA possible on consumer GPUs.
    Without this, Mistral-7B needs ~14GB just to load. With 4-bit: ~3.5GB.
    """
    mcfg = cfg["model"]
    return BitsAndBytesConfig(
        load_in_4bit=mcfg["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, mcfg["bnb_4bit_compute_dtype"]),
        # ^ getattr(torch, "float16") → torch.float16
        bnb_4bit_quant_type=mcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=mcfg["bnb_4bit_use_double_quant"],
    )


# ─────────────────────────────────────────────
# 3. Load Model + Tokenizer
# ─────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict, bnb_config: BitsAndBytesConfig):
    """
    AutoModelForCausalLM: Loads the model architecture + pretrained weights.
    "CausalLM" = Causal Language Model = predicts the NEXT token.
    (vs MaskedLM like BERT which predicts MASKED tokens)

    device_map="auto": HuggingFace Accelerate figures out which layers
    go on GPU vs CPU vs disk automatically. Essential for large models.
    """
    model_name = cfg["model"]["name"]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,  # needed for some models with custom code
    )

    # prepare_model_for_kbit_training:
    # After loading in 4-bit, some internal setup is needed before training:
    #   - Casts layer norms to float32 (stability)
    #   - Enables gradient checkpointing (trades compute for memory)
    #   - Unfreezes the embedding layer
    model = prepare_model_for_kbit_training(model)

    # AutoTokenizer: loads the tokenizer matched to the model.
    # The tokenizer converts text → token IDs (integers) and back.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Mistral tokenizer doesn't set pad_token by default.
    # pad_token is needed for batching (padding shorter sequences to equal length).
    # Convention: use eos_token as pad_token for decoder-only models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # padding_side="right": pad on the right side of sequences.
    # Required by SFTTrainer for correct loss masking.
    tokenizer.padding_side = "right"

    return model, tokenizer


# ─────────────────────────────────────────────
# 4. Inject LoRA Adapters
# ─────────────────────────────────────────────

def apply_lora(model, cfg: dict):
    """
    LoraConfig: defines where and how to add LoRA adapters.
    get_peft_model: injects the adapters into the model and freezes base weights.

    After this call:
    - Base model weights: FROZEN (no gradients)
    - LoRA adapter weights: TRAINABLE (gets gradients, gets updated)

    You can verify with: model.print_trainable_parameters()
    Typical output: "trainable params: 20,000,000 || all params: 3,750,000,000 || trainable%: 0.53"
    """
    lcfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["lora_alpha"],
        lora_dropout=lcfg["lora_dropout"],
        bias=lcfg["bias"],
        task_type=lcfg["task_type"],
        target_modules=lcfg["target_modules"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # always print this — sanity check
    return model


# ─────────────────────────────────────────────
# 5. Load and Format Dataset
# ─────────────────────────────────────────────

def format_dolly_sample(sample: dict, tokenizer) -> str:
    """
    Convert a Dolly dataset sample into Mistral's chat template format.

    Dolly sample structure:
    {
        "instruction": "What is the capital of France?",
        "context": "",           # optional background info
        "response": "Paris.",
        "category": "open_qa"
    }

    Mistral Instruct format:
    <s>[INST] {instruction}\n\n{context} [/INST] {response}</s>

    Why does format matter?
    The base model was instruction-tuned using this exact format.
    If you use a different format, the model won't recognize when to start
    responding (it'll confuse instruction tokens with response tokens).
    """
    instruction = sample["instruction"]
    context = sample.get("context", "").strip()
    response = sample["response"]

    # Build user message — include context if present
    user_msg = instruction
    if context:
        user_msg = f"{instruction}\n\nContext: {context}"

    # Apply the model's official chat template
    # This handles the [INST] / [/INST] tokens correctly
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": response},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,          # return string, not token IDs
        add_generation_prompt=False,  # we're training, not generating
    )
    return formatted


def load_and_prepare_dataset(cfg: dict, tokenizer):
    """
    Loads the dataset from HuggingFace Hub and formats each sample.
    Returns a HuggingFace Dataset with a single "text" column
    containing the fully formatted training string.
    """
    dcfg = cfg["dataset"]
    dataset = load_dataset(dcfg["name"], split=dcfg["split"])

    # Optionally subsample for faster iteration
    if dcfg.get("max_samples"):
        dataset = dataset.select(range(min(dcfg["max_samples"], len(dataset))))

    # Apply formatting to every sample
    dataset = dataset.map(
        lambda sample: {"text": format_dolly_sample(sample, tokenizer)},
        remove_columns=dataset.column_names,  # drop original columns, keep only "text"
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Example formatted sample:\n{dataset[0]['text'][:300]}...")
    return dataset


# ─────────────────────────────────────────────
# 6. Build Training Arguments
# ─────────────────────────────────────────────

def build_training_args(cfg: dict) -> TrainingArguments:
    """
    TrainingArguments: HuggingFace's container for all training hyperparameters.
    Pinned to TRL 0.8.6 API where max_seq_length/dataset_text_field/packing
    are passed directly to SFTTrainer.__init__, not here.

    Key concepts explained in the YAML config file.
    """
    tcfg = cfg["training"]
    return TrainingArguments(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        fp16=tcfg["fp16"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        report_to=tcfg["report_to"],
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        group_by_length=True,
    )


# ─────────────────────────────────────────────
# 7. Train
# ─────────────────────────────────────────────

def train(cfg: dict):
    # Build quantization config
    bnb_config = build_bnb_config(cfg)

    # Load model + tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg, bnb_config)

    # Inject LoRA adapters
    print("Applying LoRA adapters...")
    model = apply_lora(model, cfg)

    # Prepare dataset
    print("Loading and formatting dataset...")
    dataset = load_and_prepare_dataset(cfg, tokenizer)

    # Build training args
    training_args = build_training_args(cfg)

    # SFTTrainer: Supervised Fine-Tuning Trainer (from TRL library)
    # Wraps HuggingFace Trainer with:
    #   - Automatic response masking (only compute loss on assistant turns)
    #   - Dataset formatting helpers
    #   - LoRA-aware checkpointing
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["training"]["max_seq_length"],
        args=training_args,
        packing=False,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the LoRA adapter (NOT the full model — just the small adapter weights)
    output_dir = cfg["training"]["output_dir"]
    print(f"Saving adapter to {output_dir}/final-adapter")
    trainer.model.save_pretrained(f"{output_dir}/final-adapter")
    tokenizer.save_pretrained(f"{output_dir}/final-adapter")

    print("Training complete.")
    print(f"Adapter saved to: {output_dir}/final-adapter")
    print("Next step: run scripts/push_to_hub.py to upload to HuggingFace Hub")


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
