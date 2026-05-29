"""
train_runpod_medical.py — Phase 2 Medical, GPU-agnostic for RunPod
====================================================================
Auto-detects GPU and picks the optimal config:

    VRAM >= 70GB  (A100 80GB, H100 80GB):
        bf16 LoRA, batch=16, no grad ckpt, flash-attn2  → ~15 min

    VRAM >= 40GB  (A40 48GB, L40S 48GB, A6000 48GB):
        bf16 LoRA, batch=8,  no grad ckpt, flash-attn2  → ~25 min

    VRAM >= 22GB  (RTX 4090 24GB, A5000 24GB):
        4-bit QLoRA, batch=8, grad ckpt on, flash-attn2 → ~45 min

    VRAM < 22GB   (T4 16GB, RTX 3090 if shared, etc.):
        4-bit QLoRA, batch=4, grad ckpt on, no flash-attn → ~90 min

Override the auto-pick with env vars: BATCH_SIZE, USE_4BIT=1, NO_FLASH=1

Setup on RunPod:
    cd /workspace && git clone <repo> jobs-prjcts && cd jobs-prjcts/03-llm-finetuning
    pip install -r requirements_runpod.txt
    huggingface-cli login
    python scripts/train_runpod_medical.py
"""

import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


# ─────────────────────────────────────────────
# Static config
# ─────────────────────────────────────────────

BASE_MODEL     = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_NAME   = "medalpaca/medical_meadow_medqa"
OUTPUT_DIR     = "/workspace/outputs/phase2-medical-runpod"

NUM_EPOCHS     = 2
LEARNING_RATE  = 2e-4
MAX_SEQ_LENGTH = 1024
LOGGING_STEPS  = 5
SAVE_STEPS     = 100

LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

TEST_SIZE        = 0.10
MAX_EVAL_SAMPLES = 500

MEDICAL_SYSTEM = (
    "You are a knowledgeable medical AI assistant. "
    "When given a clinical multiple-choice question, analyze the case carefully, "
    "identify the correct answer (A, B, C, or D), and provide a clear explanation. "
    "Always begin your response with 'The correct answer is X)' where X is the letter."
)


# ─────────────────────────────────────────────
# GPU auto-detect → config tier
# ─────────────────────────────────────────────

def detect_gpu_tier():
    """
    Returns a dict of training config tuned to the detected GPU.
    Environment overrides:
        USE_4BIT=1         force 4-bit even on big GPUs
        NO_FLASH=1         disable Flash Attention 2 (e.g. flash-attn not installed)
        BATCH_SIZE=N       override per-device batch size
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")

    props      = torch.cuda.get_device_properties(0)
    name       = props.name
    vram_gb    = props.total_memory / 1e9
    capability = props.major + props.minor / 10
    bf16_ok    = capability >= 8.0      # Ampere and newer
    flash_ok   = capability >= 8.0 and not os.environ.get("NO_FLASH")

    print(f"GPU: {name}  ({vram_gb:.1f} GB, sm_{props.major}{props.minor})")
    print(f"bf16 supported: {bf16_ok} | flash-attn2 eligible: {flash_ok}")

    force_4bit = bool(os.environ.get("USE_4BIT"))

    # Pick tier
    if vram_gb >= 70 and not force_4bit:
        tier = dict(
            label="A100/H100 80GB tier",
            use_4bit=False,
            batch_size=16,
            grad_accum=2,
            grad_ckpt=False,
        )
    elif vram_gb >= 40 and not force_4bit:
        tier = dict(
            label="A40/L40S/A6000 48GB tier",
            use_4bit=False,
            batch_size=8,
            grad_accum=4,
            grad_ckpt=False,
        )
    elif vram_gb >= 22:
        tier = dict(
            label="RTX 4090 / A5000 24GB tier",
            use_4bit=True,
            batch_size=8,
            grad_accum=4,
            grad_ckpt=True,
        )
    else:
        tier = dict(
            label="T4 / small-VRAM tier",
            use_4bit=True,
            batch_size=4,
            grad_accum=4,
            grad_ckpt=True,
        )

    # Env overrides
    if os.environ.get("BATCH_SIZE"):
        tier["batch_size"] = int(os.environ["BATCH_SIZE"])

    # Cross-cutting flags
    tier["bf16"]     = bf16_ok
    tier["fp16"]     = not bf16_ok
    tier["flash"]    = flash_ok
    tier["vram_gb"]  = vram_gb
    tier["gpu_name"] = name

    print(f"\nSelected tier: {tier['label']}")
    print(f"  use_4bit         = {tier['use_4bit']}")
    print(f"  batch_size       = {tier['batch_size']}  (grad_accum={tier['grad_accum']}, "
          f"effective={tier['batch_size']*tier['grad_accum']})")
    print(f"  grad_checkpoint  = {tier['grad_ckpt']}")
    print(f"  precision        = {'bf16' if tier['bf16'] else 'fp16'}")
    print(f"  flash_attention2 = {tier['flash']}\n")
    return tier


# ─────────────────────────────────────────────
# Model loading — branches on tier
# ─────────────────────────────────────────────

def load_model_and_tokenizer(tier: dict):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    load_kwargs: dict = {
        "device_map": {"": 0},
        "trust_remote_code": True,
    }
    if tier["flash"]:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    if tier["use_4bit"]:
        print(f"Loading {BASE_MODEL} in 4-bit NF4...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if tier["bf16"] else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, **load_kwargs
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=tier["grad_ckpt"]
        )
    else:
        print(f"Loading {BASE_MODEL} in bf16 (no quantization)...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16 if tier["bf16"] else torch.float16,
            **load_kwargs,
        )

    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def inject_lora(model):
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

def has_5_options(sample: dict) -> bool:
    return bool(re.search(r'\bE\)', sample['input']))


def format_medqa_sample(sample: dict, tokenizer) -> str:
    messages = [
        {"role": "system",    "content": MEDICAL_SYSTEM},
        {"role": "user",      "content": sample["input"]},
        {"role": "assistant", "content": sample["output"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_data(tokenizer):
    print(f"Loading {DATASET_NAME}...")
    raw = load_dataset(DATASET_NAME, split="train")
    raw = raw.filter(lambda s: not has_5_options(s))
    print(f"After filtering 5-option samples: {len(raw)}")

    split = raw.train_test_split(test_size=TEST_SIZE, seed=42)
    train_raw, test_raw = split["train"], split["test"]
    print(f"Train: {len(train_raw)} | Test: {len(test_raw)}")

    train_dataset = train_raw.map(
        lambda s: {"text": format_medqa_sample(s, tokenizer)},
        remove_columns=train_raw.column_names,
    )
    return train_dataset, test_raw


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────

def train(model, tokenizer, train_dataset, tier: dict):
    optim = "adamw_torch_fused" if not tier["use_4bit"] else "paged_adamw_32bit"

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=tier["batch_size"],
        gradient_accumulation_steps=tier["grad_accum"],
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=tier["bf16"],
        fp16=tier["fp16"],
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        report_to="none",
        optim=optim,
        gradient_checkpointing=tier["grad_ckpt"],
        group_by_length=True,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        tf32=tier["bf16"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    eff_batch       = int(tier["batch_size"]) * int(tier["grad_accum"])
    steps_per_epoch = len(train_dataset) // eff_batch
    print(f"\nTraining: {len(train_dataset)} samples × {NUM_EPOCHS} epochs")
    print(f"Effective batch: {eff_batch}")
    print(f"Steps per epoch: ~{steps_per_epoch}  |  Total: ~{steps_per_epoch * NUM_EPOCHS}\n")

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining wall time: {elapsed/60:.1f} min")

    adapter_path = os.path.join(OUTPUT_DIR, "final-adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved: {adapter_path}")
    return trainer, adapter_path


# ─────────────────────────────────────────────
# Evaluation — MCQ accuracy
# ─────────────────────────────────────────────

def extract_answer_letter(text: str) -> str:
    text = text.strip()
    m = re.search(r'correct answer is\s+([A-D])[).]?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'[Aa]nswer[:\s]+([A-D])[).]?', text)
    if m:
        return m.group(1).upper()
    m = re.match(r'^([A-D])[).\s]', text)
    if m:
        return m.group(1).upper()
    return "?"


def extract_ground_truth(output_text: str) -> str:
    m = re.search(r'correct answer is\s+([A-D])[).]?', output_text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r'^([A-D])[).\s]', output_text.strip())
    return m.group(1).upper() if m else "?"


def generate_medical(model, tokenizer, question: str, max_new_tokens: int = 200) -> str:
    messages = [
        {"role": "system", "content": MEDICAL_SYSTEM},
        {"role": "user",   "content": question},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def evaluate(model, tokenizer, test_raw, adapter_path: str, tier: dict):
    from tqdm import tqdm

    model.eval()
    samples = test_raw.select(range(min(MAX_EVAL_SAMPLES, len(test_raw))))
    correct = total = no_answer = 0

    for sample in tqdm(samples, desc="MCQ eval"):
        gt = extract_ground_truth(sample['output'])
        if gt == "?":
            continue
        resp = generate_medical(model, tokenizer, sample['input'])
        pred = extract_answer_letter(resp)
        if pred == "?":
            no_answer += 1
        correct += (pred == gt)
        total   += 1

    accuracy = correct / total if total else 0.0
    print(f"\n{'='*50}")
    print(f"MCQ ACCURACY: {accuracy:.1%} ({correct}/{total})")
    print(f"No-answer rate: {no_answer/total:.1%}")
    print(f"{'='*50}")

    results = {
        "phase":           "phase2_medical_runpod",
        "gpu":             tier["gpu_name"],
        "vram_gb":         round(tier["vram_gb"], 1),
        "tier":            tier["label"],
        "model":           BASE_MODEL,
        "adapter":         adapter_path,
        "dataset":         DATASET_NAME,
        "num_eval":        total,
        "correct":         correct,
        "accuracy":        round(accuracy, 4),
        "no_answer_rate":  round(no_answer / total, 4) if total else 0.0,
    }
    with open(os.path.join(OUTPUT_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eval", action="store_true",
                        help="Train only, skip MCQ evaluation")
    parser.add_argument("--push-hub", default=None,
                        help="HF repo to push adapter to (e.g. anksriv/mistral-7b-medical-medqa-qlora)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}\n")

    tier = detect_gpu_tier()
    model, tokenizer = load_model_and_tokenizer(tier)
    model = inject_lora(model)
    train_dataset, test_raw = load_data(tokenizer)
    trainer, adapter_path = train(model, tokenizer, train_dataset, tier)

    if not args.skip_eval:
        evaluate(trainer.model, tokenizer, test_raw, adapter_path, tier)

    if args.push_hub:
        print(f"\nPushing adapter to {args.push_hub}...")
        trainer.model.push_to_hub(args.push_hub)
        tokenizer.push_to_hub(args.push_hub)
        print(f"Pushed: https://huggingface.co/{args.push_hub}")


if __name__ == "__main__":
    main()
