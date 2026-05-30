"""
train_runpod_coding.py — Phase 5 Coding, GPU-agnostic for RunPod
=================================================================
Fine-tunes Mistral-7B-Instruct-v0.2 on code generation tasks.

  Dataset: HuggingFaceH4/CodeAlpaca_20K
           20,111 samples — all used (no subset needed at this size)
           Fields: {prompt, completion}  ← different from Dolly/Finance!
           Languages: Python-heavy but mixed (JS, Java, C++, etc.)

  Eval:    HumanEval pass@1
           Baseline: Mistral-7B-Instruct-v0.2 ~35-40% pass@1 (no fine-tuning)
           Target:   >=40% pass@1 after fine-tuning

GPU tiers (seq_len=1024):

    VRAM >= 70GB  (A100 80GB, H100 80GB):
        bf16 LoRA, batch=8, no grad ckpt, flash-attn2  → ~20 min

    VRAM >= 40GB  (A40/L40S/A6000 48GB):
        bf16 LoRA, batch=4, no grad ckpt, flash-attn2  → ~35 min

    VRAM >= 22GB  (RTX 4090 / A5000 24GB):
        4-bit QLoRA, batch=4, grad ckpt on, flash-attn2 → ~50 min

    VRAM < 22GB   (T4 16GB, etc.):
        4-bit QLoRA, batch=2, grad ckpt on, no flash-attn → ~90 min

Setup on RunPod:
    cd /workspace && git clone <repo> jobs-prjcts && cd jobs-prjcts/03-llm-finetuning
    pip install -r requirements_runpod.txt
    pip install human-eval          # for pass@1 eval
    huggingface-cli login
    python scripts/train_runpod_coding.py
"""

import json
import os
import subprocess
import sys
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


# ─────────────────────────────────────────────
# Static config
# ─────────────────────────────────────────────

BASE_MODEL      = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_NAME    = "HuggingFaceH4/CodeAlpaca_20K"
OUTPUT_DIR      = "/workspace/outputs/phase5-coding"

MAX_SAMPLES     = None          # use full 20K dataset
SHUFFLE_SEED    = 42
NUM_EPOCHS      = 3
LEARNING_RATE   = 2.0e-4
MAX_SEQ_LENGTH  = 1024          # code completions need more headroom than finance
LOGGING_STEPS   = 10
SAVE_STEPS      = 200
TEST_SIZE       = 0.05          # ~1K held-out; most eval happens via HumanEval

LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

CODING_SYSTEM = (
    "You are an expert software engineer and programmer. When given a coding task "
    "or programming problem, write clean, correct, and well-structured code. "
    "Follow best practices for the language being used. If the task is ambiguous, "
    "state your assumptions briefly before the code. Produce working solutions."
)

# Number of HumanEval problems to evaluate (164 total; 30 is fast, 164 is thorough)
HUMANEVAL_N_PROBLEMS = 30


# ─────────────────────────────────────────────
# GPU auto-detect
# ─────────────────────────────────────────────

def detect_gpu_tier() -> dict:
    """
    seq_len=1024 — 2x finance, half of legal.
    Batch sizes are halved vs finance for same VRAM budget.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")

    props      = torch.cuda.get_device_properties(0)
    name       = props.name
    vram_gb    = props.total_memory / 1e9
    capability = props.major + props.minor / 10
    bf16_ok    = capability >= 8.0
    flash_ok   = capability >= 8.0 and not os.environ.get("NO_FLASH")
    force_4bit = bool(os.environ.get("USE_4BIT"))

    print(f"GPU: {name}  ({vram_gb:.1f} GB, sm_{props.major}{props.minor})")
    print(f"bf16 supported: {bf16_ok} | flash-attn2 eligible: {flash_ok}")

    if vram_gb >= 70 and not force_4bit:
        tier = dict(label="A100/H100 80GB tier", use_4bit=False,
                    batch_size=8, grad_accum=2, grad_ckpt=False)
    elif vram_gb >= 40 and not force_4bit:
        tier = dict(label="A40/L40S/A6000 48GB tier", use_4bit=False,
                    batch_size=4, grad_accum=4, grad_ckpt=False)
    elif vram_gb >= 22:
        tier = dict(label="RTX 4090 / A5000 24GB tier", use_4bit=True,
                    batch_size=4, grad_accum=4, grad_ckpt=True)
    else:
        tier = dict(label="T4 / small-VRAM tier", use_4bit=True,
                    batch_size=2, grad_accum=8, grad_ckpt=True)

    if os.environ.get("BATCH_SIZE"):
        tier["batch_size"] = int(os.environ["BATCH_SIZE"])

    tier.update(bf16=bf16_ok, fp16=not bf16_ok,
                flash=flash_ok, vram_gb=vram_gb, gpu_name=name)

    print(f"\nSelected tier: {tier['label']}")
    print(f"  use_4bit         = {tier['use_4bit']}")
    print(f"  batch_size       = {tier['batch_size']}  "
          f"(grad_accum={tier['grad_accum']}, "
          f"effective={int(tier['batch_size'])*int(tier['grad_accum'])})")
    print(f"  grad_checkpoint  = {tier['grad_ckpt']}")
    print(f"  precision        = {'bf16' if tier['bf16'] else 'fp16'}")
    print(f"  flash_attention2 = {tier['flash']}\n")
    return tier


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_model_and_tokenizer(tier: dict):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    load_kwargs: dict = {"device_map": {"": 0}, "trust_remote_code": True}
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
        print(f"Loading {BASE_MODEL} in bf16...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16 if tier["bf16"] else torch.float16,
            **load_kwargs,
        )

    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def inject_lora(model):
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

def filter_by_token_length(sample: dict, tokenizer) -> bool:
    """
    Drop samples whose combined prompt+completion exceeds MAX_SEQ_LENGTH.
    Avoids mid-function truncation which teaches incomplete code patterns.
    A 50-token buffer for special tokens keeps us safely under the limit.
    """
    n_tokens = len(tokenizer.encode(sample["prompt"] + sample["completion"]))
    return n_tokens <= MAX_SEQ_LENGTH - 50


def format_code_sample(sample: dict, tokenizer) -> str:
    """
    HuggingFaceH4/CodeAlpaca_20K fields: {prompt, completion}
    Unlike Dolly/Finance which use {instruction, input, output}.
    Preserve exact whitespace — indentation is semantically meaningful.
    """
    messages = [
        {"role": "system",    "content": CODING_SYSTEM},
        {"role": "user",      "content": sample["prompt"]},
        {"role": "assistant", "content": sample["completion"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_data(tokenizer):
    print(f"Loading dataset: {DATASET_NAME}...")
    raw = load_dataset(DATASET_NAME, split="train")
    print(f"Full dataset size: {len(raw)} samples")

    # Filter samples that would be truncated mid-function
    before = len(raw)
    raw = raw.filter(lambda s: filter_by_token_length(s, tokenizer))
    print(f"After length filter (max {MAX_SEQ_LENGTH} tokens): "
          f"{len(raw)} samples ({before - len(raw)} dropped)")

    raw = raw.shuffle(seed=SHUFFLE_SEED)

    split     = raw.train_test_split(test_size=TEST_SIZE, seed=42)
    train_raw = split["train"]
    test_raw  = split["test"]
    print(f"Train: {len(train_raw)} | Test (held-out): {len(test_raw)}")

    train_ds = train_raw.map(
        lambda s: {"text": format_code_sample(s, tokenizer)},
        remove_columns=train_raw.column_names,
    )

    return train_ds, test_raw


# ─────────────────────────────────────────────
# Training
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

    eff_batch       = tier["batch_size"] * tier["grad_accum"]
    steps_per_epoch = len(train_dataset) // eff_batch
    print(f"\nTraining: {len(train_dataset)} samples × {NUM_EPOCHS} epochs")
    print(f"Effective batch: {eff_batch} | Steps/epoch: ~{steps_per_epoch} | "
          f"Total: ~{steps_per_epoch * NUM_EPOCHS}\n")

    t0 = time.time()
    trainer.train()
    print(f"\nTraining wall time: {(time.time() - t0) / 60:.1f} min")

    adapter_path = os.path.join(OUTPUT_DIR, "final-adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved: {adapter_path}")
    return trainer, adapter_path


# ─────────────────────────────────────────────
# Post-training eval: HumanEval pass@1
# ─────────────────────────────────────────────

def _ensure_human_eval():
    """Install human-eval if not already available."""
    try:
        from human_eval.data import read_problems       # noqa: F401
        from human_eval.execution import check_correctness  # noqa: F401
        return True
    except ImportError:
        print("human-eval not installed. Attempting: pip install human-eval")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "human-eval"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Install failed:\n{result.stderr}")
            return False
        return True


def _generate_completion(model, tokenizer, prompt: str) -> str:
    """
    Generate a single code completion from the fine-tuned model.
    Uses greedy decoding for deterministic pass@1.
    Strips everything after the first function definition ends.
    """
    messages = [
        {"role": "system", "content": CODING_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs    = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,            # greedy for pass@1 reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


def evaluate_humaneval(model, tokenizer, adapter_path: str, tier: dict,
                       n_problems: int = HUMANEVAL_N_PROBLEMS) -> dict:
    if not _ensure_human_eval():
        print("Skipping HumanEval: human-eval package unavailable.")
        return {}

    from human_eval.data import read_problems
    from human_eval.execution import check_correctness
    from tqdm import tqdm

    problems = read_problems()
    task_ids = list(problems.keys())[:n_problems]
    print(f"\nRunning HumanEval pass@1 on {n_problems} problems...")

    model.eval()
    passed  = 0
    records = []

    for task_id in tqdm(task_ids, desc="HumanEval"):
        problem    = problems[task_id]
        completion = _generate_completion(model, tokenizer, problem["prompt"])
        result     = check_correctness(problem, completion, timeout=3.0)
        ok         = result["passed"]
        if ok:
            passed += 1
        records.append({"task_id": task_id, "passed": ok,
                        "completion": completion[:200]})

    pass_at_1 = passed / n_problems

    results = {
        "phase":        "phase5_coding",
        "gpu":          tier["gpu_name"],
        "vram_gb":      round(tier["vram_gb"], 1),
        "tier":         tier["label"],
        "model":        BASE_MODEL,
        "adapter":      adapter_path,
        "dataset":      DATASET_NAME,
        "metric":       "pass@1",
        "n_problems":   n_problems,
        "n_passed":     passed,
        "pass_at_1":    round(pass_at_1, 4),
    }

    print(f"\n{'='*50}")
    print(f"HUMANEVAL pass@1")
    print(f"Problems: {n_problems} | Passed: {passed}")
    print(f"pass@1 = {pass_at_1:.1%}")
    print(f"Baseline (Mistral base, no fine-tuning): ~35-40%")
    print(f"Target: >=40%")
    print(f"{'='*50}")

    out_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")
    return results


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eval", action="store_true",
                        help="Train only, skip HumanEval evaluation")
    parser.add_argument("--n-problems", type=int, default=HUMANEVAL_N_PROBLEMS,
                        help=f"Number of HumanEval problems (default: {HUMANEVAL_N_PROBLEMS}, max: 164)")
    parser.add_argument("--push-hub", default=None,
                        help="HF repo to push adapter (e.g. anksriv/mistral-7b-coding-qlora)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}\n")
    print("Phase 5: Coding Fine-Tuning")
    print(f"Dataset: {DATASET_NAME}  (full 20K dataset)\n")

    tier = detect_gpu_tier()
    model, tokenizer        = load_model_and_tokenizer(tier)
    model                   = inject_lora(model)
    train_dataset, test_raw = load_data(tokenizer)
    trainer, adapter_path   = train(model, tokenizer, train_dataset, tier)

    if not args.skip_eval:
        evaluate_humaneval(trainer.model, tokenizer, adapter_path, tier,
                           n_problems=args.n_problems)

    if args.push_hub:
        print(f"\nPushing adapter to {args.push_hub}...")
        trainer.model.push_to_hub(args.push_hub)
        tokenizer.push_to_hub(args.push_hub)
        print(f"Pushed: https://huggingface.co/{args.push_hub}")


if __name__ == "__main__":
    main()
