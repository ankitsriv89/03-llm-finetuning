"""
train_runpod_finance.py — Phase 4 Finance, GPU-agnostic for RunPod
===================================================================
Fine-tunes Mistral-7B-Instruct-v0.2 on general finance Q&A.

  Dataset: gbharti/finance-alpaca
           68K samples — shuffled 10K subset for breadth across:
           P/E ratios, DCF, EBITDA, portfolio theory, derivatives,
           macro indicators, risk metrics, fixed income, etc.
  Format:  {instruction, input, output} — same structure as Dolly.
           'input' is often empty; treated as optional context.

GPU tiers (seq_len=512 — much smaller than legal phases):

    VRAM >= 70GB  (A100 80GB, H100 80GB):
        bf16 LoRA, batch=16, no grad ckpt, flash-attn2  → ~15 min

    VRAM >= 40GB  (A40/L40S/A6000 48GB):
        bf16 LoRA, batch=8,  no grad ckpt, flash-attn2  → ~25 min

    VRAM >= 22GB  (RTX 4090 / A5000 24GB):
        4-bit QLoRA, batch=8, grad ckpt on, flash-attn2 → ~30 min

    VRAM < 22GB   (T4 16GB, etc.):
        4-bit QLoRA, batch=4, grad ckpt on, no flash-attn → ~50 min

Setup on RunPod:
    cd /workspace && git clone <repo> jobs-prjcts && cd jobs-prjcts/03-llm-finetuning
    pip install -r requirements_runpod.txt
    huggingface-cli login
    python scripts/train_runpod_finance.py
"""

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

BASE_MODEL      = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_NAME    = "gbharti/finance-alpaca"
OUTPUT_DIR      = "/workspace/outputs/phase4-finance"

MAX_SAMPLES     = 10000        # shuffled subset of 68K
SHUFFLE_SEED    = 42
NUM_EPOCHS      = 3
LEARNING_RATE   = 2.0e-4
MAX_SEQ_LENGTH  = 512
LOGGING_STEPS   = 10
SAVE_STEPS      = 200
TEST_SIZE       = 0.10

LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

FINANCE_SYSTEM = (
    "You are an expert financial analyst and economist with deep knowledge of "
    "corporate finance, investment analysis, financial markets, macroeconomics, "
    "and quantitative methods. Provide accurate, well-structured explanations of "
    "financial concepts, valuation techniques, and market dynamics. When performing "
    "calculations, show your reasoning step by step. "
    "This response is for educational purposes only and does not constitute "
    "financial, investment, or tax advice."
)

FINANCE_JUDGE_RUBRIC = """Rating criteria for finance responses:
1 — Wrong financial concept, incorrect direction of effect, or hallucinated metric/formula
2 — Correct general area but wrong formula, wrong units, or significant calculation error
3 — Correct concept but missing a key caveat, risk flag, or important nuance
4 — Correct and well-explained; minor omission in terminology or secondary detail
5 — Accurate, correct formula/framework, appropriate risk caveats; suitable depth for the question"""


# ─────────────────────────────────────────────
# GPU auto-detect
# ─────────────────────────────────────────────

def detect_gpu_tier() -> dict:
    """
    seq_len=512 is much smaller than legal phases (2048), so batch sizes
    can be 4x larger for the same VRAM footprint.
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
                    batch_size=16, grad_accum=1, grad_ckpt=False)
    elif vram_gb >= 40 and not force_4bit:
        tier = dict(label="A40/L40S/A6000 48GB tier", use_4bit=False,
                    batch_size=8, grad_accum=2, grad_ckpt=False)
    elif vram_gb >= 22:
        tier = dict(label="RTX 4090 / A5000 24GB tier", use_4bit=True,
                    batch_size=8, grad_accum=2, grad_ckpt=True)
    else:
        tier = dict(label="T4 / small-VRAM tier", use_4bit=True,
                    batch_size=4, grad_accum=4, grad_ckpt=True)

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

def format_finance_sample(sample: dict, tokenizer) -> str:
    """
    gbharti/finance-alpaca fields: {instruction, input, output}

    'input' is often empty — same handling as Dolly's context field.
    When non-empty it provides additional context (e.g. a financial statement
    excerpt or a specific company scenario) for the instruction.
    """
    instruction = sample.get("instruction", "").strip()
    context     = sample.get("input", "").strip()
    output      = sample.get("output", "").strip()

    user_content = instruction
    if context:
        user_content = f"{instruction}\n\nContext: {context}"

    messages = [
        {"role": "system",    "content": FINANCE_SYSTEM},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": output},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_data(tokenizer):
    """
    Load gbharti/finance-alpaca, shuffle, take 10K subset, split train/test.

    Shuffling before subsetting is important: the raw dataset has topic clusters
    (all P/E questions together, all bond questions together, etc.). Without
    shuffling the 10K subset would be biased toward early topics.
    """
    print(f"Loading dataset: {DATASET_NAME}...")
    raw = load_dataset(DATASET_NAME, split="train")
    print(f"Full dataset size: {len(raw)} samples")

    # Shuffle first to break topic clusters, then subset
    raw = raw.shuffle(seed=SHUFFLE_SEED).select(range(min(MAX_SAMPLES, len(raw))))
    print(f"After shuffle + subset: {len(raw)} samples")

    split    = raw.train_test_split(test_size=TEST_SIZE, seed=42)
    train_raw = split["train"]
    test_raw  = split["test"]
    print(f"Train: {len(train_raw)} | Test: {len(test_raw)}")

    train_ds = train_raw.map(
        lambda s: {"text": format_finance_sample(s, tokenizer)},
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
# Post-training eval (LLM-as-judge via Groq)
# ─────────────────────────────────────────────

def evaluate_finance(model, tokenizer, test_raw, adapter_path: str, tier: dict,
                     groq_api_key: str | None = None):
    try:
        from groq import Groq
    except ImportError:
        print("Skipping eval: 'groq' not installed. pip install groq")
        return {}

    api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Skipping eval: GROQ_API_KEY not set.")
        return {}

    from tqdm import tqdm

    groq_client = Groq(api_key=api_key)
    judge_model = "llama-3.3-70b-versatile"
    max_eval    = min(100, len(test_raw))
    samples     = test_raw.select(range(max_eval))

    model.eval()
    scores  = []
    records = []

    for sample in tqdm(samples, desc="Finance LLM-judge eval"):
        instruction  = sample.get("instruction", "").strip()
        context      = sample.get("input", "").strip()
        ground_truth = sample.get("output", "").strip()

        user_content = instruction
        if context:
            user_content = f"{instruction}\n\nContext: {context}"

        messages  = [{"role": "system", "content": FINANCE_SYSTEM},
                     {"role": "user",   "content": user_content}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs    = tokenizer(formatted, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=300, temperature=0.1,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

        judge_prompt = f"""You are an expert evaluator of financial knowledge responses. Rate this AI response on a 1-5 scale.

Question: {instruction[:400]}

Reference answer: {ground_truth[:500]}

AI response: {response[:500]}

{FINANCE_JUDGE_RUBRIC}

Respond with ONLY a single digit (1-5). No explanation."""

        try:
            completion = groq_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=5, temperature=0,
            )
            score_text = completion.choices[0].message.content or ""
            m = re.search(r'[1-5]', score_text.strip())
            score = int(m.group()) if m else 3
        except Exception as e:
            print(f"Groq error: {e}")
            score = 3

        scores.append(score)
        records.append({"instruction": instruction[:80], "score": score,
                        "response": response[:150]})

    avg_score  = sum(scores) / len(scores) if scores else 0.0
    score_dist = {str(i): scores.count(i) for i in range(1, 6)}

    results = {
        "phase":      "phase4_finance",
        "gpu":        tier["gpu_name"],
        "vram_gb":    round(tier["vram_gb"], 1),
        "tier":       tier["label"],
        "model":      BASE_MODEL,
        "adapter":    adapter_path,
        "dataset":    DATASET_NAME,
        "judge":      judge_model,
        "lora_r":     LORA_R,
        "num_eval":   len(scores),
        "avg_score":  round(avg_score, 3),
        "score_dist": score_dist,
    }

    print(f"\n{'='*50}")
    print(f"FINANCE LLM-JUDGE ({judge_model})")
    print(f"Average score: {avg_score:.2f}/5.0")
    print(f"Distribution:  {score_dist}")
    print(f"Baseline (Mistral base on finance): ~3.0/5.0")
    print(f"Target: >=3.5/5.0 (+0.5 pts improvement)")
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
                        help="Train only, skip LLM-judge evaluation")
    parser.add_argument("--groq-key", default=None,
                        help="Groq API key (fallback: GROQ_API_KEY env var)")
    parser.add_argument("--push-hub", default=None,
                        help="HF repo to push adapter (e.g. anksriv/mistral-7b-finance-qlora)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}\n")
    print("Phase 4: Finance Fine-Tuning")
    print(f"Dataset: {DATASET_NAME}  (shuffled {MAX_SAMPLES}-sample subset)\n")

    tier = detect_gpu_tier()
    model, tokenizer        = load_model_and_tokenizer(tier)
    model                   = inject_lora(model)
    train_dataset, test_raw = load_data(tokenizer)
    trainer, adapter_path   = train(model, tokenizer, train_dataset, tier)

    if not args.skip_eval:
        evaluate_finance(trainer.model, tokenizer, test_raw, adapter_path, tier,
                         groq_api_key=args.groq_key)

    if args.push_hub:
        print(f"\nPushing adapter to {args.push_hub}...")
        trainer.model.push_to_hub(args.push_hub)
        tokenizer.push_to_hub(args.push_hub)
        print(f"Pushed: https://huggingface.co/{args.push_hub}")


if __name__ == "__main__":
    main()
