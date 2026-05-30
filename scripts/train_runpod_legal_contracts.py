"""
train_runpod_legal_contracts.py — Phase 3a Legal Contracts, GPU-agnostic for RunPod
=====================================================================================
Fine-tunes Mistral-7B-Instruct-v0.2 on:
  Primary:   nguyen-brat/legal_contracts  (~10K contract clause Q&A from CUAD)
  Auxiliary: pile-of-law/pile-of-law subset="freelaw" (5K court opinions, 20% batch mix)

Auto-detects GPU and picks the optimal config:

    VRAM >= 70GB  (A100 80GB, H100 80GB):
        bf16 LoRA, batch=8,  no grad ckpt, flash-attn2  → ~20 min

    VRAM >= 40GB  (A40 48GB, L40S 48GB, A6000 48GB):
        bf16 LoRA, batch=4,  no grad ckpt, flash-attn2  → ~35 min

    VRAM >= 22GB  (RTX 4090 24GB, A5000 24GB):
        4-bit QLoRA, batch=4, grad ckpt on, flash-attn2 → ~60 min

    VRAM < 22GB   (T4 16GB, etc.):
        4-bit QLoRA, batch=2, grad ckpt on, no flash-attn → ~120 min

Note: seq_len=2048 (contract clauses are long) — VRAM budget is tighter than Phase 2.
Override with env vars: BATCH_SIZE, USE_4BIT=1, NO_FLASH=1

Setup on RunPod:
    cd /workspace && git clone <repo> jobs-prjcts && cd jobs-prjcts/03-llm-finetuning
    pip install -r requirements_runpod.txt
    huggingface-cli login
    python scripts/train_runpod_legal_contracts.py
"""

import json
import os
import re
import time

import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


# ─────────────────────────────────────────────
# Static config
# ─────────────────────────────────────────────

BASE_MODEL        = "mistralai/Mistral-7B-Instruct-v0.2"
PRIMARY_DATASET   = "nguyen-brat/legal_contracts"
AUX_DATASET       = "pile-of-law/pile-of-law"
AUX_SUBSET        = "freelaw"
OUTPUT_DIR        = "/workspace/outputs/phase3a-legal-contracts"

NUM_EPOCHS        = 3
LEARNING_RATE     = 1.5e-4
MAX_SEQ_LENGTH    = 2048
LOGGING_STEPS     = 10
SAVE_STEPS        = 150

LORA_R            = 16
LORA_ALPHA        = 32
LORA_DROPOUT      = 0.05
TARGET_MODULES    = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]

TEST_SIZE         = 0.10
PRIMARY_MAX_TRAIN = None       # None = use all ~9K training samples
AUX_SAMPLES       = 5000       # pile-of-law samples to mix in
AUX_WEIGHT        = 0.20       # 20% of mixed dataset from auxiliary

LEGAL_SYSTEM = (
    "You are an expert legal assistant specializing in contract law. "
    "When given a clause from a legal contract, analyze it carefully and provide "
    "a clear, accurate explanation of its legal implications, risks, and key terms. "
    "Use precise legal language while remaining accessible to non-lawyers. "
    "Cite relevant legal concepts and flag any unusual or one-sided provisions."
)


# ─────────────────────────────────────────────
# GPU auto-detect → config tier
# ─────────────────────────────────────────────

def detect_gpu_tier() -> dict:
    """
    seq_len=2048 roughly doubles VRAM vs Phase 2 (seq_len=1024).
    Batch sizes are halved accordingly to keep the same memory footprint.
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

def format_legal_contracts_sample(sample: dict, tokenizer) -> str:
    """
    nguyen-brat/legal_contracts fields:
        instruction : question about the clause (e.g. "What is the termination right here?")
        context     : raw clause text from the contract
        output      : answer explaining the clause

    We include the clause text in the user turn so the model learns to
    ground its answer in the actual contract language.
    """
    instruction = sample.get("instruction", "").strip()
    context     = sample.get("context", "").strip()
    output      = sample.get("output", "").strip()

    user_content = instruction
    if context:
        user_content = f"{instruction}\n\nContract Clause:\n{context}"

    messages = [
        {"role": "system",    "content": LEGAL_SYSTEM},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": output},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def format_pile_of_law_sample(sample: dict, tokenizer) -> str:
    """
    pile-of-law freelaw subset fields:
        text : raw court opinion text

    Reformatted as a summarization / explanation task to keep instruction-following
    format consistent with the primary dataset. Truncate to ~1500 chars to stay
    within seq_len budget when combined with the system prompt.
    """
    text = sample.get("text", "").strip()
    if len(text) > 1500:
        text = text[:1500] + "..."

    messages = [
        {"role": "system",    "content": LEGAL_SYSTEM},
        {"role": "user",      "content": f"Summarize the key legal points in this court opinion:\n\n{text}"},
        {"role": "assistant", "content": "This court opinion addresses the following key legal points:"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_data(tokenizer):
    print(f"Loading primary dataset: {PRIMARY_DATASET}...")
    raw = load_dataset(PRIMARY_DATASET, split="train")
    if PRIMARY_MAX_TRAIN:
        raw = raw.select(range(min(PRIMARY_MAX_TRAIN, len(raw))))

    split       = raw.train_test_split(test_size=TEST_SIZE, seed=42)
    train_raw   = split["train"]
    test_raw    = split["test"]
    print(f"Primary — Train: {len(train_raw)} | Test: {len(test_raw)}")

    train_primary = train_raw.map(
        lambda s: {"text": format_legal_contracts_sample(s, tokenizer)},
        remove_columns=train_raw.column_names,
    )

    print(f"Loading auxiliary dataset: {AUX_DATASET} (subset={AUX_SUBSET}, {AUX_SAMPLES} samples)...")
    try:
        aux_raw = load_dataset(AUX_DATASET, AUX_SUBSET, split="train", streaming=False)
        aux_raw = aux_raw.select(range(min(AUX_SAMPLES, len(aux_raw))))
        train_aux = aux_raw.map(
            lambda s: {"text": format_pile_of_law_sample(s, tokenizer)},
            remove_columns=aux_raw.column_names,
        )
        print(f"Auxiliary — {len(train_aux)} samples loaded")

        # Mix: keep (1-AUX_WEIGHT) primary + AUX_WEIGHT auxiliary
        n_primary = int(len(train_primary) * (1 - AUX_WEIGHT) / AUX_WEIGHT)
        n_aux     = min(AUX_SAMPLES, len(train_aux))
        n_primary = min(n_primary, len(train_primary))

        mixed = concatenate_datasets([
            train_primary.select(range(n_primary)),
            train_aux.select(range(n_aux)),
        ]).shuffle(seed=42)
        print(f"Mixed dataset: {len(mixed)} samples "
              f"({n_primary} primary + {n_aux} auxiliary)")

    except Exception as e:
        print(f"WARNING: Could not load auxiliary dataset ({e}). Training on primary only.")
        mixed = train_primary

    return mixed, test_raw


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

def evaluate_legal(model, tokenizer, test_raw, adapter_path: str, tier: dict,
                   groq_api_key: str | None = None):
    """
    LLM-as-judge evaluation on the held-out test split.
    Requires GROQ_API_KEY env var or --groq-key argument.
    Falls back gracefully if Groq is unavailable.
    """
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

    groq_client    = Groq(api_key=api_key)
    judge_model    = "llama-3.3-70b-versatile"
    max_eval       = min(100, len(test_raw))
    samples        = test_raw.select(range(max_eval))

    model.eval()
    scores  = []
    records = []

    for sample in tqdm(samples, desc="Legal LLM-judge eval"):
        instruction = sample.get("instruction", "").strip()
        context     = sample.get("context", "").strip()
        ground_truth = sample.get("output", "").strip()

        user_prompt = instruction
        if context:
            user_prompt = f"{instruction}\n\nContract Clause:\n{context}"

        # Generate response
        messages  = [{"role": "system", "content": LEGAL_SYSTEM},
                     {"role": "user",   "content": user_prompt}]
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

        # Judge
        judge_prompt = f"""You are an expert legal evaluator. Rate this AI response on a 1-5 scale.

Question: {instruction[:300]}

Contract Clause: {context[:300]}

Reference answer: {ground_truth[:400]}

AI response: {response[:400]}

Rating criteria:
1 — Wrong, irrelevant, or legally incorrect
2 — Partially correct with significant legal errors or omissions
3 — Correct interpretation but missing key legal implications
4 — Correct, complete, minor omissions; good legal precision
5 — Accurate, complete, legally precise; correctly identifies risks and obligations

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
        "phase":      "phase3a_legal_contracts",
        "gpu":        tier["gpu_name"],
        "vram_gb":    round(tier["vram_gb"], 1),
        "tier":       tier["label"],
        "model":      BASE_MODEL,
        "adapter":    adapter_path,
        "dataset":    PRIMARY_DATASET,
        "judge":      judge_model,
        "num_eval":   len(scores),
        "avg_score":  round(avg_score, 3),
        "score_dist": score_dist,
    }

    print(f"\n{'='*50}")
    print(f"LEGAL LLM-JUDGE ({judge_model})")
    print(f"Average score: {avg_score:.2f}/5.0")
    print(f"Distribution:  {score_dist}")
    print(f"Baseline (Mistral base): ~3.0/5.0")
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
                        help="HF repo to push adapter (e.g. anksriv/mistral-7b-legal-contracts-qlora)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"PyTorch: {torch.__version__}  |  CUDA: {torch.version.cuda}\n")

    tier = detect_gpu_tier()
    model, tokenizer      = load_model_and_tokenizer(tier)
    model                 = inject_lora(model)
    train_dataset, test_raw = load_data(tokenizer)
    trainer, adapter_path = train(model, tokenizer, train_dataset, tier)

    if not args.skip_eval:
        evaluate_legal(trainer.model, tokenizer, test_raw, adapter_path, tier,
                       groq_api_key=args.groq_key)

    if args.push_hub:
        print(f"\nPushing adapter to {args.push_hub}...")
        trainer.model.push_to_hub(args.push_hub)
        tokenizer.push_to_hub(args.push_hub)
        print(f"Pushed: https://huggingface.co/{args.push_hub}")


if __name__ == "__main__":
    main()
