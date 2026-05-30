---
base_model: mistralai/Mistral-7B-Instruct-v0.2
library_name: peft
tags:
  - qlora
  - lora
  - medical
  - medqa
  - mistral
datasets:
  - medalpaca/medical_meadow_medqa
language:
  - en
license: apache-2.0
---

# Mistral-7B-Instruct-v0.2 — Medical MedQA QLoRA Adapter (Partial-Epoch Checkpoint)

LoRA adapter fine-tuned on `medalpaca/medical_meadow_medqa` (USMLE-style 4-option clinical MCQs).

> **Honest status: this is a partial-training checkpoint, not the planned full run.**
> Training was interrupted at ~step 171 of 1144 (~0.3 of 2 planned epochs) because the Kaggle 2× T4 model-parallel setup ran at ~2.5 min/step instead of the expected ~5 sec/step.
> A full-epoch re-run is planned on a single-GPU RunPod instance using
> [`train_runpod_medical.py`](https://github.com/ankitsriv89/03-llm-finetuning/blob/main/scripts/train_runpod_medical.py).

## Results (held-out test split)

| Metric | Value |
|---|---|
| Eval samples | 100 |
| Accuracy | **39%** |
| No-answer rate | 0% |
| Random chance (4-option) | 25% |
| Random chance (mixed 4/5-option) | ~22% |
| Target (planned full run) | ≥50% |

Confidence interval at n=100: ±5pp (95% CI). True accuracy is plausibly in **[34%, 44%]**.

**What worked:**
- Format adherence is solid — 0% no-answer rate. The model reliably starts responses with "The correct answer is X)".
- Clear lift above chance even with ~15% of planned training compute.

**What didn't:**
- Accuracy below the +5pp-vs-base target. Primary cause: undertraining.
- Test set includes some 5-option (A–E) samples the training filter missed (regex caught `E)` but not `E:`); the model only saw 4-option questions in training, so 5-option samples are harder.

## Training details

| Item | Value |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Dataset | `medalpaca/medical_meadow_medqa` (filtered) |
| Train samples | 9,158 (after 5-option filter + 90/10 split) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Optimizer | paged_adamw_32bit |
| Learning rate | 2e-4 (cosine, 3% warmup) |
| Max seq length | 1024 |
| Effective batch size | 16 (4 × 4 grad accum) |
| Planned steps | 1144 (2 epochs) |
| **Completed steps** | **~171 (~0.3 epochs)** |
| Hardware | Kaggle 2× T4 (model-parallel) |
| Final training loss | ~0.78 (from initial 1.67) |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER = "anksriv/mistral-7b-medical-medqa-qlora"  # this repo

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)
tokenizer = AutoTokenizer.from_pretrained(BASE)

system = ("You are a knowledgeable medical AI assistant. "
          "When given a clinical multiple-choice question, analyze the case carefully, "
          "identify the correct answer (A, B, C, or D), and provide a clear explanation. "
          "Always begin your response with 'The correct answer is X)' where X is the letter.")

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": "A 32-year-old woman presents with ... A) ... B) ... C) ... D) ..."},
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
out = model.generate(inputs, max_new_tokens=200, temperature=0.1, do_sample=True)
print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Intended use & limitations

This is a **research/educational artifact**, not a clinical tool. Do not use it for medical decision-making. The 39% accuracy on USMLE-style questions, combined with partial training, makes this strictly inappropriate for any patient-facing or diagnostic application.

The adapter inherits all biases of:
- The base Mistral-7B-Instruct-v0.2 model
- The MedQA dataset (USMLE-skewed, US-centric, English-only)

## Reproducibility

- Code: https://github.com/ankitsriv89/03-llm-finetuning
- Training notebook: `notebooks/02_medical.ipynb`
- Single-GPU RunPod script: `scripts/train_runpod_medical.py`
- Eval script: `scripts/evaluate.py --mode mcq`
- Seed: 42 (train/test split)

## Planned full-run improvements

1. Run the full 2 epochs (~1144 steps) on a single A40/L40S/4090 (~30-45 min target).
2. Tighten the 5-option filter (catch both `E)` and `E:`).
3. Evaluate on the full 1K held-out set instead of 100.
4. Compare against base Mistral-7B-Instruct-v0.2 (no adapter) on the same eval set to compute the actual delta.
