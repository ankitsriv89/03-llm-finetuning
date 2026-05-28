---
title: LLM Fine-Tuning (QLoRA)
emoji: 🧠
colorFrom: purple
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Multi-domain LLM fine-tuning with QLoRA — Mistral-7B across 5 domains
---

# 🧠 LLM Fine-Tuning with QLoRA

Fine-tune **Mistral-7B-Instruct-v0.2** on domain-specific datasets using **QLoRA** (4-bit quantized LoRA). Covers 5 domains progressively — general instruction following, medical, legal, finance, and coding — with a multi-adapter Gradio demo.

> **Part 3 of 8** in a production-grade AI/ML portfolio series targeting Senior Data Scientist roles.

## What This Project Demonstrates

- **QLoRA pipeline** end-to-end: quantized model loading → LoRA adapter injection → supervised fine-tuning → HF Hub deployment
- **PEFT/TRL/bitsandbytes** ecosystem fluency
- **Domain adaptation**: same base model, 5 specialized adapters
- **Multi-adapter inference**: one loaded base model, hot-swap adapters at runtime
- **Config-driven training**: all hyperparameters in YAML, one script for all phases
- **Evaluation**: MCQ accuracy (medical), pass@1 (coding), LLM-as-judge (legal/finance)

---

## Architecture

```
Dataset (HuggingFace Hub)
    → QLoRA Training (Kaggle/Colab T4 GPU)
        → LoRA Adapter (~100MB) → HuggingFace Hub
            → Multi-Domain Gradio Demo
                → Deployed on HuggingFace Spaces
```

**Base model:** `mistralai/Mistral-7B-Instruct-v0.2` (Apache 2.0)  
**Quantization:** 4-bit NF4 via bitsandbytes (~3.5GB GPU RAM)  
**Adapter size:** ~50–200MB per domain (vs 14GB for full model in fp16)

---

## Domains and Datasets

| Phase | Domain | Dataset | Samples | Max Seq Len |
|-------|--------|---------|---------|-------------|
| 1 | General | `databricks/databricks-dolly-15k` | 5K | 512 |
| 2 | Medical | `medalpaca/medical_meadow_medqa` | 10K | 1024 |
| 3 | Legal | `nguyen-brat/legal_contracts` | 5K | 2048 |
| 4 | Finance | `gbharti/finance-alpaca` (subset) | 10K | 512 |
| 5 | Coding | `HuggingFaceH4/CodeAlpaca_20K` | 20K | 1024 |

---

## LoRA Configuration

```yaml
r: 16               # rank — adapter expressiveness
lora_alpha: 32      # scaling factor (convention: 2×r)
lora_dropout: 0.05  # regularization
target_modules:     # all attention + FFN matrices
  - q_proj, k_proj, v_proj, o_proj
  - gate_proj, up_proj, down_proj
```

Trainable parameters: ~20M out of 3.75B total (~0.53%)

---

## Project Structure

```
03-llm-finetuning/
├── configs/
│   ├── phase1_mistral_dolly.yaml      ← general (foundation)
│   ├── phase2_medical_medqa.yaml      ← medical QA
│   ├── phase3_legal.yaml              ← legal contracts
│   ├── phase4_finance.yaml            ← finance alpaca
│   └── phase5_coding.yaml             ← code generation
├── scripts/
│   ├── train.py                       ← QLoRA training (config-driven)
│   ├── inference.py                   ← load adapter + generate
│   └── push_to_hub.py                 ← upload adapter to HF Hub
├── notebooks/
│   ├── 01_phase1_mistral_dolly.ipynb  ← run on Kaggle/Colab
│   ├── 02_medical.ipynb
│   ├── 03_legal.ipynb
│   ├── 04_finance.ipynb
│   └── 05_coding.ipynb
├── app.py                             ← multi-domain Gradio demo
├── docs/
│   ├── PLAN.md                        ← architecture decisions + phases
│   ├── TUTORIAL.md                    ← concepts from scratch (QLoRA, LoRA, etc.)
│   └── DOMAIN_NOTES.md                ← dataset quirks + evaluation per domain
└── requirements.txt
```

---

## Quick Start

### Prerequisites

GPU with ≥16GB VRAM required for training. Options:
- **Kaggle Notebooks** — T4 GPU, free, 30hr/week (recommended)
- **Google Colab** — T4 free / A100 Colab Pro
- **AWS g4dn.xlarge** — T4 16GB, ~$0.53/hr (spot: ~$0.16/hr)

### 1. Clone and install

```bash
git clone https://github.com/ankitsriv89/03-llm-finetuning
cd 03-llm-finetuning
pip install -r requirements.txt
```

### 2. Train (Phase 1 — General)

```bash
python scripts/train.py --config configs/phase1_mistral_dolly.yaml
```

Or open `notebooks/01_phase1_mistral_dolly.ipynb` on Kaggle/Colab and run all cells.

### 3. Push adapter to HuggingFace Hub

```bash
export HF_TOKEN=hf_your_token_here
python scripts/push_to_hub.py \
    --adapter_path outputs/phase1-mistral-dolly/final-adapter \
    --repo_id your-username/mistral-dolly-qlora
```

### 4. Run inference

```bash
python scripts/inference.py \
    --base_model mistralai/Mistral-7B-Instruct-v0.2 \
    --adapter outputs/phase1-mistral-dolly/final-adapter
```

---

## Key Concepts

| Concept | What it means |
|---------|--------------|
| **QLoRA** | Fine-tune with 4-bit quantized base model + LoRA adapters |
| **NF4** | NormalFloat4 — optimal 4-bit format for normally-distributed weights |
| **LoRA rank (r)** | Bottleneck dimension of adapter matrices. Higher = more capacity |
| **Gradient checkpointing** | Recompute activations during backprop — trades compute for memory |
| **Paged AdamW** | Offload optimizer states to CPU when GPU is tight |
| **Response masking** | Only compute training loss on assistant response tokens, not instructions |
| **Chat template** | Model-specific prompt format — must match what the model was trained on |

Full explanations with code in [docs/TUTORIAL.md](docs/TUTORIAL.md).

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Model loading + tokenizer | `transformers` | ≥4.40 |
| LoRA adapters | `peft` | ≥0.10 |
| Supervised fine-tuning | `trl` (SFTTrainer) | ≥0.8.6 |
| 4-bit quantization | `bitsandbytes` | ≥0.43 |
| Device placement | `accelerate` | ≥0.29 |
| Datasets | `datasets` | ≥2.18 |
| Hub push/pull | `huggingface_hub` | ≥0.22 |
| Demo UI | `gradio` | ≥4.26 |
| Deep learning backend | `torch` | ≥2.2 |

---

## Docs

- [PLAN.md](docs/PLAN.md) — architecture decisions, dataset strategy, compute budget, all 6 phases
- [TUTORIAL.md](docs/TUTORIAL.md) — QLoRA from scratch: transformer weights, LoRA math, quantization, training loop, generation
- [DOMAIN_NOTES.md](docs/DOMAIN_NOTES.md) — per-domain dataset structure, formatting quirks, evaluation metrics
