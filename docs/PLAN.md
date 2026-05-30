# Project 03 — LLM Fine-Tuning (QLoRA): Plan

## Goal

Fine-tune large language models on domain-specific datasets using QLoRA (parameter-efficient, 4-bit quantized fine-tuning). Build across multiple domains — general instruction following, medical, legal, finance, and coding — then deploy all adapters behind a single multi-domain Gradio demo on HuggingFace Spaces.

---

## Problem Statement

Pretrained instruction-following models (Mistral-7B-Instruct, LLaMA-3) are generalists. For specialized domains:
- Medical: needs precise clinical reasoning, medical terminology, patient safety awareness
- Legal: needs contract clause comprehension, jurisdiction awareness, structured legal analysis
- Finance: needs earnings interpretation, ratio calculation, forward-looking statement caution
- Coding: needs syntactically correct code, language-specific idioms, debugging reasoning

Fine-tuning adapts the model's style, domain vocabulary, and reasoning patterns — without retraining the full 7B parameters.

---

## Why QLoRA Specifically?

| Approach | GPU Requirement | Cost | Quality |
|----------|----------------|------|---------|
| Full fine-tune | 8× A100 (640GB) | $$$$ | Best |
| LoRA (fp16) | 2× A100 (80GB) | $$ | Very good |
| QLoRA (4-bit + LoRA) | 1× T4 (16GB) | Free (Colab/Kaggle) | Good |

QLoRA (Dettmers et al., 2023) makes fine-tuning accessible on consumer hardware without meaningful quality loss. It is the standard approach for individual researchers and small teams.

---

## Architecture Decision Record

### Base Model: Mistral-7B-Instruct-v0.2

| Option | Size | License | Tool calls | Notes |
|--------|------|---------|-----------|-------|
| Mistral-7B-Instruct-v0.2 | 7B | Apache 2.0 | Yes | **Selected** — best balance |
| LLaMA-3-8B-Instruct | 8B | Meta license | Yes | Requires Meta agreement |
| Phi-3-mini | 3.8B | MIT | Limited | Too small for complex domains |
| Mistral-7B-v0.1 | 7B | Apache 2.0 | No | Base model, not instruct |

Mistral-7B-Instruct-v0.2 wins because: Apache 2.0 (no account needed), strong instruction following out of the box, good benchmark scores, widely supported by PEFT/TRL.

### LoRA Configuration Rationale

```
r=16, alpha=32, dropout=0.05
Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

- `r=16`: Standard starting point. Enough expressiveness for instruction tuning. Increase to 32-64 only if underfitting.
- `alpha=32`: Convention is `alpha = 2*r`. Scales the LoRA output by `alpha/r = 2`.
- All 7 projection matrices targeted: thorough coverage of attention + FFN. More parameters = better adaptation.
- `dropout=0.05`: Mild regularization. Higher dropout (0.1) if dataset is small and overfitting is a concern.

### Dataset Strategy per Domain

| Domain | Dataset | Size | Max Seq Length | Notes |
|--------|---------|------|---------------|-------|
| General | `databricks/databricks-dolly-15k` | 15K | 512 | Foundation run |
| Medical | `medalpaca/medical_meadow_medqa` | 10K | 512 | MedQA 4-option MCQ |
| Medical | `pubmed_qa` (labeled split) | 1K | 1024 | Long-context PubMed abstracts |
| Legal | `nguyen-brat/legal_contracts` | 5K | 1024 | Contract clause analysis |
| Legal | `pile-of-law/pile-of-law` (subset) | 10K | 2048 | Diverse legal text |
| Finance | `fiqa` | 6K | 512 | Finance Q&A |
| Finance | `gbharti/finance-alpaca` | 68K | 512 | Large, use 10K subset |
| Coding | `HuggingFaceH4/CodeAlpaca_20K` | 20K | 1024 | Code generation |
| Coding | `sahil2801/CodeAlpaca-20k` | 20K | 1024 | Alternative code dataset |

### Adapter Strategy: One Adapter Per Domain

Each domain gets its own fine-tuning run and its own adapter. At inference time, load the base model once and swap adapters dynamically using PEFT's multi-adapter support.

```python
model.load_adapter("adapters/medical", adapter_name="medical")
model.load_adapter("adapters/legal", adapter_name="legal")
model.set_adapter("medical")   # switch at runtime
```

This is more efficient than loading 5 separate models (each 3.5GB in 4-bit).

---

## Implementation Phases

### Phase 1: Foundation (Mistral + Dolly) — General Instruction Following
- [x] Project structure
- [x] requirements.txt
- [x] YAML config system
- [x] train.py (QLoRA training script)
- [x] inference.py (load adapter + generate)
- [x] push_to_hub.py
- [x] notebooks/01_phase1_mistral_dolly.ipynb
- [ ] Run on Kaggle T4 GPU
- [ ] Push adapter to HF Hub
- [ ] Smoke test inference

### Phase 2: Medical Domain
- [x] configs/phase2_medical_medqa.yaml
- [x] notebooks/02_medical.ipynb
- [x] scripts/evaluate.py (MCQ + Groq LLM-judge modes)
- [x] scripts/train_runpod_medical.py (GPU-agnostic single-GPU script)
- [x] Dataset: medalpaca/medical_meadow_medqa
- [x] Run on Kaggle T4 GPU (**partial: ~171/1144 steps, ~0.3 epochs**)
- [x] Evaluation: MCQ accuracy 39% on 100 held-out samples (target was ≥50%)
- [x] Push medical adapter → https://huggingface.co/anksriv/mistral-7b-medical-medqa-qlora
- [ ] **Full re-run on RunPod (target: 2 epochs, ≥50% accuracy on 500 samples)**

### Phase 3: Legal Domain
- [ ] configs/phase3_legal.yaml
- [ ] notebooks/03_legal.ipynb
- [ ] Dataset: legal contracts + pile-of-law subset
- [ ] Increase max_seq_length to 1024-2048
- [ ] Evaluation: contract clause extraction quality
- [ ] Push legal adapter

### Phase 4: Finance Domain
- [ ] configs/phase4_finance.yaml
- [ ] notebooks/04_finance.ipynb
- [ ] Dataset: finance-alpaca subset
- [ ] Evaluation: FinQA accuracy, financial reasoning quality
- [ ] Push finance adapter

### Phase 5: Coding Domain
- [ ] configs/phase5_coding.yaml
- [ ] notebooks/05_coding.ipynb
- [ ] Dataset: CodeAlpaca-20K
- [ ] Evaluation: code execution correctness (pass@1 on HumanEval subset)
- [ ] Push coding adapter

### Phase 6: Multi-Domain Demo
- [ ] app.py — Gradio demo with domain selector
- [ ] Multi-adapter loading (one base model, swap adapters)
- [ ] README.md
- [ ] Deploy to HuggingFace Spaces
- [ ] Evaluation comparison table across all domains

---

## Evaluation Strategy

### General / Medical
- Held-out test split (10% of dataset)
- For MCQ datasets: exact match accuracy
- Target: beat base Mistral-7B-Instruct by ≥5 percentage points

### Legal / Finance
- Human evaluation: 10 sample responses rated for correctness, completeness, format
- LLM-as-judge: use GPT-4o to score responses on a 1-5 scale

### Coding
- `pass@1` metric: does the generated code run and pass unit tests?
- Use a subset of HumanEval (30 problems)

---

## Compute Budget

| Phase | Dataset Size | Epochs | GPU | Estimated Time | Cost |
|-------|-------------|--------|-----|---------------|------|
| Phase 1 | 5K samples | 1 | T4 (Kaggle free) | ~45 min | Free |
| Phase 2 | 10K samples | 2 | T4 (Kaggle free) | ~90 min | Free |
| Phase 3 | 5K samples | 2 | T4 (Kaggle free) | ~60 min | Free |
| Phase 4 | 10K samples | 1 | T4 (Kaggle free) | ~60 min | Free |
| Phase 5 | 20K samples | 1 | T4 (Kaggle free) | ~120 min | Free |

Total: ~6.5 hours of GPU time across 5 runs. All fits within Kaggle's 30hr/week free quota.

---

## Files and Their Roles

```
03-llm-finetuning/
├── configs/
│   ├── phase1_mistral_dolly.yaml      ← general instruction following
│   ├── phase2_medical_medqa.yaml      ← medical QA
│   ├── phase3_legal.yaml              ← legal contracts
│   ├── phase4_finance.yaml            ← finance alpaca
│   └── phase5_coding.yaml             ← code generation
├── scripts/
│   ├── train.py                       ← QLoRA training (config-driven)
│   ├── inference.py                   ← load adapter + generate
│   ├── push_to_hub.py                 ← upload adapter to HF Hub
│   └── evaluate.py                    ← accuracy / pass@1 evaluation
├── notebooks/
│   ├── 01_phase1_mistral_dolly.ipynb  ← step-by-step (run on Kaggle/Colab)
│   ├── 02_medical.ipynb
│   ├── 03_legal.ipynb
│   ├── 04_finance.ipynb
│   └── 05_coding.ipynb
├── app.py                             ← multi-domain Gradio demo
├── docs/
│   ├── PLAN.md                        ← this file
│   ├── TUTORIAL.md                    ← concept explanations
│   └── DOMAIN_NOTES.md                ← dataset quirks per domain
├── requirements.txt
└── outputs/                           ← local adapter checkpoints (gitignored)
```

---

## Key Metrics / Success Criteria

| Metric | Target |
|--------|--------|
| All 5 adapters on HF Hub | Yes |
| Multi-domain demo deployed | Yes |
| Medical MCQ accuracy vs base | +5pp improvement |
| Coding pass@1 on HumanEval subset | ≥ 30% |
| Training runs documented with loss curves | Yes |

---

## Status

**In progress.** Phase 1 scripts and notebooks complete. GPU runs pending.
