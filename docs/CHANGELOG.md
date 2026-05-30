# Changelog

All notable changes to the QLoRA fine-tuning pipeline.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dates are absolute (YYYY-MM-DD).

---

## [Phase 6 — Multi-Domain Demo] — 2026-05-31

### Added
- `app.py` — Gradio demo serving all 5 domain adapters from a single loaded base model.
  - Domain selector (General / Medical / Legal / Finance / Coding) with hot-swap via PEFT `set_adapter()`
  - Adapters loaded lazily on first use via `PeftModel.from_pretrained` + `model.load_adapter()`; base model loaded once in 4-bit NF4
  - Streaming responses via `TextIteratorStreamer` on a background thread — Gradio displays tokens as they arrive
  - `LOCAL_ADAPTERS` env var for testing with local adapter paths before HF Hub push (format: `"medical=./outputs/phase2/final-adapter"`)
  - `NO_GPU=1` env var for CPU testing without CUDA
  - Example prompts per domain (3 per domain), temperature + max-token sliders, clear button
  - Greedy decoding for coding domain (deterministic pass@1 reproducibility); sampling for all others
- `notebooks/03_legal.ipynb` — single notebook covering both Phase 3a (contracts) and 3b (Indian law). `PHASE = "3a"` / `"3b"` config toggle at top; all subsequent cells adapt automatically. Covers 3-way dataset mixing for 3b (primary + InLegalNLI + BNS mapping), LLM-judge eval with domain-specific rubric, HF Hub push.
- `notebooks/04_finance.ipynb` — Phase 4 notebook: dataset exploration, formatter with 10% disclaimer injection, training, Groq LLM-judge eval, HF Hub push.
- `notebooks/05_coding.ipynb` — Phase 5 notebook: length filter (drops >974-token samples to prevent mid-function truncation), HumanEval pass@1 eval (30 problems by default, configurable to 164), per-problem failure breakdown, HF Hub push.

### Design notes
- Multi-adapter hot-swap: one 4-bit base model (~3.5 GB) + 5 adapters (~50–200 MB each) vs 5 separate loaded models (~17.5 GB). Typical domain switch latency: <100 ms.
- `_load_lock` prevents race conditions if multiple Gradio requests arrive before the base model finishes loading.
- Adapter resolution: `LOCAL_ADAPTERS` env var checked first, falls back to HF Hub repo ID. Useful for running the demo before GPU runs complete.

### Known limitations
- Adapters for phases 3–5 not yet run on RunPod — demo will use base Mistral until GPU runs complete and adapters are pushed to HF Hub.
- HumanEval sandbox (`check_correctness`) executes arbitrary code; safe in Kaggle/RunPod sandboxes but do not run on a production server without isolation.

---

## [Phase 5 — Coding] — 2026-05-31

### Added
- `configs/phase5_coding.yaml` — Mistral-7B + `HuggingFaceH4/CodeAlpaca_20K` (full 20K dataset). max_seq_length=1024 (code completions need headroom). LoRA r=16, 3 epochs, LR 2.0e-4. Effective batch 16.
- `scripts/train_runpod_coding.py` — GPU-agnostic RunPod runner. seq_len=1024: batch sizes 8/4/4/2 for A100/A40/4090/T4 tiers. Est. wall time ~20–90 min depending on GPU.
- Pre-training length filter: drops samples where prompt+completion > 974 tokens (MAX_SEQ_LENGTH - 50) to prevent mid-function truncation. Preserves code correctness in training signal.
- Coding system prompt: expert software engineer framing, clean/correct/well-structured code.
- Post-training HumanEval pass@1 eval: generates solutions for N HumanEval problems, runs unit tests, reports pass@1. Default N=30 (fast), max 164 (full benchmark). Greedy decoding for reproducibility. Baseline ~35–40%, target ≥40%.
- `--n-problems` CLI flag to control HumanEval problem count.

### Design notes
- Field names differ from all prior phases: CodeAlpaca uses `prompt`/`completion`, not `instruction`/`output`. The formatter reads `sample["prompt"]` and `sample["completion"]` directly.
- Full 20K dataset used without subsetting — small enough to fit in a single 3-epoch run.
- HumanEval requires the `human-eval` package (Apache 2.0). Script auto-installs it if missing.

### Known limitations
- CodeAlpaca contains some samples with bugs in the expected completion — not model error.
- HumanEval is Python-only; the CodeAlpaca training mix includes JS/Java/C++ — cross-language generalization not captured by pass@1.
- Adapter not yet run on RunPod — `outputs/phase5-coding/` will populate after GPU run.

---

## [Phase 4 — Finance] — 2026-05-31

### Added
- `configs/phase4_finance.yaml` — Mistral-7B + `gbharti/finance-alpaca` (shuffled 10K subset of 68K). Single dataset, no auxiliary mix. max_seq_length=512 (short samples). LoRA r=16, 3 epochs, LR 2.0e-4. Effective batch 16.
- `scripts/train_runpod_finance.py` — GPU-agnostic RunPod runner. seq_len=512 enables larger batch sizes vs legal phases: 16/8/8/4 for A100/A40/4090/T4 tiers. Est. wall time ~15–50 min depending on GPU.
- Finance system prompt: expert financial analyst framing + "not financial advice" disclaimer.
- Post-training LLM-as-judge eval via Groq `llama-3.3-70b-versatile` (finance rubric). Baseline ~3.0/5.0, target ≥3.5/5.0.
- Finance judge rubric: 5-point scale checking correct formula/concept, direction of effect, key caveats, appropriate risk flags.

### Design note — Phase 4 scope
Phase 4 of project 03 is intentionally simple: one dataset, one adapter, standard pipeline. An expanded multi-domain finance project (macroeconomics, Indian markets, quant econometrics, development economics — each with specialty JSONL datasets) is planned as a **separate standalone project** to keep project 03 clean and focused.

### Known limitations
- `gbharti/finance-alpaca` contains date-sensitive data (historical stock prices, past rates) — system prompt instructs model to flag figures for verification.
- Adapter not yet run on RunPod — `outputs/phase4-finance/` will populate after GPU run.

---

## [Phase 3 — Legal] — 2026-05-30

Phase 3 split into two parallel variants targeting universal contract law and Indian law respectively. Indian law variant ships with a locally-curated BNS/BNSS/BSA mapping dataset that corrects the pre-July-2024 statute references baked into public Indian legal datasets.

### Added

#### Phase 3a — Legal Contracts (universal)
- `configs/phase3a_legal_contracts.yaml` — Mistral-7B + `nguyen-brat/legal_contracts` (~10K CUAD-derived contract clause Q&A) with `pile-of-law/freelaw` court opinions mixed at 20% for legal prose pretraining signal.
- `scripts/train_runpod_legal_contracts.py` — GPU-agnostic RunPod runner (auto-detects A100/A40/4090/T4 tiers, picks 4-bit vs bf16, flash-attn2 when available).
- LoRA r=16, 3 epochs, max_seq_length=2048, LR 1.5e-4. Effective batch 16.
- Post-training LLM-as-judge eval via Groq `llama-3.3-70b-versatile` (contract clause rubric). Baseline ~3.0/5.0, target ≥3.5/5.0.

#### Phase 3b — Indian Law (JusticeAI adapter)
- `configs/phase3b_indian_law.yaml` — Mistral-7B + `viber1/indian-law-dataset` (primary, ~60%) + `Exploration-Lab/InLegalNLI` (auxiliary, 25%) + local BNS mapping (15%).
- `scripts/train_runpod_indian_law.py` — GPU-agnostic RunPod runner with 3-way weighted dataset mixing.
- `scripts/build_bns_mapping_dataset.py` — generates `data/bns_bnss_bsa_mapping.jsonl` from 109 hand-curated seed mappings: 61 IPC→BNS, 30 CrPC→BNSS, 18 IEA→BSA. Template-driven expansion produces 809 instruction-tuning samples (~7.4 per seed mapping).
- LoRA r=32 (higher than prior phases — larger domain gap from Mistral's pretraining), 4 epochs, max_seq_length=2048, LR 1.0e-4, warmup 5%.
- BNS mapping samples are never dropped during weighted sampling — model sees every IPC↔BNS mapping multiple times across 4 epochs.
- Indian law system prompt enforces BNS/BNSS/BSA vs IPC/CrPC/IEA distinction. Indian-law judge rubric checks correct statute citation.

#### Cross-cutting
- `scripts/evaluate.py` extended with `_LEGAL_JUDGE_CONFIGS` dispatch: `judge_system: "legal_contracts"` or `"indian_law"` in the config YAML selects the appropriate system prompt + rubric.
- `LEGAL_CONTRACTS_SYSTEM` and `INDIAN_LAW_SYSTEM` constants exported from `evaluate.py`.
- Eval results now record `judge_system` field for downstream analysis.

### Changed
- `scripts/evaluate.py` — `evaluate_llm_judge()` now passes a domain-specific system prompt to `generate()` (was: no system prompt). LLM-judge fallback hardened: regex match on judge output now safe-handles `None`.
- `README.md` — domains table split Phase 3 into 3a + 3b; project structure updated to list new scripts and the `data/` directory.

### Why this matters
- **Indian legal context, 2026**: BNS (Bharatiya Nyaya Sanhita), BNSS (Bharatiya Nagarik Suraksha Sanhita), and BSA (Bharatiya Sakshya Adhiniyam) replaced IPC/CrPC/IEA effective July 1, 2024. Public HuggingFace Indian legal datasets (`viber1/indian-law-dataset`, `Exploration-Lab/InLegalNLI`) were built pre-2024 and teach the model repealed section numbers as if they were current law.
- Without the BNS mapping intervention, the fine-tuned model would confidently cite IPC §302 (now repealed) instead of BNS §103 (current) for any post-July-2024 matter — a hard failure mode for JusticeAI (Project 9).
- The mapping dataset is a *fine-tuning* fix (corrects weights). Recency (post-2024 SC judgments) will be handled separately by JusticeAI's RAG retrieval layer over IndianKanoon/eCourts.

### Known limitations
- BNS mapping seed coverage is ~109 mappings — concentrated on most-litigated provisions (murder, theft, rape, bail, FIR, electronic records). Less-common procedural provisions may still be referenced via old codes.
- Both Phase 3 variants are scaffolded but not yet run on RunPod — `outputs/` will be populated once GPU run completes.
- Indian SC has flagged "phantom citations" by AI legal tools in early-2026 review; the BNS mapping addresses statute-name accuracy but not citation grounding (which requires the JusticeAI RAG layer).

### Next session
- Run Phase 4 (Finance) on RunPod and push adapter to HF Hub.
- Or run Phase 3b on RunPod and push the JusticeAI adapter to HF Hub.

---

## [Phase 2 — Medical] — 2026-04 (approx)

### Added
- `configs/phase2_medical_medqa.yaml` — `medalpaca/medical_meadow_medqa` MCQ fine-tuning, 2 epochs, max_seq_length=1024.
- `scripts/train_runpod_medical.py` — GPU-agnostic RunPod runner with tier auto-detect.
- `scripts/evaluate.py` — MCQ accuracy + LLM-as-judge modes; MCQ system prompt + answer-letter regex extraction.
- Honest partial-epoch model card after early Kaggle stop (~39% MCQ).

### Lessons learned (carried forward)
- Avoid Kaggle 2× T4 for QLoRA — model-parallel layer split caused ~30× slowdown.
- Default to single-GPU RunPod/Vast.ai for all subsequent phases.

---

## [Phase 1 — General Instruction Following] — 2026-03 (approx)

### Added
- Initial QLoRA pipeline: 4-bit Mistral-7B loading, LoRA injection (r=16), SFTTrainer wiring.
- `configs/phase1_mistral_dolly.yaml` — `databricks/databricks-dolly-15k` 5K subset, 1 epoch, max_seq_length=512.
- `scripts/train.py` — config-driven training entrypoint.
- `scripts/inference.py`, `scripts/push_to_hub.py` — adapter loading + HF Hub upload.
- `docs/PLAN.md`, `docs/TUTORIAL.md`, `docs/DOMAIN_NOTES.md`, `docs/RUNPOD.md` — architecture, concepts-from-scratch, dataset notes, GPU runbook.

### Changed
- TRL 0.13+ migration: replaced `TrainingArguments` with `SFTConfig`; moved `max_length`/`dataset_text_field`/`packing` from `SFTTrainer.__init__` into `SFTConfig`; switched `tokenizer=` to `processing_class=` per Transformers 5.x.
- Switched to bf16=True (was fp16) — Mistral loads in bfloat16 by default; fp16 caused `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16`.
