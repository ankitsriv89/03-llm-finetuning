# Changelog

All notable changes to the QLoRA fine-tuning pipeline.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dates are absolute (YYYY-MM-DD).

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
