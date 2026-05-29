# RunPod Setup — Phase 2 Medical

## Why RunPod for this phase
Kaggle's free 2× T4 setup uses model parallelism (layer-split) — only one GPU active at a time, ~90 min training.
A single A100 80GB on RunPod runs the same job in ~15-20 min by:
- Skipping 4-bit quantization (bf16 LoRA directly — no dequant overhead per forward)
- Larger batch (16 vs 4) — better GPU utilization
- Flash Attention 2 — ~2x attention speedup at 1024 seq len
- No gradient checkpointing (we have the VRAM) — saves a recompute pass
- TF32 matmul + fused AdamW

## Cost
A100 80GB Community Cloud: ~$1.89/hr. End-to-end (~20 min training + ~10 min eval) ≈ $1.00.

## Setup steps

1. **Pod creation**
   - Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
   - GPU: 1× A100 80GB
   - Container disk: 50 GB (model + dataset)
   - Volume: optional, only if you want adapter persistence across pod stops

2. **SSH in, clone repo, install**
   ```bash
   cd /workspace
   git clone <your-repo> jobs-prjcts
   cd jobs-prjcts/03-llm-finetuning
   pip install -r requirements_runpod.txt
   ```

3. **HuggingFace login** (for model download + adapter push)
   ```bash
   huggingface-cli login
   # paste your HF token
   ```

4. **Run training**
   ```bash
   # Train + eval only
   python scripts/train_runpod_medical.py

   # Train + eval + push adapter to Hub in one shot
   python scripts/train_runpod_medical.py \
     --push-hub anksriv/mistral-7b-medical-medqa-qlora

   # Train only, skip 10-min eval (push later from local)
   python scripts/train_runpod_medical.py --skip-eval
   ```

5. **Background run** (so SSH disconnect doesn't kill it)
   ```bash
   nohup python scripts/train_runpod_medical.py \
     --push-hub anksriv/mistral-7b-medical-medqa-qlora \
     > train.log 2>&1 &
   tail -f train.log
   ```

## Expected output
- Training: ~15-20 min, loss drops from ~1.8 → ~0.7
- Eval (500 samples): ~8-10 min, target accuracy ≥ 50%
- Adapter pushed to HF Hub: ~50-200 MB

## Troubleshooting
- **`flash-attn` install fails**: skip it, remove `attn_implementation="flash_attention_2"` from `load_model_and_tokenizer()`. You'll lose ~30% speed but it still works.
- **OOM during training**: reduce `BATCH_SIZE` from 16 to 8.
- **Eval is too slow**: lower `MAX_EVAL_SAMPLES` from 500 to 100 (still statistically meaningful, runs in ~2 min).
