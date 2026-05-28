# Tutorial: LLM Fine-Tuning with QLoRA — Complete Guide

This tutorial explains every concept from first principles. Work through it top-to-bottom before running any code.

---

## Part 1: How Transformers Work (the parts that matter for fine-tuning)

### The Transformer Block

Mistral-7B has 32 layers. Each layer has two sub-components:

**1. Multi-Head Self-Attention**

Self-attention answers: "for each token, which other tokens should I pay attention to?"

```
Input token embeddings: X  (shape: [seq_len, hidden_dim])

Q = X @ W_q   (queries — what am I looking for?)
K = X @ W_k   (keys   — what do I have?)
V = X @ W_v   (values — what do I pass forward?)

Attention = softmax(Q @ K.T / sqrt(d_k)) @ V
Output    = Attention @ W_o
```

`W_q`, `W_k`, `W_v`, `W_o` are the weight matrices. **LoRA is applied to these.**

**2. Feed-Forward Network (FFN)**

After attention, each token's representation passes through an FFN:

```
# Mistral uses SwiGLU activation (3 matrices instead of 2)
gate = X @ W_gate
up   = X @ W_up
out  = (SiLU(gate) * up) @ W_down
```

`W_gate`, `W_up`, `W_down` are the FFN weight matrices. **LoRA is also applied to these.**

### Why These Matrices?

The weight matrices are where "knowledge" and "behavior" live. Fine-tuning updates them so the model behaves differently. LoRA adds trainable updates to these specific matrices while keeping the original weights frozen.

---

## Part 2: LoRA — Low-Rank Adaptation

### The Key Insight

When you fine-tune a model, you're learning a weight update `ΔW`. The original weight matrix `W` is updated to `W + ΔW`.

The insight in the LoRA paper: **the weight update `ΔW` is low-rank in practice.** You don't need a full `d × d` matrix to capture the adaptation — a much smaller matrix works almost as well.

### The Math

```
W₀: original weight matrix  (d × d, e.g., 4096 × 4096 = 16.7M params)
ΔW: weight update            (d × d, full fine-tuning updates all 16.7M)

LoRA factorizes ΔW = A @ B where:
  A: d × r   (e.g., 4096 × 16 = 65K params)
  B: r × d   (e.g., 16 × 4096 = 65K params)
  Total LoRA params: 130K vs 16.7M for full fine-tuning
```

`r` is the **rank** — the bottleneck dimension. Lower rank = fewer params = less expressive. For instruction tuning, `r=16` is the standard starting point.

**At inference time**, LoRA can be merged: `W = W₀ + (alpha/r) * A @ B`

The scaling factor `alpha/r` controls how strongly the adapter influences the output. With `r=16, alpha=32`, the scaling is `32/16 = 2.0`.

### What Gets Trained

```python
# Before LoRA:
for name, param in model.named_parameters():
    print(name, param.requires_grad)
# All: requires_grad = True (full fine-tuning)

# After get_peft_model(model, lora_config):
for name, param in model.named_parameters():
    print(name, param.requires_grad)
# base.weight:    requires_grad = False  ← FROZEN
# lora_A.weight:  requires_grad = True   ← TRAINS
# lora_B.weight:  requires_grad = True   ← TRAINS
```

---

## Part 3: QLoRA — Quantized LoRA

### The Memory Problem

Even with LoRA (frozen base weights), you still need to load the base model into GPU memory. For Mistral-7B:

```
float32:  7B × 4 bytes = 28GB   (impossible on T4)
float16:  7B × 2 bytes = 14GB   (barely fits on A100)
int8:     7B × 1 byte  = 7GB    (fits on A100, tight on A10G)
nf4:      7B × 0.5 bytes = 3.5GB (fits on T4!)
```

### NF4 Quantization

NF4 (NormalFloat4) is the specific 4-bit format used in QLoRA, introduced in the QLoRA paper (Dettmers et al., 2023).

Key property: NF4 is information-theoretically optimal for normally distributed data. Neural network weights are approximately normally distributed, so NF4 preserves more information per bit than standard int4.

### Double Quantization

The quantization process requires storing "quantization constants" (scale factors). These constants themselves take up memory. Double quantization quantizes the quantization constants:

- Normal quantization constants: ~0.5 bits/param overhead
- After double quantization: ~0.1 bits/param overhead
- Saving: ~0.4 bits/param → ~350MB for a 7B model

Small saving, essentially free quality.

### Compute Dtype

The weights are stored as 4-bit integers, but computation happens in float16:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # dequantize to fp16 for matmul
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

Dequantization happens per-layer, on the fly. The GPU never holds the full model in fp16 at once — only the current layer being computed.

---

## Part 4: The Training Loop

### What SFTTrainer Does

`SFTTrainer` (from TRL) wraps HuggingFace `Trainer` with instruction-tuning specifics:

1. **Tokenizes** the formatted text strings into input_ids
2. **Creates labels** for next-token prediction
3. **Masks the instruction tokens** in the labels (sets them to -100)
   - Loss is only computed on response tokens
   - The model doesn't get penalized for "not predicting" the instruction
4. **Runs the training loop**: forward pass → loss → backward pass → optimizer step

### Response Masking — Why It Matters

Imagine training on:
```
[INST] What is the capital of France? [/INST] Paris.
```

Without masking, the model learns to predict every token including `[INST]`, `What`, `is`, etc. This wastes training signal on tokens you don't care about.

With masking:
```
Input:  [INST] What is the capital of France? [/INST] Paris.
Labels: -100   -100  -100 -100 -100 -100 -100  -100  Paris.
```

Loss is only computed on `Paris.` — the part you actually want the model to learn to generate.

### Gradient Accumulation

With `per_device_train_batch_size=4` and `gradient_accumulation_steps=4`:

```
Step 1: forward pass on samples 1-4,   compute loss, accumulate gradients
Step 2: forward pass on samples 5-8,   compute loss, accumulate gradients
Step 3: forward pass on samples 9-12,  compute loss, accumulate gradients
Step 4: forward pass on samples 13-16, compute loss, accumulate gradients
        → NOW update weights (effective batch = 16)
```

This simulates a batch size of 16 using the memory of a batch size of 4.

### Gradient Checkpointing

Normally, PyTorch stores all intermediate activations during the forward pass (needed for backprop). For a 32-layer model with long sequences, this is huge.

Gradient checkpointing trades compute for memory: it discards intermediate activations and recomputes them during the backward pass. Result: ~30% slower training, but 60-70% less activation memory.

### Paged AdamW

Standard AdamW stores two "momentum" values per parameter (m and v). For 7B params, that's 28GB just for optimizer states in float32.

`paged_adamw_32bit` uses NVIDIA unified memory to automatically offload optimizer states to CPU RAM when GPU memory is tight. Prevents Out-of-Memory crashes during training.

### Learning Rate Schedule

```
Warmup (3% of steps): lr linearly increases 0 → 2e-4
Main training:         lr follows cosine curve 2e-4 → ~0
```

Warmup prevents instability at the start — large gradient updates on a freshly initialized LoRA (random weights) can destabilize the base model. Cosine decay gives a smooth learning rate reduction rather than sudden drops.

---

## Part 5: Chat Template and Prompt Formatting

### Why the Format Matters

Every instruct model is fine-tuned with a specific conversation format. The model learned to produce responses only when it sees the exact tokens that signal "you should respond now."

Mistral Instruct v0.2 uses:
```
<s>[INST] user message here [/INST] assistant response here </s>
```

The tokenizer knows this format:
```python
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
# → "<s>[INST] What is the capital of France? [/INST] Paris.</s>"
```

During inference, use `add_generation_prompt=True` to add `[/INST]` at the end — this prompts the model to start generating:
```python
text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
# → "<s>[INST] What is the capital of France? [/INST]"
# Model then generates: "Paris.</s>"
```

---

## Part 6: Generation Parameters

### Temperature

Controls randomness. Mathematically, divides the logits before softmax:

```
logits_scaled = logits / temperature
probs = softmax(logits_scaled)
```

- `temperature=0`: argmax (always pick highest probability token) — deterministic
- `temperature=0.7`: mild randomness — good for Q&A
- `temperature=1.0`: sample proportionally — more creative
- `temperature>1.5`: very random — mostly incoherent

### Top-p (Nucleus Sampling)

Sort tokens by probability descending. Sum probabilities until you reach `p`. Only sample from that set.

With `top_p=0.9`: ignore the long tail of low-probability tokens. Focus on the top 90% probability mass.

Combined with temperature, this prevents both repetitive determinism and nonsense randomness.

### Max New Tokens

Hard cap on output length. Does NOT affect quality within the cap. Set based on expected output length:
- Short answers: 128
- Paragraphs: 256-512
- Long documents: 1024-2048

---

## Part 7: Domain-Specific Considerations

### Medical

**Dataset quirk**: MedQA samples are 4-option multiple-choice questions. Format them as:

```
[INST] Answer the following medical question by selecting the correct option.

Question: A 45-year-old patient presents with...

Options:
A. ...
B. ...
C. ...
D. ...

Select the single best answer and explain your reasoning. [/INST]

The correct answer is B. ...explanation...
```

**Why longer explanations?** Medical professionals need to see the reasoning, not just the answer. Training on explained answers produces more useful output.

**Safety note**: Always add a system prompt in the demo warning users not to use AI-generated medical advice clinically.

### Legal

**Dataset quirk**: Legal documents are long (contracts can be 50+ pages). You'll need `max_seq_length=2048` and potentially `packing=True` for efficiency.

**Format**: Legal Q&A is often about specific clause analysis:
```
[INST] Analyze the following contract clause and identify any problematic terms...
[context: clause text]
[/INST]
```

### Finance

**Dataset quirk**: Finance-alpaca contains both factual Q&A and calculation tasks. Be careful with numerical reasoning — LLMs are weak at arithmetic.

**Approach**: For calculation questions, fine-tune the model to show its work step-by-step (chain-of-thought style). Better accuracy on numerical tasks.

### Coding

**Dataset quirk**: Code must be syntactically valid. Use the `pass@1` metric — generate one solution, run it, check if tests pass.

**Format**: Code datasets typically use this structure:
```
[INST] Write a Python function that... [/INST]
def solution():
    ...
```

The tokenizer handles code whitespace (indentation) the same as regular text — just characters. Model learns to reproduce correct indentation through training.

---

## Part 8: Pushing to HuggingFace Hub

### What Gets Pushed

After training, your adapter folder contains:

```
final-adapter/
├── adapter_config.json        ← LoRA hyperparameters (r, alpha, target_modules, etc.)
├── adapter_model.safetensors  ← The actual trained weights (~50-200MB)
├── tokenizer.json             ← Tokenizer vocabulary
├── tokenizer_config.json      ← Tokenizer settings
└── special_tokens_map.json    ← Special token definitions
```

Total size: ~100-300MB per adapter. The 7B base model (~3.5GB) is NOT included — it stays on HuggingFace Hub.

### How Others Load Your Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...),
    device_map="auto",
)

# 2. Load YOUR adapter on top
model = PeftModel.from_pretrained(model, "your-username/mistral-medical-qlora")

# That's it — model is now your fine-tuned version
```

### Multi-Adapter Loading (for the demo)

```python
# Load all adapters once
model.load_adapter("your-username/mistral-dolly-qlora",   adapter_name="general")
model.load_adapter("your-username/mistral-medical-qlora", adapter_name="medical")
model.load_adapter("your-username/mistral-legal-qlora",   adapter_name="legal")
model.load_adapter("your-username/mistral-finance-qlora", adapter_name="finance")
model.load_adapter("your-username/mistral-coding-qlora",  adapter_name="coding")

# Switch domain based on user selection
model.set_adapter("medical")
response = generate(model, tokenizer, "What are symptoms of appendicitis?")
```

One 3.5GB base model + five ~100MB adapters = ~4GB total. Efficient.

---

## Key Concepts Summary

| Concept | What it is | Config param |
|---------|-----------|-------------|
| 4-bit NF4 quantization | Store weights as 4-bit NormalFloat for memory efficiency | `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"` |
| Double quantization | Quantize the quantization constants for extra savings | `bnb_4bit_use_double_quant=True` |
| LoRA rank (r) | Expressiveness of adapter. Higher = more params, more capacity | `r=16` |
| LoRA alpha | Scaling factor. Keep at 2×r | `lora_alpha=32` |
| Gradient accumulation | Simulate larger batch by accumulating gradients | `gradient_accumulation_steps=4` |
| Gradient checkpointing | Trade compute for memory (recompute activations) | `gradient_checkpointing=True` |
| Paged AdamW | Offload optimizer states to CPU to prevent OOM | `optim="paged_adamw_32bit"` |
| Response masking | Only compute loss on assistant response tokens | Handled by SFTTrainer |
| Chat template | Model-specific prompt format (e.g., `[INST]...[/INST]`) | `tokenizer.apply_chat_template()` |
| Temperature | Generation randomness control | `temperature=0.7` |
| Top-p | Nucleus sampling — ignore low-probability tokens | `top_p=0.9` |

---

## What to Explore Next

1. **RLHF / DPO**: After SFT, use Direct Preference Optimization to align the model with human preferences. DPO trains on (chosen, rejected) response pairs.
2. **Merge adapters**: Average multiple domain adapters into one using `mergekit`.
3. **Quantize the final model**: After merging, quantize to GGUF format for `llama.cpp` — runs on CPU.
4. **Ollama deployment**: Serve the quantized model locally via Ollama.
5. **Benchmark properly**: Run the full MedQA test set, HumanEval, and FinQA evaluations to measure improvement over baseline.
