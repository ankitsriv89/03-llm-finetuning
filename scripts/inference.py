"""
inference.py — Load Base Model + LoRA Adapter and Run Inference
===============================================================
After training, you have:
  - Base model: still on HuggingFace Hub (unchanged, not downloaded again)
  - LoRA adapter: small folder (~50-200MB) with adapter_model.safetensors

To run inference, you:
  1. Load the base model (quantized)
  2. Load and merge the LoRA adapter on top
  3. Run the model in generation mode

Two modes:
  - merge_and_unload: permanently fuse adapter into base model weights.
                      Faster inference, but loses the "separate adapter" structure.
  - keep separate:    load adapter on top of frozen base model each time.
                      Slower, but you can swap adapters dynamically.

For our multi-domain demo, we'll use "keep separate" so we can switch
between medical/legal/finance/coding adapters at runtime.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model_with_adapter(
    base_model_name: str,
    adapter_path: str,
    load_in_4bit: bool = True,
    merge: bool = False,
):
    """
    Load the base model + LoRA adapter.

    Args:
        base_model_name: HuggingFace model ID (e.g. "mistralai/Mistral-7B-Instruct-v0.2")
        adapter_path: local path or HF Hub ID of the trained adapter
        load_in_4bit: whether to quantize to 4-bit for memory efficiency
        merge: if True, permanently merge adapter into base model weights
               (faster inference, but can't swap adapters)
    """
    # Quantization config (same as training)
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Load base model
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapter on top of base model
    # PeftModel: wraps the base model and injects the adapter weights
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    if merge:
        # Merge adapter weights into base model.
        # After merging: model behaves like a standalone fine-tuned model.
        # Dequantize first if in 4-bit (merge requires float16/float32).
        print("Merging adapter into base model...")
        model = model.merge_and_unload()

    model.eval()  # Set to eval mode (disables dropout, etc.)
    return model, tokenizer


def generate(
    model,
    tokenizer,
    instruction: str,
    context: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Run inference on a single instruction.

    Key generation parameters:
    - max_new_tokens: max tokens to generate (not counting the input)
    - temperature: controls randomness.
        0.0 = fully deterministic (always picks highest probability token)
        1.0 = sample proportionally to probabilities
        >1.0 = more random/creative
        0.7 is a good balanced default.
    - top_p (nucleus sampling): only sample from the top-p probability mass.
        0.9 = only consider tokens that together make up 90% of probability.
        Cuts off very unlikely tokens. Works together with temperature.
    - do_sample: if False, use greedy decoding (always pick argmax).
                 if True, sample (use temperature + top_p).
    """
    # Format instruction as Mistral chat template
    user_msg = instruction
    if context:
        user_msg = f"{instruction}\n\nContext: {context}"

    messages = [{"role": "user", "content": user_msg}]

    # apply_chat_template with add_generation_prompt=True adds the [/INST] token
    # at the end, prompting the model to start generating the response.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # True during inference (unlike training)
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Generate
    with torch.no_grad():
        # torch.no_grad(): disables gradient computation.
        # During inference you don't need gradients — saves memory and compute.
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (not the input prompt)
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# ─────────────────────────────────────────────
# Demo: Interactive CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--adapter", required=True, help="Path to adapter folder or HF Hub ID")
    parser.add_argument("--merge", action="store_true", help="Merge adapter before inference")
    args = parser.parse_args()

    model, tokenizer = load_model_with_adapter(
        base_model_name=args.base_model,
        adapter_path=args.adapter,
        merge=args.merge,
    )

    print("\nModel ready. Type 'quit' to exit.\n")
    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() == "quit":
            break
        context = input("Context (optional, press Enter to skip): ").strip()
        response = generate(model, tokenizer, instruction, context)
        print(f"\nResponse:\n{response}\n{'─'*60}\n")
