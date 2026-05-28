"""
push_to_hub.py — Upload LoRA Adapter to HuggingFace Hub
=========================================================
After training, your adapter is a small folder (~50-200MB) containing:
  - adapter_config.json      : LoRA hyperparameters (r, alpha, target_modules, etc.)
  - adapter_model.safetensors: The actual trained adapter weights
  - tokenizer files           : vocab, special tokens config, etc.

You push ONLY this adapter — not the 7B base model (already on HF Hub).
Anyone who wants to use your fine-tuned model loads:
  1. The base model from HF Hub (e.g. mistralai/Mistral-7B-Instruct-v0.2)
  2. Your adapter from your HF Hub repo

This is the power of PEFT: the adapter is tiny and shareable.

Usage:
    python scripts/push_to_hub.py \
        --adapter_path outputs/phase1-mistral-dolly/final-adapter \
        --repo_id your-hf-username/mistral-dolly-qlora \
        --private false
"""

import argparse
from huggingface_hub import HfApi, login
import os


def push_adapter(adapter_path: str, repo_id: str, private: bool = False):
    """
    Push a trained LoRA adapter to HuggingFace Hub.

    The adapter folder must contain:
    - adapter_config.json
    - adapter_model.safetensors (or adapter_model.bin)
    - tokenizer files

    Args:
        adapter_path: local path to the saved adapter folder
        repo_id: HF Hub repository ID (e.g. "username/my-adapter")
        private: whether to make the repo private
    """
    # Check HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN environment variable not set.")
        print("Get your token from: https://huggingface.co/settings/tokens")
        print("Then run: export HF_TOKEN=hf_your_token_here")
        raise ValueError("HF_TOKEN not set")

    login(token=hf_token)

    api = HfApi()

    # Create the repository if it doesn't exist
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,  # don't fail if repo already exists
    )

    # Upload all files from the adapter folder
    api.upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\nAdapter successfully pushed to: https://huggingface.co/{repo_id}")
    print(f"\nTo load this adapter for inference:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{repo_id}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--repo_id", required=True, help="e.g. your-username/mistral-dolly-qlora")
    parser.add_argument("--private", action="store_true", default=False)
    args = parser.parse_args()

    push_adapter(args.adapter_path, args.repo_id, args.private)
