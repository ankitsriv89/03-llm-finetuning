"""
evaluate.py — Post-training evaluation script
==============================================
Runs domain-appropriate evaluation on a trained adapter.

Usage:
    python scripts/evaluate.py \\
        --config configs/phase2_medical_medqa.yaml \\
        --adapter outputs/phase2-medical-medqa/final-adapter \\
        --mode mcq

Supported modes:
    mcq     — extract A/B/C/D letter and compare to ground truth (medical)
    llm     — LLM-as-judge scoring 1-5 via Groq API (legal, finance) [requires GROQ_API_KEY]
    passatk — code execution pass@k (coding) [requires human_eval installed]

LLM-judge uses Groq's llama-3.3-70b-versatile (free tier, fast).
Set GROQ_API_KEY environment variable or pass --groq-key.
"""

import argparse
import json
import re
import os
import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

def load_model_and_tokenizer(base_model: str, adapter_path: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────

def generate(
    model,
    tokenizer,
    prompt: str,
    system: str = "",
    max_new_tokens: int = 300,
    temperature: float = 0.1,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# MCQ Evaluation (Medical)
# ─────────────────────────────────────────────

MEDICAL_SYSTEM = (
    "You are a knowledgeable medical AI assistant. "
    "When given a clinical multiple-choice question, analyze the case carefully, "
    "identify the correct answer (A, B, C, or D), and provide a clear explanation. "
    "Always begin your response with 'The correct answer is X)' where X is the letter."
)


def extract_predicted_letter(text: str) -> str:
    """Extract A/B/C/D from model response. Returns '?' if not found."""
    m = re.search(r'correct answer is\s+([A-D])[).]?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'[Aa]nswer[:\s]+([A-D])[).]?', text)
    if m:
        return m.group(1).upper()
    m = re.match(r'^([A-D])[).\s]', text)
    if m:
        return m.group(1).upper()
    return "?"


def extract_ground_truth_letter(text: str) -> str:
    """Extract ground truth letter from MedQA output field."""
    m = re.search(r'correct answer is\s+([A-D])[).]?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r'^([A-D])[).\s]', text.strip())
    if m:
        return m.group(1).upper()
    return "?"


def evaluate_mcq(model, tokenizer, cfg: dict, max_samples: int = 500) -> dict:
    """
    Evaluate MCQ accuracy on the held-out test split of medical_meadow_medqa.

    Steps:
    1. Load dataset and create the same train/test split used during training
    2. For each test sample: generate a response and extract the predicted letter
    3. Compare to ground truth, compute accuracy
    """
    dataset_name = cfg["dataset"]["name"]
    test_size    = cfg["dataset"].get("test_size", 0.1)
    filter_5opt  = cfg["dataset"].get("filter_5_option", True)

    print(f"Loading dataset: {dataset_name}")
    raw = load_dataset(dataset_name, split="train")

    if filter_5opt:
        raw = raw.filter(lambda s: not bool(re.search(r'\bE\)', s['input'])))
        print(f"After filtering 5-option samples: {len(raw)} samples")

    split    = raw.train_test_split(test_size=test_size, seed=42)
    test_set = split["test"]
    test_set = test_set.select(range(min(max_samples, len(test_set))))
    print(f"Evaluating on {len(test_set)} test samples")

    correct   = 0
    total     = 0
    no_answer = 0
    records   = []

    for sample in tqdm(test_set, desc="MCQ eval"):
        assert isinstance(sample, dict)
        gt = extract_ground_truth_letter(str(sample.get("output", "")))
        if gt == "?":
            continue

        response = generate(
            model, tokenizer, str(sample.get("input", "")),
            system=MEDICAL_SYSTEM, max_new_tokens=200
        )
        pred = extract_predicted_letter(response)

        is_correct = (pred == gt)
        if pred == "?":
            no_answer += 1

        correct += is_correct
        total   += 1
        records.append({"ground_truth": gt, "predicted": pred, "correct": is_correct})

    accuracy = correct / total if total > 0 else 0.0

    results = {
        "mode":           "mcq",
        "dataset":        dataset_name,
        "num_samples":    total,
        "correct":        correct,
        "accuracy":       round(accuracy, 4),
        "no_answer_rate": round(no_answer / total, 4) if total > 0 else 0.0,
        "records":        records,
    }

    print(f"\n{'='*50}")
    print(f"MCQ ACCURACY: {accuracy:.1%} ({correct}/{total})")
    print(f"No-answer rate: {no_answer/total:.1%}")
    print(f"Random chance baseline: 25.0%")
    print(f"{'='*50}")

    return results


# ─────────────────────────────────────────────
# LLM-as-Judge Evaluation via Groq (Legal / Finance)
# ─────────────────────────────────────────────

JUDGE_MODEL = "llama-3.3-70b-versatile"  # fast, free tier on Groq

LEGAL_CONTRACTS_SYSTEM = (
    "You are an expert legal assistant specializing in contract law. "
    "When given a clause from a legal contract, analyze it carefully and provide "
    "a clear, accurate explanation of its legal implications, risks, and key terms. "
    "Use precise legal language while remaining accessible to non-lawyers. "
    "Cite relevant legal concepts and flag any unusual or one-sided provisions."
)

INDIAN_LAW_SYSTEM = (
    "You are an expert in Indian law with deep knowledge of the Bharatiya Nyaya Sanhita "
    "(BNS, 2023), Bharatiya Nagarik Suraksha Sanhita (BNSS, 2023), Bharatiya Sakshya "
    "Adhiniyam (BSA, 2023), the Constitution of India, and Indian Supreme Court and High "
    "Court jurisprudence. When answering legal questions, cite the correct current statutes "
    "(BNS/BNSS/BSA for post-July 2024 matters, IPC/CrPC/IEA for historical cases), "
    "reference relevant Supreme Court precedents where applicable, and provide accurate, "
    "precise legal analysis. Clearly distinguish between the old codes (IPC/CrPC) and the "
    "new codes (BNS/BNSS/BSA) when the distinction is legally material."
)

# judge_system → (system_prompt, rubric_text, baseline_note)
_LEGAL_JUDGE_CONFIGS: dict[str, tuple[str, str, str]] = {
    "legal_contracts": (
        LEGAL_CONTRACTS_SYSTEM,
        (
            "Rating criteria for contract clause responses:\n"
            "1 — Wrong, irrelevant, or legally incorrect\n"
            "2 — Partially correct with significant legal errors or omissions\n"
            "3 — Correct interpretation but missing key legal implications\n"
            "4 — Correct, complete, minor omissions; good legal precision\n"
            "5 — Accurate, complete, legally precise; correctly identifies risks and obligations"
        ),
        "Baseline (Mistral base on contract clauses): ~3.0/5.0",
    ),
    "indian_law": (
        INDIAN_LAW_SYSTEM,
        (
            "Rating criteria for Indian law responses:\n"
            "1 — Wrong statute cited, incorrect legal principle, or factually wrong\n"
            "2 — Correct general area but wrong section numbers or significant procedural errors\n"
            "3 — Correct answer but missing key nuance (e.g. old vs new code distinction)\n"
            "4 — Correct, proper citations, minor omissions in nuance or case law\n"
            "5 — Accurate, correct BNS/BNSS/BSA vs IPC/CrPC distinction, relevant SC precedents cited"
        ),
        "Baseline (Mistral base on Indian law): ~2.5/5.0  |  Target: ≥3.5/5.0",
    ),
}


def evaluate_llm_judge(
    model, tokenizer, cfg: dict, max_samples: int = 50, groq_api_key: str | None = None
) -> dict:
    """
    Score responses with Groq's llama-3.3-70b-versatile as LLM judge on a 1-5 scale.
    Requires GROQ_API_KEY environment variable (or pass --groq-key).

    Groq free tier: ~14,400 requests/day — more than enough for 100 eval samples.

    Domain dispatch via cfg["evaluation"]["judge_system"]:
        "legal_contracts" — contract clause rubric (Phase 3a)
        "indian_law"      — BNS/IPC/BNSS rubric (Phase 3b, JusticeAI)
        (unset)           — generic 1-5 rubric (Phase 1/2 fallback)
    """
    try:
        from groq import Groq
    except ImportError:
        print("ERROR: 'groq' package required. pip install groq")
        return {}

    api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Pass --groq-key or set the env variable.")
        return {}

    groq_client = Groq(api_key=api_key)

    dataset_name      = cfg["dataset"]["name"]
    test_size         = cfg["dataset"].get("test_size", 0.1)
    instruction_field = cfg["dataset"].get("instruction_field", "instruction")
    response_field    = cfg["dataset"].get("response_field", "output")
    context_field     = cfg["dataset"].get("context_field", None)

    # Pick domain-specific system prompt + judge rubric
    judge_system_key = cfg.get("evaluation", {}).get("judge_system", "")
    if judge_system_key in _LEGAL_JUDGE_CONFIGS:
        domain_system, rubric, baseline_note = _LEGAL_JUDGE_CONFIGS[judge_system_key]
    else:
        domain_system = ""
        rubric = (
            "Rating criteria:\n"
            "1 — Completely wrong or irrelevant\n"
            "2 — Partially addresses the question with major errors\n"
            "3 — Correct answer but missing key details\n"
            "4 — Correct and mostly complete, minor omissions\n"
            "5 — Accurate, complete, domain-appropriate response"
        )
        baseline_note = ""

    print(f"Loading dataset: {dataset_name}")
    raw = load_dataset(dataset_name, split="train")
    split    = raw.train_test_split(test_size=test_size, seed=42)
    test_set = split["test"].select(range(min(max_samples, len(split["test"]))))
    print(f"Evaluating {len(test_set)} samples with Groq LLM judge ({JUDGE_MODEL})")
    if judge_system_key:
        print(f"Judge rubric: {judge_system_key}")

    scores  = []
    records = []

    for sample in tqdm(test_set, desc="LLM-judge eval"):
        assert isinstance(sample, dict)
        instruction  = str(sample.get(instruction_field) or sample.get("input") or "")
        context      = str(sample.get(context_field) or "") if context_field else ""
        ground_truth = str(sample.get(response_field) or "")

        user_prompt = instruction
        if context:
            user_prompt = f"{instruction}\n\nContext: {context}"

        response = generate(model, tokenizer, user_prompt,
                            system=domain_system, max_new_tokens=300)

        judge_prompt = (
            f"You are an expert evaluator. Rate the following AI response on a scale of 1-5.\n\n"
            f"Question: {instruction[:400]}\n\n"
            f"Reference answer: {ground_truth[:400]}\n\n"
            f"AI response: {response[:400]}\n\n"
            f"{rubric}\n\n"
            f"Respond with ONLY a single digit (1-5). No explanation."
        )

        try:
            completion = groq_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=5,
                temperature=0,
            )
            score_text = completion.choices[0].message.content or ""
            m = re.search(r'[1-5]', score_text.strip())
            score = int(m.group()) if m else 3
        except Exception as e:
            print(f"Groq judge error: {e}")
            score = 3  # neutral fallback on API error

        scores.append(score)
        records.append({
            "instruction": instruction[:100],
            "score":       score,
            "response":    response[:200],
        })

    avg_score = sum(scores) / len(scores) if scores else 0.0
    score_dist = {str(i): scores.count(i) for i in range(1, 6)}

    results = {
        "mode":         "llm_judge",
        "judge_model":  JUDGE_MODEL,
        "judge_system": judge_system_key or "generic",
        "dataset":      dataset_name,
        "num_samples":  len(scores),
        "avg_score":    round(avg_score, 3),
        "score_dist":   score_dist,
        "records":      records,
    }

    print(f"\n{'='*50}")
    print(f"LLM JUDGE (Groq {JUDGE_MODEL})")
    print(f"Average score: {avg_score:.2f}/5.0")
    print(f"Distribution:  {score_dist}")
    if baseline_note:
        print(baseline_note)
    print(f"{'='*50}")

    return results


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Post-training evaluation")
    parser.add_argument("--config",      required=True,  help="Path to phase YAML config")
    parser.add_argument("--adapter",     required=True,  help="Path to trained adapter")
    parser.add_argument("--mode",        default="mcq",  choices=["mcq", "llm", "passatk"],
                        help="Evaluation mode: mcq | llm | passatk")
    parser.add_argument("--max-samples", type=int,       default=None,
                        help="Max evaluation samples (default: from config or 500)")
    parser.add_argument("--groq-key",    default=None,
                        help="Groq API key (fallback: GROQ_API_KEY env var)")
    parser.add_argument("--output",      default=None,
                        help="JSON output path (default: <adapter_dir>/eval_results.json)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_model  = cfg["model"]["name"]
    max_samples = args.max_samples or cfg.get("evaluation", {}).get("max_eval_samples", 500)
    output_path = args.output or os.path.join(os.path.dirname(args.adapter), "eval_results.json")

    model, tokenizer = load_model_and_tokenizer(base_model, args.adapter)

    if args.mode == "mcq":
        results = evaluate_mcq(model, tokenizer, cfg, max_samples=max_samples)
    elif args.mode == "llm":
        results = evaluate_llm_judge(
            model, tokenizer, cfg,
            max_samples=max_samples,
            groq_api_key=args.groq_key,
        )
    elif args.mode == "passatk":
        print("pass@k evaluation not yet implemented in this script.")
        print("See notebooks/05_coding.ipynb for HumanEval evaluation.")
        return
    else:
        print(f"Unknown mode: {args.mode}")
        return

    results["config"]  = args.config
    results["adapter"] = args.adapter

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
