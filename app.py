"""
app.py — Multi-Domain QLoRA Demo
=================================
Gradio demo for 5 domain-specific Mistral-7B QLoRA adapters.

Adapters hot-swap at runtime — only one 7B base model loaded in 4-bit (~3.5 GB).

Domains:
    general   — databricks-dolly-15k (general instruction following)
    medical   — medical_meadow_medqa (clinical MCQ reasoning)
    legal     — legal contracts (CUAD contract clause analysis)
    finance   — finance-alpaca (financial analysis + ratios)
    coding    — CodeAlpaca-20K (code generation)

HF Hub adapter repos (set HF_USERNAME in env or edit ADAPTER_REPOS below):
    {HF_USERNAME}/mistral-7b-dolly-qlora
    {HF_USERNAME}/mistral-7b-medical-medqa-qlora
    {HF_USERNAME}/mistral-7b-legal-contracts-qlora
    {HF_USERNAME}/mistral-7b-finance-qlora
    {HF_USERNAME}/mistral-7b-coding-qlora

Environment:
    HF_TOKEN       — HuggingFace token (read-only is sufficient)
    HF_USERNAME    — your HF username (default: anksriv)
    LOCAL_ADAPTERS — comma-separated local adapter paths (overrides Hub for each domain)
                     e.g. "medical=./outputs/phase2/final-adapter,coding=./outputs/phase5/final-adapter"
    NO_GPU         — set to "1" to force CPU (slow, for local testing without GPU)
"""

import os
import threading
from typing import Iterator

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

BASE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"
HF_USERNAME  = os.environ.get("HF_USERNAME", "anksriv")

DOMAINS = ["general", "medical", "legal", "finance", "coding"]

ADAPTER_REPOS = {
    "general": f"{HF_USERNAME}/mistral-7b-dolly-qlora",
    "medical": f"{HF_USERNAME}/mistral-7b-medical-medqa-qlora",
    "legal":   f"{HF_USERNAME}/mistral-7b-legal-contracts-qlora",
    "finance": f"{HF_USERNAME}/mistral-7b-finance-qlora",
    "coding":  f"{HF_USERNAME}/mistral-7b-coding-qlora",
}

DOMAIN_SYSTEMS = {
    "general": (
        "You are a helpful, accurate, and concise assistant. Answer questions clearly and directly."
    ),
    "medical": (
        "You are a knowledgeable medical AI assistant. "
        "When given a clinical multiple-choice question, analyze the case carefully, "
        "identify the correct answer (A, B, C, or D), and provide a clear explanation. "
        "Always begin your response with 'The correct answer is X)' where X is the letter."
    ),
    "legal": (
        "You are an expert legal assistant specializing in contract law. "
        "When given a clause from a legal contract, analyze it carefully and provide "
        "a clear, accurate explanation of its legal implications, risks, and key terms. "
        "Use precise legal language while remaining accessible to non-lawyers. "
        "Cite relevant legal concepts and flag any unusual or one-sided provisions."
    ),
    "finance": (
        "You are an expert financial analyst and economist. "
        "When given a financial or economic question, provide accurate, well-reasoned analysis. "
        "Show your calculations when relevant. Cite key financial concepts and metrics. "
        "Note: this is for educational purposes only — not financial advice."
    ),
    "coding": (
        "You are an expert software engineer and programmer. When given a coding task "
        "or programming problem, write clean, correct, and well-structured code. "
        "Follow best practices for the language being used. If the task is ambiguous, "
        "state your assumptions briefly before the code. Produce working solutions."
    ),
}

DOMAIN_EXAMPLES = {
    "general": [
        "Explain the difference between supervised and unsupervised learning.",
        "What are the key principles of good software design?",
        "Summarize the causes of the 2008 financial crisis.",
    ],
    "medical": [
        (
            "A 65-year-old man presents with sudden-onset crushing chest pain radiating to the left arm. "
            "ECG shows ST elevation in leads II, III, and aVF. Which artery is most likely occluded?\n"
            "A) Left anterior descending\nB) Right coronary artery\nC) Left circumflex\nD) Left main coronary"
        ),
        (
            "A 28-year-old woman presents with polyuria, polydipsia, and a fasting glucose of 250 mg/dL. "
            "Anti-GAD antibodies are positive. What is the most appropriate initial treatment?\n"
            "A) Metformin\nB) Glipizide\nC) Insulin\nD) Lifestyle modification alone"
        ),
    ],
    "legal": [
        (
            "Analyze this clause: 'The Indemnifying Party shall defend, indemnify, and hold harmless "
            "the Indemnified Party from any claims arising from the Indemnifying Party's gross negligence "
            "or willful misconduct, but excluding any claims arising from the Indemnified Party's "
            "own negligence.'"
        ),
        (
            "Analyze this non-compete clause: 'Employee agrees not to engage in any business that competes "
            "with Employer within a 100-mile radius for a period of 3 years following termination, "
            "regardless of the reason for termination.'"
        ),
    ],
    "finance": [
        "Explain the relationship between bond prices and interest rates, with an example.",
        "A company has revenue of $10M, COGS of $6M, operating expenses of $2M, and tax rate of 25%. Calculate EBIT, net income, and gross margin.",
        "What is the difference between systematic and unsystematic risk? How does diversification help?",
    ],
    "coding": [
        "Write a Python function that finds the longest common subsequence of two strings.",
        "Implement a binary search tree with insert, search, and in-order traversal methods in Python.",
        "Write a Python decorator that retries a function up to 3 times on exception, with exponential backoff.",
    ],
}

DOMAIN_LABELS = {
    "general": "🤖 General",
    "medical": "🏥 Medical",
    "legal":   "⚖️  Legal",
    "finance": "📈 Finance",
    "coding":  "💻 Coding",
}

DOMAIN_DESCRIPTIONS = {
    "general": "General instruction following — trained on Databricks Dolly-15K",
    "medical": "Clinical MCQ reasoning — trained on MedAlpaca MedQA (USMLE-style questions)",
    "legal":   "Contract clause analysis — trained on CUAD-derived legal_contracts dataset",
    "finance": "Financial analysis — trained on finance-alpaca (economic Q&A and ratio analysis)",
    "coding":  "Code generation — trained on CodeAlpaca-20K (Python, JS, Java, C++)",
}


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

_model     = None
_tokenizer = None
_loaded_adapters: set[str] = set()
_load_lock = threading.Lock()


def _resolve_adapter_path(domain: str) -> str:
    """
    Check LOCAL_ADAPTERS env var first (e.g. for testing before HF push),
    then fall back to HF Hub repo ID.
    """
    local_map: dict[str, str] = {}
    raw = os.environ.get("LOCAL_ADAPTERS", "")
    if raw:
        for part in raw.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                local_map[k.strip()] = v.strip()

    if domain in local_map and os.path.isdir(local_map[domain]):
        return local_map[domain]
    return ADAPTER_REPOS[domain]


def _load_base():
    """Load the base model once. Called lazily on first inference request."""
    global _model, _tokenizer

    use_cpu = os.environ.get("NO_GPU") == "1" or not torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if use_cpu:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if _bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
        )

    _tokenizer = tokenizer
    _model     = model


def _bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return (props.major + props.minor / 10) >= 8.0


def _ensure_adapter(domain: str):
    """Load a domain adapter into the base model if not already loaded."""
    global _model

    if domain in _loaded_adapters:
        return

    adapter_path = _resolve_adapter_path(domain)
    print(f"Loading adapter '{domain}' from: {adapter_path}")

    if not _loaded_adapters:
        # First adapter — use PeftModel.from_pretrained
        _model = PeftModel.from_pretrained(
            _model, adapter_path, adapter_name=domain
        )
    else:
        # Subsequent adapters — load into existing PeftModel
        _model.load_adapter(adapter_path, adapter_name=domain)

    _loaded_adapters.add(domain)
    print(f"Adapter '{domain}' loaded. Active adapters: {sorted(_loaded_adapters)}")


def get_model_and_tokenizer(domain: str):
    """Ensure base model + requested adapter are loaded; set active adapter."""
    with _load_lock:
        if _model is None:
            print("Loading base model...")
            _load_base()
            print("Base model ready.")

        _ensure_adapter(domain)
        _model.set_adapter(domain)
        _model.eval()

    return _model, _tokenizer


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def _build_prompt(domain: str, user_message: str) -> str:
    messages = [
        {"role": "system",    "content": DOMAIN_SYSTEMS[domain]},
        {"role": "user",      "content": user_message},
    ]
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_streaming(
    domain: str,
    user_message: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> Iterator[str]:
    """
    Yield partial response tokens as they are generated (streaming via TextIteratorStreamer).
    Gradio displays these incrementally.
    """
    if not user_message.strip():
        yield "Please enter a message."
        return

    try:
        model, tokenizer = get_model_and_tokenizer(domain)
    except Exception as e:
        yield f"Error loading model: {e}"
        return

    prompt  = _build_prompt(domain, user_message)
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Greedy for coding (reproducible), sampling for other domains
    do_sample = domain != "coding"

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=0.9 if do_sample else 1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial

    thread.join()


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

_CSS = """
.domain-btn { font-size: 0.9rem !important; }
.domain-info { font-size: 0.85rem; color: #6b7280; margin-bottom: 0.5rem; }
footer { display: none !important; }
"""

def _make_example_buttons(domain: str) -> list[str]:
    return DOMAIN_EXAMPLES.get(domain, [])


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="QLoRA Multi-Domain Demo", css=_CSS, theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            """
# 🧠 LLM Fine-Tuning with QLoRA — Multi-Domain Demo

**Mistral-7B-Instruct-v0.2** fine-tuned on 5 domains using QLoRA (4-bit quantization + LoRA adapters).
One base model loaded in memory; domain adapters hot-swap at runtime.

| Domain | Dataset | Eval Metric |
|--------|---------|-------------|
| 🤖 General | Dolly-15K | LLM-as-judge |
| 🏥 Medical | MedAlpaca MedQA | MCQ accuracy |
| ⚖️  Legal | CUAD legal contracts | LLM-as-judge |
| 📈 Finance | Finance-Alpaca | LLM-as-judge |
| 💻 Coding | CodeAlpaca-20K | HumanEval pass@1 |
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Select Domain")

                domain_state = gr.State("general")

                domain_btns = {}
                for d in DOMAINS:
                    domain_btns[d] = gr.Button(
                        DOMAIN_LABELS[d],
                        variant="primary" if d == "general" else "secondary",
                        elem_classes=["domain-btn"],
                    )

                domain_info = gr.Markdown(
                    DOMAIN_DESCRIPTIONS["general"],
                    elem_classes=["domain-info"],
                )

                gr.Markdown("### Settings")
                max_tokens_slider = gr.Slider(
                    minimum=64, maximum=1024, value=512, step=64,
                    label="Max new tokens",
                )
                temperature_slider = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.05,
                    label="Temperature (ignored for coding — greedy)",
                )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Response",
                    height=480,
                    show_copy_button=True,
                    type="messages",
                )
                user_input = gr.Textbox(
                    label="Your message",
                    placeholder="Ask anything in the selected domain...",
                    lines=3,
                    max_lines=8,
                )
                with gr.Row():
                    submit_btn = gr.Button("Generate", variant="primary")
                    clear_btn  = gr.Button("Clear")

                gr.Markdown("#### Example prompts")
                example_btns_row = gr.Row()
                with example_btns_row:
                    example_btns = [
                        gr.Button(f"Example {i+1}", size="sm", visible=False)
                        for i in range(3)
                    ]

        # ── Domain switching ────────────────────────────────────

        def switch_domain(new_domain: str):
            updates = {}
            for d in DOMAINS:
                updates[domain_btns[d]] = gr.update(
                    variant="primary" if d == new_domain else "secondary"
                )
            updates[domain_info]  = gr.update(value=DOMAIN_DESCRIPTIONS[new_domain])
            updates[domain_state] = new_domain

            examples = DOMAIN_EXAMPLES.get(new_domain, [])
            for i, btn in enumerate(example_btns):
                if i < len(examples):
                    # Shorten long examples for the button label
                    short = examples[i][:60].replace("\n", " ")
                    updates[btn] = gr.update(value=f"Ex {i+1}: {short}…", visible=True)
                else:
                    updates[btn] = gr.update(visible=False)

            return updates

        all_outputs = [domain_state, domain_info] + list(domain_btns.values()) + example_btns

        for d in DOMAINS:
            domain_btns[d].click(
                fn=lambda nd=d: switch_domain(nd),
                outputs=all_outputs,
            )

        # ── Initialize example buttons for the default domain ──
        demo.load(
            fn=lambda: switch_domain("general"),
            outputs=all_outputs,
        )

        # ── Example button click → fill input ──────────────────

        def fill_example(domain: str, idx: int) -> str:
            examples = DOMAIN_EXAMPLES.get(domain, [])
            return examples[idx] if idx < len(examples) else ""

        for i, btn in enumerate(example_btns):
            btn.click(
                fn=lambda d, i=i: fill_example(d, i),
                inputs=[domain_state],
                outputs=[user_input],
            )

        # ── Chat submission ────────────────────────────────────

        def chat(message: str, history: list, domain: str,
                 max_tok: int, temp: float):
            if not message.strip():
                return history, ""

            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": ""})

            partial_response = ""
            for token_chunk in generate_streaming(domain, message, max_tok, temp):
                partial_response = token_chunk
                history[-1]["content"] = partial_response
                yield history, ""

        submit_btn.click(
            fn=chat,
            inputs=[user_input, chatbot, domain_state, max_tokens_slider, temperature_slider],
            outputs=[chatbot, user_input],
        )
        user_input.submit(
            fn=chat,
            inputs=[user_input, chatbot, domain_state, max_tokens_slider, temperature_slider],
            outputs=[chatbot, user_input],
        )
        clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, user_input])

    return demo


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
