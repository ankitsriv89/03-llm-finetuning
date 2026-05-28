# Domain Notes — Datasets, Quirks, and Formatting

Practical notes for each fine-tuning phase. Read before starting each notebook.

---

## Phase 1: General — Databricks Dolly-15K

**HuggingFace:** `databricks/databricks-dolly-15k`  
**License:** CC BY-SA 3.0 (commercial use allowed)  
**Size:** 15,000 samples  
**Use:** 5,000 for foundation run, 15,000 for full run  

### Sample Structure
```json
{
  "instruction": "Explain what a transformer is in machine learning.",
  "context": "",
  "response": "A transformer is a neural network architecture...",
  "category": "open_qa"
}
```

### Categories Distribution
| Category | % | Notes |
|----------|---|-------|
| open_qa | 25% | No context needed |
| closed_qa | 12% | Requires provided context |
| summarization | 11% | Long context → response |
| brainstorming | 19% | List generation |
| information_extraction | 11% | Extract facts from context |
| creative_writing | 11% | Open-ended generation |
| classification | 11% | Label/categorize text |

### Formatting Notes
- Include context when non-empty: `f"{instruction}\n\nContext: {context}"`
- Skip samples with empty responses (a few exist)
- Most samples are short — `max_seq_length=512` covers >95%

---

## Phase 2: Medical — MedAlpaca / MedQA

**Primary:** `medalpaca/medical_meadow_medqa`  
**License:** CC BY-NC 4.0 (non-commercial)  
**Size:** ~10,000 samples  
**Alternative:** `lavita/ChatDoctor-HealthCareMagic-100k` (patient Q&A, larger)

### Sample Structure
```json
{
  "input": "A 67-year-old woman presents with chest pain...\nA) Myocardial infarction\nB) Pulmonary embolism\nC) Aortic dissection\nD) Pericarditis",
  "output": "The correct answer is A) Myocardial infarction.\n\nThe clinical presentation..."
}
```

### Formatting Notes
- Samples are already MCQ format — just wrap in the chat template
- Responses include full explanations (not just the letter) — important for learning
- Increase max_seq_length to 1024 (some clinical descriptions are long)
- Add safety disclaimer to system prompt in demo

### Evaluation Metric
Extract the predicted letter (A/B/C/D) from the model's response and compare to ground truth. Accuracy = correct / total.

```python
import re
def extract_answer(text: str) -> str:
    match = re.search(r'\b([A-D])\)', text)
    return match.group(1) if match else "?"
```

### Common Issues
- Some samples have formatting inconsistencies — preprocess to normalize
- A few samples have 5 options (E) — filter these out or handle separately
- Don't train on the test split (MedQA has an official split)

---

## Phase 3: Legal — Contract Analysis

**Primary:** `nguyen-brat/legal_contracts` or `rceborg/legal-contracts`  
**Alternative (large):** `pile-of-law/pile-of-law` (subset `us_contracts`)  
**License:** Check per dataset — most are research-only  
**Size:** Use 5,000-10,000 samples  

### Sample Structure (legal_contracts)
```json
{
  "contract_text": "This Agreement is entered into as of...",
  "question": "What are the termination conditions?",
  "answer": "Either party may terminate this agreement by providing 30 days written notice..."
}
```

### Formatting Notes
- **Increase max_seq_length to 2048** — contracts are long
- Consider `packing=True` for efficiency (many short answers, pad waste otherwise)
- Format: `contract_text` goes in the context field, `question` is the instruction
- Truncate very long contracts to `max_seq_length - 200` tokens, keep the question

### Preprocessing for Long Contracts
```python
def format_legal_sample(sample, tokenizer, max_ctx_tokens=1800):
    # Truncate contract text to fit in context window
    ctx_tokens = tokenizer.encode(sample["contract_text"])[:max_ctx_tokens]
    ctx_text = tokenizer.decode(ctx_tokens)
    
    user_msg = f"{sample['question']}\n\nContract:\n{ctx_text}"
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": sample["answer"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
```

### Evaluation Metric
No standard MCQ — use LLM-as-judge:
- GPT-4o rates responses 1-5 on: accuracy, completeness, citation of specific clauses
- Average score across 50 held-out samples

---

## Phase 4: Finance — Finance Alpaca

**Primary:** `gbharti/finance-alpaca`  
**License:** CC BY 4.0  
**Size:** 68,634 samples — use 10,000  

### Sample Structure
```json
{
  "instruction": "What is the P/E ratio and how is it calculated?",
  "input": "",
  "output": "The Price-to-Earnings (P/E) ratio is a valuation metric calculated by dividing..."
}
```

### Formatting Notes
- Same structure as Dolly — use the same formatting function
- `input` field is often empty — handle like Dolly's `context`
- Shuffle before selecting 10K subset (dataset has topic clusters; shuffling ensures diversity)
- max_seq_length=512 is fine for most samples

### Finance-Specific Considerations
- Numerical reasoning: LLMs are weak at arithmetic. For calculation Q&A, train on step-by-step solutions
- Include disclaimer: "This is not financial advice" — add to system prompt in demo
- Watch out for date-specific information (stock prices, interest rates) — may become stale

### Alternative Datasets
- `sujet-ai/Sujet-Finance-Instruct-177k` — larger, higher quality
- `FinGPT/fingpt-sentiment-train` — financial sentiment specifically
- `TheFinAI/flare-ner` — financial named entity recognition

### Evaluation Metric
FinQA (`ibm/finqa`) has a test split with numerical Q&A. Extract numbers from responses and compare exact match. Harder but more meaningful than human eval.

---

## Phase 5: Coding — CodeAlpaca

**Primary:** `HuggingFaceH4/CodeAlpaca_20K`  
**License:** Apache 2.0  
**Size:** 20,111 samples  

### Sample Structure
```json
{
  "prompt": "Create a function to calculate the sum of a sequence of integers.\n[1, 2, 3, 4, 5]",
  "completion": "def sum_sequence(sequence):\n  sum = 0\n  for num in sequence:\n    sum += num\n  return sum"
}
```

### Formatting Notes
- Column names are `prompt`/`completion` (not `instruction`/`response`) — adjust formatter
- max_seq_length=1024 — code completions can be long
- Don't truncate mid-function — the model learns incomplete code if you do
- Preserve exact whitespace (indentation is semantically meaningful in Python)

```python
def format_code_sample(sample, tokenizer):
    messages = [
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["completion"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
```

### Evaluation Metric: pass@1

HumanEval (`openai/openai_humaneval`) is the standard benchmark. For each problem:
1. Generate one solution from your fine-tuned model
2. Run the provided unit tests
3. `pass@1` = fraction of problems where the generated code passes all tests

```python
# Simplified evaluation loop
from human_eval.data import read_problems
from human_eval.execution import check_correctness

problems = read_problems()
results = []
for task_id, problem in list(problems.items())[:30]:  # use 30 problems
    completion = generate(model, tokenizer, problem["prompt"])
    result = check_correctness(problem, completion, timeout=3.0)
    results.append(result["passed"])

print(f"pass@1: {sum(results)/len(results):.1%}")
```

**Baseline:** Mistral-7B-Instruct-v0.2 scores ~35-40% pass@1 on HumanEval without fine-tuning. Target: ≥40% after fine-tuning on CodeAlpaca.

### Common Issues
- Some CodeAlpaca samples have bugs in the expected output — not your model's fault
- Multi-language samples (JavaScript, Java, etc.) — either filter to Python-only or keep mixed
- Very long algorithmic problems may exceed max_seq_length — filter by token count before training

---

## Cross-Domain Comparison Checklist

After completing all phases, evaluate all adapters on these same 10 prompts to see domain specialization:

1. "Explain what a neural network is." (general)
2. "What are symptoms of type 2 diabetes?" (medical)
3. "What is an indemnification clause?" (legal)
4. "What does a high P/E ratio indicate?" (finance)
5. "Write a Python function to reverse a linked list." (coding)
6. "Summarize the importance of sleep." (general)
7. "What is the standard of care for hypertension?" (medical)
8. "Explain force majeure in contract law." (legal)
9. "What is EBITDA and why does it matter?" (finance)
10. "Implement binary search in Python." (coding)

Rate each response for: correctness, domain-appropriate language, depth.

The general adapter should handle everything adequately. Domain adapters should show improved vocabulary, reasoning style, and accuracy in their specific domain.
