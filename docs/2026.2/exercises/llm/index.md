!!! success inline end "Deadline and Submission"

    :date: TBD
    
    :clock1: Commits until 23:59

    :material-account-group: Team (2–3 members)

    :simple-github: GitHub Pages link via [insper.blackboard.com](https://insper.blackboard.com){:target="_blank"}.

**Activity: Fine-Tuning a Large Language Model with LoRA**

In this activity you will fine-tune a pre-trained LLM on a custom task using **LoRA (Low-Rank Adaptation)**, one of the most widely used Parameter-Efficient Fine-Tuning (PEFT) techniques in the industry. You will use the Hugging Face ecosystem: `transformers`, `peft`, and `trl`.

---

## Learning Objectives

By the end of this activity you will be able to:

1. Load and inspect a pre-trained LLM and its tokenizer
2. Configure and apply LoRA adapters to specific attention modules
3. Fine-tune the model on a custom instruction dataset using `SFTTrainer`
4. Evaluate the fine-tuned model qualitatively and quantitatively
5. Understand the trade-off between trainable parameters, memory, and quality

---

## Prerequisites

Install the required packages:

```bash
pip install transformers peft trl datasets accelerate bitsandbytes
```

You will need access to a GPU (Google Colab Pro or similar). For a minimal experiment, `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters) or `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B) work on a T4 GPU.

---

## Exercise 1 — Model Inspection and Baseline

### Instructions

1. **Load a pre-trained model** and its tokenizer:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
```

2. **Count total parameters** and report:
   - Total parameters
   - Parameters per layer type (embedding, attention, FFN)
   - Memory usage (use `model.get_memory_footprint()`)

3. **Run a baseline inference** with 5 prompts related to your chosen task (e.g., sentiment analysis, Q&A, code generation). Record the raw model responses.

4. **Evaluate baseline** using an appropriate metric for your task (accuracy for classification, BLEU/ROUGE for generation). This establishes the baseline before fine-tuning.

---

## Exercise 2 — Dataset Preparation

### Instructions

1. **Choose or create a task dataset** with at least 500 instruction-following examples. Suggested sources:
   - Hugging Face Datasets (e.g., `financial_phrasebank`, `medical_questions_pairs`, `code_x_glue_ct_code_to_code_trans`)
   - Custom dataset relevant to your domain
   - Avoid: general chat datasets (too broad), datasets already in the model's training data

2. **Format in instruction-following format** (compatible with the model's chat template):

```python
def format_example(example):
    return {
        "text": tokenizer.apply_chat_template([
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]}
        ], tokenize=False, add_generation_prompt=False)
    }
```

3. **Split** 80% train / 10% validation / 10% test. Report:
   - Dataset size, label distribution (if classification)
   - Average input/output length in tokens
   - Example of a formatted training sample

---

## Exercise 3 — LoRA Configuration and Fine-Tuning

### Instructions

1. **Configure LoRA** using `peft`:

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,                         # rank — experiment with 4, 8, 16
    lora_alpha=16,               # scaling factor (typically 2×r)
    target_modules=["q_proj", "v_proj"],  # modules to adapt
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

2. **Fine-tune with SFTTrainer**:

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
    ),
)
trainer.train()
```

3. **Run experiments** with at least **two different LoRA configurations** and compare:
   - `r=4` vs `r=16`
   - `target_modules=["q_proj", "v_proj"]` vs `["q_proj", "k_proj", "v_proj", "o_proj"]`

4. **Plot training and validation loss curves** for each configuration.

### Required reporting

| Config | Rank r | Target modules | Trainable params | Train loss | Val loss | Test metric |
|--------|--------|---------------|-----------------|-----------|---------|------------|
| A | 4 | q, v | ? | ? | ? | ? |
| B | 16 | q, k, v, o | ? | ? | ? | ? |

---

## Exercise 4 — Evaluation and Analysis

### Instructions

1. **Quantitative evaluation** on the test set:
   - Classification: accuracy, F1
   - Generation: BLEU-4, ROUGE-L, or task-specific metric
   - Compare baseline vs. each LoRA configuration

2. **Qualitative evaluation**: run 10 test prompts through baseline and best fine-tuned model. Show 3 examples where fine-tuning clearly improved the output and 1–2 where it did not.

3. **Error analysis**: what types of inputs does the fine-tuned model still struggle with? Is it a data problem, prompt problem, or capacity problem?

4. **Ablation — rank analysis**: if time allows, test `r ∈ {2, 4, 8, 16, 32}` and plot test metric vs. trainable parameter count. At what rank does performance plateau?

---

## Exercise 5 — Reflection

Answer the following questions in your report (1 paragraph each):

1. How many parameters did LoRA actually train? What percentage of the full model? Why is this enough to adapt the model to your task?

2. What would full fine-tuning have required in terms of memory? Why is that prohibitive for most teams?

3. Did the fine-tuned model "forget" any general capabilities of the base model? Give evidence from your qualitative evaluation.

4. If you were deploying this fine-tuned model in production, what additional steps would you take before launching?

---

## Evaluation Criteria

!!! danger "Important Constraints"
    - Use **only open-source models** (no GPT-4 API, no Claude API). The model must be loadable from Hugging Face Hub.
    - Report **GPU hours used** and estimated cost (Google Colab compute units or equivalent).
    - All code must be reproducible: set random seeds, pin library versions.

| Criteria | Points |
|:---:|---|
| **2 pts** | Dataset selection, preparation, and formatting |
| **2 pts** | LoRA configuration and successful fine-tuning |
| **2 pts** | Quantitative evaluation and comparison of configurations |
| **2 pts** | Qualitative evaluation and error analysis |
| **2 pts** | Reflection questions and report quality |

**Submission format:** GitHub Pages report + link to training notebook (Google Colab or similar). Include all plots and tables.

**AI Collaboration:** Allowed, but you must understand every configuration parameter. The report must be your own analysis.
