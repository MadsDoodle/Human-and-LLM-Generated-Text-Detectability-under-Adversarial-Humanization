# ================================================================
# src/utils/helpers.py
# Shared utilities: label encoding, sampling, checkpoint helpers,
# HF Hub push, README generation, checkpoint cleanup
# ================================================================

import os
import glob
import json
import torch
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, create_repo, snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEED = 42


# ================================================================
# LABEL ENCODING
# Defined once here — import into all detector modules instead
# of redefining across statistical.py, neural.py, cnn.py etc.
# ================================================================

def encode_labels(labels):
    """Binary encode label series: 'llm' → 1, 'human' → 0."""
    return (labels == 'llm').astype(int)


# ================================================================
# SAMPLING UTILITIES
# Defined once here — import into all LLM detector modules
# instead of redefining in each model's notebook cell.
# ================================================================

def sample_balanced(df, n, seed=SEED):
    """
    Sample n//2 human and n//2 llm rows, shuffle, reset index.
    Used by all LLM-as-detector notebooks for eval set construction.
    """
    h = df[df['label'] == 'human'].sample(n // 2, random_state=seed)
    l = df[df['label'] == 'llm'].sample(n // 2, random_state=seed)
    return pd.concat([h, l]).sample(frac=1, random_state=seed).reset_index(drop=True)


def get_pool(df, n=10, seed=SEED):
    """
    Sample n human and n llm rows for few-shot pool construction.
    Used by all LLM-as-detector notebooks.
    """
    h = df[df['label'] == 'human'].sample(n, random_state=seed)
    l = df[df['label'] == 'llm'].sample(n, random_state=seed)
    return pd.concat([h, l]).reset_index(drop=True)


# ================================================================
# CHECKPOINT UTILITIES
# find_best_checkpoint is shared by neural.py, generalisation.py,
# and the HF Hub push script below.
# ================================================================

def find_best_checkpoint(model_dir):
    """
    Priority 1: Read trainer_state.json → best_model_checkpoint by ROC-AUC.
    Priority 2: No trainer_state → take highest-numbered checkpoint.
    Priority 3: No checkpoint subdirs → return root dir (DeBERTa case —
                model saved directly to root with no checkpoint subdirs).
    """
    ckpts = glob.glob(os.path.join(model_dir, "checkpoint-*"))

    if not ckpts:
        # DeBERTa path — model saved directly in root, no checkpoints
        print(f"      No checkpoint subdirs — using root dir: {model_dir}")
        return model_dir

    # Try to find trainer_state.json and read best_model_checkpoint
    for ckpt in ckpts:
        state_file = os.path.join(ckpt, "trainer_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
            best = state.get("best_model_checkpoint")
            if best and os.path.exists(best):
                print(f"      ✅ Best checkpoint (by ROC-AUC): {best}")
                return best
            else:
                print(f"      ⚠️  trainer_state.json found but best_model_checkpoint "
                      f"missing or invalid: {best}")

    # Fallback — no valid trainer_state, use highest-numbered checkpoint
    latest = max(ckpts, key=lambda x: int(x.split("-")[-1]))
    print(f"      ⚠️  No trainer_state found — falling back to latest: {latest}")
    return latest


def clear_checkpoints(results_dir):
    """
    Delete all ckpt_*.json resume checkpoints from a results directory.
    Equivalent to the inline cleanup cells (Cells 20, 22, 25) that
    appeared before each LLM-as-detector scoring loop.
    """
    ckpts = glob.glob(os.path.join(results_dir, 'ckpt_*.json'))
    for f in ckpts:
        os.remove(f)
        print(f"  🗑  Removed: {os.path.basename(f)}")
    if not ckpts:
        print(f"  (no checkpoints found in {results_dir})")
    return len(ckpts)


# ================================================================
# HF HUB PUSH
# ================================================================

# Registry of all trained detector models to push.
# Each entry: (local_dir, repo_suffix, description)
MODEL_PUSH_REGISTRY = [
    ("./models/BERT_hc3",       "bert-detector-hc3",
     "BERT (bert-base-uncased) fine-tuned on HC3 for AI-text detection. "
     "Binary classifier: human (0) vs LLM-generated (1). "
     "Trained with 1 epoch, dropout=0.2, early stopping on ROC-AUC."),

    ("./models/BERT_eli5",      "bert-detector-eli5",
     "BERT (bert-base-uncased) fine-tuned on ELI5 for AI-text detection. "
     "Binary classifier: human (0) vs LLM-generated (1). "
     "Trained with 1 epoch, dropout=0.2, early stopping on ROC-AUC."),

    ("./models/RoBERTa_hc3",    "roberta-detector-hc3",
     "RoBERTa (roberta-base) fine-tuned on HC3 for AI-text detection. "
     "Binary classifier: human (0) vs LLM-generated (1). "
     "Trained with 1 epoch, dropout=0.2, early stopping on ROC-AUC."),

    ("./models/RoBERTa_eli5",   "roberta-detector-eli5",
     "RoBERTa (roberta-base) fine-tuned on ELI5 for AI-text detection. "
     "Binary classifier: human (0) vs LLM-generated (1). "
     "Trained with 1 epoch, dropout=0.2, early stopping on ROC-AUC."),

    ("./models/ELECTRA_hc3",    "electra-detector-hc3",
     "ELECTRA (google/electra-base-discriminator) fine-tuned on HC3 for AI-text detection. "
     "Binary classifier: human (0) vs LLM-generated (1). "
     "Trained with 1 epoch, dropout=0.2, early stopping on ROC-AUC."),

    ("./models/ELECTRA_eli5",   "electra-detector-eli5",
     "ELECTRA (google/electra-base-discriminator) fine-tuned on ELI5 for AI-text detection. "
     "Binary classifier: human (0) vs LLM-generated (1). "
     "Trained with 1 epoch, dropout=0.2, early stopping on ROC-AUC."),

    ("./models/DistilBERT_hc3", "distilbert-detector-hc3",
     "DistilBERT (distilbert-base-uncased) fine-tuned on HC3 for AI-text detection. "
     "Binary classifier: human (0) vs LLM-generated (1). "
     "Trained with 1 epoch, dropout=0.2, early stopping on ROC-AUC. "
     "Note: uses correct DistilBERT-specific dropout param names."),

    ("./models/DistilBERT_eli5","distilbert-detector-eli5",
     "DistilBERT (distilbert-base-uncased) fine-tuned on ELI5 for AI-text detection. "
     "Binary classifier: human (0) vs LLM-generated (1). "
     "Trained with 1 epoch, dropout=0.2, early stopping on ROC-AUC. "
     "Note: uses correct DistilBERT-specific dropout param names."),

    ("./models/DeBERTa_hc3",    "deberta-v3-detector-hc3",
     "DeBERTa-v3-base (microsoft/deberta-v3-base) fine-tuned on HC3 for AI-text detection. "
     "Binary classifier: human (0) vs LLM-generated (1). "
     "Trained in full fp32 (bf16/fp16 disabled due to disentangled attention gradient issues). "
     "No intermediate checkpointing — final in-memory weights used directly."),

    ("./models/DeBERTa_eli5",   "deberta-v3-detector-eli5",
     "DeBERTa-v3-base (microsoft/deberta-v3-base) fine-tuned on ELI5 for AI-text detection. "
     "Binary classifier: human (0) vs LLM-generated (1). "
     "Trained in full fp32 (bf16/fp16 disabled due to disentangled attention gradient issues). "
     "No intermediate checkpointing — final in-memory weights used directly."),
]


def make_readme(repo_id, local_dir, description, hf_user):
    """
    Generate a model card README for a pushed detector checkpoint.
    Called once per model in push_detectors_to_hub().
    """
    return f"""---
language: en
tags:
  - text-classification
  - ai-text-detection
  - pytorch
license: mit
---

# {repo_id}

## What is this?
This model was fine-tuned as part of a research project comparing transformer-based
AI-text detectors across two benchmark datasets: **HC3** and **ELI5**.

The task is binary classification:
- **Label 0** → Human-written text
- **Label 1** → LLM-generated text

## Model details
{description}

## Training setup
| Setting | Value |
|---|---|
| Epochs | 1 |
| Batch size (train) | 16 |
| Learning rate | 2e-5 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Dropout | 0.2 |
| Max seq length | 512 |
| Validation split | 10% |
| Best model metric | ROC-AUC |

## Datasets
- **HC3** — Human ChatGPT Comparison Corpus
- **ELI5** — Explain Like I'm 5 (Reddit QA dataset)
Cross-dataset evaluation (e.g. trained on HC3, tested on ELI5) was used to
measure generalisability of each detector.

## How to load
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("{hf_user}/{repo_id.split('/')[1]}")
tokenizer = AutoTokenizer.from_pretrained("{hf_user}/{repo_id.split('/')[1]}")

text = "Your input text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
prob_llm = torch.softmax(logits, dim=-1)[0][1].item()
print(f"P(LLM-generated): {{prob_llm:.4f}}")
```

## Notes
- Local training dir: `{local_dir}`
- All models in this series are private repos under `{hf_user}`.
- Part of a larger study — do not use for production content moderation without further evaluation.
"""


def push_detectors_to_hub(
    hf_user=None,
    hf_token=None,
    registry=None,
):
    """
    Push all trained detector checkpoints to HuggingFace Hub.

    Args:
        hf_user  : HF username. Defaults to HF_USER env var.
        hf_token : HF token. Defaults to HF_TOKEN env var.
        registry : list of (local_dir, repo_suffix, description) tuples.
                   Defaults to MODEL_PUSH_REGISTRY above.
    """
    hf_user  = hf_user  or os.getenv("HF_USER",  "Moodlerz")
    hf_token = hf_token or os.getenv("HF_TOKEN")
    registry = registry or MODEL_PUSH_REGISTRY

    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    api = HfApi()
    print(f"✅ Authenticated as: {hf_user}")

    push_summary = []

    for local_dir, repo_suffix, description in registry:
        repo_id = f"{hf_user}/{repo_suffix}"

        print(f"\n{'='*60}")
        print(f"Processing : {repo_id}")
        print(f"Local dir  : {local_dir}")

        if not os.path.exists(local_dir):
            print(f"⚠️  Skipping — local dir not found: {local_dir}")
            push_summary.append((repo_id, "SKIPPED — dir not found"))
            continue

        try:
            # ── Resolve best checkpoint ───────────────────────
            resolved_path = find_best_checkpoint(local_dir)
            print(f"📂 Loading from : {resolved_path}")

            # ── Create HF repo ────────────────────────────────
            create_repo(repo_id=repo_id, token=hf_token,
                        private=True, exist_ok=True)
            print(f"📁 Repo ready   : {repo_id}")

            # ── Write README into root local_dir ──────────────
            readme_path = os.path.join(local_dir, "README.md")
            with open(readme_path, "w") as f:
                f.write(make_readme(repo_id, local_dir, description, hf_user))
            print(f"📝 README written")

            # ── Load from resolved checkpoint path ────────────
            print(f"⏳ Loading model ...")
            model     = AutoModelForSequenceClassification.from_pretrained(
                            resolved_path)
            tokenizer = AutoTokenizer.from_pretrained(resolved_path)

            # ── Push model + tokenizer ────────────────────────
            print(f"⏳ Pushing model to hub ...")
            model.push_to_hub(repo_id, token=hf_token, private=True)

            print(f"⏳ Pushing tokenizer to hub ...")
            tokenizer.push_to_hub(repo_id, token=hf_token, private=True)

            # ── Push README ───────────────────────────────────
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                token=hf_token,
            )

            print(f"✅ Done: https://huggingface.co/{repo_id}")
            push_summary.append((repo_id, "✅ SUCCESS"))

            del model, tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            push_summary.append((repo_id, f"❌ FAILED — {str(e)}"))

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("PUSH SUMMARY")
    print("="*60)
    for repo_id, status in push_summary:
        print(f"  {status:30s}  {repo_id}")

    succeeded = sum(1 for _, s in push_summary if "SUCCESS" in s)
    skipped   = sum(1 for _, s in push_summary if "SKIPPED" in s)
    failed    = sum(1 for _, s in push_summary if "FAILED"  in s)

    print(f"\n  ✅ Succeeded : {succeeded}")
    print(f"  ⚠️  Skipped   : {skipped}")
    print(f"  ❌ Failed    : {failed}")
    print(f"\n🔗 All your models: https://huggingface.co/{hf_user}")

    return push_summary