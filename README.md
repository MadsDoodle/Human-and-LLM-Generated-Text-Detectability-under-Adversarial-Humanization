# Detecting the Machine
### A Comprehensive Benchmark of AI-Generated Text Detectors Across Architectures, Domains, and Adversarial Conditions

<p align="center">
  <img src="assets/model_arch.png" alt="Benchmark Pipeline" width="850"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.17522"><img src="https://img.shields.io/badge/arXiv-Preprint-red?logo=arxiv" alt="arXiv"/></a>
  <a href="https://huggingface.co/Moodlerz"><img src="https://img.shields.io/badge/🤗%20HuggingFace-Moodlerz-yellow" alt="HuggingFace"/></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-orange?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20A100%20%7C%20L4-76b900?logo=nvidia" alt="GPU"/>
</p>

---

## 📄 Abstract

The rapid proliferation of large language models has created an urgent need for robust, generalizable detectors of machine-generated text. Existing benchmarks evaluate a single detector family on a single dataset under ideal conditions, leaving critical questions about cross-domain transfer, cross-LLM generalization, and adversarial robustness unanswered.

This work presents a **comprehensive, multi-stage benchmark** evaluating **six detector families** across two corpora: **HC3** (Human–ChatGPT, multi-domain, 46,726 samples) and **ELI5** (Human–Mistral-7B, conversational, 30,000 samples).

**Central findings:**
- Fine-tuned transformers achieve near-perfect in-distribution AUROC (≥0.994) but degrade universally under domain shift
- An XGBoost stylometric hybrid matches transformer performance while remaining interpretable via SHAP
- LLM-as-detector prompting lags far behind fine-tuned approaches — best open-source result LLaMA-2-13B-Chat CoT at AUROC 0.898, GPT-4o-mini zero-shot at 0.909 on ELI5
- Perplexity-based detectors reveal a **critical polarity inversion** that, once corrected, yields AUROC ≈0.91
- No detector family generalizes robustly across LLM sources and domains simultaneously

---

## 👥 Authors

| Name | Affiliation | Contact |
|------|------------|---------|
| **Madhav S. Baidya** | Indian Institute of Technology (BHU), Varanasi, India | madhavsukla.baidya.chy22@itbhu.ac.in |
| **S. S. Baidya** | Indian Institute of Technology Guwahati, India | saurav.baidya@iitg.ac.in |
| **Chirag Chawla** | Indian Institute of Technology (BHU), Varanasi, India | chirag.chawla.chy22@itbhu.ac.in |

---

## 🗺️ Project Structure

```
detecting-the-machine/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md                  ← Instructions to download HC3 and ELI5
│
├── notebooks/
│   ├── 00_dataset_preparation.ipynb
│   ├── 01_statistical_detectors.ipynb
│   ├── 02_neural_detectors.ipynb
│   ├── 03_cnn_detector.ipynb
│   ├── 04_stylometric_detector.ipynb
│   ├── 05_llm_as_detector.ipynb
│   ├── 06_perplexity_detectors.ipynb
│   ├── 07_contrastive_likelihood.ipynb
│   ├── 08_cross_llm_generalisation.ipynb
│   └── 09_adversarial_humanization.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py              ← HC3/ELI5 loading and preprocessing
│   │   └── generator.py           ← Mistral-7B ELI5 answer generation
│   ├── detectors/
│   │   ├── statistical.py         ← LR, RF, SVM
│   │   ├── neural.py              ← BERT, RoBERTa, ELECTRA, DistilBERT, DeBERTa
│   │   ├── cnn.py                 ← 1D-CNN
│   │   ├── stylometric.py         ← Stylometric hybrid + SHAP
│   │   ├── perplexity.py          ← GPT-2/GPT-Neo perplexity detectors
│   │   ├── contrastive.py         ← Contrastive likelihood
│   │   └── llm_detector/
│   │       ├── prompts.py         ← All zero-shot, few-shot, CoT prompt builders
│   │       ├── scoring.py         ← Constrained decoding, prior calibration, ensemble
│   │       └── models.py          ← Per-model config and loading
│   ├── evaluation/
│   │   ├── metrics.py             ← AUROC, AUPRC, EER, Brier, FPR@95, bootstrap CIs
│   │   └── generalisation.py      ← Cross-LLM eval, embedding distance metrics
│   └── utils/
│       └── helpers.py             ← Sampling, label encoding, checkpoint utils
│
├── results/                       ← Gitignored, populated at runtime
│   └── .gitkeep
│
├── configs/
│   ├── neural_detectors.yaml
│   ├── llm_detectors.yaml
│   └── evaluation.yaml
│
└── assets/
    └── pipeline_figure.png
```

---

## 🖥️ Compute Environment

> **All experiments in this paper were run on Google Colab with NVIDIA A100 and L4 GPUs.**

| Stage | GPU Used | Notes |
|-------|----------|-------|
| Dataset preparation & ELI5 generation (Mistral-7B) | **NVIDIA A100 (40GB)** | Flash Attention 2, `torch.compile`, batch size 48 |
| Neural detector training (BERT, RoBERTa, ELECTRA, DistilBERT, DeBERTa) | **NVIDIA A100 (40GB)** | FP16 throughout; DeBERTa in full FP32 |
| 1D-CNN training | **NVIDIA L4 (24GB)** | < 5M parameters, fast convergence |
| Stylometric + perplexity detectors | **NVIDIA L4 (24GB)** | GPT-2 Small for sentence-level PPL |
| LLM-as-detector (7B–14B models) | **NVIDIA A100 (40GB)** | 4-bit NF4 quantization via BitsAndBytes |
| Cross-LLM generalization (corpus generation) | **NVIDIA A100 (40GB)** | One model at a time, checkpoint-resumable |
| Adversarial humanization | **NVIDIA L4 (24GB)** | Qwen2.5-1.5B as rewriter, 4-bit NF4 |

If you are running locally, an **A100 or equivalent (≥40GB VRAM)** is recommended for the LLM-as-detector and cross-LLM generalization stages. An **RTX 3090/4090 (24GB)** is sufficient for all other stages with minor batch size adjustments.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MadsDoodle/Human-and-LLM-Generated-Text-Detectability-under-Adversarial-Humanization.git
cd detecting-the-machine
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Then download the spaCy model (required for stylometric features):

```bash
python -m spacy download en_core_web_sm
```

### 4. Set Environment Variables

Create a `.env` file in the project root:

```bash
# HuggingFace token — required for gated models (LLaMA-2, LLaMA-3)
HF_TOKEN=hf_your_token_here
HF_USER=your_huggingface_username

# OpenAI API key — required only for GPT-4o-mini detector notebook
OPENAI_API_KEY=sk-your_key_here

# Optional: data directory override
DATA_DIR=./data
```

Then export them before running:

```bash
export $(cat .env | xargs)
```

### 5. Download the Datasets

See [`data/README.md`](data/README.md) for detailed instructions. In brief:

**HC3:**
```bash
# Automatic via HuggingFace datasets (Method 1)
python -c "from datasets import load_dataset; load_dataset('Hello-SimpleAI/HC3')"

# Or manual clone (Method 2 — used in this project)
git lfs install
git clone https://huggingface.co/datasets/Hello-SimpleAI/HC3
```

**ELI5:**
```bash
python -c "from datasets import load_dataset; load_dataset('sentence-transformers/eli5')"
```

> **Note:** The ELI5 LLM-generated answers (Mistral-7B) are **not** part of the original dataset. Run `notebooks/00_dataset_preparation.ipynb` to generate them, or download the pre-generated CSV from the HuggingFace Hub (link in `data/README.md`).

---

## 📓 Running the Notebooks

The notebooks are designed to be run **in order**. Each notebook saves intermediate outputs (CSVs, pickles) that subsequent notebooks depend on.

```
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
```

### Notebook 00 — Dataset Preparation

Loads HC3, generates ELI5 Mistral-7B answers, deduplicates, length-matches, and produces train/test splits.

**Outputs:** `hc3_train.csv`, `hc3_test.csv`, `eli5_train.csv`, `eli5_test.csv`

```bash
jupyter nbconvert --to notebook --execute notebooks/00_dataset_preparation.ipynb
```

### Notebook 01 — Statistical Detectors

Logistic Regression, Random Forest, SVM with RBF kernel on 22 hand-crafted linguistic features.

**Outputs:** `results/detector_family1_results.csv`

### Notebook 02 — Neural Transformer Detectors

Fine-tunes BERT, RoBERTa, ELECTRA, DistilBERT, and DeBERTa-v3-base. Each model is trained on both HC3 and ELI5 separately, then evaluated under all four cross-domain conditions.

> ⚠️ **DeBERTa-v3 note:** Must be trained in **full FP32** — BF16 silently zeros gradients, FP16 crashes the grad scaler. Checkpointing is disabled; final in-memory weights are used directly.

**Outputs:** `./models/BERT_hc3/`, `./models/RoBERTa_hc3/`, ... (10 model directories)

### Notebook 03 — 1D-CNN Detector

Shallow multi-filter 1D-CNN (<5M parameters). Includes degradation curve analysis under progressive token mixing.

**Outputs:** `results/cnn/`

### Notebook 04 — Stylometric Hybrid Detector

60+ feature stylometric pipeline with Logistic Regression, Random Forest, and XGBoost. Includes SHAP TreeExplainer feature attribution.

**Outputs:** `results/stylometric/`, SHAP plots

### Notebook 05 — LLM-as-Detector

Zero-shot, few-shot, and Chain-of-Thought detection using TinyLlama-1.1B-Chat-v1.0, Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct,  LLaMA-2-13B-Chat, Qwen2.5-14B-Instruct, and GPT-4o-mini.

> ⚠️ **Gated models** (LLaMA-2, LLaMA-3): Requires a HuggingFace token with access granted at [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf).

> ⚠️ **GPT-4o-mini**: Requires a valid `OPENAI_API_KEY`. Estimated cost at default settings (n=200 ZS/FS, n=50 CoT): **~$0.22**.

**Outputs:** `results/tinyllama/`, `results/qwen25_1p5b/`, `results/qwen25_7b_detector/`, `results/llama2_13b_detector/`, `results/qwen25_14b_detector/`, `results/llm_detector_gpt4omini/`

### Notebook 06 — Perplexity-Based Detectors

Unsupervised detection using GPT-2 Small/Medium/XL and GPT-Neo-125M/1.3B with sliding window perplexity and four normalization strategies.

**Outputs:** `detector_family3_perplexity_results.csv`

### Notebook 07 — Contrastive Likelihood Detection

`S(x) = log P_large(x) − log P_small(x)` with base, multi-scale, token variance, and hybrid scoring variants.

**Outputs:** `results/contrastive/`

### Notebook 08 — Cross-LLM Generalization

Stage 3 evaluation: trained neural detectors vs. five unseen LLMs, embedding-space generalization matrix (MiniLM + LR/SVM/RF), DeBERTa penultimate-layer distribution shift analysis (KL, Wasserstein-2, Fréchet), and confidence collapse profiling.

**Outputs:** `stage3_results/3A/`, `stage3_results/3B/`, `stage3_results/3D/`, `stage3_results/3E/`

### Notebook 09 — Adversarial Humanization

Three-level humanization (L0 → L1 → L2) using Qwen2.5-1.5B-Instruct as the rewriting model. All trained detectors are evaluated at each level.

**Outputs:** Adversarial AUROC / detection rate tables

---

## 📊 Key Results

### Stage 1: In-Distribution vs. Cross-Domain AUROC

| Detector | HC3→HC3 | HC3→ELI5 | ELI5→ELI5 | ELI5→HC3 |
|----------|---------|----------|-----------|----------|
| Logistic Regression | 0.888 | 0.741 | 0.845 | 0.743 |
| Random Forest | 0.977 | 0.783 | 0.962 | 0.634 |
| SVM (RBF) | 0.799 | 0.693 | 0.792 | 0.599 |
| BERT | 0.995 | 0.949 | 0.994 | 0.908 |
| RoBERTa | **0.999** | 0.974 | **1.000** | **0.966** |
| ELECTRA | 0.997 | 0.960 | 0.998 | 0.932 |
| DistilBERT | 0.997 | 0.958 | 0.998 | 0.931 |
| DeBERTa-v3 | 0.991 | 0.876 | 0.953 | 0.889 |
| 1D-CNN | 0.999 | 0.830 | 0.998 | 0.843 |
| Stylometric XGBoost | **0.9996** | 0.863 | 0.997 | 0.904 |

### Stage 1: LLM-as-Detector Summary

| Model | Regime | HC3 AUROC | ELI5 AUROC |
|-------|--------|-----------|------------|
| TinyLlama-1.1B-Chat-v1.0 | Zero-Shot | 0.565 | 0.507 |
| Qwen2.5-1.5B-Instruct | Zero-Shot | 0.522 | 0.521 |
| Llama-3.1-8B-Instruct | Zero-Shot | 0.730 | 0.751 |
| Qwen2.5-7B-Instruct | CoT | 0.639 | **0.781** |
| LLaMA-2-13B-Chat | CoT | **0.878** | **0.898** |
| Qwen2.5-14B-Instruct | CoT | 0.662 | 0.800 |
| GPT-4o-mini | Zero-Shot | 0.847 | **0.909** |

### Stage 1: Perplexity Detectors (Best per Model)

| Reference Model | HC3 AUROC | ELI5 AUROC |
|----------------|-----------|------------|
| GPT-2 Small | 0.910 | 0.907 |
| GPT-2 Medium | 0.905 | 0.928 |
| GPT-2 XL | 0.892 | 0.931 |
| GPT-Neo-125M | **0.917** | 0.897 |
| GPT-Neo-1.3B | 0.900 | 0.926 |

### Stage 3: Adversarial Humanization (RoBERTa-HC3)

| Level | HC3 AUROC | Detection Rate |
|-------|-----------|----------------|
| L0 (original AI) | 0.990 | 100.0% |
| L1 (light humanization) | 0.991 | 100.0% |
| L2 (heavy humanization) | **0.962** | 91.0% |

> Light humanization **increases** detectability — the rewriter superimposes additional model-specific patterns.

---

## 🏗️ Using the Source Modules

All `src/` modules can be imported directly for custom experiments:

```python
# Load and preprocess HC3
from src.data.loader import flatten_hc3_robust, deduplicate_hc3

# Train a statistical detector
from src.detectors.statistical import LinguisticFeatureExtractor, encode_labels
from sklearn.ensemble import RandomForestClassifier

extractor = LinguisticFeatureExtractor()
X_train   = extractor.extract_features(train_texts)
clf       = RandomForestClassifier(n_estimators=100, max_depth=10)
clf.fit(X_train, encode_labels(train_labels))

# Train a neural detector
from src.detectors.neural import train_and_evaluate_detector, DetectorConfig

results = train_and_evaluate_detector(
    model_name       = "BERT_HC3",
    model_checkpoint = "bert-base-uncased",
    train_data       = hc3_train,
    val_data         = hc3_val,
    test_data_dict   = {"hc3_to_hc3": hc3_test, "hc3_to_eli5": eli5_test},
    output_dir       = "./models/BERT_hc3",
)

# Evaluate with the five-metric suite + bootstrap CIs
from src.evaluation.metrics import five_metrics, bootstrap_five

m  = five_metrics(y_true, y_score)
ci = bootstrap_five(y_true, y_score, n_boot=1000)
print(f"AUROC: {m['auroc']:.4f} [{ci['auroc'][0]:.3f}, {ci['auroc'][1]:.3f}]")

# Run perplexity-based detection
from src.detectors.perplexity import PerplexityCalculator, perplexity_to_detectability

calc  = PerplexityCalculator("gpt2", device="cuda")
ppls  = calc.calculate_batch_perplexity(texts)
scores, labels_clean, _ = perplexity_to_detectability(ppls, y_true, method="log_rank")
calc.cleanup()

# LLM-as-detector (Qwen2.5-7B example)
from src.detectors.llm_detector.models import load_causal_model_4bit, get_label_token_ids
from src.detectors.llm_detector.prompts import qwen_zero_shot
from src.detectors.llm_detector.scoring import constrained_score, compute_task_prior

model, tokenizer = load_causal_model_4bit("Qwen/Qwen2.5-7B-Instruct")
yes_ids, no_ids  = get_label_token_ids(tokenizer)
yes_prior, no_prior = compute_task_prior(
    model, tokenizer, eval_df, yes_ids, no_ids,
    prompt_fn=qwen_zero_shot
)
score = constrained_score(
    model, tokenizer, qwen_zero_shot(text, tokenizer),
    yes_ids, no_ids, yes_prior, no_prior, flip=False
)
```

---

## 🔬 Detector Architecture Details

### Statistical / Classical

22 hand-crafted features across 7 categories: surface statistics, lexical diversity, punctuation, repetition metrics, entropy measures, syntactic complexity, and discourse markers. Classifiers: Logistic Regression, Random Forest (100 trees, depth 10), SVM with RBF kernel (Platt-scaled).

### Fine-Tuned Encoder Transformers

| Model | Params | Precision | Batch | Warmup | Notes |
|-------|--------|-----------|-------|--------|-------|
| BERT | 110M | FP16 | 32/64 | 6% ratio | Standard MLM |
| RoBERTa | 125M | FP16 | 32/64 | 6% ratio | Dynamic masking, no NSP |
| ELECTRA | 110M | FP16 | 32/64 | 6% ratio | Replaced-token detection |
| DistilBERT | 66M | FP16 | 32/64 | 6% ratio | Knowledge-distilled BERT |
| DeBERTa-v3 | 184M | **FP32** | 16/32 | 500 steps | Disentangled attention; no checkpointing |

Shared: AdamW (lr=2e-5, wd=0.01), 1 epoch, dropout=0.2, max_seq_len=512, 10% val split.

### Shallow 1D-CNN

Embedding (vocab=30k, dim=128) → 4 parallel Conv1D branches (kernel sizes {2,3,4,5}, 128 filters each, BatchNorm+ReLU) → Global Max Pool → Dense (512→256→1). Total params: <5M. Trained with Adam (lr=1e-3), early stopping (patience=3), max 10 epochs.

### Stylometric Hybrid

60+ features extending Family 1 with: POS tag distribution (spaCy), dependency tree depth, function word profiles, punctuation entropy, AI hedge phrase density, 6 readability indices (Flesch, Gunning Fog, etc.), sentence-level GPT-2 perplexity statistics (mean, variance, CV). Classifiers: LR, RF (300 trees), XGBoost (400 estimators, lr=0.05). SHAP TreeExplainer for attribution.

### LLM-as-Detector

Constrained next-token logit decoding at the `Answer:` position. Key pipeline components:
- **Polarity correction**: Qwen/LLaMA-2 use swapped prompts (yes=human, no=AI) due to unconditional no-bias
- **Task prior calibration**: Subtract averaged yes/no logits over 50 real task prompts
- **CoT ensemble**: 0.6×conf + 0.4×logit, with a per-model dead zone where only the logit is used

### Perplexity-Based (Unsupervised)

5 reference models (GPT-2 S/M/XL, GPT-Neo-125M/1.3B). Sliding window (512 tokens, stride 256). Outlier clip at 10,000 PPL. Four normalization methods evaluated (rank, log-rank, minmax, sigmoid); best selected per condition by AUROC.

---

## 📏 Evaluation Protocol

All detectors output a continuous score in [0, 1] representing P(LLM-generated). The five-metric evaluation suite:

| Metric | Description |
|--------|-------------|
| **AUROC** | Area under the ROC curve — primary metric |
| **AUPRC** | Area under the precision-recall curve |
| **EER** | Equal Error Rate — threshold-free |
| **Brier Score** | Mean squared probability error |
| **FPR@95%TPR** | False positive rate at 95% recall |

Bootstrap 95% confidence intervals (1,000 iterations) are reported for all five metrics. DeLong paired AUROC tests are used for pairwise detector comparisons.

Four evaluation conditions are reported for supervised families:

```
HC3 → HC3    (in-distribution)
HC3 → ELI5   (cross-domain)
ELI5 → ELI5  (in-distribution)
ELI5 → HC3   (cross-domain)
```

---

## 🤗 Pre-Trained Models

All fine-tuned transformer detectors are available as private repositories on HuggingFace Hub under [Moodlerz](https://huggingface.co/Moodlerz):

| Repo | Base Model | Trained On |
|------|-----------|-----------|
| `Moodlerz/bert-detector-hc3` | bert-base-uncased | HC3 |
| `Moodlerz/bert-detector-eli5` | bert-base-uncased | ELI5 |
| `Moodlerz/roberta-detector-hc3` | roberta-base | HC3 |
| `Moodlerz/roberta-detector-eli5` | roberta-base | ELI5 |
| `Moodlerz/electra-detector-hc3` | electra-base-discriminator | HC3 |
| `Moodlerz/electra-detector-eli5` | electra-base-discriminator | ELI5 |
| `Moodlerz/distilbert-detector-hc3` | distilbert-base-uncased | HC3 |
| `Moodlerz/distilbert-detector-eli5` | distilbert-base-uncased | ELI5 |
| `Moodlerz/deberta-v3-detector-hc3` | deberta-v3-base | HC3 |
| `Moodlerz/deberta-v3-detector-eli5` | deberta-v3-base | ELI5 |

**Loading a pre-trained detector:**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id  = "Moodlerz/roberta-detector-hc3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

text   = "Your input text here."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    logits = model(**inputs).logits
prob_llm = torch.softmax(logits, dim=-1)[0][1].item()
print(f"P(LLM-generated): {prob_llm:.4f}")
```

---

## ⚙️ Configuration Files

Key hyperparameters are stored in `configs/`:

**`configs/neural_detectors.yaml`** — training hyperparameters for transformer family  
**`configs/llm_detectors.yaml`** — model IDs, prior N, dead zones, ensemble weights  
**`configs/evaluation.yaml`** — eval conditions, bootstrap iterations

---

## 💡 Key Implementation Notes

### DeBERTa-v3 Precision Bug
DeBERTa-v3's disentangled attention produces small gradient magnitudes that **underflow in BF16** (7-bit mantissa) and crash the FP16 grad scaler. Always use full FP32:
```python
model = AutoModelForSequenceClassification.from_pretrained(...).float()
# TrainingArguments: fp16=False, bf16=False
```

### Perplexity Polarity Inversion
GPT-2/GPT-Neo assign **lower** perplexity to LLM-generated text (not higher). A naive detector will flag human text as AI-generated. The fix — rank-inversion of the raw PPL signal — is implemented in `src/detectors/perplexity.py::perplexity_to_detectability()`.

### LLM-as-Detector Prior Calibration
Without task prior subtraction, RLHF-aligned models collapse all outputs to near-uniform scores. The prior must be computed from the **exact same prompt template** used at inference time. See `src/detectors/llm_detector/scoring.py::compute_task_prior()`.

### Qwen2.5-14B-Instruct Generation Fix
Using `eos_token_id` as `pad_token_id` in `model.generate()` causes **premature termination** and a ~90% unknown verdict rate in CoT. Always set `pad_token_id=tokenizer.pad_token_id` explicitly.

### Length Matching
Without the ±20% word-count length matching step, classical detectors trivially exploit the length disparity between human and LLM answers. This step is applied in `notebooks/00_dataset_preparation.ipynb` before train/test splitting.

---

## 🔧 Troubleshooting

**`OutOfMemoryError` on LLM-as-detector notebooks:**
Reduce batch size or switch to a smaller model. All 7B+ models require 4-bit NF4 quantization which is enabled by default.

**`flash_attn` install fails:**
Flash Attention 2 is optional and only used for Mistral-7B generation. You can comment out `!pip install flash-attn` in `notebooks/00_dataset_preparation.ipynb` — the generation will still run (slightly slower).

**DeBERTa AUROC collapses to ~0.5 after loading a checkpoint:**
This is a known HuggingFace issue with DeBERTa-v3's LayerNorm key naming. The fix is `save_strategy="no"` during training, which is already implemented. Never save and reload intermediate DeBERTa checkpoints.

**LLaMA-2/LLaMA-3 gated model access denied:**
Request access at [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) and ensure your `HF_TOKEN` environment variable is set.

**Spacy `en_core_web_sm` not found:**
```bash
python -m spacy download en_core_web_sm
```

---

## 📚 Citation

If you use this codebase or findings in your research, please cite:

```bibtex
@article{baidya2024detectingmachine,
  title   = {Detecting the Machine: A Comprehensive Benchmark of AI-Generated Text
             Detectors Across Architectures, Domains, and Adversarial Conditions},
  author  = {Baidya, Madhav S. and Baidya, S. S. and Chawla, Chirag},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.17522}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

The HC3 dataset is released under its original license by [Hello-SimpleAI](https://huggingface.co/Hello-SimpleAI/HC3). The ELI5 dataset is released by [Fan et al. (2019)](https://aclanthology.org/P19-1346/). All pre-trained base models are subject to their respective licenses on HuggingFace Hub.

---

## 🙏 Acknowledgements

The authors thank the **Indian Institute of Technology (BHU), Varanasi** and **IIT Guwahati** for computational resources and institutional support.

We also acknowledge:
- The maintainers of the **HC3** corpus ([Guo et al., 2023](https://arxiv.org/abs/2301.07597)) and the **ELI5** dataset ([Fan et al., 2019](https://aclanthology.org/P19-1346/))
- The **HuggingFace** open-source ecosystem and all model contributors
- The developers of **TinyLlama**, **Qwen2.5**, **LLaMA-2/3**, and **Mistral-7B**
- **Google Colab** for providing access to A100 and L4 GPU instances used throughout this work

---

## 📬 Contact

For questions, issues, or collaboration inquiries:

- **Madhav S. Baidya** — madhavsukla.baidya.chy22@itbhu.ac.in
- **S. S. Baidya** — saurav.baidya@iitg.ac.in
- **Chirag Chawla** — chirag.chawla.chy22@itbhu.ac.in

Please open a [GitHub Issue](https://github.com/MadsDoodle/Human-and-LLM-Generated-Text-Detectability-under-Adversarial-Humanization.git) for bugs or feature requests.