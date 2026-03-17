# ================================================================
# src/detectors/perplexity.py
# STAGE 2C: Perplexity-Based Detectors
# Covers: GPT-2 Small/Medium/XL, GPT-Neo-125M, GPT-Neo-1.3B
# ================================================================

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, accuracy_score
from sklearn.calibration import calibration_curve
import time
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ──────────────────────────────────────────────────────
MAX_LENGTH  = 512
STRIDE      = 256
BATCH_SIZE  = 8
PPL_CLIP    = 1e4
NORM_METHOD = 'log_rank'

REFERENCE_MODELS = {
    'GPT2-Small'   : 'gpt2',
    'GPT2-Medium'  : 'gpt2-medium',
    'GPT2-XL'      : 'gpt2-xl',
    'GPT-Neo-125M' : 'EleutherAI/gpt-neo-125m',
    'GPT-Neo-1.3B' : 'EleutherAI/gpt-neo-1.3b',
}

NORM_METHODS = ['rank', 'log_rank', 'minmax', 'sigmoid']


# ── Label Encoder ───────────────────────────────────────────────
def encode_labels(labels):
    return (labels == 'llm').astype(int)


# ── Perplexity Calculator ───────────────────────────────────────
class PerplexityCalculator:
    """
    Calculates perplexity using a reference language model.
    Uses sliding window for texts longer than MAX_LENGTH to avoid
    silent truncation bias (LLM text tends to be longer).
    Clips extreme outlier perplexities to PPL_CLIP for rank stability.
    Lower perplexity = model finds text more predictable.
    """

    def __init__(self, model_name, device='cuda',
                 max_length=MAX_LENGTH, stride=STRIDE, ppl_clip=PPL_CLIP):
        print(f"Loading {model_name}...")
        self.model_name  = model_name
        self.device      = device
        self.max_length  = max_length
        self.stride      = stride
        self.ppl_clip    = ppl_clip

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✅ Loaded {model_name}")

    def calculate_perplexity(self, text):
        """
        Sliding window perplexity — handles texts longer than max_length.
        Without this, long LLM texts get silently truncated, biasing results.
        """
        encodings = self.tokenizer(text, return_tensors='pt', truncation=False)
        input_ids = encodings.input_ids.to(self.device)
        seq_len   = input_ids.shape[1]

        if seq_len == 0:
            return float('nan')

        # Short text — single forward pass
        if seq_len <= self.max_length:
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
            ppl = torch.exp(outputs.loss).item()
            return min(ppl, self.ppl_clip)

        # Long text — sliding window
        nlls       = []
        total_toks = 0
        for begin in range(0, seq_len, self.stride):
            end   = min(begin + self.max_length, seq_len)
            chunk = input_ids[:, begin:end]
            toks  = end - begin
            with torch.no_grad():
                outputs = self.model(chunk, labels=chunk)
            nlls.append(outputs.loss.item() * toks)
            total_toks += toks
            if end == seq_len:
                break

        avg_nll = sum(nlls) / total_toks
        ppl     = min(np.exp(avg_nll), self.ppl_clip)
        return ppl

    def calculate_batch_perplexity(self, texts, batch_size=BATCH_SIZE):
        """Calculate perplexity for a list of texts."""
        perplexities = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc=f"PPL ({self.model_name.split('/')[-1]})"):
            batch = texts[i:i + batch_size]
            for text in batch:
                try:
                    ppl = self.calculate_perplexity(str(text))
                    perplexities.append(ppl)
                except Exception as e:
                    print(f"  ⚠ Error on sample {i}: {e}")
                    perplexities.append(np.nan)
        return np.array(perplexities)

    def cleanup(self):
        """Free GPU memory before loading next model."""
        del self.model
        torch.cuda.empty_cache()
        print(f"  🧹 {self.model_name} unloaded from GPU")


# ── Perplexity → Detectability Conversion ──────────────────────
def perplexity_to_detectability(perplexities, labels, method='log_rank'):
    """
    Convert raw perplexity to [0,1] P(llm) detectability score.

    KEY INSIGHT (bug fix from v1):
      GPT2/GPT-Neo assign LOWER perplexity to LLM text (both are transformer LMs).
      So higher PPL → more human-like → LOWER P(llm).
      All methods invert the relationship: high PPL → low detectability score.

    Methods:
      rank     : rank-based, robust to outliers
      log_rank : rank on log(PPL), more robust to extreme spikes
      minmax   : linear rescaling, sensitive to outliers
      sigmoid  : sigmoid around median, smooth calibration
    """
    valid_mask   = ~np.isnan(perplexities)
    ppl_clean    = perplexities[valid_mask]
    labels_clean = labels[valid_mask]

    if method == 'rank':
        ranks = np.argsort(np.argsort(ppl_clean))
        detectability = 1.0 - (ranks / (len(ranks) - 1))

    elif method == 'log_rank':
        log_ppl = np.log(ppl_clean + 1e-8)
        ranks   = np.argsort(np.argsort(log_ppl))
        detectability = 1.0 - (ranks / (len(ranks) - 1))

    elif method == 'minmax':
        ppl_min = ppl_clean.min()
        ppl_max = ppl_clean.max()
        detectability = 1.0 - (ppl_clean - ppl_min) / (ppl_max - ppl_min + 1e-8)

    elif method == 'sigmoid':
        median = np.median(ppl_clean)
        scale  = np.std(ppl_clean) + 1e-8
        detectability = 1.0 / (1.0 + np.exp((ppl_clean - median) / scale))

    else:
        raise ValueError(f"Unknown method: {method}")

    detectability = np.clip(detectability, 0.0, 1.0)
    return detectability, labels_clean, valid_mask


def best_threshold_accuracy(y_true, scores):
    """Find optimal decision threshold via Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores    = tpr - fpr
    best_idx    = j_scores.argmax()
    best_thresh = float(thresholds[best_idx])
    preds = (scores >= best_thresh).astype(int)
    acc   = accuracy_score(y_true, preds)
    return best_thresh, acc