# ================================================================
# src/detectors/contrastive.py
# NOTEBOOK 3: Contrastive Likelihood Detection
# ================================================================
# Replaces raw perplexity (Stage 2C) with:
#   S(x) = log P_small(x) - log P_large(x)
#
# Intuition (distribution-shift theory):
#   LLM text is optimised under a large model → anomalously smooth
#   to large model but appears rough to small model → measurable gap.
#   Human text does NOT exhibit this systematic gap.
#
# Extensions:
#   1. Base contrastive score (small vs large)
#   2. Multi-scale contrast (3 model sizes)
#   3. Layer-wise likelihood contrast (hidden-state log-probs at depth)
#   4. Token-level contrast variance (per-token score variability)
# ================================================================

import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "./results/contrastive"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model Scale Groups ──────────────────────────────────────────
# Three-scale ladder for multi-scale contrast.
# All are decoder-only causal LMs trained on similar corpora (OpenWebText / Pile).
SCALE_MODELS = {
    "small":  "gpt2",            # 117M
    "medium": "gpt2-medium",     # 345M
    "large":  "gpt2-xl",         # 1.5B  — use 4-bit on Colab T4/A100
}


# ── Label Encoder ───────────────────────────────────────────────
def encode_labels(series):
    return (series == "llm").astype(int)


# ── Token-Level Log-Prob Engine ─────────────────────────────────
class TokenLogProbCalculator:
    """
    Computes per-token log-probabilities under a causal LM.
    Returns both document-level (mean) and per-token arrays,
    enabling token-level variance analysis.
    """

    def __init__(self, model_name: str, use_fp16: bool = True):
        self.model_name = model_name
        print(f"  Loading {model_name} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if (use_fp16 and torch.cuda.is_available()) else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        print(f"  ✅ {model_name} ready")

    def get_token_log_probs(self, text: str, max_length: int = 512):
        """
        Returns:
            mean_log_prob   : float  — average log P(token | context)
            token_log_probs : ndarray — per-token log probabilities
            num_tokens      : int
        """
        enc = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.model.device)

        if input_ids.shape[1] < 2:
            return np.nan, np.array([]), 0

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits  = outputs.logits  # (1, T, V)

        # Shift: predict token t+1 from context up to t
        shift_logits = logits[:, :-1, :]           # (1, T-1, V)
        shift_labels = input_ids[:, 1:]            # (1, T-1)

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_lp  = log_probs[0, torch.arange(shift_labels.shape[1]), shift_labels[0]]
        token_lp_np = token_lp.cpu().float().numpy()

        return float(token_lp_np.mean()), token_lp_np, len(token_lp_np)

    def score_corpus(self, texts, batch_desc: str = ""):
        """Score a list of texts. Returns (mean_log_probs, token_lp_lists)."""
        mean_lps, token_lps_all = [], []
        for text in tqdm(texts, desc=f"  logP [{batch_desc}]", leave=False):
            m, t, _ = self.get_token_log_probs(str(text))
            mean_lps.append(m)
            token_lps_all.append(t)
        return np.array(mean_lps), token_lps_all

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()


# ── Contrastive Score Derivations ──────────────────────────────
def contrastive_score(small_lps, large_lps):
    """
    S(x) = log P_small(x) - log P_large(x)
    Positive → small model finds text *relatively* more probable than large.
    Human text: gap ≈ 0 or random.
    LLM text:   large model assigns much higher prob → S(x) is more negative.
    We negate so higher score = more likely AI.
    """
    score = -(small_lps - large_lps)   # negate so higher = more AI-like
    return score


def multi_scale_score(small_lps, medium_lps, large_lps):
    """
    Combine three pairwise contrasts:
      S_12 = small vs medium
      S_23 = medium vs large
      S_13 = small vs large
    Final = weighted average (equal weights here).
    """
    s12 = -(small_lps - medium_lps)
    s23 = -(medium_lps - large_lps)
    s13 = -(small_lps - large_lps)
    return (s12 + s23 + s13) / 3.0


def token_contrast_variance(small_token_lps, large_token_lps):
    """
    Per-document: variance of (log P_large(t) - log P_small(t)) across tokens.
    LLM text tends to have more uniform token-level likelihoods under the
    large model (smooth generation) → lower variance. Human text is noisier.
    Returns: negated variance so higher = more AI-like.
    """
    scores = []
    for s_tok, l_tok in zip(small_token_lps, large_token_lps):
        min_len = min(len(s_tok), len(l_tok))
        if min_len < 2:
            scores.append(np.nan)
            continue
        diff = l_tok[:min_len] - s_tok[:min_len]
        scores.append(float(np.var(diff)))
    # Negate so higher score = lower variance = more AI-like
    arr = np.array(scores)
    return -arr


def rankscale(arr):
    """Map raw scores → [0, 1] via rank normalisation."""
    valid = ~np.isnan(arr)
    out   = np.full_like(arr, fill_value=np.nan, dtype=float)
    ranked = np.argsort(np.argsort(arr[valid]))
    out[valid] = ranked / (ranked.max() + 1e-9)
    return out