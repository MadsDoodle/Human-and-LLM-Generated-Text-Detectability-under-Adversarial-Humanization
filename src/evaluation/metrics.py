# ================================================================
# src/evaluation/metrics.py
# Core evaluation metrics: AUROC, AUPRC, EER, Brier, FPR@95,
# bootstrap CIs, DeLong test, detector inference engine,
# confidence profile, ECE
# ================================================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, roc_curve,
)
from scipy import stats as scipy_stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42


# ================================================================
# INFERENCE ENGINE
# ================================================================

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts   = texts
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            str(self.texts[i]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
        }


def run_detector(texts, ckpt_path, base_model):
    """
    Load a trained sequence classifier from ckpt_path and score texts.
    Returns a numpy array of P(llm) scores.
    """
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    ckpt_path).to(device)
    model.eval()

    ds     = InferenceDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    scores = []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            scores.extend(
                torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())

    del model, tokenizer
    torch.cuda.empty_cache()
    return np.array(scores)


# ================================================================
# FIVE CORE METRICS
# ================================================================

def five_metrics(y_true, y_score):
    """
    Compute the 5 core metrics used throughout Stage 3:
    AUROC, AUPRC, EER, Brier Score, FPR@95TPR.
    """
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)

    # EER — point where FPR ≈ FNR (1 - TPR)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr     = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer     = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    # FPR @ 95% TPR — false positive rate when TPR = 0.95
    tpr_95_idx = np.argmin(np.abs(tpr - 0.95))
    fpr95      = float(fpr[tpr_95_idx])

    return {
        "auroc": auroc,
        "auprc": auprc,
        "eer":   eer,
        "brier": brier,
        "fpr95": fpr95,
    }


def bootstrap_five(y_true, y_score, n_boot=1000, seed=SEED):
    """
    Bootstrap 95% CIs for all five metrics.
    Returns dict of {metric: (lower, upper)}.
    """
    rng   = np.random.default_rng(seed)
    n     = len(y_true)
    keys  = ["auroc", "auprc", "eer", "brier", "fpr95"]
    boots = {k: [] for k in keys}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        ys  = y_score[idx]

        # Skip degenerate samples (only one class present)
        if len(np.unique(yt)) < 2:
            continue

        m = five_metrics(yt, ys)
        for k in keys:
            boots[k].append(m[k])

    ci = {}
    for k in keys:
        arr  = np.array(boots[k])
        ci[k] = (
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        )
    return ci


def delong_p(y_true, y_score_a, y_score_b):
    """
    DeLong test for comparing two AUROCs on the same test set.
    Returns: (auc_a, auc_b, z_stat, p_value)

    Based on: DeLong et al. (1988) — Comparing the Areas Under
    Two or More Correlated Receiver Operating Characteristic Curves.
    """
    def auc_and_structural_components(y_true, y_score):
        pos   = y_score[y_true == 1]
        neg   = y_score[y_true == 0]
        n_pos = len(pos)
        n_neg = len(neg)

        # Placement values
        v10 = np.array([
            (np.sum(p > neg) + 0.5 * np.sum(p == neg)) / n_neg
            for p in pos
        ])
        v01 = np.array([
            (np.sum(n < pos) + 0.5 * np.sum(n == pos)) / n_pos
            for n in neg
        ])

        auc = v10.mean()
        return auc, v10, v01, n_pos, n_neg

    auc_a, v10_a, v01_a, n_pos, n_neg = auc_and_structural_components(
        y_true, y_score_a)
    auc_b, v10_b, v01_b, _,     _     = auc_and_structural_components(
        y_true, y_score_b)

    # Variance components
    s10 = np.cov(v10_a, v10_b)   # 2×2 covariance matrix
    s01 = np.cov(v01_a, v01_b)

    var_a  = s10[0, 0] / n_pos + s01[0, 0] / n_neg
    var_b  = s10[1, 1] / n_pos + s01[1, 1] / n_neg
    cov_ab = s10[0, 1] / n_pos + s01[0, 1] / n_neg

    var_diff = var_a + var_b - 2 * cov_ab

    if var_diff <= 0:
        # Degenerate case — return p=1 (no detectable difference)
        return auc_a, auc_b, 0.0, 1.0

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))   # two-tailed

    return auc_a, auc_b, float(z), float(p)


# ================================================================
# CONFIDENCE PROFILE & ECE (Stage 3E)
# ================================================================

def confidence_profile(scores):
    """
    Compute a confidence profile for a detector's score distribution.
    Returns a dict of distributional statistics used in Stage 3E
    to detect confidence collapse across unseen LLMs.

    A healthy detector shows high separation between human and LLM
    score distributions with low within-class variance.
    Confidence collapse = scores converge toward 0.5 regardless of label.
    """
    scores = np.asarray(scores)

    # Distance from decision boundary — higher = more confident
    margin = np.abs(scores - 0.5)

    return {
        "mean":               float(scores.mean()),
        "std":                float(scores.std()),
        "median":             float(np.median(scores)),
        # Fraction of predictions in the uncertain zone [0.4, 0.6]
        "uncertain_frac":     float(((scores >= 0.4) & (scores <= 0.6)).mean()),
        # Fraction of predictions with high confidence (>0.8 or <0.2)
        "high_conf_frac":     float(((scores > 0.8) | (scores < 0.2)).mean()),
        # Mean margin from decision boundary
        "mean_margin":        float(margin.mean()),
        "std_margin":         float(margin.std()),
        # Percentile profile
        "p10":                float(np.percentile(scores, 10)),
        "p25":                float(np.percentile(scores, 25)),
        "p75":                float(np.percentile(scores, 75)),
        "p90":                float(np.percentile(scores, 90)),
    }


def ece_score(probs, labels, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Measures how well predicted probabilities match empirical frequencies.

    A perfectly calibrated detector has ECE = 0:
    among all predictions with P(llm) = p, exactly fraction p
    should actually be LLM-generated.

    Args:
        probs  : array of predicted P(llm) ∈ [0, 1]
        labels : array of ground-truth binary labels (1=llm, 0=human)
        n_bins : number of equal-width bins across [0, 1]

    Returns:
        ece    : float — weighted mean |accuracy - confidence| per bin
        bin_data: list of dicts with per-bin statistics for plotting
    """
    probs  = np.asarray(probs)
    labels = np.asarray(labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece       = 0.0
    n_total   = len(probs)
    bin_data  = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Bin membership — use half-open intervals [lo, hi)
        # except the last bin which includes hi=1.0
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        n_bin = mask.sum()
        if n_bin == 0:
            bin_data.append({
                "lo": lo, "hi": hi, "n": 0,
                "confidence": (lo + hi) / 2,
                "accuracy":   0.0,
                "gap":        0.0,
            })
            continue

        confidence = float(probs[mask].mean())
        accuracy   = float(labels[mask].mean())
        gap        = abs(accuracy - confidence)

        ece += (n_bin / n_total) * gap

        bin_data.append({
            "lo":         lo,
            "hi":         hi,
            "n":          int(n_bin),
            "confidence": confidence,
            "accuracy":   accuracy,
            "gap":        gap,
        })

    return float(ece), bin_data