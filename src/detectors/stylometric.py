# ================================================================
# src/detectors/stylometric.py
# NOTEBOOK 5: Stylometric and Statistical Hybrid Detector
# ================================================================
# Substantially extends Stage 2A feature set with:
#   - POS tag distribution (spaCy)
#   - Dependency tree depth (avg + max)
#   - Function word frequency profiles
#   - Punctuation entropy
#   - Per-sentence perplexity mean + variance (GPT-2 small)
#   - Readability indices (Flesch-Kincaid, Gunning Fog)
#
# Classifiers: Logistic Regression, Random Forest, XGBoost
# ================================================================

import os
import json
import pickle
import re
import math
from collections import Counter

import pandas as pd
import numpy as np
import torch
import spacy
import textstat
from scipy import stats as scipy_stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    accuracy_score, roc_curve, classification_report,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp = spacy.load("en_core_web_sm", disable=["ner"])  # keep tagger + parser

RESULTS_DIR = "./results/stylometric"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Sentence Perplexity Engine (GPT-2 Small) ───────────────────
ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
ppl_model     = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
ppl_model.eval()
ppl_tokenizer.pad_token = ppl_tokenizer.eos_token


def sentence_perplexity(sentence: str, max_len: int = 256) -> float:
    """Compute GPT-2 perplexity for a single sentence."""
    enc = ppl_tokenizer(sentence, return_tensors="pt",
                        max_length=max_len, truncation=True)
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return np.nan
    with torch.no_grad():
        loss = ppl_model(input_ids, labels=input_ids).loss
    return float(torch.exp(loss).item())


# ── Function Word Lists ─────────────────────────────────────────
FUNCTION_WORDS = set([
    "the","a","an","of","in","on","at","to","for",
    "with","by","from","as","is","are","was","were",
    "be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might",
    "shall","can","that","this","these","those","which",
    "and","but","or","nor","so","yet","both","either",
    "not","no","nor","rather","quite",
])

HEDGING_WORDS = [
    "maybe","perhaps","possibly","probably","might","could",
    "seem","appear","approximately","around","roughly","likely",
    "suggests","indicates","implies","arguably",
]
CERTAINTY_WORDS = [
    "definitely","certainly","obviously","clearly","always","never",
    "absolutely","undoubtedly","evidently","plainly",
]
CONNECTOR_WORDS = [
    "however","therefore","moreover","furthermore","additionally",
    "consequently","nevertheless","nonetheless","meanwhile","thus",
    "hence","whereby","thereby","accordingly","subsequently",
]
AI_HEDGE_PHRASES = [
    "it is worth noting","it is important to","as an ai","note that",
    "in summary","in conclusion","to summarize","first,","second,",
    "third,","finally,","overall,","in general,","generally speaking",
]


# ── Full Stylometric Feature Extractor ──────────────────────────
class FullStylometricExtractor:
    """
    Extended feature set combining Stage 2A features with
    POS, syntax, function words, readability, and sentence-level PPL.
    """

    def __init__(self, use_sentence_ppl: bool = True):
        self.use_sentence_ppl = use_sentence_ppl
        self.feature_names_   = []

    def extract_batch(self, texts):
        rows = []
        for text in tqdm(texts, desc="Extracting features", leave=False):
            rows.append(self._extract_single(str(text)))
        df = pd.DataFrame(rows)
        self.feature_names_ = df.columns.tolist()
        return df

    def _extract_single(self, text: str) -> dict:
        words     = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        f = {}

        # ─── STAGE 2A FEATURES (preserved) ──────────────────────
        f["word_count"]          = len(words)
        f["char_count"]          = len(text)
        f["sentence_count"]      = len(sentences)
        f["avg_word_len"]        = np.mean([len(w) for w in words]) if words else 0
        f["avg_sentence_len"]    = len(words) / len(sentences) if sentences else 0
        f["type_token_ratio"]    = len(set(words)) / len(words) if words else 0
        word_freq = Counter(words)
        f["hapax_ratio"]         = sum(1 for c in word_freq.values() if c==1) / len(words) if words else 0
        f["comma_density"]       = text.count(",") / len(words) if words else 0
        f["period_density"]      = text.count(".") / len(words) if words else 0
        f["question_mark_ratio"] = text.count("?") / len(sentences) if sentences else 0
        f["exclamation_ratio"]   = text.count("!") / len(sentences) if sentences else 0
        bigrams  = list(zip(words[:-1], words[1:]))
        trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
        f["bigram_repetition"]   = 1 - (len(set(bigrams))/len(bigrams))   if bigrams  else 0
        f["trigram_repetition"]  = 1 - (len(set(trigrams))/len(trigrams)) if trigrams else 0
        wprobs = np.array(list(word_freq.values())) / len(words)
        f["word_entropy"]        = scipy_stats.entropy(wprobs)
        sent_lens = [len(s.split()) for s in sentences]
        f["sentence_len_variance"] = np.var(sent_lens)  if sent_lens else 0
        f["sentence_len_std"]      = np.std(sent_lens)  if sent_lens else 0
        tl = text.lower()
        f["hedging_density"]    = sum(tl.count(w) for w in HEDGING_WORDS)    / len(words) if words else 0
        f["certainty_density"]  = sum(tl.count(w) for w in CERTAINTY_WORDS)  / len(words) if words else 0
        f["connector_density"]  = sum(tl.count(w) for w in CONNECTOR_WORDS)  / len(words) if words else 0
        contractions = ["n't","'ll","'re","'ve","'d","'m"]
        f["contraction_ratio"]  = sum(text.count(c) for c in contractions) / len(words) if words else 0

        # ─── NEW: AI PHRASE DENSITY ──────────────────────────────
        f["ai_phrase_density"]  = sum(tl.count(p) for p in AI_HEDGE_PHRASES) / max(len(sentences), 1)

        # ─── NEW: FUNCTION WORD FEATURES ─────────────────────────
        fw_count = sum(1 for w in words if w.lower() in FUNCTION_WORDS)
        f["function_word_ratio"] = fw_count / len(words) if words else 0
        top_fw = ["the","a","of","in","and","to","is","that","it","as"]
        for fw in top_fw:
            f[f"fw_{fw}"] = tl.count(f" {fw} ") / len(words) if words else 0

        # ─── NEW: PUNCTUATION ENTROPY ────────────────────────────
        punct_chars = [c for c in text if not c.isalnum() and not c.isspace()]
        if punct_chars:
            pc = Counter(punct_chars)
            pp = np.array(list(pc.values())) / len(punct_chars)
            f["punctuation_entropy"] = scipy_stats.entropy(pp)
        else:
            f["punctuation_entropy"] = 0.0

        # ─── NEW: READABILITY INDICES ────────────────────────────
        try:
            f["flesch_reading_ease"]   = textstat.flesch_reading_ease(text)
            f["flesch_kincaid_grade"]  = textstat.flesch_kincaid_grade(text)
            f["gunning_fog"]           = textstat.gunning_fog(text)
            f["smog_index"]            = textstat.smog_index(text)
            f["automated_readability"] = textstat.automated_readability_index(text)
            f["coleman_liau"]          = textstat.coleman_liau_index(text)
        except Exception:
            for k in ["flesch_reading_ease","flesch_kincaid_grade","gunning_fog",
                      "smog_index","automated_readability","coleman_liau"]:
                f[k] = np.nan

        # ─── NEW: POS TAG DISTRIBUTION (spaCy) ──────────────────
        doc  = nlp(text[:5000])   # cap for speed
        tags = [t.pos_ for t in doc]
        total_tags = len(tags) or 1
        for pos_tag in ["NOUN","VERB","ADJ","ADV","DET","ADP","PUNCT","PROPN","PRON","NUM"]:
            f[f"pos_{pos_tag.lower()}"] = tags.count(pos_tag) / total_tags

        # ─── NEW: DEPENDENCY TREE DEPTH ──────────────────────────
        depths = []
        for sent in doc.sents:
            root = [t for t in sent if t.head == t]
            if root:
                depths.append(self._tree_depth(root[0]))
        f["dep_depth_mean"] = float(np.mean(depths)) if depths else 0
        f["dep_depth_max"]  = float(np.max(depths))  if depths else 0

        # ─── NEW: SENTENCE-LEVEL PERPLEXITY (mean + variance) ────
        if self.use_sentence_ppl and sentences:
            ppls = []
            for sent in sentences[:15]:  # cap for speed
                p = sentence_perplexity(sent)
                if not np.isnan(p) and p < 1e6:
                    ppls.append(p)
            f["sent_ppl_mean"] = float(np.mean(ppls)) if ppls else np.nan
            f["sent_ppl_var"]  = float(np.var(ppls))  if ppls else np.nan
            f["sent_ppl_std"]  = float(np.std(ppls))  if ppls else np.nan
            f["sent_ppl_cv"]   = (f["sent_ppl_std"] / f["sent_ppl_mean"]
                                  if f["sent_ppl_mean"] and f["sent_ppl_mean"] > 0
                                  else np.nan)
        else:
            for k in ["sent_ppl_mean","sent_ppl_var","sent_ppl_std","sent_ppl_cv"]:
                f[k] = np.nan

        # ─── BURSTINESS ──────────────────────────────────────────
        f["burstiness"] = self._burstiness(words)

        return f

    def _tree_depth(self, token, depth=0):
        children = list(token.children)
        if not children:
            return depth
        return max(self._tree_depth(c, depth+1) for c in children)

    def _burstiness(self, words):
        if len(words) < 2:
            return 0
        pos_map = {}
        for i, w in enumerate(words):
            pos_map.setdefault(w, []).append(i)
        vars_ = []
        for positions in pos_map.values():
            if len(positions) > 1:
                vars_.append(np.var(np.diff(positions)))
        return float(np.mean(vars_)) if vars_ else 0


# ── Median Imputation ───────────────────────────────────────────
def impute_median(train_df, test_dfs):
    """Impute NaNs with column medians computed on train set."""
    medians  = train_df.median()
    train_df = train_df.fillna(medians)
    test_out = [df.fillna(medians) for df in test_dfs]
    return train_df, test_out, medians