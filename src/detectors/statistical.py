# ================================================================
# src/detectors/statistical.py
# STAGE 2A: Statistical / Classical Detectors
# Covers: Logistic Regression, Random Forest, SVM
# ================================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    brier_score_loss, log_loss, classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import scipy.stats as stats
from tqdm import tqdm


# ── Label Encoder ───────────────────────────────────────────────
def encode_labels(labels):
    return (labels == 'llm').astype(int)


# ── Linguistic Feature Extractor ────────────────────────────────
class LinguisticFeatureExtractor:
    """
    Extract interpretable linguistic features for detectability analysis.
    """

    def __init__(self):
        self.feature_names = []

    def extract_features(self, texts):
        """Extract features from list of texts."""
        features = []
        for text in tqdm(texts, desc="Extracting features"):
            features.append(self._extract_single(text))
        df = pd.DataFrame(features)
        self.feature_names = df.columns.tolist()
        return df

    def _extract_single(self, text):
        """Extract features from a single text."""
        words     = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        features = {}

        # ===== SURFACE STATISTICS =====
        features['word_count']       = len(words)
        features['char_count']       = len(text)
        features['sentence_count']   = len(sentences)
        features['avg_word_len']     = np.mean([len(w) for w in words]) if words else 0
        features['avg_sentence_len'] = len(words) / len(sentences) if sentences else 0

        # ===== LEXICAL DIVERSITY =====
        features['type_token_ratio'] = len(set(words)) / len(words) if words else 0
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0

        word_freq = Counter(words)
        features['hapax_ratio'] = sum(1 for c in word_freq.values() if c == 1) / len(words) if words else 0

        # ===== PUNCTUATION & FORMATTING =====
        features['comma_density']       = text.count(',') / len(words) if words else 0
        features['period_density']      = text.count('.') / len(words) if words else 0
        features['question_mark_ratio'] = text.count('?') / len(sentences) if sentences else 0
        features['exclamation_ratio']   = text.count('!') / len(sentences) if sentences else 0

        # ===== REPETITION METRICS =====
        bigrams  = list(zip(words[:-1], words[1:]))
        features['bigram_repetition']  = 1 - (len(set(bigrams)) / len(bigrams))   if bigrams  else 0
        trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
        features['trigram_repetition'] = 1 - (len(set(trigrams)) / len(trigrams)) if trigrams else 0

        # ===== ENTROPY MEASURES =====
        word_probs = np.array(list(word_freq.values())) / len(words)
        features['word_entropy'] = stats.entropy(word_probs)

        sent_lens = [len(s.split()) for s in sentences]
        if len(set(sent_lens)) > 1:
            sent_len_counts = Counter(sent_lens)
            sent_len_probs  = np.array(list(sent_len_counts.values())) / len(sent_lens)
            features['sentence_len_entropy'] = stats.entropy(sent_len_probs)
        else:
            features['sentence_len_entropy'] = 0

        # ===== SYNTACTIC COMPLEXITY =====
        features['sentence_len_variance'] = np.var(sent_lens)  if sent_lens else 0
        features['sentence_len_std']      = np.std(sent_lens)  if sent_lens else 0

        # ===== DISCOURSE MARKERS =====
        hedging_words   = ['maybe','perhaps','possibly','probably','might','could','seem','appear']
        features['hedging_density']   = sum(text.lower().count(w) for w in hedging_words)   / len(words) if words else 0

        certainty_words = ['definitely','certainly','obviously','clearly','always','never']
        features['certainty_density'] = sum(text.lower().count(w) for w in certainty_words) / len(words) if words else 0

        connector_words = ['however','therefore','moreover','furthermore','additionally','consequently']
        features['connector_density'] = sum(text.lower().count(w) for w in connector_words) / len(words) if words else 0

        # ===== FORMALITY MARKERS =====
        contractions = ["n't","'ll","'re","'ve","'d","'m"]
        features['contraction_ratio'] = sum(text.count(c) for c in contractions) / len(words) if words else 0

        # ===== BURSTINESS =====
        features['burstiness'] = self._calculate_burstiness(words)

        return features

    def _calculate_burstiness(self, words):
        """Calculate burstiness metric (tendency for words to appear in bursts)."""
        if len(words) < 2:
            return 0
        word_positions = {}
        for i, word in enumerate(words):
            word_positions.setdefault(word, []).append(i)
        gap_vars = []
        for positions in word_positions.values():
            if len(positions) > 1:
                gap_vars.append(np.var(np.diff(positions)))
        return np.mean(gap_vars) if gap_vars else 0