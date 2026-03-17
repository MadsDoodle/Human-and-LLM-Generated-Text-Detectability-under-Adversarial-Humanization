# ================================================================
# src/detectors/cnn.py
# NOTEBOOK 4: 1D-CNN Shallow Neural Detector
# ================================================================
# Architecture: Embedding → 1D Conv (multi-filter) → Global Max Pool
#               → Dense → Sigmoid
# < 5M parameters — bridges handcrafted features (Stage 2A) and
# full transformer fine-tuning (Stage 2B).
# ================================================================

import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    accuracy_score, roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "./results/cnn"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Config ─────────────────────────────────────────────────────
class CNNConfig:
    MAX_VOCAB_SIZE  = 30_000
    MAX_SEQ_LENGTH  = 256
    MIN_WORD_FREQ   = 2
    EMBED_DIM       = 128
    FILTER_SIZES    = [2, 3, 4, 5]
    NUM_FILTERS     = 128
    DROPOUT         = 0.4
    HIDDEN_DIM      = 256
    BATCH_SIZE      = 64
    EPOCHS          = 10
    LR              = 1e-3
    WEIGHT_DECAY    = 1e-4
    PATIENCE        = 3
    VAL_SPLIT       = 0.1
    SEED            = 42

    @property
    def total_filter_dim(self):
        return len(self.FILTER_SIZES) * self.NUM_FILTERS

cfg = CNNConfig()


# ── Vocabulary Builder ──────────────────────────────────────────
class Vocabulary:
    PAD = "<PAD>"; UNK = "<UNK>"

    def __init__(self, max_size=30_000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(str(text).lower().split())
        vocab = [w for w, f in counter.most_common(self.max_size)
                 if f >= self.min_freq]
        special = [self.PAD, self.UNK]
        all_tokens = special + vocab
        self.word2idx = {w: i for i, w in enumerate(all_tokens)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        print(f"  Vocabulary size: {len(self.word2idx):,}")
        return self

    def encode(self, text, max_length):
        tokens = str(text).lower().split()[:max_length]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        ids = ids + [0] * (max_length - len(ids))
        return ids


# ── Dataset Class ───────────────────────────────────────────────
class CNNTextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.encodings = [vocab.encode(t, max_length) for t in texts]
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],    dtype=torch.float),
        )


# ── Model Architecture ──────────────────────────────────────────
class MultiFilterCNN(nn.Module):
    """
    Embedding → N parallel 1D Conv layers (different kernel sizes)
    → Global Max Pool per conv → Concatenate → Dropout → Dense → Sigmoid
    Parameter count target: < 5M
    """

    def __init__(self, vocab_size, embed_dim, filter_sizes,
                 num_filters, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels  = embed_dim,
                    out_channels = num_filters,
                    kernel_size  = ks,
                    padding      = ks // 2,
                ),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            )
            for ks in filter_sizes
        ])
        total_filters = len(filter_sizes) * num_filters
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_filters, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.permute(0, 2, 1)
        pooled = []
        for conv in self.convs:
            out  = conv(emb)
            pool = F.max_pool1d(out, out.size(2)).squeeze(2)
            pooled.append(pool)
        cat = torch.cat(pooled, dim=1)
        logit = self.classifier(cat)
        return logit.squeeze(1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Label Encoder ───────────────────────────────────────────────
def encode_labels(series):
    return (series == "llm").astype(int).tolist()


# ── Training Loop ───────────────────────────────────────────────
def train_cnn(train_df, val_df, test_data_dict, tag=""):
    """
    Full training loop with early stopping.
    Returns: trained model, tokenizer vocab, results dict, history.
    """
    print(f"\n{'='*60}")
    print(f"Training 1D-CNN  [{tag}]")
    print(f"{'='*60}")

    vocab = Vocabulary(cfg.MAX_VOCAB_SIZE, cfg.MIN_WORD_FREQ)
    vocab.build(train_df["text"].tolist())

    tr_ds  = CNNTextDataset(train_df["text"], encode_labels(train_df["label"]), vocab, cfg.MAX_SEQ_LENGTH)
    val_ds = CNNTextDataset(val_df["text"],   encode_labels(val_df["label"]),   vocab, cfg.MAX_SEQ_LENGTH)

    tr_loader  = DataLoader(tr_ds,  batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    model = MultiFilterCNN(
        vocab_size   = len(vocab.word2idx),
        embed_dim    = cfg.EMBED_DIM,
        filter_sizes = cfg.FILTER_SIZES,
        num_filters  = cfg.NUM_FILTERS,
        hidden_dim   = cfg.HIDDEN_DIM,
        dropout      = cfg.DROPOUT,
    ).to(device)

    n_params = model.count_parameters()
    print(f"  Parameters : {n_params:,}  ({'<5M ✅' if n_params < 5_000_000 else '>5M ⚠️'})")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)

    best_val_auc = 0.0
    best_state   = None
    patience_ctr = 0
    history      = []

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(tr_loader, desc=f"Epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                probs  = torch.sigmoid(logits).cpu().numpy()
                val_scores.extend(probs)
                val_labels.extend(y.numpy())

        val_auc = roc_auc_score(val_labels, val_scores)
        scheduler.step(val_auc)

        avg_loss = total_loss / len(tr_loader)
        print(f"  Epoch {epoch:2d} | Loss={avg_loss:.4f} | Val AUC={val_auc:.4f}")
        history.append({"epoch": epoch, "loss": avg_loss, "val_auc": val_auc})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    results = {}
    for test_name, test_df in test_data_dict.items():
        te_ds = CNNTextDataset(test_df["text"], encode_labels(test_df["label"]),
                               vocab, cfg.MAX_SEQ_LENGTH)
        te_loader = DataLoader(te_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

        scores, labels = [], []
        with torch.no_grad():
            for x, y in te_loader:
                probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
                scores.extend(probs)
                labels.extend(y.numpy())

        scores = np.array(scores)
        labels = np.array(labels)
        preds  = (scores > 0.5).astype(int)

        results[test_name] = {
            "y_true":               labels,
            "y_pred":               preds,
            "detectability_scores": scores,
            "roc_auc":              roc_auc_score(labels, scores),
            "brier_score":          brier_score_loss(labels, scores),
            "log_loss":             log_loss(labels, scores),
            "accuracy":             accuracy_score(labels, preds),
        }
        print(f"  {test_name:20s} → AUC={results[test_name]['roc_auc']:.4f}  "
              f"Acc={results[test_name]['accuracy']:.4f}")

    return model, vocab, results, history


# ── Degradation Curve Helper ────────────────────────────────────
def interpolate_texts(llm_texts, human_texts, mix_ratio):
    """
    Return texts where mix_ratio fraction of tokens come from human text.
    0.0 = pure LLM, 1.0 = pure human.
    """
    mixed = []
    for llm, human in zip(llm_texts, human_texts):
        llm_toks   = str(llm).split()
        human_toks = str(human).split()
        n_total    = len(llm_toks)
        n_human    = int(n_total * mix_ratio)
        n_llm      = n_total - n_human
        combined   = llm_toks[:n_llm] + human_toks[:n_human]
        mixed.append(" ".join(combined))
    return mixed


# ── Filter Visualisation ────────────────────────────────────────
def get_top_ngrams_for_filter(model, vocab, texts, filter_idx, filter_size,
                               top_k=10, n_examples=500):
    """Find input n-grams that maximally activate a specific filter."""
    model.eval()
    activation_records = []

    for text in texts[:n_examples]:
        tokens = str(text).lower().split()[:cfg.MAX_SEQ_LENGTH]
        ids    = [vocab.word2idx.get(t, 1) for t in tokens]
        ids    += [0] * (cfg.MAX_SEQ_LENGTH - len(ids))
        x      = torch.tensor([ids], dtype=torch.long).to(device)

        with torch.no_grad():
            emb = model.embedding(x).permute(0, 2, 1)
            for cb in model.convs:
                if cb[0].kernel_size[0] == filter_size:
                    out = cb(emb).squeeze(0)
                    break
            max_val, max_pos = out[filter_idx % cfg.NUM_FILTERS].max(0)
            pos = max_pos.item()
            ngram = tokens[max(0, pos - filter_size//2):
                           max(0, pos - filter_size//2) + filter_size]
            activation_records.append((max_val.item(), " ".join(ngram)))

    activation_records.sort(reverse=True)
    return activation_records[:top_k]