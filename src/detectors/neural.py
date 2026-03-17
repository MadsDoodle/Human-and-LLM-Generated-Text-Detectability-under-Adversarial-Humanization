# ================================================================
# src/detectors/neural.py
# Family 2: Neural Transformer Detectors
# Covers: BERT, RoBERTa, ELECTRA, DistilBERT, DeBERTa-v3-base
# ================================================================

import os
import json
import pickle
import shutil
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    roc_curve, accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# SECTION A: BERT / RoBERTa / ELECTRA
# ================================================================

class DetectorConfig:

    MODELS = {
        'BERT':    'bert-base-uncased',
        'RoBERTa': 'roberta-base',
        'ELECTRA': 'google/electra-base-discriminator',
    }

    HC3_TRAIN  = "hc3_train.csv"
    HC3_TEST   = "hc3_test.csv"
    ELI5_TRAIN = "eli5_train.csv"
    ELI5_TEST  = "eli5_test.csv"

    NUM_EPOCHS       = 1
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_EVAL  = 64
    LEARNING_RATE    = 2e-5
    WARMUP_RATIO     = 0.06
    WEIGHT_DECAY     = 0.01
    DROPOUT          = 0.2
    VAL_SPLIT_RATIO  = 0.1
    LOGGING_STEPS    = 100
    MAX_SEQ_LENGTH   = 512
    FP16             = True
    FORCE_RETRAIN    = False

    OUTPUT_DIR_BASE = "./models"
    RESULTS_DIR     = "./results"

    @classmethod
    def get_output_dir(cls, model_name, dataset_name):
        return os.path.join(cls.OUTPUT_DIR_BASE, f"{model_name}_{dataset_name}")

os.makedirs(DetectorConfig.OUTPUT_DIR_BASE, exist_ok=True)
os.makedirs(DetectorConfig.RESULTS_DIR, exist_ok=True)


# ── Checkpoint Utilities ────────────────────────────────────────

def check_model_exists(output_dir):
    """
    For DeBERTa-style training, model is saved directly to root dir.
    Just check if model weights exist in root dir.
    """
    if not os.path.exists(output_dir):
        return False, None

    model_files = ['pytorch_model.bin', 'model.safetensors']
    for mf in model_files:
        if os.path.exists(os.path.join(output_dir, mf)):
            print(f"   ✅ Found existing model in root: {output_dir}")
            return True, output_dir

    return False, None


def load_trained_model(model_checkpoint_path, base_model_name):
    print(f"   ⏳ Loading trained model from: {model_checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    model_checkpoint_path)
    print(f"   ✅ Loaded successfully")
    return model, tokenizer


def save_checkpoint_status(output_dir, status_info):
    status_file = os.path.join(output_dir, "training_status.json")
    with open(status_file, 'w') as f:
        json.dump(status_info, f, indent=2)


# ── Dataset Class ───────────────────────────────────────────────

class TextDetectionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length,
            padding='max_length', truncation=True,
            return_tensors='pt')
        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels':         torch.tensor(label, dtype=torch.long)
        }


# ── Label Encoder ───────────────────────────────────────────────

def encode_labels(labels):
    return (labels == 'llm').astype(int)


# ── Metrics ─────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs       = torch.nn.functional.softmax(
                    torch.tensor(logits), dim=-1).numpy()
    predictions = probs[:, 1]
    pred_labels = (predictions > 0.5).astype(int)
    return {
        'roc_auc':     roc_auc_score(labels, predictions),
        'brier_score': brier_score_loss(labels, predictions),
        'accuracy':    accuracy_score(labels, pred_labels)
    }


# ── Training Function (BERT / RoBERTa / ELECTRA) ────────────────

def train_and_evaluate_detector(
    model_name,
    model_checkpoint,
    train_data,
    val_data,
    test_data_dict,
    output_dir,
    force_retrain=False
):
    print(f"\n{'#'*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'#'*70}")

    # ── Checkpoint check ─────────────────────────────────────
    model_exists, checkpoint_path = check_model_exists(output_dir)

    if model_exists and not force_retrain and not DetectorConfig.FORCE_RETRAIN:
        print(f"\n🔄 Found existing model — loading and skipping training")
        try:
            model, tokenizer = load_trained_model(checkpoint_path, model_checkpoint)
            model.to(device)
            skip_training = True
        except Exception as e:
            print(f"   ⚠️ Load failed: {e} — retraining from scratch")
            skip_training = False
    else:
        print(f"\n🆕 Training from scratch")
        skip_training = False

    # ── Training ─────────────────────────────────────────────
    if not skip_training:
        print(f"\n⏳ Loading {model_checkpoint} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model     = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=2,
            problem_type="single_label_classification",
            hidden_dropout_prob=DetectorConfig.DROPOUT,
            attention_probs_dropout_prob=DetectorConfig.DROPOUT
        )
        print(f"✅ Model loaded")

        train_dataset = TextDetectionDataset(
            train_data['text'].tolist(),
            train_data['label_encoded'].tolist(),
            tokenizer, DetectorConfig.MAX_SEQ_LENGTH)

        val_dataset = TextDetectionDataset(
            val_data['text'].tolist(),
            val_data['label_encoded'].tolist(),
            tokenizer, DetectorConfig.MAX_SEQ_LENGTH)

        # ── DeBERTa-style: NO checkpointing, NO early stopping ──
        # Runs full epoch, saves final in-memory weights to root dir
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=DetectorConfig.NUM_EPOCHS,
            per_device_train_batch_size=DetectorConfig.BATCH_SIZE_TRAIN,
            per_device_eval_batch_size=DetectorConfig.BATCH_SIZE_EVAL,
            learning_rate=DetectorConfig.LEARNING_RATE,
            warmup_ratio=DetectorConfig.WARMUP_RATIO,
            weight_decay=DetectorConfig.WEIGHT_DECAY,
            logging_steps=DetectorConfig.LOGGING_STEPS,

            # ── Eval during training for visibility, but no checkpointing ──
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="no",
            load_best_model_at_end=False,

            fp16=DetectorConfig.FP16,
            dataloader_num_workers=2,
            report_to="none",
            disable_tqdm=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        print(f"\n🚀 Starting training ...")
        print(f"   Train samples : {len(train_dataset):,}")
        print(f"   Val samples   : {len(val_dataset):,}")
        print(f"   Epochs        : {DetectorConfig.NUM_EPOCHS}")
        print(f"   Batch size    : {DetectorConfig.BATCH_SIZE_TRAIN}")
        print(f"   LR            : {DetectorConfig.LEARNING_RATE}")
        print(f"   Warmup        : {DetectorConfig.WARMUP_RATIO*100}% of steps")

        try:
            trainer.train()
            print(f"\n✅ Training complete!")

            print(f"⏳ Saving model to: {output_dir}")
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"✅ Saved to root dir — no checkpoint subdirs")

            # ── Post-training sanity check ───────────────────
            print("\n⏳ Post-training sanity check ...")
            dummy_texts = [
                "yeah i just grabbed some lunch and honestly it was pretty mid tbh",
                "Certainly! The mitochondria is the powerhouse of the cell, responsible for producing adenosine triphosphate (ATP) through the process of cellular respiration, which involves glycolysis, the Krebs cycle, and oxidative phosphorylation."
            ]
            dummy_enc = tokenizer(
                dummy_texts, return_tensors="pt",
                padding=True, truncation=True).to(device)
            with torch.no_grad():
                dummy_logits = model(**dummy_enc).logits
            dummy_probs = torch.softmax(dummy_logits, dim=-1)[:, 1].cpu().numpy()
            dummy_std   = dummy_probs.std()

            print(f"   Sanity scores : {dummy_probs.round(4)}")
            print(f"   Score std     : {dummy_std:.4f}")

            if dummy_std < 0.05:
                print(f"   ⚠️  WARNING: Model collapsed (std={dummy_std:.4f})")
                print(f"   Check training logs — model may not have converged")
            else:
                print(f"   ✅ Model looks healthy")

            save_checkpoint_status(output_dir, {
                'model_name':    model_name,
                'status':        'completed',
                'output_dir':    output_dir,
                'sanity_std':    float(dummy_std),
                'sanity_scores': dummy_probs.tolist(),
                'collapsed':     bool(dummy_std < 0.05),
                'timestamp':     pd.Timestamp.now().isoformat()
            })

        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            save_checkpoint_status(output_dir, {
                'model_name': model_name,
                'status':     'failed',
                'error':      str(e),
                'timestamp':  pd.Timestamp.now().isoformat()
            })
            raise

    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=DetectorConfig.BATCH_SIZE_EVAL,
            fp16=DetectorConfig.FP16,
            report_to="none"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics
        )

    # ── Evaluate on test sets ─────────────────────────────────
    results = {}
    for test_name, test_data in test_data_dict.items():
        print(f"\n📊 Evaluating on {test_name} ...")
        test_dataset = TextDetectionDataset(
            test_data['text'].tolist(),
            test_data['label_encoded'].tolist(),
            tokenizer, DetectorConfig.MAX_SEQ_LENGTH)

        predictions          = trainer.predict(test_dataset)
        logits               = predictions.predictions
        probs                = torch.nn.functional.softmax(
                                torch.tensor(logits), dim=-1).numpy()
        detectability_scores = probs[:, 1]

        y_true = test_data['label_encoded'].values
        y_pred = (detectability_scores > 0.5).astype(int)

        results[test_name] = {
            'y_true':               y_true,
            'y_pred':               y_pred,
            'detectability_scores': detectability_scores,
            'roc_auc':    roc_auc_score(y_true, detectability_scores),
            'brier_score':brier_score_loss(y_true, detectability_scores),
            'log_loss':   log_loss(y_true, detectability_scores),
            'accuracy':   accuracy_score(y_true, y_pred)
        }
        print(f"   ROC-AUC  : {results[test_name]['roc_auc']:.4f}")
        print(f"   Brier    : {results[test_name]['brier_score']:.4f}")
        print(f"   Accuracy : {results[test_name]['accuracy']:.4f}")

    del model, trainer
    torch.cuda.empty_cache()
    return results


# ================================================================
# SECTION B: DistilBERT
# ================================================================

class DistilBERTConfig:
    MODEL_NAME = 'distilbert-base-uncased'

    HC3_TRAIN  = "hc3_train.csv"
    HC3_TEST   = "hc3_test.csv"
    ELI5_TRAIN = "eli5_train.csv"
    ELI5_TEST  = "eli5_test.csv"

    NUM_EPOCHS         = 1
    BATCH_SIZE_TRAIN   = 32
    BATCH_SIZE_EVAL    = 64
    LEARNING_RATE      = 2e-5
    WARMUP_RATIO       = 0.06
    WEIGHT_DECAY       = 0.01
    DROPOUT            = 0.2
    ATTENTION_DROPOUT  = 0.2
    VAL_SPLIT_RATIO    = 0.1
    LOGGING_STEPS      = 100
    MAX_SEQ_LENGTH     = 512
    FP16               = torch.cuda.is_available()

    OUTPUT_DIR_BASE = "./models"
    RESULTS_DIR     = "./results"

    @classmethod
    def get_output_dir(cls, dataset_name):
        return os.path.join(cls.OUTPUT_DIR_BASE, f"DistilBERT_{dataset_name}")


def train_distilbert_detector(train_data, val_data, test_data_dict,
                               output_dir, dataset_name):
    print(f"\n{'#'*70}")
    print(f"TRAINING: DistilBERT on {dataset_name}")
    print(f"{'#'*70}")

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(DistilBERTConfig.MODEL_NAME)

    # DistilBERT-specific dropout param names
    model = AutoModelForSequenceClassification.from_pretrained(
        DistilBERTConfig.MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification",
        dropout=DistilBERTConfig.DROPOUT,
        attention_dropout=DistilBERTConfig.ATTENTION_DROPOUT
    )
    print(f"✅ Model loaded — dropout={DistilBERTConfig.DROPOUT}")

    train_dataset = TextDetectionDataset(
        train_data['text'].tolist(),
        train_data['label_encoded'].tolist(),
        tokenizer, DistilBERTConfig.MAX_SEQ_LENGTH)

    val_dataset = TextDetectionDataset(
        val_data['text'].tolist(),
        val_data['label_encoded'].tolist(),
        tokenizer, DistilBERTConfig.MAX_SEQ_LENGTH)

    # ── DeBERTa-style: no checkpointing, no early stopping ──
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=DistilBERTConfig.NUM_EPOCHS,
        per_device_train_batch_size=DistilBERTConfig.BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=DistilBERTConfig.BATCH_SIZE_EVAL,
        learning_rate=DistilBERTConfig.LEARNING_RATE,
        warmup_ratio=DistilBERTConfig.WARMUP_RATIO,
        weight_decay=DistilBERTConfig.WEIGHT_DECAY,
        logging_steps=DistilBERTConfig.LOGGING_STEPS,

        eval_strategy="steps",
        eval_steps=200,

        save_strategy="no",
        load_best_model_at_end=False,

        fp16=DistilBERTConfig.FP16,
        dataloader_num_workers=2,
        report_to="none",
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\n🚀 Starting training ...")
    print(f"   Train samples : {len(train_dataset):,}")
    print(f"   Val samples   : {len(val_dataset):,}")
    print(f"   Epochs        : {DistilBERTConfig.NUM_EPOCHS}")
    print(f"   Batch size    : {DistilBERTConfig.BATCH_SIZE_TRAIN}")
    print(f"   LR            : {DistilBERTConfig.LEARNING_RATE}")

    trainer.train()
    print(f"\n✅ Training complete!")

    print(f"⏳ Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved to root dir — no checkpoint subdirs")

    # ── Sanity check using real test samples ────────────────
    print("\n⏳ Post-training sanity check ...")
    hc3_test_loaded = pd.read_csv(DistilBERTConfig.HC3_TEST)
    hc3_test_loaded['label_encoded'] = encode_labels(hc3_test_loaded['label'])
    human_sample = hc3_test_loaded[hc3_test_loaded['label'] == 'human']['text'].iloc[0][:300]
    llm_sample   = hc3_test_loaded[hc3_test_loaded['label'] == 'llm']['text'].iloc[0][:300]

    dummy_enc = tokenizer(
        [human_sample, llm_sample],
        return_tensors="pt", padding=True,
        truncation=True, max_length=512).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        dummy_logits = model(**dummy_enc).logits
    dummy_probs = torch.softmax(dummy_logits, dim=-1)[:, 1].cpu().numpy()
    dummy_std   = dummy_probs.std()

    print(f"   Human score   : {dummy_probs[0]:.4f}")
    print(f"   LLM score     : {dummy_probs[1]:.4f}")
    print(f"   Score std     : {dummy_std:.4f}")

    if dummy_std < 0.05:
        print(f"   ⚠️  WARNING: Model may be collapsed (std={dummy_std:.4f})")
    else:
        print(f"   ✅ Model looks healthy")

    with open(os.path.join(output_dir, "training_status.json"), 'w') as f:
        json.dump({
            'model_name':    f'DistilBERT_{dataset_name}',
            'status':        'completed',
            'sanity_std':    float(dummy_std),
            'sanity_scores': dummy_probs.tolist(),
            'collapsed':     bool(dummy_std < 0.05),
            'timestamp':     pd.Timestamp.now().isoformat()
        }, f, indent=2)

    # ── Evaluate on test sets ────────────────────────────────
    results = {}
    for test_name, test_data in test_data_dict.items():
        print(f"\n📊 Evaluating on {test_name} ...")
        test_dataset = TextDetectionDataset(
            test_data['text'].tolist(),
            test_data['label_encoded'].tolist(),
            tokenizer, DistilBERTConfig.MAX_SEQ_LENGTH)

        predictions          = trainer.predict(test_dataset)
        probs                = torch.nn.functional.softmax(
                                torch.tensor(predictions.predictions),
                                dim=-1).numpy()
        detectability_scores = probs[:, 1]
        y_true               = test_data['label_encoded'].values
        y_pred               = (detectability_scores > 0.5).astype(int)

        results[test_name] = {
            'y_true':               y_true,
            'y_pred':               y_pred,
            'detectability_scores': detectability_scores,
            'roc_auc':    roc_auc_score(y_true, detectability_scores),
            'brier_score':brier_score_loss(y_true, detectability_scores),
            'log_loss':   log_loss(y_true, detectability_scores),
            'accuracy':   accuracy_score(y_true, y_pred)
        }
        print(f"   ROC-AUC  : {results[test_name]['roc_auc']:.4f}")
        print(f"   Brier    : {results[test_name]['brier_score']:.4f}")
        print(f"   Accuracy : {results[test_name]['accuracy']:.4f}")

    del model, trainer
    torch.cuda.empty_cache()
    return results


# ================================================================
# SECTION C: DeBERTa-v3-base
# ================================================================
# Precision strategy:
#   fp32 throughout — bf16 silently zeroes gradients for this
#   architecture; fp16 crashes the grad scaler.
# Checkpointing strategy:
#   Disabled entirely (save_strategy="no",
#   load_best_model_at_end=False). Eliminates the LayerNorm
#   gamma/beta key mismatch that caused AUC ≈ 0.50.
# ================================================================

class DeBERTaConfig:
    MODEL_CHECKPOINT = "microsoft/deberta-v3-base"

    HC3_TRAIN  = "hc3_train.csv"
    HC3_TEST   = "hc3_test.csv"
    ELI5_TRAIN = "eli5_train.csv"
    ELI5_TEST  = "eli5_test.csv"

    NUM_EPOCHS              = 1
    BATCH_SIZE_TRAIN        = 16
    BATCH_SIZE_EVAL         = 32
    LEARNING_RATE           = 2e-5
    WARMUP_STEPS            = 500
    WEIGHT_DECAY            = 0.01
    DROPOUT                 = 0.2
    VAL_SPLIT_RATIO         = 0.1
    LOGGING_STEPS           = 100
    MAX_SEQ_LENGTH          = 512

    # ── Precision: full fp32 ──────────────────────────────────
    # bf16 silently zeroes DeBERTa-v3 gradients (disentangled
    # attention produces small gradient magnitudes that underflow
    # in bf16's 7-bit mantissa). fp16 causes unscaling crash.
    FP16 = False
    BF16 = False

    # ── No checkpoint reloading ───────────────────────────────
    # load_best_model_at_end reloads a saved checkpoint whose
    # LayerNorm keys use old gamma/beta naming → all 24 LayerNorm
    # layers reinitialise to random → AUC collapses to 0.50.
    SAVE_STRATEGY           = "no"
    LOAD_BEST_MODEL_AT_END  = False

    # Gradient clipping — important for DeBERTa-v3 stability
    MAX_GRAD_NORM           = 1.0

    OUTPUT_DIR_BASE = "./models"
    RESULTS_DIR     = "./results"

    @classmethod
    def get_output_dir(cls, dataset_name):
        return os.path.join(cls.OUTPUT_DIR_BASE, f"DeBERTa_{dataset_name}")


def train_deberta(train_data, val_data, test_data_dict,
                  output_dir, tag):
    """
    Train DeBERTa-v3-base for binary AI-text detection.

    Precision strategy:
        fp32 throughout — bf16 silently zeroes gradients for this
        architecture; fp16 crashes the grad scaler. No mixed
        precision is the safest and only reliable option here.

    Checkpointing strategy:
        Disabled entirely (save_strategy="no",
        load_best_model_at_end=False). The Trainer runs one full
        epoch and the final in-memory model is used directly for
        prediction. This eliminates the LayerNorm gamma/beta key
        mismatch that caused AUC ≈ 0.50 in previous runs.
    """
    print(f"\n{'='*60}")
    print(f"Training DeBERTa-v3-base  [dataset: {tag}]")
    print(f"{'='*60}")

    # Delete any stale checkpoints from previous failed runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"🗑️  Cleared stale output dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(DeBERTaConfig.MODEL_CHECKPOINT)

    # ── Model — forced to fp32 ────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        DeBERTaConfig.MODEL_CHECKPOINT,
        num_labels=2,
        hidden_dropout_prob=DeBERTaConfig.DROPOUT,
        attention_probs_dropout_prob=DeBERTaConfig.DROPOUT,
        ignore_mismatched_sizes=True,
    ).float()   # explicit fp32 cast — do not remove

    # ── Datasets ──────────────────────────────────────────────
    train_ds = TextDetectionDataset(
        train_data["text"].tolist(),
        train_data["label_encoded"].tolist(),
        tokenizer, DeBERTaConfig.MAX_SEQ_LENGTH)

    val_ds = TextDetectionDataset(
        val_data["text"].tolist(),
        val_data["label_encoded"].tolist(),
        tokenizer, DeBERTaConfig.MAX_SEQ_LENGTH)

    # ── TrainingArguments ─────────────────────────────────────
    args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = DeBERTaConfig.NUM_EPOCHS,
        per_device_train_batch_size = DeBERTaConfig.BATCH_SIZE_TRAIN,
        per_device_eval_batch_size  = DeBERTaConfig.BATCH_SIZE_EVAL,
        learning_rate               = DeBERTaConfig.LEARNING_RATE,
        warmup_steps                = DeBERTaConfig.WARMUP_STEPS,
        weight_decay                = DeBERTaConfig.WEIGHT_DECAY,
        logging_steps               = DeBERTaConfig.LOGGING_STEPS,
        # ── Evaluation ────────────────────────────────────────
        eval_strategy               = "steps",
        eval_steps                  = 200,
        # ── Checkpointing: DISABLED ───────────────────────────
        save_strategy               = "no",
        load_best_model_at_end      = False,
        # ── Precision: full fp32 ──────────────────────────────
        fp16                        = False,
        bf16                        = False,
        # ── Stability ─────────────────────────────────────────
        max_grad_norm               = DeBERTaConfig.MAX_GRAD_NORM,
        dataloader_num_workers      = 2,
        report_to                   = "none",
    )

    # ── Trainer ───────────────────────────────────────────────
    # No EarlyStoppingCallback — incompatible with save_strategy="no"
    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────
    print("\nTraining ... (fp32, no checkpointing)")
    print("Expect loss to drop below 0.5 by step 400 if working.\n")
    trainer.train()
    print("\n✅ Training complete — using final in-memory weights")

    # ── Sanity check on validation set ───────────────────────
    val_out = trainer.evaluate()
    val_auc = val_out.get("eval_roc_auc", float("nan"))
    print(f"\n  Validation AUC : {val_auc:.4f}  "
          f"{'✅ looks good' if val_auc > 0.7 else '⚠️  still low — check logs'}")

    # ── Evaluate on all test splits ───────────────────────────
    print("\nRunning test-set predictions ...")
    results = {}

    for test_name, test_df in test_data_dict.items():
        test_ds = TextDetectionDataset(
            test_df["text"].tolist(),
            test_df["label_encoded"].tolist(),
            tokenizer, DeBERTaConfig.MAX_SEQ_LENGTH)

        preds  = trainer.predict(test_ds)
        probs  = torch.nn.functional.softmax(
                     torch.tensor(preds.predictions),
                     dim=-1).numpy()
        scores = probs[:, 1]
        y_true = test_df["label_encoded"].values
        y_pred = (scores > 0.5).astype(int)

        auc  = roc_auc_score(y_true, scores)
        acc  = accuracy_score(y_true, y_pred)
        bri  = brier_score_loss(y_true, scores)
        ll   = log_loss(y_true, scores)
        mh   = scores[y_true == 0].mean()
        ml   = scores[y_true == 1].mean()
        sep  = ml - mh

        results[test_name] = {
            "y_true":               y_true,
            "y_pred":               y_pred,
            "detectability_scores": scores,
            "roc_auc":              auc,
            "brier_score":          bri,
            "log_loss":             ll,
            "accuracy":             acc,
            "mean_human_score":     mh,
            "mean_llm_score":       ml,
            "score_separation":     sep,
        }
        print(f"  {test_name:20s}  AUC={auc:.4f}  Acc={acc:.4f}  "
              f"MeanH={mh:.3f}  MeanL={ml:.3f}  Sep={sep:.3f}")

    # ── Save final model & tokenizer ──────────────────────────
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Model saved → {output_dir}")

    # ── Save npy score files for Notebook 6 ──────────────────
    for eval_name, d in results.items():
        np.save(
            f"deberta_scores_{eval_name}.npy",
            {"y_true":  d["y_true"],
             "y_score": d["detectability_scores"]}
        )

    del model, trainer
    torch.cuda.empty_cache()
    return results