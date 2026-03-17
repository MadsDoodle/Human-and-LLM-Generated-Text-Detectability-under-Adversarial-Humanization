# ================================================================
# src/detectors/llm_detector/models.py
# Per-model configs and shared loading/sampling utilities
# ================================================================

import os
import random
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, set_seed,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Global seed ─────────────────────────────────────────────────
SEED = 42

# ================================================================
# MODEL CONFIGS
# ================================================================

TINYLLAMA_CONFIG = {
    'MODEL_NAME': 'TinyLlama-1.1B',
    'HF_ID':      'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'EVAL_N':     500,
    'K_SHOT':     3,
    'MAX_LEN':    1024,
    'RESULTS':    './results/tinyllama',
}

QWEN1P5B_CONFIG = {
    'MODEL_NAME': 'Qwen2.5-1.5B',
    'HF_ID':      'Qwen/Qwen2.5-1.5B-Instruct',
    'EVAL_N':     500,
    'K_SHOT':     3,
    'MAX_LEN':    2048,
    'RESULTS':    './results/qwen25_1p5b',
}

QWEN7B_CONFIG = {
    'MODEL_NAME':       'Qwen2.5-7B',
    'HF_ID':            'Qwen/Qwen2.5-7B-Instruct',
    'EVAL_N':           500,
    'COT_N':            70,
    'K_SHOT':           3,
    'MAX_LEN':          2048,
    'RESULTS_DIR':      './results/qwen25_7b_detector',
    'COT_CONF_WEIGHT':  0.6,
    'COT_LOGIT_WEIGHT': 0.4,
    'PRIOR_N':          50,
    'FLIP':             False,
    'DEAD_ZONE_LO':     0.35,
    'DEAD_ZONE_HI':     0.65,
    'VERDICT_LO':       0.35,
    'VERDICT_HI':       0.65,
}

LLAMA2_13B_CONFIG = {
    'MODEL_NAME':       'LLaMA-2-13B',
    'HF_ID':            'meta-llama/Llama-2-13b-chat-hf',
    'EVAL_N':           200,
    'COT_N':            30,
    'K_SHOT':           3,
    'MAX_LEN':          3072,
    'RESULTS_DIR':      './results/llama2_13b_detector',
    'COT_CONF_WEIGHT':  0.6,
    'COT_LOGIT_WEIGHT': 0.4,
    'PRIOR_N':          50,
    'FLIP':             False,
    'DEAD_ZONE_LO':     0.40,
    'DEAD_ZONE_HI':     0.60,
    'VERDICT_LO':       0.35,
    'VERDICT_HI':       0.65,
}

QWEN14B_CONFIG = {
    'MODEL_NAME':       'Qwen2.5-14B',
    'HF_ID':            'Qwen/Qwen2.5-14B-Instruct',
    'EVAL_N':           200,
    'COT_N':            30,
    'K_SHOT':           3,
    'MAX_LEN':          3072,
    'RESULTS_DIR':      './results/qwen25_14b_detector',
    'COT_CONF_WEIGHT':  0.6,
    'COT_LOGIT_WEIGHT': 0.4,
    'PRIOR_N':          50,
    'FLIP':             False,
    'DEAD_ZONE_LO':     0.35,
    'DEAD_ZONE_HI':     0.65,
    'VERDICT_LO':       0.35,
    'VERDICT_HI':       0.65,
}

GPT4OMINI_CONFIG = {
    'MODEL_ID':          'gpt-4o-mini',
    'EVAL_SAMPLE_SIZE':  200,
    'COT_SAMPLE_SIZE':   50,
    'K_SHOT':            5,
    'RESULTS_DIR':       './results/llm_detector_gpt4omini',
    'TOKEN_BUDGET':      4_000_000,
}


# ================================================================
# SHARED DATA UTILITIES
# ================================================================

def sample_balanced(df, n, seed=SEED):
    h = df[df['label'] == 'human'].sample(n // 2, random_state=seed)
    l = df[df['label'] == 'llm'].sample(n // 2, random_state=seed)
    return pd.concat([h, l]).sample(frac=1, random_state=seed).reset_index(drop=True)


def get_pool(df, n=10, seed=SEED):
    h = df[df['label'] == 'human'].sample(n, random_state=seed)
    l = df[df['label'] == 'llm'].sample(n, random_state=seed)
    return pd.concat([h, l]).reset_index(drop=True)


# ================================================================
# SHARED MODEL UTILITIES
# ================================================================

def get_label_token_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for s in ['yes', 'Yes', 'YES', ' yes', ' Yes']:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.add(ids[0])
    for s in ['no', 'No', 'NO', ' no', ' No']:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.add(ids[0])

    print(f'yes token IDs : {list(yes_ids)}')
    print(f'  -> tokens   : {[tokenizer.decode([i]) for i in yes_ids]}')
    print(f'no token IDs  : {list(no_ids)}')
    print(f'  -> tokens   : {[tokenizer.decode([i]) for i in no_ids]}')

    if not yes_ids or not no_ids:
        raise RuntimeError('No valid single-token IDs found')
    return list(yes_ids), list(no_ids)


def load_causal_model_fp16(hf_id, trust_remote_code=True):
    """Load a causal LM in fp16 (for smaller models: TinyLlama, Qwen1.5B)."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def load_causal_model_4bit(hf_id, compute_dtype=torch.float16,
                            padding_side='left', trust_remote_code=True):
    """Load a causal LM in 4-bit NF4 (for large models: Qwen7B, LLaMA13B, Qwen14B)."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=trust_remote_code,
        padding_side=padding_side)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb,
        device_map='auto',
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


# ================================================================
# TF-IDF RETRIEVERS (shared by all models using few-shot retrieval)
# ================================================================

def build_retriever(pool_df):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vec.fit_transform(pool_df['text'].astype(str))
    return vec, matrix


def retrieve_shots(text, pool_df, vec, matrix, k=3):
    q    = vec.transform([text])
    sims = cosine_similarity(q, matrix).flatten()
    results  = []
    per_class = [k // 2 + (1 if i < k % 2 else 0) for i in range(2)]
    random.shuffle(per_class)
    for n, lbl in zip(per_class, ['human', 'llm']):
        mask       = (pool_df['label'] == lbl).values
        class_sims = sims.copy()
        class_sims[~mask] = -1
        top_idx = class_sims.argsort()[-n:][::-1]
        results.extend(top_idx.tolist())
    return pool_df.iloc[results]