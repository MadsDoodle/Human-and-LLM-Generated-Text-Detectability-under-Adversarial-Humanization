# ================================================================
# src/evaluation/generalisation.py
# Cross-LLM generalisation evaluation:
#   - Checkpoint resolution (local-first, HF Hub fallback)
#   - Multi-LLM corpus generation (format_prompt, load_causal_model,
#     generate_texts)
#   - Distribution shift metrics (KL, Wasserstein, Fréchet)
#   - DeBERTa penultimate-layer embeddings
# ================================================================

import os
import glob as glob_mod
import json
import pickle
import warnings
import numpy as np
import torch
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig, set_seed,
)
from tqdm import tqdm
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED           = 42
MAX_NEW_TOKENS = 150

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question clearly "
    "and concisely in 3-5 sentences."
)


# ================================================================
# SECTION A: CHECKPOINT UTILITIES
# ================================================================

def find_best_checkpoint(model_dir):
    """
    Priority 1: Read trainer_state.json → best_model_checkpoint by ROC-AUC
    Priority 2: No trainer_state → take highest-numbered checkpoint
    Priority 3: No checkpoint subdirs → return root dir (DeBERTa case)
    """
    ckpts = glob_mod.glob(os.path.join(model_dir, "checkpoint-*"))

    if not ckpts:
        # DeBERTa — saved directly to root, no checkpoint subdirs
        print(f"      No checkpoint subdirs — using root dir: {model_dir}")
        return model_dir

    # Try to read best_model_checkpoint from trainer_state.json
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
                print(f"      ⚠️  trainer_state found but best_model_checkpoint "
                      f"missing or invalid: {best}")

    # Fallback — no valid trainer_state, use highest-numbered
    latest = max(ckpts, key=lambda x: int(x.split("-")[-1]))
    print(f"      ⚠️  No trainer_state found — falling back to latest: {latest}")
    return latest


def resolve_detector_path(det_name, local_dir, hf_repo_suffix,
                           hf_user, hf_token=None):
    """
    1. Check if local_dir exists and has model weights → use it.
    2. Otherwise pull from HF Hub into local_dir and use that.
    Returns the resolved checkpoint path ready for from_pretrained.
    """
    from huggingface_hub import snapshot_download

    model_files = ["pytorch_model.bin", "model.safetensors"]

    # ── Step 1: Check local ────────────────────────────────────
    print(f"\n  [{det_name}] Checking local: {local_dir}")

    local_has_model = False
    if os.path.exists(local_dir):
        # Check root dir
        for mf in model_files:
            if os.path.exists(os.path.join(local_dir, mf)):
                local_has_model = True
                break
        # Check checkpoint subdirs
        if not local_has_model:
            ckpts = glob_mod.glob(os.path.join(local_dir, "checkpoint-*"))
            for ckpt in ckpts:
                for mf in model_files:
                    if os.path.exists(os.path.join(ckpt, mf)):
                        local_has_model = True
                        break

    if local_has_model:
        print(f"  ✅ Found locally — using local weights")
        return find_best_checkpoint(local_dir)

    # ── Step 2: Pull from HF Hub ───────────────────────────────
    repo_id = f"{hf_user}/{hf_repo_suffix}"
    print(f"  ⚠️  Not found locally")
    print(f"  🔄 Pulling from HF Hub: {repo_id}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=hf_token,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
        )
        print(f"  ✅ Downloaded from HF Hub → {local_dir}")
        return find_best_checkpoint(local_dir)

    except Exception as e:
        print(f"  ❌ HF Hub pull failed for {repo_id}: {str(e)}")
        return None


# ================================================================
# SECTION B: MULTI-LLM CORPUS GENERATION
# ================================================================

MODEL_REGISTRY = {
    "TinyLlama-1.1B": {
        "hf_id":        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gated":        False,
        "use_4bit":     False,
        "prompt_style": "chatml",
    },
    "Qwen2.5-1.5B": {
        "hf_id":        "Qwen/Qwen2.5-1.5B-Instruct",
        "gated":        False,
        "use_4bit":     False,
        "prompt_style": "chatml",
    },
    "Qwen2.5-7B": {
        "hf_id":        "Qwen/Qwen2.5-7B-Instruct",
        "gated":        False,
        "use_4bit":     True,
        "prompt_style": "chatml",
    },
    "LLaMA-3.1-8B": {
        "hf_id":        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "gated":        True,
        "use_4bit":     True,
        "prompt_style": "llama3",
    },
    "LLaMA-2-13B": {
        "hf_id":        "meta-llama/Llama-2-13b-chat-hf",
        "gated":        True,
        "use_4bit":     True,
        "prompt_style": "llama2",
    },
}


def format_prompt(user_text, prompt_style, tokenizer):
    """
    Format a raw user question into the correct prompt format
    for each model family. Uses apply_chat_template where
    available, falls back to manual string formatting.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]

    if prompt_style == "chatml":
        # Qwen2.5, TinyLlama — all support apply_chat_template
        # add_generation_prompt=True appends the assistant turn
        # opener so the model knows to start generating
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    elif prompt_style == "llama3":
        # LLaMA-3.x supports apply_chat_template but we verify
        # the tokenizer actually has a chat_template set;
        # if not, fall back to the manual format
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return (
                f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{SYSTEM_PROMPT}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_text}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

    elif prompt_style == "llama2":
        # LLaMA-2 apply_chat_template is inconsistent across
        # versions — manual format is always safer here
        return (
            f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"{user_text} [/INST]"
        )

    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")


def load_causal_model(model_cfg, hf_token=None):
    """
    Load a causal LM with optional 4-bit quantization.
    Handles tokenizer padding side correctly for generation.
    """
    hf_id    = model_cfg["hf_id"]
    use_4bit = model_cfg["use_4bit"]

    print(f"  Loading tokenizer : {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        token=hf_token,
        trust_remote_code=True,
    )

    # Left-pad for generation — right-pad is for training only.
    # Without this, batched generation produces shifted/garbage
    # outputs for padded sequences in the batch.
    tokenizer.padding_side = "left"

    # Some tokenizers have no pad token — use eos as fallback
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Loading model     : {hf_id}  (4bit={use_4bit})")

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )

    model.eval()
    print(f"  ✅ Loaded: {hf_id}")
    return model, tokenizer


def generate_texts(prompts, model, tokenizer,
                   prompt_style, batch_size=8):
    """
    Generate responses for a list of raw prompt strings.
    Processes in batches to avoid OOM.
    Returns a list of generated strings (input prompt stripped).
    """
    generated = []

    for i in tqdm(range(0, len(prompts), batch_size),
                  desc="Generating"):
        batch_raw = prompts[i : i + batch_size]

        # Format each prompt correctly for this model family
        batch_formatted = [
            format_prompt(p, prompt_style, tokenizer)
            for p in batch_raw
        ]

        inputs = tokenizer(
            batch_formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Strip input tokens — keep only the newly generated part
        input_len = inputs["input_ids"].shape[1]
        for out in outputs:
            new_tokens = out[input_len:]
            text = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()
            generated.append(text)

    return generated


# ================================================================
# SECTION C: DISTRIBUTION SHIFT METRICS (Stage 3D)
# ================================================================

def embed_deberta(texts, deberta_tok, deberta_model,
                  batch_size=32):
    """
    Extract penultimate hidden state [CLS] vector from DeBERTa.
    Used to measure distribution shift between training and test LLMs.
    """
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = deberta_tok(
            batch,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = deberta_model(**enc)
        cls_emb = out.hidden_states[-2][:, 0, :].cpu().numpy()
        all_embs.append(cls_emb)
    return np.vstack(all_embs)


def kl_gaussian(mu1, cov1, mu2, cov2):
    """
    KL divergence KL(N1 || N2) — numerically stable via slogdet.
    N1 = reference distribution (training/ChatGPT)
    N2 = test distribution (unseen LLM)
    """
    d        = len(mu1)
    cov2_inv = np.linalg.pinv(cov2)
    diff     = mu2 - mu1
    term1    = np.trace(cov2_inv @ cov1)
    term2    = float(diff @ cov2_inv @ diff)
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    term3    = logdet2 - logdet1
    return float(0.5 * (term1 + term2 - d + term3))


def wasserstein2(X, Y):
    """
    Approximate W2 distance between two sets of embeddings
    via Gaussian mean + covariance.
    """
    mu1, mu2 = X.mean(0), Y.mean(0)
    S1,  S2  = np.cov(X.T), np.cov(Y.T)
    sq_S2    = sqrtm(S2)
    M        = sqrtm(sq_S2 @ S1 @ sq_S2)
    w2_sq    = np.sum((mu1 - mu2) ** 2) + np.trace(S1 + S2 - 2 * M.real)
    return float(np.sqrt(max(w2_sq, 0)))


def frechet_distance(X, Y):
    """
    Fréchet distance (FID-style) on embedding space.
    Lower = closer to training distribution.
    """
    mu1, mu2 = X.mean(0), Y.mean(0)
    S1,  S2  = np.cov(X.T), np.cov(Y.T)
    covmean  = sqrtm(S1 @ S2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = np.sum((mu1 - mu2) ** 2) + np.trace(S1 + S2 - 2 * covmean)
    return float(max(fd, 0))


def pca_project(embs, n_comp=64, seed=SEED):
    """
    Project embeddings to n_comp principal components before
    computing distribution distances. Reduces noise and makes
    covariance estimation more stable for high-dim embeddings.
    """
    return PCA(n_components=n_comp, random_state=seed).fit_transform(embs)