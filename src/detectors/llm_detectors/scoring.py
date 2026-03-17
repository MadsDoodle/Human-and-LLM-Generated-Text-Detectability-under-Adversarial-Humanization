# ================================================================
# src/detectors/llm_detector/scoring.py
# Constrained decoding, prior calibration, CoT ensemble, GPT-4o scoring
# ================================================================

import re
import os
import json
import time
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import set_seed
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

SEED = 42


# ================================================================
# CONSTRAINED SCORING
# Shared by TinyLlama, Qwen1.5B, Qwen7B, LLaMA-2-13B, Qwen14B
#
# Polarity notes:
#   Standard (TinyLlama, Qwen1.5B): yes=AI  → flip=True  → P(llm)=P(yes)
#   Swapped  (Qwen7B, LLaMA13B, Qwen14B):
#     yes=human, no=AI → flip=False → P(llm)=P(no) after prior correction
# ================================================================

def constrained_score(model, tokenizer, prompt_text, yes_ids, no_ids,
                      yes_prior=None, no_prior=None,
                      flip=False, use_prior=True, max_len=1024):
    """
    Returns P(llm) ∈ [0, 1].

    flip=False (swapped prompt, yes=human):
        raw = softmax([no_logit, yes_logit])[1] = P(yes=human)
        After task-prior correction the ordering inverts empirically
        → raw is returned as P(llm) without additional inversion.
    flip=True (standard prompt, yes=AI):
        raw = P(yes=AI) = P(llm) directly.
    """
    enc = tokenizer(prompt_text, return_tensors='pt',
                    truncation=True, max_length=max_len).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1, :]

    yes_logit = torch.stack([logits[i] for i in yes_ids]).max()
    no_logit  = torch.stack([logits[i] for i in no_ids]).max()

    if use_prior and yes_prior is not None and no_prior is not None:
        yes_logit = yes_logit - yes_prior
        no_logit  = no_logit  - no_prior

    raw = torch.softmax(torch.stack([no_logit, yes_logit]), dim=0)[1].item()
    return 1.0 - raw if flip else raw


# ================================================================
# TASK PRIOR CALIBRATION
# Shared by Qwen7B, LLaMA-2-13B, Qwen14B
# ================================================================

def compute_task_prior(model, tokenizer, eval_df, yes_ids, no_ids,
                       prompt_fn, n_samples=50, max_len=2048, seed=SEED):
    """
    Average yes/no logits over n_samples real task prompts.
    prompt_fn must be the zero-shot function for the current model
    (prior must match the exact prompt context used at evaluation).

    Args:
        prompt_fn: callable(text, tokenizer) → prompt string
    """
    sample = eval_df.sample(min(n_samples, len(eval_df)), random_state=seed)
    yes_logits_list, no_logits_list = [], []

    for _, row in tqdm(sample.iterrows(), total=len(sample),
                       desc='Computing task prior'):
        text   = str(row['text'])
        prompt = prompt_fn(text, tokenizer)
        enc    = tokenizer(prompt, return_tensors='pt',
                           truncation=True,
                           max_length=max_len).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1, :]

        yes_logits_list.append(
            torch.stack([logits[i] for i in yes_ids]).max().item())
        no_logits_list.append(
            torch.stack([logits[i] for i in no_ids]).max().item())

    task_yes_prior = torch.tensor(yes_logits_list).mean()
    task_no_prior  = torch.tensor(no_logits_list).mean()

    print(f'\nTask prior — yes: {task_yes_prior:.3f}  '
          f'no: {task_no_prior:.3f}  '
          f'gap: {(task_no_prior - task_yes_prior):.3f}')
    return task_yes_prior, task_no_prior


# ================================================================
# CoT OUTPUT PARSERS
# parse_cot_output_v2 — Qwen7B (AI_CONFIDENCE tag)
# parse_cot_output    — LLaMA-2-13B and Qwen14B (same logic)
# ================================================================

def parse_cot_output_v2(raw: str):
    """CoT parser for Qwen7B. Verdict: yes=AI, no=human."""
    raw_lower = raw.lower()

    # AI_CONFIDENCE explicit tag
    conf = None
    m_conf = re.search(
        r'ai_confidence[:\s]+([0-9]+(?:\.[0-9]+)?)', raw_lower)
    if m_conf:
        raw_score = float(m_conf.group(1))
        conf = min(raw_score / 10.0, 1.0)

    # Fallback: x/10 patterns
    if conf is None:
        scores_x_of_10 = re.findall(
            r'([0-9](?:\.[0-9]+)?)\s*/\s*10', raw_lower)
        if scores_x_of_10:
            vals = [float(s) for s in scores_x_of_10 if float(s) <= 10]
            if len(vals) >= 3:
                conf = min(sum(vals) / len(vals) / 10.0, 1.0)

    # Fallback: dimension keyword scores
    if conf is None:
        dim_scores = re.findall(
            r'(?:structure|completeness|hedging|personal voice|lexical|'
            r'response fit|short.form)[^\n]*?([0-9](?:\.[0-9]+)?)\b',
            raw_lower
        )
        if dim_scores:
            vals = [float(s) for s in dim_scores if float(s) <= 10]
            if vals:
                conf = min(sum(vals) / len(vals) / 10.0, 1.0)

    # Verdict (CoT prompt: yes=AI, no=human)
    m_verdict = re.search(r'verdict:\s*(yes|no)', raw_lower)
    if m_verdict:
        verdict = 'llm' if m_verdict.group(1) == 'yes' else 'human'
    else:
        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        last  = lines[-1].lower() if lines else ''
        if last.startswith('yes'):
            verdict = 'llm'
        elif last.startswith('no'):
            verdict = 'human'
        else:
            verdict = 'unknown'

    # Sanity check: trust conf over verdict when strongly divergent
    if conf is not None and verdict != 'unknown':
        if conf > 0.75 and verdict == 'human':
            verdict = 'llm'
        elif conf < 0.25 and verdict == 'llm':
            verdict = 'human'

    return verdict, conf


def parse_cot_output(raw: str):
    """CoT parser for LLaMA-2-13B and Qwen14B. Verdict: yes=AI, no=human."""
    raw_lower = raw.lower()

    conf = None
    m = re.search(r'ai_confidence[:\s]+([0-9]+(?:\.[0-9]+)?)', raw_lower)
    if m:
        conf = min(float(m.group(1)) / 10.0, 1.0)

    if conf is None:
        vals_10 = re.findall(r'([0-9](?:\.[0-9]+)?)\s*/\s*10', raw_lower)
        if vals_10:
            vals = [float(v) for v in vals_10 if float(v) <= 10]
            if len(vals) >= 3:
                conf = min(sum(vals) / len(vals) / 10.0, 1.0)

    if conf is None:
        dim_scores = re.findall(
            r'(?:structural|lexical|coverage|hedging|personal|'
            r'alignment|formulaic)[^\n]*?([0-9](?:\.[0-9]+)?)\b',
            raw_lower
        )
        if dim_scores:
            vals = [float(s) for s in dim_scores if float(s) <= 10]
            if vals:
                conf = min(sum(vals) / len(vals) / 10.0, 1.0)

    m_v = re.search(r'verdict:\s*(yes|no)', raw_lower)
    if m_v:
        verdict = 'llm' if m_v.group(1) == 'yes' else 'human'
    else:
        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        last  = lines[-1].lower() if lines else ''
        if last.startswith('yes'):
            verdict = 'llm'
        elif last.startswith('no'):
            verdict = 'human'
        else:
            verdict = 'unknown'

    if conf is not None and verdict != 'unknown':
        if conf > 0.75 and verdict == 'human':
            verdict = 'llm'
        elif conf < 0.25 and verdict == 'llm':
            verdict = 'human'

    return verdict, conf


# ================================================================
# CoT ENSEMBLE SCORER
# Shared by Qwen7B, LLaMA-2-13B, Qwen14B
# Dead zone width varies per model — pass explicitly
# ================================================================

def cot_ensemble_score(conf, logit_score,
                       conf_weight=0.6, logit_weight=0.4,
                       dead_zone_lo=0.40, dead_zone_hi=0.60):
    """
    Dead zone [dead_zone_lo, dead_zone_hi]:
      conf inside  → uninformative, use logit only
      conf outside → weighted ensemble

    Dead zone widths by model:
      LLaMA-2-13B : [0.40, 0.60]
      Qwen7B      : [0.35, 0.65]  (conf less reliable at 7B)
      Qwen14B     : [0.35, 0.65]
    """
    if conf is None or (dead_zone_lo <= conf <= dead_zone_hi):
        score     = logit_score
        used_conf = False
    else:
        score     = conf_weight * conf + logit_weight * logit_score
        used_conf = True

    return float(max(0.01, min(0.99, score))), used_conf


# ================================================================
# GPT-4o-mini API ENGINE & PARSERS
# ================================================================

MAX_RETRIES  = 5
RETRY_DELAY  = 2.0
tokens_used_total = 0


def call_gpt(client, messages: list, model_id: str = "gpt-4o-mini",
             max_tokens: int = 120, token_budget: int = 4_000_000) -> dict:
    global tokens_used_total
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model       = model_id,
                messages    = messages,
                temperature = 0,
                seed        = SEED,
                max_tokens  = max_tokens,
            )
            raw_text          = resp.choices[0].message.content.strip()
            tokens_used_total += resp.usage.total_tokens
            if tokens_used_total > token_budget * 0.8:
                print(f"  ⚠️  TOKEN WARNING: {tokens_used_total:,} / "
                      f"{token_budget:,} used")
            return {
                "raw_text": raw_text,
                "usage":    resp.usage.total_tokens,
                "error":    None,
            }
        except Exception as e:
            wait = RETRY_DELAY * (2 ** attempt)
            print(f"    API error (attempt {attempt+1}/{MAX_RETRIES}): "
                  f"{e} — retrying in {wait:.0f}s")
            time.sleep(wait)

    return {"raw_text": "", "usage": 0, "error": "max_retries_exceeded"}


def parse_ai_score(raw: str):
    """
    Extract AI_SCORE from zero-shot / few-shot GPT-4o-mini response.
    Returns [0.0, 1.0] or None.
    AI_SCORE: 0 = certainly human, 100 = certainly AI.
    """
    # Primary: explicit AI_SCORE tag
    m = re.search(r'ai_score[:\s]+([0-9]+(?:\.[0-9]+)?)', raw.lower())
    if m:
        val = float(m.group(1))
        if val <= 10:
            return val / 10.0
        elif val <= 100:
            return val / 100.0

    # Fallback 1: parse 7 dimension lines and average
    dim_scores = re.findall(
        r'^[1-7][.:]\s*(?:[^\n]*?:)?\s*([0-9](?:\.[0-9]+)?)\s*$',
        raw, re.MULTILINE
    )
    if len(dim_scores) >= 4:
        vals = [float(s) for s in dim_scores if float(s) <= 10]
        if vals:
            return min(sum(vals) / len(vals) / 10.0, 1.0)

    # Fallback 2: any x/10 pattern
    vals_10 = re.findall(r'\b([0-9](?:\.[0-9]+)?)\s*/\s*10\b', raw)
    if vals_10:
        vals = [float(v) for v in vals_10 if float(v) <= 10]
        if len(vals) >= 3:
            return min(sum(vals) / len(vals) / 10.0, 1.0)

    # Fallback 3: last standalone integer in range
    integers = re.findall(r'\b([0-9]{1,3})\b', raw)
    if integers:
        val = int(integers[-1])
        if 0 <= val <= 10:
            return val / 10.0
        elif 10 < val <= 100:
            return val / 100.0

    return None


def parse_cot_response(raw: str):
    """
    GPT-4o-mini CoT parser. Uses same AI_SCORE format as zero-shot.
    Verdict: ai/llm → 'llm', human → 'human'.
    """
    conf = parse_ai_score(raw)

    raw_lower = raw.lower()
    m_v = re.search(r'verdict:\s*(ai|llm|human)', raw_lower)
    if m_v:
        verdict = 'llm' if m_v.group(1) in ['ai', 'llm'] else 'human'
    else:
        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        last  = lines[-1].lower() if lines else ''
        if 'ai' in last or 'llm' in last:
            verdict = 'llm'
        elif 'human' in last:
            verdict = 'human'
        else:
            verdict = 'unknown'

    if conf is not None and verdict != 'unknown':
        if conf > 0.75 and verdict == 'human':
            verdict = 'llm'
        elif conf < 0.25 and verdict == 'llm':
            verdict = 'human'

    return verdict, conf


def score_gpt(client, eval_df, pool_df, regime, ds_name,
              model_id, results_dir, k_shot=5,
              eval_sample_size=200, cot_sample_size=50):
    """
    Scores one (regime, dataset) pair against GPT-4o-mini.
    Saves checkpoint after every call — interrupt-safe.
    Auto-resumes from checkpoint if re-run.
    """
    from src.detectors.llm_detector.prompts import (
        build_zero_shot, build_few_shot, build_cot
    )
    global tokens_used_total

    ckpt_path = f"{results_dir}/ckpt_GPT4oMini_{regime}_{ds_name}.json"

    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            records = json.load(f)
        done = {r["idx"] for r in records}
        print(f"    Resuming from checkpoint: {len(done)}/{len(eval_df)} done")
    else:
        records, done = [], set()

    session_tokens = 0

    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df),
                         desc=f"{regime}/{ds_name}", leave=True):
        if idx in done:
            continue

        text     = str(row["text"])
        true_lbl = row["label"]

        if regime == "zero_shot":
            messages = build_zero_shot(text)
            max_tok  = 180
        elif regime == "few_shot":
            messages = build_few_shot(text, pool_df, k=k_shot)
            max_tok  = 180
        elif regime == "cot":
            messages = build_cot(text)
            max_tok  = 600
        else:
            continue

        result         = call_gpt(client, messages, model_id=model_id,
                                  max_tokens=max_tok)
        session_tokens += result["usage"]
        raw_text       = result["raw_text"]

        cot_verdict = None
        cot_conf    = None

        if regime in ("zero_shot", "few_shot"):
            score = parse_ai_score(raw_text)
            if score is None:
                raw_lower = raw_text.lower().strip()
                if any(w in raw_lower for w in
                       ['ai-generated', 'ai generated',
                        'generated by', 'llm', 'language model']):
                    score = 0.85
                elif 'human' in raw_lower:
                    score = 0.15
                else:
                    score = 0.5

        elif regime == "cot":
            verdict, conf = parse_cot_response(raw_text)
            cot_verdict   = verdict
            cot_conf      = float(conf) if conf is not None else None
            if conf is not None:
                score = conf
            else:
                score = (1.0 if verdict == 'llm'
                         else 0.0 if verdict == 'human'
                         else 0.5)

        pred_lbl = 'llm' if score >= 0.45 else 'human'
        correct  = int(pred_lbl == true_lbl)

        failure_mode = (
            "correct"        if pred_lbl == true_lbl else
            "false_positive" if pred_lbl == "llm" and true_lbl == "human"
            else "false_negative" if pred_lbl == "human" and true_lbl == "llm"
            else "unknown_output"
        )

        records.append({
            "idx":              idx,
            "true_label":       true_lbl,
            "pred_label":       pred_lbl,
            "score":            float(score),
            "correct":          correct,
            "regime":           regime,
            "dataset":          ds_name,
            "text_preview":     text[:300],
            "raw_model_output": raw_text,
            "tokens_used":      result["usage"],
            "failure_mode":     failure_mode,
            "cot_verdict":      cot_verdict,
            "cot_conf":         cot_conf,
        })

        with open(ckpt_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"    ✅ Done — session tokens: {session_tokens:,}  "
          f"cumulative: {tokens_used_total:,}")
    return __import__('pandas').DataFrame(records)