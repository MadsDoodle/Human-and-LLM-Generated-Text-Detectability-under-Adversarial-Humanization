import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = os.getenv('HF_TOKEN')

def create_prompt(question):
    """Simple, neutral prompt"""
    prompt = f"""<s>[INST] Answer the following question clearly and concisely.

Question: {question}

Answer: [/INST]"""
    return prompt

def generate_batch_answers(questions, batch_size, max_new_tokens):
    """Generate answers in batches"""
    answers = []

    # Change this line:
    for i in tqdm(range(0, len(questions), batch_size),
                  desc="Generating",
                  position=0,           # ADD THIS
                  leave=True):          # ADD THIS
        batch_questions = questions[i:i+batch_size]
        prompts = [create_prompt(q) for q in batch_questions]

        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONFIG["MAX_INPUT_LENGTH"]
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=CONFIG["TEMPERATURE"],
                top_p=CONFIG["TOP_P"],
                do_sample=CONFIG["DO_SAMPLE"],
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode (remove prompt)
        for j, output in enumerate(outputs):
            prompt_length = inputs['input_ids'][j].shape[0]
            generated_tokens = output[prompt_length:]
            answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            answers.append(answer.strip())

    return answers

if __name__ == '__main__':
    df = pd.read_csv('./data/hc3_paired_dataset.csv')
    # load model, call generate_batch_answers, save output