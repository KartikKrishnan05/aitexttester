import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import logging as hf_logging

# --- SILENCE WARNINGS ---
# This tells the library: "Don't show me warnings, only show me errors."
hf_logging.set_verbosity_error()

def load_model():
    print("Loading model...")
    model_id = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    print("Model loaded successfully!")
    return model, tokenizer

def calculate_perplexity(text, model, tokenizer):
    # 1. Clean input
    text = text.strip()
    if not text:
        return 0.0

    encodings = tokenizer(text, return_tensors='pt')
    
    # 2. Safety check for short text
    if encodings.input_ids.size(1) < 2:
        print("Text is too short to analyze.")
        return 0.0

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        
        if begin_loc > 0:
            target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            
            # 3. Handle Math Errors (NaN)
            if torch.isnan(outputs.loss):
                return 999.0
                
            nlls.append(outputs.loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if not nlls:
        return 0.0

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

if __name__ == "__main__":
    model, tokenizer = load_model()
    
    while True:
        print("\n" + "="*40)
        user_input = input("Paste text to check (or type 'exit'):\n")
        
        if user_input.lower() in ['exit', 'quit']:
            break
            
        score = calculate_perplexity(user_input, model, tokenizer)
        
        # Interpretation
        if score > 1000:
             print(f"\nPerplexity Score: High (Math Overflow)")
             print("Prediction: Likely Human (Complex or chaotic text)")
        else:
            print(f"\nPerplexity Score: {score:.2f}")
            if score < 35:
                print("Prediction: Likely AI (Low Perplexity)")
            else:
                print("Prediction: Likely Human (High Perplexity)")