import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import tqdm
import numpy as np


# takes in a EvalPrediction and returns a dictionary string to metric values.
def mse(eval_preds):
    predictions, labels = eval_preds    # (eval set size, 1)
    mse_metric = evaluate.load("mse")
    return mse_metric.compute(predictions=predictions, references=labels)


def distinctness(generations):
    unigrams, bigrams, trigrams = set(), set(), set()
    total_words = 0
    
    for gen in generations:
        o = gen.split(' ')
        total_words += len(o)
        unigrams.update(o)
        for i in range(len(o) - 1):
            bigrams.add(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
    
    return len(unigrams) / total_words, len(bigrams) / total_words, len(trigrams) / total_words


def compute_perplexity(args, generation, rad, device='cuda'):
    if "gpt2" in args.lm:
        model = AutoModelForCausalLM.from_pretrained('gpt2-xl', device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
    
    else:   # use lm itself for ppl evaluation
        model = rad._lm
        tokenizer = rad._lm_tokenizer
        
    perplexities = []
    
    pbar = tqdm(generation, total=len(generation), desc='Evaluate Fluency')
    for row in pbar:
        prompt = row['prompt']['text']
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.inference_mode():
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0]
            prompt_loss *= (prompt_input_ids.shape[1]-1)
            
            for cont in row['generations']:
                cont = cont['text']
                full_input_ids = tokenizer.encode(prompt+cont, return_tensors='pt').to(device)
                full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
                loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])
                ppl = torch.exp(loss).item()
                
                if ppl < 1e5:
                    perplexities.append(ppl)
                    
        pbar.set_description(
            f'mean ppl = {np.mean(perplexities):.3f}'
        )
        
    return perplexities