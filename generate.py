import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
import argparse
from reward_modeling.reward_model import GPT2RewardModel
from utils.utils import prepare_lm, get_rm_tokenizer
from utils.metrics import distinctness, compute_perplexity
import torch
from rad import RewardAugmentedDecoder
from tqdm.auto import tqdm
import numpy as np
import json


def generate_on_prompts(args, rad, eval_prompts):

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    dist_n = []
    generation = []
    report = {}

    if args.test:
        eval_prompts = eval_prompts[:100]
        
    eval_prompt_chunks = list(chunks(eval_prompts, args.batch_size))

    pbar = tqdm(eval_prompt_chunks)
    for chunk in pbar:
        with torch.inference_mode():
            generated_texts = rad.sample(
                chunk,
                max_new_tokens=args.max_new_tokens,
                topk=args.topk,
                beta=args.beta,
                num_return_sequences=args.num_return_sequences,
            )
            
        for i, samples in enumerate(generated_texts):
            dist_n.append(distinctness(samples))
            generation.append({
                'prompt': {"text": chunk[i]},
                'generations': 
                    [{"text": sp} for sp in samples]
            })

        pbar.set_description(
            f'dist-n = {["{:.3f}".format(x) for x in np.nanmean(np.array(dist_n), axis=0)]}'
        )

    ppl = compute_perplexity(args, generation, rad)
    
    report.update({
        'dist_n': np.nanmean(np.array(dist_n), axis=0).tolist(),
        "perplexity": np.mean(ppl)
    })
    
    return report, generation


def load_rad(args):
    lm, lm_tokenizer, max_length = prepare_lm(args.lm)

    # rm
    if args.rm == 'gpt2':
        rm_tokenizer = AutoTokenizer.from_pretrained(args.rm)
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        rm_tokenizer.padding_side = 'right'
        rm_tokenizer.max_length = 1024
        
        rm = GPT2RewardModel(reward_model_name=args.rm, out_features=1)
        
        state_dict = torch.load(args.rm_dir)
        rm.load_state_dict(state_dict)
        rm = rm.to('cuda')

    rad = RewardAugmentedDecoder(
        lm, 
        lm_tokenizer, 
        rm, 
        rm_tokenizer, 
        max_length, 
        num_gpus=torch.cuda.device_count(),
        inverse=args.inverse)
    return rad


# ADD CUSTOM DATASET HERE
def load_dataset(args) -> list[str]:
    prompts = []
    if args.dataset == 'negative':
        file_dir = "datasets/sentiment_prompts-10k/negative_prompts.jsonl"
    elif args.dataset == 'neutral':
        file_dir = "datasets/sentiment_prompts-10k/neutral_prompts.jsonl"
    elif args.dataset == 'positive':
        file_dir = "datasets/sentiment_prompts-10k/positive_prompts.jsonl"
    with open(file_dir) as f:
        for line in f:
            prompts.append(json.loads(line)['prompt']['text'])
    return prompts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs/")
    parser.add_argument("--dataset", choices=['negative','neutral','positive'], default='negative')
    
    parser.add_argument("--beta", default=10, type=int)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--inverse", default=False, type=bool)      # steer toward lower reward
    
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--num_return_sequences", default=1, type=int)
    parser.add_argument("--max_new_tokens", default=20, type=int)
    
    parser.add_argument("--lm", default="gpt2-large", choices=
        ["gpt2-large","gpt-neox-20b","Llama-2-7b-hf", "Llama-2-13b-hf", "Llama-2-70b-hf"])
    parser.add_argument("--rm", default="gpt2", choices=["gpt2"])
    parser.add_argument("--rm_dir", default="reward_modeling/saved_models/gpt2_sentiment/pytorch_model.bin")
    
    parser.add_argument("--test", default=False, type=bool)
    
    args = parser.parse_args()
    return args


def main(args):
    set_seed(1)
    prompts = load_dataset(args)
    rad = load_rad(args)
    results, generation = generate_on_prompts(args, rad, prompts)
    
    with open(
        os.path.join(
            args.outdir,
            f'custom_task_report_{args.lm}_{args.rm}_top{args.topk}_beta{args.beta}_{args.dataset}.json'
        ), 'w'
    ) as f:
        json.dump(results, f)
    
    with open(
        os.path.join(
            args.outdir,
            f'custom_task_generation_{args.lm}_{args.rm}_top{args.topk}_beta{args.beta}_{args.dataset}.jsonl'
        ), 'w'
    ) as f:
        for entry in generation:
            json.dump(entry, f)
            f.write("\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
