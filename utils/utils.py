from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    set_seed,
    pipeline,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)
from datasets import load_dataset
from torch import nn
import torch
from pathlib import Path
import yaml
import random
import copy
from torch.utils.data import ConcatDataset, Subset
from reward_model import GPT2RewardModel
from distutils.util import strtobool
import json


def get_dataset_name_and_kwargs_from_data_config(data_config):
    if isinstance(data_config, dict):
        name = list(data_config.keys())[0]

        # first copy the dict, then remove the size and fraction
        kwargs = copy.deepcopy(data_config[name])

        kwargs.pop("fraction", None)
        kwargs.pop("size", None)
        return name, kwargs
    else:
        return data_config, {}


def get_dataset(
    args,
    tokenizer
) -> tuple[ConcatDataset, dict[str, Subset]]:
    train_datasets, evals = [], {}

    for data_config in args.datasets:
        dataset_name, kwargs = get_dataset_name_and_kwargs_from_data_config(data_config)
        train, val = get_one_dataset(args, dataset_name, tokenizer)
        train_datasets.append(train)

        if val is not None:
            evals[dataset_name] = Subset(val, list(range(min(len(val), args.eval_size)))) if args.eval_size else val

    train = ConcatDataset(train_datasets)
    return train, evals


def get_one_dataset(
    args, 
    dataset_name, 
    tokenizer
):
    if dataset_name == "sst2":
        dataset = load_dataset("sst2")
        dataset = dataset.rename_columns({"label": "labels", "sentence": "text"})
        
        columns = dataset['train'].column_names
        columns_to_keep = ["text", "labels"]
        dataset = dataset.remove_columns(list(set(columns)-set(columns_to_keep)))
        
        def tokenize_dataset(examples):
            # remove the space at the end of each sentence
            return tokenizer([e[:-1] for e in examples["text"]], truncation=True, max_length=args.max_length)
        
        dataset = dataset.map(tokenize_dataset, batched=True)
        train, eval = dataset['train'], dataset['validation']
        
    elif dataset_name == "amazon_polarity":
        dataset = load_dataset("amazon_polarity")
        dataset = dataset.rename_columns({"label": "labels", "content": "text"})
        
        columns = dataset['train'].column_names
        columns_to_keep = ["text", "labels"]
        dataset = dataset.remove_columns(list(set(columns)-set(columns_to_keep)))
        
        def tokenize_dataset(examples):
            return tokenizer(examples["text"], truncation=True, max_length=args.max_length)
        
        dataset = dataset.map(tokenize_dataset, batched=True)
        train, eval = dataset['train'], dataset['test']
    
    elif dataset_name == "jigsaw_unintended_bias":
        dataset = load_dataset("jigsaw_unintended_bias", data_dir=args.jigsaw_dir)
        columns = dataset['train'].column_names
        columns_to_keep = [
            "comment_text", "target", "severe_toxicity", "obscene",
            "identity_attack", "insult", "threat", "sexual_explicit"
        ]
        dataset = dataset.remove_columns(list(set(columns)-set(columns_to_keep)))
        dataset = dataset.map(
            lambda example: {"labels": [example["target"],
                                        example["severe_toxicity"],
                                        example["obscene"],
                                        example["identity_attack"],
                                        example["insult"],
                                        example["threat"],
                                        example["sexual_explicit"]]},
            remove_columns=columns_to_keep[1:]  # keep "comment_text" and "labels" only
        )
        dataset = dataset.rename_columns({"comment_text": "text"})
            
        def tokenize_dataset(examples):
            return tokenizer(examples["text"], truncation=True, max_length=args.max_length)
        
        dataset = dataset.map(tokenize_dataset, batched=True)
        train, eval = dataset['train'], dataset['test_public_leaderboard']
        
    return train, eval


def prepare_lm(model_name):

    if model_name == "gpt2-large":
        lm_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        lm = AutoModelForCausalLM.from_pretrained("gpt2-large", device_map='balanced_low_0')
        max_length = 1024
    elif model_name == 'gpt-neox-20b':
        lm_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        # max_memory={i: "32GiB" for i in range(4)}       # assume 4 GPUs
        # max_memory[0] = "16GiB" 
        # configuration = GPTNeoXConfig()
        # with init_empty_weights():
        #     model = GPTNeoXForCausalLM(configuration)
        #     device_map = infer_auto_device_map(model, no_split_module_classes=["GPTNeoXLayer"], max_memory=max_memory)
        # device_map['embed_out'] = device_map['gpt_neox.embed_in']       # put output layer on the same device as the input layer
        # lm = GPTNeoXForCausalLM.from_pretrained(
        #     "EleutherAI/gpt-neox-20b", device_map=device_map, torch_dtype=torch.float16)
        lm = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/gpt-neox-20b", device_map='balanced_low_0', torch_dtype=torch.float16)
        max_length = 2048
    
    elif "llama" in model_name or "Llama" in model_name:
        model_name = f"meta-llama/{model_name}"
        lm_tokenizer = LlamaTokenizer.from_pretrained(model_name)
        lm = LlamaForCausalLM.from_pretrained(
            model_name, device_map='balanced_low_0', torch_dtype=torch.bfloat16)
        max_length = 4096
        
    # set pad_token_id to eos_token_id because GPT2/Llama does not have a PAD token
    lm_tokenizer.pad_token = lm_tokenizer.eos_token
    lm_tokenizer.padding_side = 'left'                  # left padding while generating
    lm.config.pad_token_id = lm.config.eos_token_id
    
    return lm, lm_tokenizer, max_length 


def get_rm_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.max_length = args.max_length
    return tokenizer


def get_reward_model(args):
    if "gpt2" in args.reward_model_name:
        model = GPT2RewardModel(
            reward_model_name=args.reward_model_name,
            out_features=args.out_features, 
            loss_fn=args.loss_fn
        )
    return model


def _strtobool(x):
    return bool(strtobool(x))


def read_yamls(dir):
    args = {}
    no_conf = True

    for config_file in Path(dir).glob("**/*.yaml"):
        no_conf = False
        with config_file.open("r") as f:
            args.update(yaml.safe_load(f))

    if no_conf:
        print(f"WARNING: No yaml files found in {dir}")

    return args