import sys
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AlbertTokenizer
import torch
from torch import nn
import os
import random
from tqdm import tqdm
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default= '../experiment/analysis_experiment/RARC_1B_MultiChunk', required=False, help='Directory including the configuration file')
    return parser.parse_args()


def get_long_text_list(dataset_repo, output_dir, min_len, max_len):
    # cache long text for preventing full dataset traversal on each preparation. 
    if os.path.exists(f'{output_dir}/long_text.json'):
        with open(f'{output_dir}/long_text.json', 'r', encoding='utf-8') as f:
            long_text_list =  json.load(f)
        return long_text_list

    dataset = load_dataset(dataset_repo, split="train", streaming=True)

    long_text_list = []
    for example in tqdm(dataset, desc="Processing examples"):
        if 0 <= len(example["text"]) <= max_len*6:  # one token \approx 2~6 char, here filter very long and very short text
            long_text_list.append(example["text"])
        
    with open(f'{output_dir}/long_text.json', 'w', encoding='utf-8') as f:
        json.dump(long_text_list, f, ensure_ascii=False)

    return long_text_list

    

def get_examples(model_id, dataset_repo, samples_num, min_len, max_len, instruction_dataset_repo, output_dir):
    model_name = model_id.split('/')[-1]
    train_data_name = f"{output_dir}/train_"+model_name+"_"+str(samples_num)+f"samples_{min_len}-{max_len}len_albert.pt"
    eval_data_name = f"{output_dir}/eval_"+model_name+"_"+str(samples_num)+f"samples_{min_len}-{max_len}len_albert.pt"

    if os.path.exists(train_data_name):
        print("loading data...")
        return torch.load(train_data_name), torch.load(eval_data_name)
    print(f"preparing data :train_data_name:{train_data_name}")

    tokenizer_albert = AlbertTokenizer.from_pretrained("/mnt/zhaorunsong/models/albert-xlarge-v2")
    tokenizer_llama = AutoTokenizer.from_pretrained(model_id)
    
    long_text_list = get_long_text_list(dataset_repo, output_dir, min_len, max_len)

    examples = []
    for text in tqdm(long_text_list, desc="Processing examples"):

        half1, half2 = split_in_half(text)
        half1 = tokenizer_albert(half1, add_special_tokens=False)["input_ids"]
        half2 = tokenizer_llama(half2, add_special_tokens=False)["input_ids"]

        if (len(half1)+len(half2))<min_len:
            continue
        if (len(half1)+len(half2))>max_len:
            continue

        # half for prefix, half for LM
        # last_start = len(ids) // 2

        inputs = [tokenizer_albert.bos_token_id] + half1
        ae_target = inputs + [tokenizer_llama.eos_token_id]
        lm_target = half2 + [tokenizer_llama.eos_token_id]
        
        inputs = torch.LongTensor(inputs)
        ae_target = torch.LongTensor(ae_target)
        lm_target = torch.LongTensor(lm_target)
        examples.append({"inputs":inputs, "ae_target":ae_target, "lm_target":lm_target})

        if len(examples) == samples_num+1000:
            break

    # 1k for validation
    torch.save(examples[1000:], train_data_name)
    torch.save(examples[:1000], eval_data_name)
    
    return examples[1000:], examples[:1000]

def split_in_half(s):
    length = len(s)
    mid = (length + 1) // 2  # 处理奇数长度时前半部分多1个字符
    first_half = s[:mid]
    second_half = s[mid:]
    return first_half, second_half



    
if __name__ == "__main__":

    args = parse_args()
    with open(args.work_dir+"/config.json") as f:
        config=json.load(f)
    
    training_config = config["pretrain_training_config"]
    config["data_config"]["model_id"] = training_config["model_id"]

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config["data_config"]["output_dir"] = output_dir

    train_examples, eval_examples = get_examples(**config["data_config"])

"""
cd pretrain
python pre_prepare_data.py --work_dir '../experiment/debug/quick'

"""