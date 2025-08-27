import bisect
import re
import sys
import os
from collections import defaultdict

import math
from datasets import load_dataset
from llmlingua import PromptCompressor
from transformers import AutoTokenizer
import torch
from torch import nn
import os
import random
from tqdm import tqdm
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='../experiment/experiment_15x_8gpu/RARC_lingua2',
                        required=False, help='Directory including the configuration file')
    return parser.parse_args()


def get_long_text_list(dataset_repo, output_dir, min_len, max_len):
    # cache long text for preventing full dataset traversal on each preparation.
    if os.path.exists(f'{output_dir}/long_text.json'):
        with open(f'{output_dir}/long_text.json', 'r', encoding='utf-8') as f:
            long_text_list = json.load(f)
        return long_text_list

    dataset = load_dataset(dataset_repo, split="train", streaming=True)

    long_text_list = []
    for example in tqdm(dataset, desc="Processing examples"):
        if 0 <= len(example[
                        "text"]) <= max_len * 6:  # one token \approx 2~6 char, here filter very long and very short text
            long_text_list.append(example["text"])

    with open(f'{output_dir}/long_text.json', 'w', encoding='utf-8') as f:
        json.dump(long_text_list, f, ensure_ascii=False)

    return long_text_list

def map_compressed_words_to_blocks_indices(tokenizer, original_text, compressed_text, block_size=510):
    """
    将压缩文本按空格分词后，映射到原文 token 索引，并按块分组。
    找不到的词直接跳过。

    返回:
        block_indices: dict，key=块号，value=[块内索引, ...]
    """
    # 1. 原文 tokenize + offset mapping
    enc_orig = tokenizer(original_text, add_special_tokens=False, return_offsets_mapping=True)
    orig_token_ids = enc_orig["input_ids"]
    offsets = enc_orig["offset_mapping"]

    # 2. 压缩文本分词
    compressed_words = compressed_text.split()

    block_indices = defaultdict(list)
    last_end = 0

    for word in compressed_words:
        pos = original_text.find(word, last_end)
        if pos == -1:
            continue  # 找不到就跳过

        # 找到原文 token 对应索引
        found_idx = -1
        for idx, (s, e) in enumerate(offsets):
            if s <= pos < e:
                found_idx = idx
                break

        if found_idx != -1:
            block_id = found_idx // block_size
            intra_idx = found_idx % block_size
            block_indices[block_id].append(intra_idx)
            last_end = pos + len(word)

        # 4. 转成 tensor
    block_indices_tensor = {k: torch.tensor(v, dtype=torch.long) for k, v in block_indices.items()}
    lingua2_list = []
    for tensor in block_indices_tensor.values():
        # 在每个 tensor 末尾加 [128000]
        tensor_with_sep = torch.cat([tensor, torch.tensor([128000], dtype=torch.long)])
        lingua2_list.append(tensor_with_sep)
    if len(lingua2_list) == 0:
        lingua2_list.append(torch.tensor([128000], dtype=torch.long))
    lingua2 = torch.cat(lingua2_list, dim=0)
    return lingua2


def get_examples(model_id, dataset_repo, samples_num, min_len, max_len, instruction_dataset_repo, output_dir):
    model_name = model_id.split('/')[-1]
    train_data_name = f"{output_dir}/train_" + model_name + "_" + str(samples_num) + f"samples_{min_len}-{max_len}len_rarc_lingua2_51x.pt"
    eval_data_name = f"{output_dir}/eval_" + model_name + "_" + str(samples_num) + f"samples_{min_len}-{max_len}len_rarc_lingua2_51x.pt"

    if os.path.exists(train_data_name):
        print("loading data...")
        return torch.load(train_data_name), torch.load(eval_data_name)
    print(f"preparing data :train_data_name:{train_data_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm_lingua = PromptCompressor(model_name="/mnt/zhaorunsong/models/llmlingua-2-xlm-roberta-large-meetingbank",use_llmlingua2=True)

    long_text_list = get_long_text_list(dataset_repo, output_dir, min_len, max_len)

    examples = []
    for text in tqdm(long_text_list, desc="Processing examples"):

        ids = tokenizer(text, add_special_tokens=False)['input_ids']

        if len(ids) < min_len:
            continue
        if len(ids) > max_len:
            continue

        # half for prefix, half for LM
        last_start = len(ids) // 2

        inputs = [tokenizer.bos_token_id] + ids[:last_start]
        ae_target = inputs + [tokenizer.eos_token_id]
        lm_target = ids[last_start:] + [tokenizer.eos_token_id]

        org_text = tokenizer.decode(inputs, add_special_tokens=False)
        compressed_chunk_text = llm_lingua.compress_prompt(org_text, rate=0.02, target_token=10)["compressed_prompt"]
        mem_real_idx = map_compressed_words_to_blocks_indices(tokenizer, org_text, compressed_chunk_text)

        inputs = torch.LongTensor(inputs)
        ae_target = torch.LongTensor(ae_target)
        lm_target = torch.LongTensor(lm_target)


        examples.append({"inputs": inputs, "ae_target": ae_target, "lm_target": lm_target, "lingua2": mem_real_idx})

        if len(examples) == samples_num + 1000:
            break

    # 1k for validation
    torch.save(examples[1000:], train_data_name)
    torch.save(examples[:1000], eval_data_name)

    return examples[1000:], examples[:1000]


if __name__ == "__main__":

    args = parse_args()
    with open(args.work_dir + "/config.json") as f:
        config = json.load(f)

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