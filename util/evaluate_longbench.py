import json
import sys
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", type=str, required=True, help="Directory containing the data files")
args = parser.parse_args()

input_path = os.path.join(args.work_dir, "output", "instruction_inference_results.json")
output_path = os.path.join(args.work_dir, "output", "longbench_eval_results.json")

sys.setrecursionlimit(10000)

# 初始化存储F1, BLEU-4和EM分数的列表
metrics = ['rouge-f1', 'bleu4', "exact_match"]
datasets = ["2wikimqa", "gov_report", "hotpotqa", "multi_news", "multifieldqa_en", "musique", "narrativeqa", "qasper", "qmsum"]
results = {dataset: {metric: [] for metric in metrics} for dataset in datasets}
total_results = {metric: [] for metric in metrics}

# 读取数据并计算各项指标
with open(input_path, 'r') as file:
    data = json.load(file)
    for example in data:
        dataset = example["dataset"]
        if dataset in datasets:
            for metric in metrics:
                results[dataset][metric].append(example[metric])
                total_results[metric].append(example[metric])

# 计算平均分数并打印结果
iid_results = {}
for dataset in datasets:
    for metric in metrics:
        avg_score = np.mean(results[dataset][metric])
        print(f"{dataset}_{metric}: {avg_score}")
        iid_results[f"{dataset}_{metric}"] = avg_score.item()

for metric in metrics:
    avg_total_score = np.mean(total_results[metric])
    print(f"total_{metric}: {avg_total_score}")
    iid_results[f"total_{metric}"] = avg_total_score.item()
    iid_results[f"iid_test_samples_num_{metric}"] = len(total_results[metric])

# 写入结果到输出文件
with open(output_path, 'w') as json_file:
    json.dump(iid_results, json_file, indent=4)
