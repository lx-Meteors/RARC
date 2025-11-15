import os
import json



def get_default_input(dataset_name):
    mapping = {
        "2wikimqa": "",
        "gov_report": "You are given a report by a government agency.  Write a one-page summary of the report.",
        "hotpotqa": "",
        "multi_news": "You are given several news passages. Write a one-page summary of all news. ",
        "multifieldqa_en": "",
        "musique": "",
        "narrativeqa": "",
        "qasper": "",
        "qmsum": "",
    }
    return mapping.get(dataset_name, "input_text")

data_root = "/mnt/zhaorunsong/dataset/longbench_lx_test"    # 存放 jsonl 的目录

# 你想加载并处理的数据集名字
datasets_to_load = ["2wikimqa", "gov_report", "hotpotqa", "multi_news", "multifieldqa_en", "musique", "narrativeqa", "qasper", "qmsum"]

loaded_data = {}

for dataset in datasets_to_load:
    file_path = os.path.join(data_root, f"{dataset}.jsonl")

    if not os.path.exists(file_path):
        print(f"[Warning] 文件不存在: {file_path}")
        continue

    items = []
    # ===== 读取阶段 =====
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            # --- 数据必须是 dict ---
            if not isinstance(item, dict):
                print(f"[跳过] 格式错误（不是 dict）：{item}")
                continue

            # input 为空时赋值
            if not item.get("input") or str(item["input"]).strip() == "":
                item["input"] = get_default_input(dataset)

            items.append(item)

    loaded_data[dataset] = items
    print(f"[Loaded] {dataset}: {len(items)} 条数据")

    # ===== 保存阶段（覆盖原文件）=====
    save_path = os.path.join(data_root, f"{dataset}.jsonl")
    with open(save_path, "w", encoding="utf-8") as fout:
        for item in items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ===== 保存阶段（覆盖原文件）=====
    save_path = os.path.join(data_root, "longbench_test_instruction_dataset.jsonl")
    with open(save_path, "a", encoding="utf-8") as fout:
        for item in items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Saved] {dataset} 已回写到 {save_path}")

print("\n[Done] 所有选择的数据集已处理并保存。")
