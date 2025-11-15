# longbench评测


# 修改config

- 将config的 "instruction_dataset_repo"："/mnt/zhaorunsong/dataset/bigData/mrqa-workshop_mrqa" 改成 "/mnt/zhaorunsong/dataset/bigData/longbench"
- 将处理好的longbench_test__instruction_dataset.jsonl放到sft/output下面(我发给你)
- 然后可以运行 instruction_prepare_data.py 处理数据
  - python instruction_prepare_data.py --work_dir  '../experiment/rebuttal/15x_8gpu/SAC_Long'