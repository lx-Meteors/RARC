# python instruction_prepare_data.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_trainer.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL' --port 14527
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_evaluator.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL' --batch_size 1
# python ../util/evaluate_ood.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL'
# python ../util/evaluate_iid.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL'

# python instruction_prepare_data.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_trainer.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL' --port 14527
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_evaluator.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL' --batch_size 1
# python ../util/evaluate_ood.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL'
# python ../util/evaluate_iid.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL'


# python instruction_prepare_data.py --work_dir  '../experiment/debug/quick'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_trainer.py --work_dir  '../experiment/debug/quick' --port 14527
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_evaluator.py --work_dir  '../experiment/debug/quick' --batch_size 1
# python ../util/evaluate_ood.py --work_dir '../experiment/debug/quick'
# python ../util/evaluate_iid.py --work_dir '../experiment/debug/quick'


# python instruction_prepare_data.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_trainer.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL' --port 14527
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_evaluator.py --work_dir  '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL' --batch_size 1
# python ../util/evaluate_ood.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL'
# python ../util/evaluate_iid.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL'

#python instruction_prepare_data.py --work_dir  '../experiment/main_experiment/ICAE_EPL_1B_SingleChunk'
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python ./instruction_trainer.py --work_dir   '../experiment/main_experiment/ICAE_EPL_1B_SingleChunk' --port 14527 > train.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./instruction_evaluator.py --work_dir   '../experiment/main_experiment/ICAE_EPL_1B_SingleChunk' --batch_size 1
python ../util/evaluate_ood.py --work_dir   '../experiment/rebuttal/DAST/15x_DAST'
python ../util/evaluate_iid.py --work_dir   '../experiment/rebuttal/DAST/15x_DAST'

# 分析实验
CUDA_VISIBLE_DEVICES=1,2,3,4  nohup python ./instruction_trainer.py --work_dir   '../experiment/analysis_experiment/ICAE_EPL_1B_DynamicMemTokens' --port 14527 > train.log 2>&1 &
CUDA_VISIBLE_DEVICES=1,2,3,4 python ./instruction_evaluator.py --work_dir   '../experiment/analysis_experiment/ICAE_EPL_1B_DynamicMemTokens' --batch_size 1
# 后台启动方式
#nohup python ./instruction_trainer.py --work_dir   '../experiment/main_experiment/ICAE_EPL_1B_SingleChunk' --port 14527 > train.log 2>&1 &
# tail -f train.log

