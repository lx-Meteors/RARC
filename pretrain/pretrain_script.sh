
# python pre_prepare_data.py --work_dir '../experiment/no-pe'
# python ./pre_trainer.py --work_dir '../experiment/no-pe' --port 14525
# python ./pre_evaluator.py --work_dir '../experiment/no-pe' --batch_size 1

# python ./pre_trainer.py --work_dir '../experiment/you-pe' --port 14526
# python ./pre_evaluator.py --work_dir '../experiment/you-pe' --batch_size 1


# python pre_prepare_data.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pre_trainer.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL' --port 14574
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pre_evaluator.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL' --batch_size 1


# python pre_prepare_data.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pre_trainer.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL' --port 14574
# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./pre_evaluator.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_UPL' --batch_size 1

# nohup python pre_prepare_data.py --work_dir '../experiment/analysis_experiment/RARC_1B_MultiChunk' > pre_evaluator.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python ./pre_trainer.py --work_dir '../experiment/analysis_experiment/RARC_1B_Only_KVCache' --port 14572
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./pre_evaluator.py --work_dir '../experiment/analysis_experiment/RARC_1B_Only_KVCache' --batch_size 1
#
cd ..
cd sft
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./instruction_trainer.py --work_dir   '../experiment/analysis_experiment/RARC_1B_Only_KVCache' --port 14527
CUDA_VISIBLE_DEVICES=4,5,6,7 python ./instruction_evaluator.py --work_dir   '../experiment/analysis_experiment/RARC_1B_Only_KVCache' --batch_size 1

# nohup bash pretrain_script.sh  > bash.log 2>&1 &
# 后台启动方式
#nohup python ./pre_trainer.py --work_dir '../experiment/icae_15x_upl_sure' --port 14571 > train.log 2>&1 &
# tail -f bash.log
# tail -f RARC_1B_MultiChunk.log
