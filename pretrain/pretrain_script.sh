
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

#python pre_prepare_data.py --work_dir '../experiment/local_experiment/ICAE_Llama-3.2-1B_DPL_AEweight-0.75'
#python ./pre_trainer.py --work_dir '../experiment/rebuttal/DAST/15x_DAST' --port 14571
#python ./pre_evaluator.py --work_dir  '../experiment/rebuttal/DAST/15x_DAST'  --batch_size 1
cd ..
cd sft
python ./instruction_trainer.py --work_dir  '../experiment/rebuttal/DAST/15x_DAST' --port 14527
python ./instruction_evaluator.py --work_dir  '../experiment/rebuttal/DAST/15x_DAST' --batch_size 1
# /mnt/zhaorunsong/lx/RARC/experiment/rebuttal/DAST/15x_DAST
# nohup bash pretrain_script.sh > bash.log 2>&1 &
# 后台启动方式
# nohup python ./instruction_trainer.py --work_dir   '../experiment/main_experiment/ICAE_EPL_1B_SingleChunk' --port 14527 > train.log 2>&1 &
# tail -f bash.log
# /mnt/zhaorunsong/lx/RARC/experiment/rebuttal/DAST/15x_DAST