#!/bin/bash --login

#SBATCH --job-name=4rlCtx5MCFT

#SBATCH --output=logs/mcrae_finetune/out_ps_mcrae_prop_conj_num_prop4_ctx5_roberta_large_je_fintune_clean_cslb.txt
#SBATCH --error=logs/mcrae_finetune/err_ps_mcrae_prop_conj_num_prop4_ctx5_roberta_large_je_fintune_clean_cslb.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu,gpu_v100
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

#SBATCH --time 0-6:00:00

conda activate venv

python3 model/lm_prop_conj.py --config_file configs/3_finetune/ps_mcrae_prop_conj_num_prop4_ctx5_roberta_large_je_fintune_clean_cslb.json


echo 'Job Finished!'
