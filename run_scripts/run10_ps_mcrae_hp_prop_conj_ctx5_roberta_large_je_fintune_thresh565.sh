#!/bin/bash --login

#SBATCH --job-name=hprlCtx5MCFT

#SBATCH --output=logs/mcrae_finetune/out_ps_mcrae_hp_prop_conj_ctx5_roberta_large_je_fintune_thresh565.txt
#SBATCH --error=logs/mcrae_finetune/err_ps_mcrae_hp_prop_conj_ctx5_roberta_large_je_fintune_thresh565.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu
#SBATCH --mem=0
#SBATCH --gres=gpu:1
##SBATCH --qos=gpu7d
#SBATCH --exclusive

#SBATCH --time 2-00:00:00

conda activate venv

python3 model/lm_prop_conj.py --config_file configs/3_finetune/ps_mcrae_hp_prop_conj_ctx5_roberta_large_je_fintune_thresh565.json

echo 'Job Finished!'
