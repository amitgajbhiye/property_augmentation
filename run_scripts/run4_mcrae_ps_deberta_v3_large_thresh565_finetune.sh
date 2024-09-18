#!/bin/bash --login

#SBATCH --job-name=dbv3MCFT

#SBATCH --output=logs/mcrae_finetune/out_ps_mcrae_prop_conj_deberta_v3_large_je_fintune_thresh565.txt
#SBATCH --error=logs/mcrae_finetune/err_ps_mcrae_prop_conj_deberta_v3_large_je_fintune_thresh565.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu

#SBATCH --mem=0
#SBATCH --gres=gpu:1

#SBATCH --exclusive
#SBATCH --time 0-10:00:00

conda activate venv

python3 model/lm_prop_conj.py --config_file configs/3_finetune/ps_mcrae_prop_conj_deberta_v3_large_je_fintune_thresh565.json

echo 'Job Finished!'
