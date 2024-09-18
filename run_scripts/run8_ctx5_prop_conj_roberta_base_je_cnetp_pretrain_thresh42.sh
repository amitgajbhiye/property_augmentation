#!/bin/bash --login

#SBATCH --job-name=PreRBCtx5

#SBATCH --output=logs/pretrain/out_ctx5_prop_conj_roberta_base_je_cnetp_pretrain_thresh42.txt
#SBATCH --error=logs/pretrain/err_ctx5_prop_conj_roberta_base_je_cnetp_pretrain_thresh42.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu

#SBATCH --mem=10G
#SBATCH --gres=gpu:1

#SBATCH --time 2-00:00:00

conda activate venv

python3 model/lm_prop_conj.py --config_file configs/1_configs_pretrain/ctx5_prop_conj_roberta_base_je_cnetp_pretrain.json


echo 'Job Finished!'
