#!/bin/bash --login

#SBATCH --job-name=rbCTX3pt

#SBATCH --output=logs/pretrain/out_ctx3_hp_roberta_base_je_pretrain_mask_id_con_prop_cnetp.txt
#SBATCH --error=logs/pretrain/err_ctx3_hp_roberta_base_je_pretrain_mask_id_con_prop_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

#SBATCH --qos="gpu7d"

#SBATCH -t 7-00:00:00

conda activate venv

python3 model/lm_con_prop.py --config_file configs/1_configs_pretrain/ctx3_hp_roberta_base_je_pretrain_mask_id_con_prop_cnetp.json

echo 'Job Finished!'
