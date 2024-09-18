#!/bin/bash --login

#SBATCH --job-name=rbConPropPretrain

#SBATCH --output=logs/pretrain/out_roberta_base_je_pretrain_mask_id_ctx1_con_prop_mscg_cnetp.txt
#SBATCH --error=logs/pretrain/err_roberta_base_je_pretrain_mask_id_ctx1_con_prop_mscg_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu
#SBATCH --mem=9g
#SBATCH --gres=gpu:1

##SBATCH --qos="gpu7d"

#SBATCH -t 2-00:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 model/roberta_con_prop.py --config_file configs/1_configs_pretrain/roberta_base_je_pretrain_mask_id_ctx1_con_prop_mscg_cnetp.json

echo 'Job Finished!'