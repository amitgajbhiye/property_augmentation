#!/bin/bash --login

#SBATCH --job-name=mcFTpsMCSGcnetp

#SBATCH --output=logs/mcrae_finetune/out_roberta_base_mask_id_ctx1_con_prop_mscg_cnetp_mcrae_finetune.txt
#SBATCH --error=logs/mcrae_finetune/err_roberta_base_mask_id_ctx1_con_prop_mscg_cnetp_mcrae_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu
#SBATCH --mem=9g
#SBATCH --gres=gpu:1

##SBATCH --qos="gpu7d"

#SBATCH -t 0-05:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 model/roberta_con_prop.py --config_file configs/3_finetune/roberta_base_mask_id_ctx1_con_prop_mscg_cnetp_mcrae_finetune.json

echo 'Job Finished!'