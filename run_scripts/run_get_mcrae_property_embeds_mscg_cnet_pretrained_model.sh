#!/bin/bash --login

#SBATCH --job-name=McPropEmbeds_mscg_cnet_pretrained_model

#SBATCH --output=logs/embeddings/out_mcrae_get_property_embeds_mscg_cnetp_pretraind_model.txt
#SBATCH --error=logs/embeddings/err_mcrae_get_property_embeds_mscg_cnetp_pretraind_model.txt

#SBATCH --tasks-per-node=1
#SBATCH --ntasks=1
#SBATCH -A scw1858

#SBATCH -p gpu
#SBATCH --mem=6g
#SBATCH --gres=gpu:1

#SBATCH -t 0-00:45:00

echo 'This script is running on:' hostname

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mcrae_get_property_embeds_mscg_cnetp_model.json

echo 'Job Finished!'
