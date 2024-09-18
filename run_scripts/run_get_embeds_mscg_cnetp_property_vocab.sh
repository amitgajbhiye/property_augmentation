#!/bin/bash --login

#SBATCH --job-name=mscgCnetpPropVocabEmbeds

#SBATCH --output=logs/embeddings/out_get_embeds_property_vocab_mscg_cnetp.txt
#SBATCH --error=logs/embeddings/err_get_embeds_property_vocab_mscg_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/2_get_embeds_property_vocab_mscg_cnetp.json


echo 'Job Finished!'