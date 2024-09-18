#!/bin/bash --login

#SBATCH --job-name=mscgCnetConPropEmnbeds

#SBATCH --output=logs/embeddings/out_get_embeds_mscg_cnetp_concepts_and_properties.txt
#SBATCH --error=logs/embeddings/err_get_embeds_mscg_cnetp_concepts_and_properties.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:1

#SBATCH -t 0-02:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/1_get_embeds_mscg_cnetp_concepts.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/3_get_embeds_mscg_cnetp_properties.json

echo 'Job Finished!'