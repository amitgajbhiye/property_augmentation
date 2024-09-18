#!/bin/bash --login

#SBATCH --job-name=mscgEmbeds

#SBATCH --output=logs/mscg_100k/out_mscg_con_prop_and_vocab.txt
#SBATCH --error=logs/mscg_100k/err_mscg_con_prop_and_vocab.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:1

#SBATCH -t 0-03:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mscg_100k/mscg_get_con_embeddings.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mscg_100k/mscg_get_prop_embeddings.json 
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mscg_100k/mscg_prop_vocab_embeddings.json

echo 'finished!'
