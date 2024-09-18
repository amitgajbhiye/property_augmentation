#!/bin/bash --login

#SBATCH --job-name=getEmbeds

#SBATCH --output=logs/mcrae_clean_vocab_data/out_mcrae_get_embeds_clean_data.txt
#SBATCH --error=logs/mcrae_clean_vocab_data/err_mcrae_get_embeds_clean_data.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu
##SBATCH --gres=gpu:1


#SBATCH -p compute
#SBATCH --mem=8G
#SBATCH -t 0-02:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mcrae_get_con_embeddings.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mcrae_get_prop_embeddings.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/get_clean_prop_vocab_embeddings.json

echo 'Job Finished!'