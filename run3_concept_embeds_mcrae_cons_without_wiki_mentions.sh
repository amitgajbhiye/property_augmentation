#!/bin/bash --login

#SBATCH --job-name=McRaeEmb

#SBATCH --output=logs/mention_encoder/out_concept_embeds_mcrae_cons_without_wiki_mentions.txt
#SBATCH --error=logs/mention_encoder/err_concept_embeds_mcrae_cons_without_wiki_mentions.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10G
#SBATCH --gres=gpu:1

#SBATCH -t 0-1:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/15_for_mention_concept_embeds_mcrae_cons_without_wiki_mentions.json

echo 'finished!'