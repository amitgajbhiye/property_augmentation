#!/bin/bash --login

#SBATCH --job-name=DefMenEncVocabEntropy

#SBATCH --output=logs/mention_encoder/out_for_mention_definition_encoder_mscgcnetpchatgpt_entropy_model.txt
#SBATCH --error=logs/mention_encoder/err_for_mention_definition_encoder_mscgcnetpchatgpt_entropy_model.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-01:00:00

conda activate venv

# Mention Embeds
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/13_for_mention_encoder_get_con_vocab_embed_entropy_bert_large_mscg_cnetp_chatgpt100k_pt_model.json

# Definition Embeds
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/14_for_definition_encoder_get_con_vocab_embed_entropy_bert_large_mscg_cnetp_chatgpt100k_pt_model.json


echo 'Job Finished !!!'
