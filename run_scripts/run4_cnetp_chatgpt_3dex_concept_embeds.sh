#!/bin/bash --login

#SBATCH --job-name=cnetp_chatgpt_dex_con_embeds

#SBATCH --output=logs/mention_encoder/out_cnetp_chatgpt_dex_con_embeds.txt
#SBATCH --error=logs/mention_encoder/err_cnetp_chatgpt_dex_con_embeds.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10G
#SBATCH --gres=gpu:1

#SBATCH -t 0-6:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/16_for_mention_encoder_get_con_vocab_embed_entropy_bert_large_cnetp_chatgpt100k_pt_model.json

echo 'finished!'