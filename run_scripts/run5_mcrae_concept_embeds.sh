#!/bin/bash --login

#SBATCH --job-name=setup1_clean_mcrae_bienc_concept_embedding

#SBATCH --output=logs/mention_encoder/out_setup1_bienc_clean_mcrae_concept_embedding
#SBATCH --error=logs/mention_encoder/err_setup1_bienc_clean_mcrae_concept_embedding

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10G
#SBATCH --gres=gpu:1

#SBATCH -t 0-6:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/17_for_mention_encoder_get_mcrae_con_vocab_embed_entropy_bert_large_cnetp_chatgpt100k_pt_model.json

echo 'finished!'