#!/bin/bash --login

#SBATCH --job-name=menEncVocab

#SBATCH --output=logs/definition_encoder/out_definition_enc_wordnet_codwoe_con_vocabembed_entropy_bert_large_cnetpchatgpt100k_entropy_model.txt
#SBATCH --error=logs/definition_encoder/err_definition_enc_wordnet_codwoe_con_vocabembed_entropy_bert_large_cnetpchatgpt100k_entropy_model.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-06:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/8_for_definition_encoder_get_con_vocab_embed_entropy_bert_large_cnetp_chatgpt100k_pt_model.json

echo 'Job Finished !!!'
