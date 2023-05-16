#!/bin/bash --login

#SBATCH --job-name=cnetpChatgpt100k_UFET_type_prop_embeds

#SBATCH --output=logs/ufet_exp/out_ufet_cnetp_chatgpt100k_bert_large_ufet_type_prop_vocab_embeds.txt
#SBATCH --error=logs/ufet_exp/err_ufet_cnetp_chatgpt100k_bert_large_ufet_type_prop_vocab_embeds.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=16g
#SBATCH --gres=gpu:1

#SBATCH -t 0-02:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ufet_get_type_embed_bert_large_cnetp_chatgpt100k_pretrained_model.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ufet_get_cnetp_prop_vocab_embed_bert_large_cnetp_chatgpt100k_pretrained_model.json

echo 'Job Finished !!!'
