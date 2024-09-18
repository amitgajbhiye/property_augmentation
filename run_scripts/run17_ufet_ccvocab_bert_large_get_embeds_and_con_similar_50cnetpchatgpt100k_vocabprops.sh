#!/bin/bash --login

#SBATCH --job-name=ufetCcVOcab_type_prop_similatdata

#SBATCH --output=logs/ufet_exp/entropy_ccvocab/out_ccvocab_ufet_bert_large_get_type_propvocab_embeds_and_con_similar_50cnetchatgpt_vocab_props.txt
#SBATCH --error=logs/ufet_exp/entropy_ccvocab/err_ccvocab_ufet_bert_large_get_type_propvocab_embeds_and_con_similar_50cnetchatgpt_vocab_props.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-02:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/1_ccvocab_get_type_embed_entropy_bert_large_cnetp_chatgpt100k_pt_model.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/2_ccvocab_get_vocab_prop_embed_entropy_bert_large_cnetp_chatgpt100k_pt_model.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/ufet/ce_cnetp_chatgpt_vocab/3_ccvocab_get_type_sim50_vocabprop_entropy_bert_large_cnetp_chatgpt100k_pt_model.json

echo 'Job Finished !!!'
