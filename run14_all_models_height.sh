#!/bin/bash --login

#SBATCH --job-name=heightAllModels

#SBATCH --output=logs/height/out_height_all_models.txt
#SBATCH --error=logs/height/err_height_all_models.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 0-01:00:00

conda activate venv


python3 je_filter_similar_props.py --config_file configs/height/1_mcrae_cslb_deb3l.json

python3 get_embeds_and_train_data.py --config_file configs/height/2_bienc_entropy_bert_large_cnetpchatgpt.json
python3 get_embeds_and_train_data.py --config_file configs/height/3_bienc_contra_conceptfix_bert_large_cnetpchatgpt.json
python3 get_embeds_and_train_data.py --config_file configs/height/4_bienc_entropy_bert_large_cnetp.json
python3 get_embeds_and_train_data.py --config_file configs/height/5_bienc_entropy_bert_large_coling_best.json
python3 get_embeds_and_train_data.py --config_file configs/height/6_bienc_entropy_bert_large_contra_propfix.json
python3 get_embeds_and_train_data.py --config_file configs/height/7_bienc_entropy_bert_large_chatgpt100k.json

echo 'Job Finished !!!'
