#!/bin/bash --login

#SBATCH --job-name=entContra

#SBATCH --output=logs/food_taste/out_entropy_contra_food_taste.txt
#SBATCH --error=logs/food_taste/err_entropy_contra_food_taste.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 0-1:30:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/food_taste/3_bienc_entropy_bert_large_cnetpchatgpt.json
python3 get_embeds_and_train_data.py --config_file configs/food_taste/4_food_adj_taste_bienc_entropy_bert_large_cnetpchatgpt.json

python3 get_embeds_and_train_data.py --config_file configs/food_taste/5_bienc_contra_bert_large_cnetpchatgpt.json
python3 get_embeds_and_train_data.py --config_file configs/food_taste/6_food_adj_taste_bienc_contra_bert_large_cnetpchatgpt.json

echo 'Job Finished !!!'
