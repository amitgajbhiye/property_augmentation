#!/bin/bash --login

#SBATCH --job-name=entColing

#SBATCH --output=logs/food_taste/out_entropy_coling_best_food_taste.txt
#SBATCH --error=logs/food_taste/err_entropy_coling_best_food_taste.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 0-00:30:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/food_taste/9_bienc_entropy_bert_large_coling_best_adj_taste.json
python3 get_embeds_and_train_data.py --config_file configs/food_taste/10_bienc_entropy_bert_large_coling_best.json

echo 'Job Finished !!!'
