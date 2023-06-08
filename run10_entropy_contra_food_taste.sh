#!/bin/bash --login

#SBATCH --job-name=entCnetp

#SBATCH --output=logs/food_taste/out_cnetp_vocab_pt_bert_large_entropy_food_taste.txt
#SBATCH --error=logs/food_taste/err_cnetp_vocab_pt_bert_large_entropy_food_taste.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 0-00:30:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/food_taste/7_bienc_entropy_bert_large_cnetp.json
python3 get_embeds_and_train_data.py --config_file configs/food_taste/8_bienc_entropy_bert_large_cnetp_adj_taste.json

echo 'Job Finished !!!'
