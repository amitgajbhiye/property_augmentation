#!/bin/bash --login

#SBATCH --job-name=10chat100kPretrain

#SBATCH --output=logs/chatgpt_pretrain/out_bienc_bb_chatgpt100k_pretrain_patience10.txt
#SBATCH --error=logs/chatgpt_pretrain/err_bienc_bb_chatgpt100k_pretrain_patience10.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100

#SBATCH --gres=gpu:1
#SBATCH --mem=16G

#SBATCH --time 2-00:00:00

conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/bienc_bb_chatgpt100k_pretrain_patience10.json

echo 'Job Finished!'
