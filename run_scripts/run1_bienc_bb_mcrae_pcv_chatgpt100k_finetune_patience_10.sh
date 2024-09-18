#!/bin/bash --login

#SBATCH --job-name=10McChatGptFT

#SBATCH --output=logs/chatgpt_finetune/out_bienc_bb_mcrae_pcv_chatgpt100k_finetune_patience_10.txt
#SBATCH --error=logs/chatgpt_finetune/err_bienc_bb_mcrae_pcv_chatgpt100k_finetune_patience_10.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 0-06:00:00

conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/bienc_bb_mcrae_pcv_chatgpt100k_finetune_config_patience_10.json

echo 'finished!'