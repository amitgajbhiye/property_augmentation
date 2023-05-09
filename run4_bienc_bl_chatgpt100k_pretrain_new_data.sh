#!/bin/bash --login

#SBATCH --job-name=blChatPre

#SBATCH --output=logs/chatgpt_pretrain/out_bienc_bert_large_chatgpt100k_pretrain_new_data.txt
#SBATCH --error=logs/chatgpt_pretrain/err_bienc_bert_large_chatgpt100k_pretrain_new_data.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

#SBATCH --time 2-00:00:00

conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/bienc_bert_large_chatgpt100k_pretrain_new_data.json

echo 'Job Finished!'
