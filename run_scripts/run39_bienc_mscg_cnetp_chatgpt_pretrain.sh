#!/bin/bash --login

#SBATCH --job-name=blMscgCnetpChatGPT_pretrain

#SBATCH --output=logs/mscg_cnetp_chatgpt_pretrain/out_bienc_bert_large_mscg_cnetp_chatgpt100k_pretrain.txt
#SBATCH --error=logs/mscg_cnetp_chatgpt_pretrain/err_bienc_bert_large_mscg_cnetp_chatgpt100k_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --mem=25G

#SBATCH --time 2-00:00:00

conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/bienc_bert_large_mscg_cnetp_chatgpt100k_pretrain.json

echo 'Job Finished !!!'