#!/bin/bash --login

#SBATCH --job-name=CnetChatPtrnGrid

#SBATCH --output=logs/cnetp_chatgpt_pretrain/out_bienc_grid_search_bb_cnetp_chatgpt20k_pretrain.txt
#SBATCH --error=logs/cnetp_chatgpt_pretrain/err_bienc_grid_search_bb_cnetp_chatgpt20k_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

#SBATCH --time 2-00:00:00

conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/bienc_grid_search_bb_cnetp_chatgpt20k_pretrain.json

echo 'Job Finished!'
