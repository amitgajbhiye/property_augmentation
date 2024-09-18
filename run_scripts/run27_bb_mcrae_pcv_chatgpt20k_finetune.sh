#!/bin/bash --login

#SBATCH --job-name=G2McChatGptFT

#SBATCH --output=logs/chatgpt_finetune/out_bienc_grid2_bestmodel_bb_mcrae_pcv_cnetp_chatgpt20k_finetune_config.txt
#SBATCH --error=logs/chatgpt_finetune/err_bienc_grid2_bestmodel_bb_mcrae_pcv_cnetp_chatgpt20k_finetune_config.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 0-06:00:00

conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/bienc_grid2_bestmodel_bb_mcrae_pcv_cnetp_chatgpt20k_finetune_config.json

echo 'Job Finished!'