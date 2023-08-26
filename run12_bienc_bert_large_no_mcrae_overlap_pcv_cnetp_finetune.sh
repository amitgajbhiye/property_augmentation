#!/bin/bash --login

#SBATCH --job-name=blCnetP_No_mcrae_overlap_finetune

#SBATCH --output=logs/bienc_mcrae_fine_tune/out_bert_large_no_mcrae_overlap_pcv_cnetp_finetune.txt
#SBATCH --error=logs/bienc_mcrae_fine_tune/err_bert_large_no_mcrae_overlap_pcv_cnetp_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=16g
#SBATCH --gres=gpu:1

#SBATCH -t 0-15:00:00

conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/bert_large_no_mcrae_overlap_pcv_cnetp_finetune_config.json

echo 'Job Finished !!!'