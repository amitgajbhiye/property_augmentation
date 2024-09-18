#!/bin/bash --login

#SBATCH --job-name=blMcFtlr15

#SBATCH --output=logs/bienc_mcrae_fine_tune/out_ps_bienc_bert_large_mcrae_cnetp_lr15Epoch50_pretrained_model_ftlr15.txt
#SBATCH --error=logs/bienc_mcrae_fine_tune/err_ps_bienc_bert_large_mcrae_cnetp_lr15Epoch50_pretrained_model_ftlr15.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=16g
#SBATCH --gres=gpu:1

#SBATCH -t 0-8:00:00

conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/ps_bienc_bert_large_mcrae_cnetp_lr15Epoch50_pretrained_model_ftlr15.json

echo 'Job Finished !!!'