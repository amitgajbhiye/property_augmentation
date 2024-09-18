#!/bin/bash --login

#SBATCH --job-name=bllr15McFt

#SBATCH --output=logs/bienc_mcrae_fine_tune/out_ps_bienc_bert_large_mcrae_cnetp_lr15_pretrained_model.txt
#SBATCH --error=logs/bienc_mcrae_fine_tune/err_ps_bienc_bert_large_mcrae_cnetp_lr15_pretrained_model.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=16g
#SBATCH --gres=gpu:1

#SBATCH -t 0-8:00:00


conda activate venv


python3 bienc_fine_tune.py --config_file configs/3_finetune/ps_bienc_bert_large_mcrae_cnetp_lr15_pretrained_model.json

echo 'Job Finished !!!'