#!/bin/bash --login

#SBATCH --job-name=BLpCP26

#SBATCH --output=logs/pretrain/out_bienc_bert_large_cnetp_pretrain_lr26.txt
#SBATCH --error=logs/pretrain/err_bienc_bert_large_cnetp_pretrain_lr26.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=16g
#SBATCH --gres=gpu:1

#SBATCH --time 1-00:00:00

##SBATCH --qos=gu7d

conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/bienc_bert_large_cnetp_pretrain_lr26.json

echo 'Job Finished!'
