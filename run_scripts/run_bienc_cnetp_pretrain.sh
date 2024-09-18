#!/bin/bash --login

#SBATCH --job-name=PtrCnetP

#SBATCH --output=logs/pretrain/out_bienc_bb_cnetp_5neg_pretrain.txt
#SBATCH --error=logs/pretrain/err_bienc_bb_cnetp_5neg_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH --qos=gpu7d
#SBATCH --time 7-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 bienc_run_model.py --config_file configs/1_configs_pretrain/bienc_bb_cnetp.json

echo 'finished!'