#!/bin/bash --login

#SBATCH --job-name=biPSmr

#SBATCH --output=logs/mcrae_finetune/out_mcrae_ps_bienc_cnetp.txt
#SBATCH --error=logs/mcrae_finetune/err_mcrae_ps_bienc_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH --time 0-10:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 bienc_fine_tune.py --config_file configs/3_finetune/bienc_ps_mcrae_cnetp_pretrained_model.json

echo 'finished!'