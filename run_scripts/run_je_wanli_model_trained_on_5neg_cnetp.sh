#!/bin/bash --login

#SBATCH --job-name=WaNLICnetPretrain

#SBATCH --output=logs/pretrain/out_je_wanli_model_trained_on_5neg_cnetp.txt
#SBATCH --error=logs/pretrain/err_je_je_wanli_model_trained_on_5neg_cnetp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=15g
#SBATCH --gres=gpu:1

##SBATCH --qos="gpu7d"

#SBATCH -t 01-20:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_con_prop.py --config_file configs/1_configs_pretrain/je_wanli_model_trained_on_5neg_cnetp.json

echo 'Job Finished!'