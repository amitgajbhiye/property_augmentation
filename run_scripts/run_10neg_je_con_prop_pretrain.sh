#!/bin/bash --login

#SBATCH --job-name=JEcp10Np

#SBATCH --output=logs/pretrain/out_je_conprop_bb_cnetp_10neg_pretrain.txt
#SBATCH --error=logs/pretrain/err_je_conprop_bb_cnetp_10neg_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH --qos="gpu7d"

#SBATCH -t 6-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_con_prop.py --config_file configs/1_configs_pretrain/je_conprop_bb_cnetp_10neg.json

echo 'Job Finished!'