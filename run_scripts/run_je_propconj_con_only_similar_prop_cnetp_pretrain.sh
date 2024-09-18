#!/bin/bash --login

#SBATCH --job-name=conOnlyPretrain

#SBATCH --output=logs/pretrain/out_run_je_propconj_con_only_similar_prop_cnetp_pretrain.txt
#SBATCH --error=logs/pretrain/err_run_je_propconj_con_only_similar_prop_cnetp_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:1
#SBATCH --time 1-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/1_configs_pretrain/je_propconj_con_only_similar_prop_cnetp_pretrain.json

echo 'Job Finished!'
