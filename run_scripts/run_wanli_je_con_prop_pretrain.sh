#!/bin/bash --login

#SBATCH --job-name=WnliPTph

#SBATCH --output=logs/pretrain/out_je_pre_hypo_wanli_bb_pretrain.txt
#SBATCH --error=logs/pretrain/err_je_pre_hypo_wanli_bb_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100,gpu
#SBATCH --mem=20g
#SBATCH --gres=gpu:1

##SBATCH --qos="gpu7d"

#SBATCH -t 01-15:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_con_prop.py --config_file configs/1_configs_pretrain/je_pre_hypo_wanli_bb_pretrain.json

echo 'Job Finished!'