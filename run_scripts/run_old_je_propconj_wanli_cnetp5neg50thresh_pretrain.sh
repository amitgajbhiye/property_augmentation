#!/bin/bash --login

#SBATCH --job-name=oldPT

#SBATCH --output=logs/pretrain/out_old_je_propconj_wanli_cnetp5neg50thresh_pretrain.txt
#SBATCH --error=logs/pretrain/err_old_je_propconj_wanli_cnetp5neg50thresh_pretrain.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858
#SBATCH --partition gpu_v100

#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH --time 2-00:00:00

##SBATCH --qos="gpu7d"

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/1_configs_pretrain/old_je_propconj_wanli_cnetp5neg50thresh_pretrain.json

echo 'Job Finished!'