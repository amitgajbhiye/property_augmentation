#!/bin/bash --login

#SBATCH --job-name=580McFT

#SBATCH --output=logs/mcrae_finetune/out_mcrae_5neg80thres_je_prop_conj_prop_split_finetune.txt
#SBATCH --error=logs/mcrae_finetune/err_mcrae_5neg80thres_je_prop_conj_prop_split_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=9G
#SBATCH --gres=gpu:1

#SBATCH --time 0-5:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/3_finetune/ps_5neg80thres_prop_conj_je_mcrae_fintune.json

echo 'Job Finished !!!'