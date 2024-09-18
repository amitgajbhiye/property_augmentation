#!/bin/bash --login

#SBATCH --job-name=conOnlyMcFT

#SBATCH --output=logs/mcrae_finetune/out_ps_con_only_5sim_props_conj_je_mcrae_fintune.txt
#SBATCH --error=logs/mcrae_finetune/err_ps_con_only_5sim_props_conj_je_mcrae_fintune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu
#SBATCH --mem=9G
#SBATCH --gres=gpu:1

#SBATCH --time 0-5:00:00

echo 'This script is running on:'
hostname

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/3_finetune/ps_con_only_5sim_props_conj_je_mcrae_fintune.json

echo 'Job Finished !!!'