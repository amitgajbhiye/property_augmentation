#!/bin/bash --login

#SBATCH --job-name=Mc3simFt

#SBATCH --output=logs/mcrae_finetune/out_mcrae_ps_deberta_nli_con_only_3sim_prop_conj_mcrae_finetune.txt
#SBATCH --error=logs/mcrae_finetune/err_mcrae_ps_deberta_nli_con_only_3sim_prop_conj_mcrae_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu
#SBATCH --mem=9G
#SBATCH --gres=gpu:1

#SBATCH --time 0-5:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/3_finetune/ps_deberta_nli_con_only_3sim_prop_conj_mcrae_finetune.json

echo 'Job Finished !!!'
