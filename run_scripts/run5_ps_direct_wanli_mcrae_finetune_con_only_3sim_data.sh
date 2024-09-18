#!/bin/bash --login

#SBATCH --job-name=WftCS

#SBATCH --output=logs/mcrae_finetune/out_ps_mcrae_wanli_pre_hypo_pretrained_je_finetuned_on_mcrae_con_only_3sim_data.txt
#SBATCH --error=logs/mcrae_finetune/err_ps_mcrae_wanli_pre_hypo_pretrained_je_finetuned_on_mcrae_con_only_3sim_data.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu
#SBATCH --mem=9G
#SBATCH --gres=gpu:1

#SBATCH --time 0-5:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/3_finetune/ps_mcrae_wanli_pre_hypo_pretrained_je_finetuned_on_mcrae_con_only_3sim_data.json

echo 'Job Finished !!!'
