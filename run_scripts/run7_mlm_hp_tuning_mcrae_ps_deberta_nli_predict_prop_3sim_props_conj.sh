#!/bin/bash --login

#SBATCH --job-name=MLMhpt3pp

#SBATCH --output=logs/mcrae_finetune/out_hp_mlm_tuning_ps_deberta_nli_predict_prop_3sim_prop_conj_mcrae_finetune.txt
#SBATCH --error=logs/mcrae_finetune/err_hp_mlm_tuning_ps_deberta_nli_predict_prop_3sim_prop_conj_mcrae_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu
#SBATCH --mem=9G
#SBATCH --gres=gpu:1

#SBATCH --time 2-00:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 model/je_property_conjuction.py --config_file configs/3_finetune/hp_mlm_tuning_ps_deberta_nli_predict_prop_3sim_prop_conj_mcrae_finetune.json

echo 'Job Finished !!!'
