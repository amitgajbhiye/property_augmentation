#!/bin/bash --login

#SBATCH --job-name=hpPPCT3ft

#SBATCH --output=logs/mcrae_finetune/out_roberta_base_hp_mlm_ctx3_tuning_ps_deberta_nli_predict_prop_3sim_prop_conj_mcrae_finetune.txt
#SBATCH --error=logs/mcrae_finetune/err_roberta_base_hp_mlm_ctx3_tuning_ps_deberta_nli_predict_prop_3sim_prop_conj_mcrae_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu
#SBATCH --mem=9G
#SBATCH --gres=gpu:1

#SBATCH --time 2-00:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 model/roberta_prop_aug.py --config_file configs/3_finetune/roberta_base_hp_mlm_ctx3_tuning_ps_deberta_nli_predict_prop_3sim_prop_conj_mcrae_finetune.json

echo 'Job Finished !!!'
