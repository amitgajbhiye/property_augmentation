#!/bin/bash --login

#SBATCH --job-name=Mc3simConOnly

#SBATCH --output=logs/data_sampling/out_mcrae_get_deberta_nli_con_only_3sim_mscg_ncetp_prop_conj.txt
#SBATCH --error=logs/data_sampling/err_mcrae_get_deberta_nli_con_only_3sim_mscg_ncetp_prop_conj.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

##SBATCH --partition gpu_v100,gpu
##SBATCH --mem=8g
##SBATCH --gres=gpu:1

#SBATCH --partition compute
#SBATCH --time 0-5:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mcrae_ps_get_deberta_nli_con_only_3sim_mscg_ncetp_prop_conj.json

echo 'Job Finished!'