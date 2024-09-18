#!/bin/bash --login

#SBATCH --job-name=3SimPP

#SBATCH --output=logs/data_sampling/out_get_pretrain_debert_nli_mscg_cnetp_predict_property_3sim_prop_conj_data.txt
#SBATCH --error=logs/data_sampling/err_get_pretrain_debert_nli_mscg_cnetp_predict_property_3sim_prop_conj_data.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

##SBATCH --partition gpu_v100,gpu
##SBATCH --mem=8g
##SBATCH --gres=gpu:1

#SBATCH --partition compute
#SBATCH --time 0-10:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/5_get_predict_prop_sim_data/get_predict_prop_sim_3props_mscg_cnetp_deberta_nli.json

echo 'Job Finished!'