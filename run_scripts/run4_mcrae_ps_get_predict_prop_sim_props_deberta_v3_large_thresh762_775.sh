#!/bin/bash --login

#SBATCH --job-name=dbv3_thres762_775

#SBATCH --output=logs/mcrae_clean_vocab_data/out_mcrae_ps_get_predict_prop_sim_props_deberta_v3_large_thresh762_775.txt
#SBATCH --error=logs/mcrae_clean_vocab_data/err_mcrae_ps_get_predict_prop_sim_props_deberta_v3_large_thresh762_775.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu,gpu_v100
#SBATCH --mem=6g
#SBATCH --gres=gpu:1

#SBATCH --time 0-03:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mcrae_data_clean_vocab/mcrae_ps_get_predict_prop_sim_props_deberta_v3_large_thresh762.json
python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mcrae_data_clean_vocab/mcrae_ps_get_predict_prop_sim_props_deberta_v3_large_thresh775.json

echo 'Job Finished!'
