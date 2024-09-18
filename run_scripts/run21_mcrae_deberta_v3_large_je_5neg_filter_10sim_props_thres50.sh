#!/bin/bash --login

#SBATCH --job-name=McFilterCnetpDeberta 

#SBATCH --output=logs/mcrae_clean_vocab_data/out_mcrae_deberta_v3_large_je_5neg_filter_10sim_props_thres50.txt
#SBATCH --error=logs/mcrae_clean_vocab_data/err_mcrae_deberta_v3_large_je_5neg_filter_10sim_props_thres50.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -t 0-00:45:00

conda activate venv

python3 je_filter_similar_props.py --config_file configs/2_configs_get_embeds_and_train_data/mcrae_data_clean_vocab/mcrae_deberta_v3_large_je_5neg_filter_10sim_props_thres50.json

echo 'Job Finished!'