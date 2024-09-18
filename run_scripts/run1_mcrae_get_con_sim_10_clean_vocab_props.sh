#!/bin/bash --login

#SBATCH --job-name=getConSim10Props

#SBATCH --output=logs/mcrae_clean_vocab_data/out_mcrae_get_con_sim_10_clean_vocab_props.txt
#SBATCH --error=logs/mcrae_clean_vocab_data/err_mcrae_get_con_sim_10_clean_vocab_props.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu
##SBATCH --gres=gpu:1

#SBATCH -p compute
#SBATCH --mem=8G
#SBATCH -t 0-02:00:00

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mcrae_data_clean_vocab/mcrae_con_sim_10_clean_vocab_props.json

echo 'Job Finished!'