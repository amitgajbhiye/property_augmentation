#!/bin/bash --login

#SBATCH --job-name=McConSim50_mscg_cnet_pretrained_model

#SBATCH --output=logs/data_sampling/out_mcrae_get_con_50sim_vocab_mscg_cnept_prop.txt
#SBATCH --error=logs/data_sampling/err_mcrae_get_con_50sim_vocab_mscg_cnept_prop.txt

#SBATCH --tasks-per-node=1
#SBATCH --ntasks=1
#SBATCH -A scw1858

#SBATCH -p gpu
#SBATCH --mem=6g
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/mcrae_get_con_50sim_vocab_mscg_cnept_prop.json

echo 'Job Finished!'
