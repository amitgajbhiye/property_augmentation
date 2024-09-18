#!/bin/bash --login

#SBATCH --job-name=MscgCnetPConSim50Props

#SBATCH --output=logs/data_sampling/out_get_mscg_cnet_con_50sim_vocab_prop.txt
#SBATCH --error=logs/data_sampling/err_get_mscg_cnet_con_50sim_vocab_prop.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:1

#SBATCH -t 0-05:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/4_get_mscg_cnet_con_50sim_vocab_prop.json

echo 'Job Finished!'