#!/bin/bash --login

#SBATCH --job-name=conSim15Emb

#SBATCH --output=logs/data_sampling/out_pretrain_cnetp_get_cnet_con_15sim_vocab_prop.txt
#SBATCH --error=logs/data_sampling/err_pretrain_cnetp_get_cnet_con_15sim_vocab_prop.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu_v100,gpu
##SBATCH --mem=8g
##SBATCH --gres=gpu:1

#SBATCH -p compute
#SBATCH -t 0-05:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/2_configs_get_embeds_and_train_data/5_get_cnet_con_15sim_vocab_prop.json


echo 'finished!'