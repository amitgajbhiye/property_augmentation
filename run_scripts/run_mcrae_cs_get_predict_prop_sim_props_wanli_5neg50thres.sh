#!/bin/bash --login

#SBATCH --job-name=mcCSwanli_5neg50thres

#SBATCH --output=logs/data_sampling/out_mcrae_cs_get_predict_prop_sim_props_wanli_5neg50thres.txt
#SBATCH --error=logs/data_sampling/err_mcrae_cs_get_predict_prop_sim_props_wanli_5neg50thres.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu_v100,gpu
##SBATCH --mem=8g
##SBATCH --gres=gpu:1

#SBATCH -p compute
#SBATCH -t 0-2:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/5_get_predict_prop_sim_data/mcrae_cs_get_predict_prop_sim_props_wanli_5neg50thres.json

echo 'Job Finished!'
