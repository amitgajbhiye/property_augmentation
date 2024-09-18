#!/bin/bash --login

#SBATCH --job-name=5neg80thres

#SBATCH --output=logs/data_sampling/out_mcrae_get_predict_prop_sim_props_5neg80thres.txt
#SBATCH --error=logs/data_sampling/err_mcrae_get_predict_prop_sim_props_5neg80thres.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100,gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:1

#SBATCH --time 0-4:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/5_get_predict_prop_sim_data/mcrae_get_predict_prop_sim_props_5neg80thres.json

echo 'Job Finished!'