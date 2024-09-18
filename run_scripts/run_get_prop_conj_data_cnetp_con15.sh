#!/bin/bash --login

#SBATCH --job-name=con15Conj

#SBATCH --output=logs/data_sampling/out_pretrain_cnetp_get_predict_prop_sim_con15_props.txt
#SBATCH --error=logs/data_sampling/err_pretrain_cnetp_get_predict_prop_sim_con15_props.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

##SBATCH --partition gpu_v100,gpu
##SBATCH --mem=8g
##SBATCH --gres=gpu:1

#SBATCH --partition compute
#SBATCH --time 0-12:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/5_get_predict_prop_sim_data/get_predict_prop_sim_con15_props.json

echo 'Job Finished!'