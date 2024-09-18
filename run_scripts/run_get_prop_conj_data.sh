#!/bin/bash --login

#SBATCH --job-name=10nPT1050

#SBATCH --output=logs/data_sampling/out_get_predict_prop_sim_vocab_props_pt10neg50thres_10neg_inputdata.txt
#SBATCH --error=logs/data_sampling/err_get_predict_prop_sim_vocab_props_pt10neg50thres_10neg_inputdata.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu_v100,gpu
##SBATCH --mem=8g
##SBATCH --gres=gpu:1

#SBATCH -p compute
#SBATCH -t 0-20:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/5_get_predict_prop_sim_data/get_predict_prop_sim_props_10neg50thresh_10neg_inputdata.json


echo 'Job Finished!'