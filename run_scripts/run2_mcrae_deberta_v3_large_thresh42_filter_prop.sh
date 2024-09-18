#!/bin/bash --login

#SBATCH --job-name=dbv342thresMcRae

#SBATCH --output=logs/data_sampling/out_mcrae_deberta_v3_large_je_5neg_filter_sim_props_thres42.txt
#SBATCH --error=logs/data_sampling/err_mcrae_deberta_v3_large_je_5neg_filter_sim_props_thres42.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --time 0-04:00:00

conda activate venv

python3 je_filter_similar_props.py --config_file configs/4_filter_sim_props/mcrae_deberta_v3_large_je_5neg_filter_sim_props_thres42.json


echo 'Job Finished!'
