#!/bin/bash --login

#SBATCH --job-name=dbV3

#SBATCH --output=logs/data_sampling/out_deberta_v3_large_je_5neg_filter_sim_props_thres40.txt
#SBATCH --error=logs/data_sampling/err_deberta_v3_large_je_5neg_filter_sim_props_thres40.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH --partition gpu_v100
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

#SBATCH --time 0-02:30:00

conda activate venv

python3 je_filter_similar_props.py --config_file configs/4_filter_sim_props/deberta_v3_large_je_5neg_filter_sim_props_thres40.json

echo 'Job Finished!'
