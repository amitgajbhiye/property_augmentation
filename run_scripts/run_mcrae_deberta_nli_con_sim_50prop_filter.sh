#!/bin/bash --login

#SBATCH --job-name=MCnliFilSim

#SBATCH --output=logs/data_sampling/out_mcrae_deberta_nli_con_sim_50prop_filter.txt
#SBATCH --error=logs/data_sampling/err_mcrae_deberta_nli_con_sim_50prop_filter.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH --account scw1858

#SBATCH -p gpu
#SBATCH --mem=25G
##SBATCH --exclusive

#SBATCH --gres=gpu:1

#SBATCH -t 0-03:00:00

echo 'This script is running on:' hostname

conda activate venv

python3 je_filter_similar_props.py --config_file configs/4_filter_sim_props/mcrae_deberta_nli_con_sim_50prop_filter.json

echo 'Job Finished!'
