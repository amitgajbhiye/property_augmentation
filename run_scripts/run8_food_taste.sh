#!/bin/bash --login

#SBATCH --job-name=debFoodTaste

#SBATCH --output=logs/food_taste/mcrae_cslb_debv3l_filter.txt
#SBATCH --error=logs/food_taste/mcrae_cslb_debv3l_filter.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 0-2:00:00

conda activate venv

python3 je_filter_similar_props.py --config_file configs/food_taste/1_mcrae_cslb_deb3l_filter_food_taste.json
python3 je_filter_similar_props.py --config_file configs/food_taste/2_mcrae_cslb_deb3l_filter_food_adjective_taste.json

echo 'Job Finished !!!'
