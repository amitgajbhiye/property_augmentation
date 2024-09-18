#!/bin/bash --login

#SBATCH --job-name=deb_no_bitter_sour_no_adj

#SBATCH --output=logs/je_deberta_no_mcrae_data_ablation/out_deb_no_bitter_sour_no_adj.txt
#SBATCH --error=logs/je_deberta_no_mcrae_data_ablation/err_deb_no_bitter_sour_no_adj.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 0-01:00:00

conda activate venv

python3 je_filter_similar_props.py --config_file configs/food_taste/2_no_mcrae_cslb_overlap_deb3l_filter_food_bitter_taste.json
python3 je_filter_similar_props.py --config_file configs/food_taste/2_no_mcrae_cslb_overlap_deb3l_filter_food_sour_taste.json

echo "Job Finished !!!"



