#!/bin/bash --login

#SBATCH --job-name=sft_llama2

#SBATCH --output=logs/commonality/out_sft_llama2_props.txt
#SBATCH --error=logs/commonality/err_sft_llama2_props.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --gres=gpu:1
##SBATCH --exclusive

#SBATCH --mem=20G
#SBATCH -t 0-02:00:00

conda activate venv

python3 for_llm_ranking_je_filter_similar_props.py --config_file configs/commonality_llama/1_filter_llama2_props_mcrae_cslb_deb3l.json

echo 'Job Finished !!!'
