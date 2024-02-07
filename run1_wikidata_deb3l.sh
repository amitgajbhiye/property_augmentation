#!/bin/bash --login

#SBATCH --job-name=wikidata

#SBATCH --output=logs/wikidata/out_all_wikidata.txt
#SBATCH --error=logs/wikidata/err_all_wikidata.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=20G
#SBATCH -t 0-02:00:00

conda activate venv

python3 for_llm_ranking_je_filter_similar_props.py --config_file configs/wikidata/wikidata_mcrae_cslb_deb3l.json


echo 'Job Finished !!!'
