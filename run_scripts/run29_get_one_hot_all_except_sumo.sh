#!/bin/bash --login

#SBATCH --job-name=ExSumoOneHot

#SBATCH --output=logs/ontology_completion/out_get_one_hot_all_except_sumo.txt
#SBATCH --error=logs/ontology_completion/err_get_one_hot_all_except_sumo.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute,highmem

#SBATCH --mem=100G
#SBATCH -t 3-00:00:00

conda activate venv

python3 one_hot_converter.py

echo 'Job Finished !!!'