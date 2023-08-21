#!/bin/bash --login

#SBATCH --job-name=sumoOneHot

#SBATCH --output=logs/ontology_completion/sumo/out_get_one_hot_enc.txt
#SBATCH --error=logs/ontology_completion/sumo/err_get_one_hot_enc.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem

#SBATCH --mem=100G
#SBATCH -t 0-04:00:00

conda activate venv

python3 one_hot_converter.py

echo 'Job Finished !!!'